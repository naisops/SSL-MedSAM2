# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 18:24:00 2025

@author: naisops
"""

import os
import numpy as np
import copy
import torch
from collections import defaultdict
from pathlib import Path
from scipy.special import expit
import SimpleITK as sitk
import re
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.ndimage as ndi
import nibabel as nib
from skimage import measure

'''
SAM 2 treats each obj_id as its own binary classification task. 
It produces C separate mask-logit maps—even when you track multiple objects 
in one pass—without inter-object coupling 
'''


def add_prompts(predictor, inference_state,
                mask_3d,  # in (d,H,W)
                prompt_ponits: dict,
                prompt_bbox: dict,
                cl_lst=None, # list of foreground class id [1,2,...]
                ):
    """
        prompt_points dict:{
            sl:{
                cl:[
                    [(x,y),label], ...
                    ]
                }
            }
        prompt_bbox dict:{
            sl:{
                cl:[x_min, y_min, x_max, y_max]
                }
            }

        Note: the prompts and masks use absolute pixel coords relative to the original scan (before resizing)
        The mask should also be its original size
        """

    # add mask
    # print(f'mask index in adding prompt:{np.unique(mask_3d)}')
    pt_frames = []
    for cl in cl_lst:
        if mask_3d is not None:
            if not isinstance(mask_3d, torch.Tensor):
                mask_3d = torch.from_numpy(mask_3d)
            for sl in range(mask_3d.size(0)):
                mask_cl = (mask_3d[sl] == cl).int()
                if torch.sum(mask_cl).item() > 0:
                    predictor.add_new_mask(inference_state, frame_idx=sl, obj_id=cl, mask=mask_cl)
                    pt_frames.append(sl)
                    # if sl == mask_3d.size(0)//2:
                    #     plt.imshow(mask_cl)
                    #     plt.axis("off")
                    #     plt.savefig(f"mask_sl{sl}_cl{cl}.png", bbox_inches="tight", dpi=150)

        # add points
        if prompt_ponits is not None:
            for sl in prompt_ponits:
                if cl-1 in prompt_ponits[sl]:
                    points = [p[0] for p in prompt_ponits[sl][cl-1]]
                    labels = [p[1] for p in prompt_ponits[sl][cl-1]]
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=sl,
                        obj_id=cl,
                        points=points,
                        labels=labels,
                    )
                    pt_frames.append(sl)

        # add bbox
        if prompt_bbox is not None:
            for sl in prompt_bbox:
                if cl-1 in prompt_bbox[sl]:
                    bbox = prompt_bbox[sl][cl-1]
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=sl,
                        obj_id=cl,
                        box=bbox,
                    )
                    pt_frames.append(sl)

    return predictor, inference_state, pt_frames


def run_propagate(predictor, inference_state, segs_lst, pt_frames, cl_lst:list = None, reverse=False):
    # return list of logits in [(c,h,w)]
    out_segs = copy.deepcopy(segs_lst)
    min_pt_fm = min(pt_frames)
    max_pt_fm = max(pt_frames)

    if reverse:
        start_frame_idx = max_pt_fm
    else:
        start_frame_idx = min_pt_fm

    objid_to_idx = inference_state['obj_id_to_idx']
    for out_frame_idx, out_obj_ids, out_mask_logits in \
            predictor.propagate_in_video(inference_state, start_frame_idx=start_frame_idx, reverse=reverse):
        if len(out_obj_ids) != len(cl_lst):
            zero_logits = torch.zeros_like(out_mask_logits[0,0])
            logits_stack = [
                out_mask_logits[objid_to_idx[cl], 0] if cl in out_obj_ids else zero_logits
                for cl in cl_lst
            ]
            logits_stack = torch.stack(logits_stack, dim=0)
            out_segs[out_frame_idx] = logits_stack
        else:
            out_segs[out_frame_idx] = out_mask_logits[:,0]

    return out_segs  # [(c,h,w)]


def multi_class_mask_slice(probs):
    # probs in (C,H,W)
    max_idx = np.argmax(probs, axis=0)  # shape (H, W)
    out = np.zeros_like(probs)  # shape (C, H, W)

    h_idx, w_idx = np.indices(max_idx.shape)
    out[max_idx, h_idx, w_idx] = probs[max_idx, h_idx, w_idx]
    out = (out > 0.5).astype(np.uint8)  # shape (C, H, W)

    outmask = np.zeros_like(out[0], dtype=np.uint8)  # shape (H, W)
    for cl in range(out.shape[0]):
        outmask += (cl + 1) * out[cl]

    return outmask  # (H,W)

def probs_to_binary(probs):
    # probs in (d,C,H,W)
    max_idx = np.argmax(probs, axis=1)  # shape (d, H, W)
    out = np.zeros_like(probs)  # shape (d, C, H, W)

    d_idx, h_idx, w_idx = np.indices(max_idx.shape)
    out[d_idx, max_idx, h_idx, w_idx] = probs[d_idx, max_idx, h_idx, w_idx]
    out = (out > 0.5).astype(np.uint8)  # shape (d, C, H, W)

    return out  # (d,c,H,W)

def multichannel_binary_to_mask(binary_mask):
    outmask = np.zeros_like(binary_mask[:,0], dtype=np.uint8)  # shape (d,H, W)
    for cl in range(binary_mask.shape[1]):
        outmask += (cl + 1) * binary_mask[:,cl]

    return outmask


# get prediction based on a 3D volume where only some slices have masks
def generate_pseudo_logits(volume,  # shape is (d, 3, image_size, image_size)
                           predictor,
                           video_height: int,
                           video_width: int,
                           cl:list =None,
                           mask_3d=None,  # shape is (d, video_width, video_height)
                           prompt_ponits: dict = None,  # dict of {'slice': [[(x,y),label], ...]}
                           prompt_bbox: dict = None,  # dict of {'slice': [[(), ()]]}
                           ):
    # return the pseudo logits in (d,c,h,w)

    segs_3D_cl = np.zeros((volume.size(0),len(cl),video_height, video_width), dtype=np.float32)  # will contain the pred logts for the entile volume
    segs_3D_cl = torch.from_numpy(segs_3D_cl).to('cuda')  # (d,h,w)

    ## run the predictor for each class one time (save memory)
    segs_lst = [None for i in range(volume.size(0))]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(volume, video_height, video_width)
        predictor, inference_state, pt_frames = add_prompts(predictor, inference_state,
                                                            mask_3d, prompt_ponits, prompt_bbox, cl_lst = cl)
        out_segs_forward = run_propagate(predictor, inference_state, segs_lst, pt_frames,
                                         cl_lst=cl, reverse=False)  # [(c,h,w)]

        predictor.reset_state(inference_state)
        inference_state = predictor.init_state(volume, video_height, video_width)
        predictor, inference_state, pt_frames = add_prompts(predictor, inference_state,
                                                            mask_3d, prompt_ponits, prompt_bbox, cl_lst = cl)
        out_segs_backward = run_propagate(predictor, inference_state, segs_lst, pt_frames,
                                          cl_lst=cl, reverse=True)  # [(c,h,w)]
        predictor.reset_state(inference_state)

    for sl in range(volume.size(0)):
        if out_segs_forward[sl] is None or out_segs_backward[sl] is None:
            segs_3D_cl[sl] = out_segs_forward[sl] if out_segs_backward[sl] is None else out_segs_backward[sl]
        else:
            segs_3D_cl[sl] = (out_segs_forward[sl] + out_segs_backward[sl]) / 2

    return segs_3D_cl  # (d,c,h,w)


def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def generate_uncert_and_pred(
        objID, ref_img_nii, n_slices,
        ori_size=(0, 0),
        img_size=512,
        pred_npz_dir: str = '',
        uncertainty_dir: str = '',
        pred_dir: str = '',
        only_largest_comp = True,
        save_multi_channel_mask = False,
):
    # Compute final predictions and uncertainty
    nii_files = os.listdir(pred_npz_dir)
    cl_groups = defaultdict(list)
    for f in nii_files:
        cl = int(f.split('_ope_')[0])
        cl_groups[cl].append(f)

    out_probs = []
    out_uncert = []
    for sl in range(n_slices):
        probs_stack = []
        for cl in sorted(cl_groups.keys()):
            probs_cl = []
            for nii_file in cl_groups[cl]:
                nii_log = nib.load(os.path.join(pred_npz_dir, nii_file), mmap=True)
                proxy = nii_log.dataobj
                probs_cl.append(proxy[sl])
            probs_stack.append(np.stack(probs_cl, axis=0))  # [(n,C,H,W),...]

        probs_stack = np.concatenate(probs_stack, axis=1)  # (n,C,H,W)
        assert probs_stack.shape[2:] == ori_size, 'the logits size do not match'

        # --- 1. Final prediction mask from average logits ---
        mean_probs = np.mean(probs_stack, axis=0)  # shape: [C, H, W]
        # pred_mask = multi_class_mask_slice(mean_probs)  # (H,W)
        # print(f'class index of final mask: {np.unique(pred_mask)}')
        out_probs.append(mean_probs)

        # --- 2. Uncertainty Map (entropy over mean probability) ---
        mean_probs = torch.nn.functional.interpolate(
            torch.from_numpy(mean_probs).unsqueeze(0),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).numpy()
        entropy_binary = -(mean_probs * np.log(mean_probs + 1e-8) + (1 - mean_probs) * np.log(1 - mean_probs + 1e-8))
        assert len(entropy_binary.shape) == 3, 'processing slice by slice, the shape of entropy is wrong'
        out_uncert.append(entropy_binary)

    # --- 3. save uncertainty
    out_uncert = np.stack(out_uncert, axis=0)  # shape: [d, C, H, W]
    nii_uncert = nib.Nifti1Image(out_uncert, np.eye(4))
    uncert_file = f'{objID}_uncertMap.nii.gz'
    nib.save(nii_uncert, os.path.join(uncertainty_dir, uncert_file))  # shape: [d, C, H, W]
    del out_uncert

    # --- 4. save multi-label mask
    pred_mask_3d_path = os.path.join(pred_dir, objID)
    out_probs = np.stack(out_probs, axis=0)  # (d,c,H,W)
    out_mask_bi = probs_to_binary(out_probs) # (d,c,H,W), binary
    if only_largest_comp:
        for clid in range(out_mask_bi.shape[1]):
            if np.max(out_mask_bi[:,clid]) > 0:
                out_mask_bi[:,clid] = getLargestCC(out_mask_bi[:,clid])
                out_mask_bi[:,clid] = np.uint8(out_mask_bi[:,clid])
    if save_multi_channel_mask:
        Path(pred_mask_3d_path).mkdir(parents=True, exist_ok=True)
        for cid in range(out_mask_bi.shape[1]):
            save_nii_cl = sitk.GetImageFromArray(out_mask_bi[:,cid])
            save_nii_cl.CopyInformation(ref_img_nii)
            sitk.WriteImage(save_nii_cl, os.path.join(pred_mask_3d_path, f'{cid}.nii.gz'))

    out_mask = multichannel_binary_to_mask(out_mask_bi) # (d,H,W)
    print(f'class index for output mask: {np.unique(out_mask)}')

    save_nii = sitk.GetImageFromArray(out_mask)
    # Set spatial metadata from reference image
    save_nii.CopyInformation(ref_img_nii)

    # save_nii.SetSpacing(ref_img_nii.GetSpacing())
    # save_nii.SetOrigin(ref_img_nii.GetOrigin())
    # save_nii.SetDirection(ref_img_nii.GetDirection())
    # save as 3d volume
    sitk.WriteImage(save_nii, pred_mask_3d_path + '.nii.gz')
    # save as 2d images
    # Path(pred_mask_3d_path).mkdir(parents=True, exist_ok=True)
    # for sls in range(out_mask.shape[0]):
    #     plt.imsave(os.path.join(pred_mask_3d_path, f'{sls}.png'), out_mask[sls], format='png')


# def topk_indices(a, k=10):
#     # a: 2D array of shape (h, w)
#     flat = a.ravel()
#     # find the k largest entries (unsorted)
#     idx_unsorted = np.argpartition(flat, -k)[-k:]
#     # sort those k entries in descending order
#     idx_sorted = idx_unsorted[np.argsort(-flat[idx_unsorted])]
#     # unravel to 2D coordinates
#     rows, cols = np.unravel_index(idx_sorted, a.shape)
#     # build a list of (row, col) tuples
#     return list(zip(rows.tolist(), cols.tolist()))

def sample_certainty_points(uncert: np.ndarray,
                            mask: np.ndarray,
                            k_pos: int = 5,
                            k_neg: int = 5):
    """
    Return two lists of up to k (row, col) points:
      - pos_pts: the k most-certain points inside mask (mask==1)
      - neg_pts: the k most-certain points outside mask (mask==0)

    uncert: 2D array of shape (H, W)
    mask:   2D binary array of shape (H, W)
    """
    # --- helper to pick up to k smallest entries in vals, returning coords from ys,xs
    def _topk_least(vals, ys, xs, k):
        n = len(vals)
        if n == 0 or k <= 0:
            return []
        k = min(k, n)
        # find the indices of the k smallest entries (unsorted)
        idx_unsorted = np.argpartition(vals, k-1)[:k]
        # map back to (col, row)
        return [(int(xs[i]), int(ys[i])) for i in idx_unsorted]

    # positive region
    ys_pos, xs_pos = np.nonzero(mask)
    vals_pos      = uncert[ys_pos, xs_pos]
    pos_pts = _topk_least(vals_pos, ys_pos, xs_pos, k_pos)

    # negative region
    ys_neg, xs_neg = np.nonzero(mask == 0)
    vals_neg       = uncert[ys_neg, xs_neg]
    neg_pts = _topk_least(vals_neg, ys_neg, xs_neg, k_neg)

    return pos_pts, neg_pts


def largest_component_bbox(mask):
    """
    Find the largest connected component in a binary mask and return its bounding box.

    Parameters
    ----------
    mask : np.ndarray
        2D binary array of shape (H, W), with True/1 indicating foreground.

    Returns
    -------
    bbox : (x_min, y_min, x_max, y_max) or None
        Coordinates of the tightest bounding box around the largest component.
        Returns None if there are no foreground pixels.
    """
    # Label connected components
    labeled, num_features = ndi.label(mask > 0)
    if num_features == 0:
        return None

    # Compute sizes of components 1..num_features
    sizes = ndi.sum(mask > 0, labeled, index=np.arange(1, num_features + 1))
    # Identify the largest component label
    largest_label = sizes.argmax() + 1  # +1 because index 0 corresponds to label=1

    # Build a mask of just that component
    largest_mask = (labeled == largest_label)

    # Find bounding box of that component
    ys, xs = np.where(largest_mask)
    y_min, y_max = ys.min().item(), ys.max().item()
    x_min, x_max = xs.min().item(), xs.max().item()

    return [x_min, y_min, x_max, y_max]


def bounding_box_all(mask: np.ndarray):
    """
    Return the bounding box of *all* positive pixels in `mask`.

    mask : 2D binary array (H, W), where >0 indicates foreground.
    original 3D mask should have been post-processed by largest component.

    Returns [x_min, y_min, x_max, y_max] or None if mask is empty.
    """
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0:
        return None
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    return [x_min, y_min, x_max, y_max]

def generate_prompts_from_mo(
        mo_mask,  # np array, (d, c+1, h, w) or (1,h,w) in original size
        mo_uncert, # (d,h,w)
        sam_uncert = None,  # np array, (d, c,h,w) c is the number of foreground classes
        num_points_by_mo_cert: int = 5,  # number of positive and negtive points
        num_points_by_diff_cert: int = 5,  # number of positive and negtive points
):
    """
    prompt_points dict:{
        sl:{
            cl:[
                [(x,y),label], ...
                ]
            }
        }
    prompt_bbox dict:{
        sl:{
            cl:[x_min, y_min, x_max, y_max]
            }
        }
    """
    if len(mo_mask.shape) == 3:
        mo_mask = np.expand_dims(mo_mask, axis=1)

    num_classes = mo_mask.shape[1]
    num_slices = mo_mask.shape[0]
    prompt_points = defaultdict(lambda: defaultdict(list))
    prompt_boxs = defaultdict(lambda: defaultdict(list))

    for sl in range(num_slices):
        for cl in range(num_classes):
            # --- 1. get the points prompt
            pos_by_mo, neg_by_mo = sample_certainty_points(mo_uncert[sl],mo_mask[sl, cl],
                                                           k_pos = num_points_by_mo_cert,
                                                           k_neg=num_points_by_mo_cert)

            if sam_uncert is not None:
                uncert_diff_pos = (sam_uncert[sl, cl] - mo_uncert[sl]) * mo_mask[sl, cl]
                uncert_diff_neg = (sam_uncert[sl, cl] - mo_uncert[sl]) * (1 - mo_mask[sl, cl])

                k_pos = min(np.sum(uncert_diff_pos > 0), num_points_by_diff_cert)
                k_neg = min(np.sum(uncert_diff_neg > 0), num_points_by_diff_cert)

                pos_by_diff, neg_by_diff = sample_certainty_points(mo_uncert[sl] - sam_uncert[sl, cl], mo_mask[sl, cl],
                                                                    k_pos=k_pos,
                                                                    k_neg=k_neg)
            else:
                pos_by_diff = []
                neg_by_diff = []

            pos_total = set(pos_by_mo + pos_by_diff)
            neg_total = set(neg_by_mo + neg_by_diff)

            pos_total = [[p, 1] for p in pos_total]
            neg_total = [[p, 0] for p in neg_total]

            prompt_points[sl][cl] = pos_total + neg_total

            # --- 2. get the bbox prompt
            if np.sum(mo_mask[sl, cl]) > 0:
                bbox = bounding_box_all(mo_mask[sl, cl])
                prompt_boxs[sl][cl] = bbox

    return prompt_points, prompt_boxs
