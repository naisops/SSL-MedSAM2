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

from util_3d import generate_pseudo_logits



"""
------------ 1. random switch between un and la ------------
"""
def switch_part_for_la_un(
        volume_un,  # shape in (d_un,3,512,512)
        volume_la,  # shape in (d_la,3,512,512)
        mask_la,  # shape in (d_la, Cl, w, h)
        switch_begin: int = 0,  # the begining index of the slice in la waiting to be switched
        switch_length: int = 10,  # the switch length, number of slices being switched
):
    """
    No matter how switch, since we choose a part in labelled slices to replace the unlabelled slices, after switch
    The index of the unlabelled slice on unlabelled data will not be changed.
    The index of the un_slices_switched will be the same on unlabelled data and labelled data.
    """
    d_volume_un = volume_un.size(0)
    d_volume_la = volume_la.shape[0]
    switch_end = switch_begin + switch_length
    mask_un = np.zeros((volume_un.size(0), *mask_la.shape[1:]), dtype=mask_la.dtype)

    # get the volume_un after switch
    volume_un_start = volume_un[:min(switch_begin, d_volume_un)].clone()
    slices_switch_la = volume_la[switch_begin:switch_end].clone()
    volume_un_end = volume_un[switch_end:d_volume_un].clone() if switch_end < d_volume_un else None
    arrays_total = [volume_un_start, slices_switch_la, volume_un_end]
    arrays_total = [a for a in arrays_total if a is not None]
    volume_un_switched = torch.cat(arrays_total, dim=0)

    # get the mask_un after switch
    mask_un_start = mask_un[:min(switch_begin, d_volume_un)].copy()
    mask_switch_la = mask_la[switch_begin:switch_end].copy()
    mask_un_end = mask_un[switch_end:d_volume_un].copy() if switch_end < d_volume_un else None
    arrays_total = [mask_un_start, mask_switch_la, mask_un_end]
    arrays_total = [a for a in arrays_total if a is not None]
    mask_un_switched = np.concatenate(arrays_total, axis=0)

    assert mask_un_switched.shape[0] == volume_un_switched.size(
        0), 'for the unlabelled data, the mask and volume have no same slices after switch'

    # get the frame indexes of the unlabelled slice in volume_un_switched
    index_un_slice_in_un_switch = list(range(volume_un_switched.size(0)))
    for idx in range(volume_un_start.size(0), volume_un_start.size(0) + switch_length):
        index_un_slice_in_un_switch.remove(idx)

    # get the slices of un waiting to be switched
    if switch_begin >= d_volume_un:
        slices_switch_un = None
    else:
        slices_switch_un = volume_la[switch_begin:min(switch_end, d_volume_un)].clone()

    # get the volume_la and mask_la after switch
    if slices_switch_un is None:
        volume_la_switched = None
        mask_la_switched = None
        index_un_slice_in_la_switch = []
    else:
        # get the volume_la after switch
        volume_la_start = volume_la[:switch_begin].clone()
        volume_la_end = volume_la[switch_end:].clone() if switch_end < d_volume_la - 1 else None
        arrays_total = [volume_la_start, slices_switch_un, volume_la_end]
        arrays_total = [a for a in arrays_total if a is not None]
        volume_la_switched = torch.cat(arrays_total, dim=0)

        # get the frame indexes of the unlabelled slice in volume_un_switched
        index_un_slice_in_la_switch = list(
            range(switch_begin, switch_begin + slices_switch_un.size(0)))

        # get the mask_la after switch
        mask_la_switched = mask_la.copy()
        mask_la_switched[switch_begin:switch_end] = 0
        if slices_switch_un.shape[0] < switch_length:
            mask_la_switched = np.delete(mask_la_switched, np.s_[switch_begin + slices_switch_un.shape[0]:switch_end],
                                         axis=0)

        assert mask_la_switched.shape[0] == volume_la_switched.size(
            0), 'for the labelled data, the mask and volume have no same slices after switch'
        assert mask_la_switched.shape[1:] == mask_la.shape[
                                             1:], 'for the labelled data, the mask slice shape changes after switch'

    assert set(index_un_slice_in_un_switch).isdisjoint(
        index_un_slice_in_la_switch), "Index lists for un_slice in un and la have common items!"
    assert sorted(index_un_slice_in_un_switch + index_un_slice_in_la_switch) == sorted(
        list(range(volume_un.size(0)))), 'Index of un_slices will not rebuild un'

    # When rebuild pred_un after switch and pred, cat(part1_un_pred,switch_la_pred if exist,part2_un_pred if exist)
    return {'volume_un_switched': volume_un_switched, 'mask_un_switched': mask_un_switched,
            'index_un_slice_in_un_switch': index_un_slice_in_un_switch,
            'volume_la_switched': volume_la_switched, 'mask_la_switched': mask_la_switched,
            'index_un_slice_in_la_switch': index_un_slice_in_la_switch}


def pred_on_switch_and_rebuild(
        predictor, cl,
        volume_un_switched,
        mask_un_switched,
        index_un_slice_in_un_switch: list,
        volume_la_switched,
        mask_la_switched,
        index_un_slice_in_la_switch: list,
        video_height_un: int,
        video_width_un: int,
        prompt_ponits: dict = None,  # the prompts on original unlabelled data before switch
        prompt_bbox: dict = None,  # the prompts on original unlabelled data before switch
):  # return the rebuild logits in (d,h,w)

    prompt_ponits_un_switch = {sl: prompt_ponits[sl] for sl in
                               index_un_slice_in_un_switch} if prompt_ponits is not None else None
    prompt_bbox_un_switch = {sl: prompt_bbox[sl] for sl in
                             index_un_slice_in_un_switch} if prompt_bbox is not None else None

    logits_un_switch = generate_pseudo_logits(volume_un_switched, predictor, cl,
                                              video_height_un, video_width_un,
                                              mask_3d=mask_un_switched, prompt_ponits=prompt_ponits_un_switch,
                                              prompt_bbox=prompt_bbox_un_switch)  # (d,h,w)

    if volume_la_switched is not None:
        prompt_ponits_la_switch = {sl: prompt_ponits[sl] for sl in
                                   index_un_slice_in_la_switch} if prompt_ponits is not None else None
        prompt_bbox_la_switch = {sl: prompt_bbox[sl] for sl in
                                 index_un_slice_in_la_switch} if prompt_bbox is not None else None

        # when predict on labelled image, we still use video width/height of unlabelled image,
        # thus we can easily get the pred logits of switch part on the labelled image
        logits_la_switch = generate_pseudo_logits(volume_la_switched, predictor, cl,
                                                  video_height_un, video_width_un,
                                                  mask_3d=mask_la_switched, prompt_ponits=prompt_ponits_la_switch,
                                                  prompt_bbox=prompt_bbox_la_switch)  # (d,h,w)
    else:
        logits_la_switch = None

    logits_rebuild_un_switch = {sl: logits_un_switch[sl] for sl in index_un_slice_in_un_switch}
    if volume_la_switched is not None:
        logits_rebuild_la_switch = {sl: logits_la_switch[sl] for sl in index_un_slice_in_la_switch}
    else:
        logits_rebuild_la_switch = {}

    logits_rebuild_total = {**logits_rebuild_un_switch, **logits_rebuild_la_switch}
    logits_rebuild_total = [logits_rebuild_total[i] for i in sorted(logits_rebuild_total.keys())]  # [(h,w)]
    logits_rebuild_total = torch.stack(logits_rebuild_total, dim=0)  # (d,h,w)

    return logits_rebuild_total  # (d,h,w)


def generate_switch_region(
        rand_gen,
        mask,  # the GT mask for the labelled volume
        cl: int,  # the class index waiting to be processed
        n_sampling: int,  # the number of sampled switch_begin indexes
):
    """
    Get the switch region per class
    Assumption: the GT mask for each class is continues through slices (or only missing a few middle slices)
    """

    label_region = []
    ns = mask.shape[0]
    for sl in range(ns):
        if np.sum(mask[sl, cl]) > 0:
            label_region.append(sl)

    label_length = max(label_region) - min(label_region)
    switch_length = label_length // 2
    scan_lenth = label_length - switch_length
    scan_region = list(range(min(label_region), min(label_region) + scan_lenth))
    switch_begin_lst = rand_gen.sample(scan_region, min(len(scan_region), n_sampling))

    return switch_begin_lst, switch_length


def random_switch_and_rebuild(
        my_rng, cl, obj_la,
        volume_un=None,
        volume_la=None,
        mask_la=None,
        n_sampling_on_each_mask=None,
        predictor=None,
        video_height_un=None,
        video_width_un=None,
        n_slices=None,
        save_npz_dir=None,
        prompt_ponits=None,
        prompt_bbox=None,
):
    switch_begin_lst, switch_length = generate_switch_region(
        my_rng, mask_la, cl, n_sampling_on_each_mask,
    )
    for switch_begin in switch_begin_lst:
        switch_results = switch_part_for_la_un(
            volume_un, volume_la, mask_la,
            switch_begin=switch_begin, switch_length=switch_length,
        )

        logits_rebuild_total = pred_on_switch_and_rebuild(
            predictor, cl,
            switch_results['volume_un_switched'],
            switch_results['mask_un_switched'],
            switch_results['index_un_slice_in_un_switch'],
            switch_results['volume_la_switched'],
            switch_results['mask_la_switched'],
            switch_results['index_un_slice_in_la_switch'],
            video_height_un, video_width_un,
            prompt_ponits=prompt_ponits,
            prompt_bbox=prompt_bbox,
        )  # (d,h,w)

        assert logits_rebuild_total.size() == (n_slices, video_height_un, video_width_un)
        probs = torch.sigmoid(logits_rebuild_total)
        nii_file = f'{cl}_ope_{obj_la}_onloc_{switch_begin}.nii'
        probs_nii = sitk.GetImageFromArray(probs.cpu().numpy())
        sitk.WriteImage(probs_nii, Path(save_npz_dir) / nii_file)


"""
------------ 2. random insert la to un or un to la ------------
"""
def new_prompt_after_insert(
        prompt,
        serted_length=0,
        mod='un_to_la',
        insert_loc=0,
):
    if mod == 'un_to_la':
        new_prompt = {sl + insert_loc: value for sl, value in prompt.items()}
    elif mod == 'la_to_un':
        new_prompt = {}
        for sl, value in prompt.items():
            if sl < insert_loc:
                new_prompt[sl] = prompt[sl]
            else:
                new_prompt[sl + serted_length] = prompt[sl]
    else:
        raise TypeError('mod is not supported')

    return new_prompt

def frames_index_un_after_insert(
        un_length=0,
        serted_length=0,
        mod='un_to_la',
        insert_loc=0,
):
    ori_index = list(range(un_length))
    if mod == 'un_to_la':
        new_index = [sl+insert_loc for sl in ori_index]
    elif mod == 'la_to_un':
        new_index = []
        for sl in ori_index:
            if sl < insert_loc:
                new_index.append(sl)
            else:
                new_index.append(sl+serted_length)
    else:
        raise TypeError('mod is not supported')

    return sorted(new_index)
def whole_insert_a_to_b(
        my_rng, obj_la,
        cl: list=None,
        mod='un_to_la',
        volume_a=None,  # torch tensor (n,3,h,w)
        volume_b=None,  # torch tensor (n,3,h,w)
        mask_la=None,  # nparray (n,h,w)
        n_sampling_on_each_mask=None,
        predictor=None,
        video_height_un=None,
        video_width_un=None,
        prompt_points: dict = None,
        prompt_bbox: dict = None,
        save_npz_dir=None,
):
    """
    will sert a between b[insert-1] and b[insert]
    """
    insert_locs = my_rng.sample(list(range(volume_b.size(0))), n_sampling_on_each_mask)
    serted_length = volume_a.size(0)
    if mod == 'un_to_la':
        mask_a = np.zeros((volume_a.size(0), *mask_la.shape[1:]), dtype=mask_la.dtype)
        mask_b = mask_la.copy()
        un_length = volume_a.size(0)
    elif mod == 'la_to_un':
        mask_a = mask_la.copy()
        mask_b = np.zeros((volume_b.size(0), *mask_la.shape[1:]), dtype=mask_la.dtype)
        un_length = volume_b.size(0)
    else:
        raise TypeError('mod is not supported')
    for insert in insert_locs:
        volume_b_inserted = torch.cat((volume_b[:insert], volume_a, volume_b[insert:]), dim=0)
        mask_b_inserted = np.concatenate((mask_b[:insert], mask_a, mask_b[insert:]), axis=0)

        new_prompt_points = new_prompt_after_insert(
            prompt_points, serted_length=serted_length, mod=mod, insert_loc=insert,) if prompt_points else None
        new_prompt_bbox = new_prompt_after_insert(
            prompt_bbox, serted_length=serted_length, mod=mod, insert_loc=insert,) if prompt_bbox else None
        return_frame_index = frames_index_un_after_insert(
            un_length=un_length, serted_length=serted_length, mod=mod, insert_loc=insert,)

        logits_un_insert = generate_pseudo_logits(volume_b_inserted, predictor,
                                                  video_height_un, video_width_un,
                                                  cl = cl,
                                                  mask_3d=mask_b_inserted, prompt_ponits=new_prompt_points,
                                                  prompt_bbox=new_prompt_bbox) # (n, h, w)

        logits_un_rebuild = [logits_un_insert[sl] for sl in return_frame_index]
        logits_un_rebuild = torch.stack(logits_un_rebuild,dim=0)

        assert logits_un_rebuild.size() == (un_length, len(cl), video_height_un, video_width_un)
        probs = torch.sigmoid(logits_un_rebuild)
        nii_file = f'{cl[0]}_ope_{obj_la}_onloc_{insert}.nii'
        probs_nii = nib.Nifti1Image(probs.cpu().numpy(), np.eye(4, dtype=np.float32))
        nib.save(probs_nii, Path(save_npz_dir) / nii_file)



"""
------------ 3. one by one switch from un to la ------------
"""
def get_label_region_of_mask(
        cl,
        mask_la, # (n,c,h,w)
):
    mask_lst = []
    for sl in range(mask_la.shape[0]):
        if np.sum(mask_la[sl,cl]) > 0:
            mask_lst.append(sl)

    return len(mask_lst)

def chunk_list(lst, chunk_size=5):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_logits_for_switch_part(
        predictor, cl,
        video_height_un=0,
        video_width_un=0,
        volume_un=None,
        volume_la=None,
        mask_la=None,
        switch=None,
        prompt_points=None,
        prompt_bbox=None,
):
    mask_un = np.zeros((volume_un.size(0), *mask_la.shape[1:]), dtype=mask_la.dtype)
    vol_switch_un = volume_un[switch]
    mask_switch_un = mask_un[switch]
    switch_start = switch[0]
    switch_end = switch[-1]
    la_length = volume_la.size(0)

    volume_la_1 = volume_la[:min(switch_start,la_length)]
    volume_la_2 = volume_la[switch_end+1:] if switch_end + 1 < la_length else None
    volume_la_switched = [volume_la_1, vol_switch_un, volume_la_2]
    volume_la_switched = [v for v in volume_la_switched if v is not None]
    volume_la_switched = torch.cat(volume_la_switched,dim=0)

    mask_la_1 = mask_la[:min(switch_start, la_length)]
    mask_la_2 = mask_la[switch_end + 1:] if switch_end + 1 < la_length else None
    mask_la_switched = [mask_la_1, mask_switch_un, mask_la_2]
    mask_la_switched = [v for v in mask_la_switched if v is not None]
    mask_la_switched = np.concatenate(mask_la_switched, axis=0)

    return_frames = [sl - max(0, switch_start-la_length) for sl in switch]
    new_prompt_points = {sl - max(0, switch_start-la_length): prompt_points[sl] for sl in switch} if prompt_points else None
    new_prompt_bbox = {sl - max(0, switch_start-la_length): prompt_bbox[sl] for sl in switch} if prompt_points else None

    logits_la_switched = generate_pseudo_logits(volume_la_switched, predictor, cl,
                                              video_height_un, video_width_un,
                                              mask_3d=mask_la_switched, prompt_ponits=new_prompt_points,
                                              prompt_bbox=new_prompt_bbox)  # (n, h, w)

    return_logits_un = {}
    for i, fm in enumerate(return_frames):
        return_logits_un[switch[i]]=logits_la_switched[fm]

    return return_logits_un # {sl: (h,w)}

def one_by_one_switch_un_to_la(
        cl, obj_la,
        n_chunk = 5,
        volume_un=None,  # torch tensor (n,3,h,w)
        volume_la=None,  # torch tensor (n,3,h,w)
        mask_la=None,  # nparray (n,c,h,w)
        n_slices=None,
        predictor=None,
        video_height_un=None,
        video_width_un=None,
        prompt_points: dict = None,
        prompt_bbox: dict = None,
        save_npz_dir=None,
):
    un_length = volume_un.size(0)
    label_length = get_label_region_of_mask(cl,mask_la)
    chunk_size = min(un_length//n_chunk, label_length//2)

    switch_parts = chunk_list(list(range(un_length)), chunk_size=chunk_size)
    logits_un_out = {}

    for switch in switch_parts:
        return_logits_un = get_logits_for_switch_part(
                predictor, cl,
                video_height_un=video_height_un,
                video_width_un=video_width_un,
                volume_un=volume_un,
                volume_la=volume_la,
                mask_la=mask_la,
                switch=switch,
                prompt_points=prompt_points,
                prompt_bbox=prompt_bbox,
        ) # {sl: (h,w)}

        logits_un_out.update(return_logits_un)

    logits_un_out = [logits_un_out[sl] for sl in sorted((logits_un_out.keys()))]
    logits_un_out = torch.stack(logits_un_out, dim=0)

    assert logits_un_out.size() == (n_slices, video_height_un, video_width_un)
    probs = torch.sigmoid(logits_un_out)
    nii_file = f'{cl}_ope_{obj_la}.nii'
    probs_nii = sitk.GetImageFromArray(probs.cpu().numpy())
    sitk.WriteImage(probs_nii, Path(save_npz_dir) / nii_file)
