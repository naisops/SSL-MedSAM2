# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 18:24:00 2025

@author: naisops
"""

from tqdm import tqdm
import os
import numpy as np

import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from sam2.build_sam import build_sam2_video_predictor_npz
from dataloader_3d import MedicalVolumeDataset, get_objs_and_paths
from pathlib import Path
import random
import shutil
import util_3d
import random_mix_method
import sys
import datetime

# import warnings
# warnings.simplefilter('error')
from gen_3d_config import Config as custom_Config

if __name__ == "__main__":
    print("sys.argv:", sys.argv)
    custom_config = custom_Config.get_config()

'''
SAM 2 treats each obj_id as its own binary classification task. 
It produces C separate mask-logit maps—even when you track multiple objects 
in one pass—without inter-object coupling 
'''

MODEL_CONFIG = "configs/sam2.1_hiera_t512.yaml"
MODEL_CHECKPOINT = "checkpoints/MedSAM2_latest.pt"
predictor = build_sam2_video_predictor_npz(MODEL_CONFIG, MODEL_CHECKPOINT)
my_rng = random.Random(42)
img_size = 512

def condition_care_ged4(full_path):
    return 'GED4.nii.gz' == Path(full_path).name

def condition_care_dwi800(full_path):
    return 'DWI_800.nii.gz' == Path(full_path).name

def condition_care_t1(full_path):
    return 'T1.nii.gz' == Path(full_path).name

def condition_care_t2(full_path):
    return 'T2.nii.gz' == Path(full_path).name

def condition_care_mask(full_path):
    return 'mask_GED4.nii.gz' == Path(full_path).name

labelled_objs_vendors = {
    'Vendor_A':['1009-A-S4','1053-A-S3', '1920-A-S1', '1927-A-S1', '1967-A-S2',
                 '1971-A-S2', '1991-A-S4', '1998-A-S4', '1406-A-S1', '1432-A-S4'],
    'Vendor_B1':['0241-B1-S2', '0482-B1-S2', '0487-B1-S2', '0496-B1-S3', '1144-B1-S1',
                 '1158-B1-S2', '1194-B1-S4', '1002-B1-S1', '1012-B1-S4', '1029-B1-S4'],
    'Vendor_B2':['1031-B2-S1', '1041-B2-S3', '1053-B2-S3', '1070-B2-S4', '1075-B2-S4',
                 '1076-B2-S4', '1086-B2-S4', '1097-B2-S4', '1098-B2-S4', '1115-B2-S4'],
}

seq_cond = {'ged4': condition_care_ged4,'dwi800':condition_care_dwi800,
            't1': condition_care_t1, 't2': condition_care_t2}

total_part = custom_config.total_part
vendor = custom_config.vendor
run_part = custom_config.run_part
n_sampling_on_each_mask = custom_config.n_sampling_on_each_mask
sequence = custom_config.sequence
random_mix_method_name = custom_config.random_mix_method_name
insert_mod = custom_config.insert_mod

## --- 0. Define saving dirs
data_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# The path to save ensembled predictions
save_npz_dir = f'/home/psxzg6/pred_npz/CARE2025/{vendor}/{sequence}/{random_mix_method_name}/run_{run_part}_{data_str}'
if random_mix_method_name == 'random_whole_insert':
    save_npz_dir = f'/home/psxzg6/pred_npz/CARE2025/{vendor}/{sequence}/{random_mix_method_name}_{insert_mod}/run_{run_part}_{data_str}'
if os.path.exists(save_npz_dir):
    shutil.rmtree(save_npz_dir)
Path(save_npz_dir).mkdir(parents=True, exist_ok=True)

save_pred_dir = f'/home/psxzg6/pred/CARE2025/{random_mix_method_name}/{vendor}/{sequence}'
if random_mix_method_name == 'random_whole_insert':
    save_pred_dir = f'/home/psxzg6/pred/CARE2025/{random_mix_method_name}_{insert_mod}/{vendor}/{sequence}'
Path(save_pred_dir).mkdir(parents=True, exist_ok=True)

save_uncert_dir = f'/home/psxzg6/uncertainty/CARE2025/{random_mix_method_name}/{vendor}/{sequence}'
if random_mix_method_name == 'random_whole_insert':
    save_uncert_dir = f'/home/psxzg6/uncertainty/CARE2025/{random_mix_method_name}_{insert_mod}/{vendor}/{sequence}'
Path(save_uncert_dir).mkdir(parents=True, exist_ok=True)

## --- 1. Generate datasets (un_dataset, la_dataset)
root_dir_la = os.path.join('/home/psxzg6/datasets/CARE2025/', vendor)
labelled_objs = labelled_objs_vendors[vendor]
# get path dict for mask data
mask_path_dict = get_objs_and_paths(labelled_objs,condition_care_mask, root_dir_la)
# get path dict for labelled image data
img_path_dict_la = get_objs_and_paths(labelled_objs,condition_care_ged4, root_dir_la)

# get path dict for unlabelled image data
root_dir_un = os.path.join('/home/psxzg6/datasets/CARE2025/', vendor)
# unlabelled_objs = sorted([obj for obj in os.listdir(root_dir_un) if obj not in labelled_objs])
unlabelled_objs = labelled_objs.copy()
img_path_dict_un = get_objs_and_paths(unlabelled_objs, seq_cond[sequence], root_dir_un)

# divide the unlabelled set to several parts for efficient processing
parts = np.array_split(unlabelled_objs, total_part)
unlabelled_objs = parts[run_part]
print(unlabelled_objs)

# create datasets
labelled_dataset = MedicalVolumeDataset(
             labelled_objs, # list of object names
             img_path_dict_la, # dict of ['obj': img_path]
             img_size=img_size,
             modality = 'MRI', # CT, MRI
             z_axis = 0, # along which axis to run predictor (the slices axis)
             )
unlabelled_dataset = MedicalVolumeDataset(
             unlabelled_objs, # list of object names
             img_path_dict_un, # dict of ['obj': img_path]
             img_size = img_size,
             modality = 'MRI', # CT, MRI
             z_axis = 0, # along which axis to run predictor (the slices axis)
             )

'''
sample={'obj':obj, 'img_data': nii_image_data, 
                'img_nii': nii_image, 'video_width': w, 'video_height':h}
'''
for sample_un in unlabelled_dataset:
    if sample_un is None:
        continue
    obj_un = sample_un['obj']
    volume_un = sample_un['img_data']
    nii_un = sample_un['img_nii']
    n_slices = volume_un.size(0)
    video_width_un,video_height_un = sample_un['video_width'], sample_un['video_height']
    print('generating logits and pred on', obj_un)
    for sample_la in labelled_dataset:
        obj_la = sample_la['obj']
        volume_la = sample_la['img_data']
        path_mask = mask_path_dict[obj_la]
        mask_la = sitk.ReadImage(path_mask)
        mask_la = sitk.GetArrayFromImage(mask_la)
        mask_la = (mask_la>0).astype(np.uint8)

        # resize the mask to unlabelled data size, for easily get logits for unlabelled slices
        mask_la = F.interpolate(torch.from_numpy(mask_la).unsqueeze(1),
                                size=(video_height_un, video_width_un), mode='nearest').squeeze(1).numpy().astype(np.uint8)
        # (n,h,w)

        # for cl in range(num_class):
        if random_mix_method_name == 'random_whole_insert':
            if insert_mod == 'la_to_un':
                random_mix_method.whole_insert_a_to_b(
                    my_rng, obj_la,
                    cl=[1],
                    mod=insert_mod,
                    volume_a=volume_la,  # torch tensor (n,3,h,w)
                    volume_b=volume_un,  # torch tensor (n,3,h,w)
                    mask_la=mask_la,  # nparray (n,h,w)
                    n_sampling_on_each_mask=n_sampling_on_each_mask,
                    predictor=predictor,
                    video_height_un=video_height_un,
                    video_width_un=video_width_un,
                    prompt_points=None,
                    prompt_bbox=None,
                    save_npz_dir=save_npz_dir,
                )


    util_3d.generate_uncert_and_pred(obj_un, nii_un, n_slices,
                                     ori_size=(video_height_un, video_width_un),
                                     img_size=img_size,
                                     pred_npz_dir=save_npz_dir,
                                     uncertainty_dir=save_uncert_dir,
                                     pred_dir=save_pred_dir,
                                     only_largest_comp=True,
                                     )
    shutil.rmtree(save_npz_dir)
    Path(save_npz_dir).mkdir(parents=True, exist_ok=True)

print(f'all logits are saved to {save_npz_dir}, all preds are saved to {save_pred_dir}, all uncert are saved to {save_uncert_dir}')

        
