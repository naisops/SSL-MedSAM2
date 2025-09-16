# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 17:16:15 2025

@author: naisops
"""
import os
import glob
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import cv2
import SimpleITK as sitk

def find_data(data_dir, condition):
    """
    Find the data paths that meet a custom condition.
    Args:
        data_dir (str): Root directory to search.
        condition (callable): A callable that applies custom conditions to filenames.
    Returns:
        list: Located paths of the data that meet the condition.
    """
    data_path = []
    
    for file in os.listdir(data_dir):
        full_path = os.path.join(data_dir, file)
        if os.path.isdir(full_path):
            # Recursively search subdirectories and collect paths
            data_path.extend(find_data(full_path, condition))
        elif condition(full_path):
            data_path.append(os.path.normpath(full_path))
            
    return data_path

def get_objs_and_paths(objs, cond, root_dir):
    data_paths = find_data(root_dir, cond)
    img_path_dict = {}
    for obj in objs:
        for path in data_paths:
            if obj in path:
                img_path_dict[obj]=path

    return img_path_dict
def preprocess(image_data, modality="CT", window_level=-750, window_width=1500):
    if modality == "CT":
        assert window_level is not None and window_width is not None, "CT modality requires window_level and window_width"
        lower_bound = window_level - window_width / 2
        upper_bound = window_level + window_width / 2
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
    else:
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0
    
    return image_data_pre

def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    if h == image_size and w == image_size:
        resized_array = array[:,None].repeat(3, axis=1)
    else:
        resized_array = np.zeros((d, 3, image_size, image_size))
        
        for i in range(d):
            img_pil = Image.fromarray(array[i].astype(np.uint8))
            img_rgb = img_pil.convert("RGB")
            img_resized = img_rgb.resize((image_size, image_size))
            img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
            resized_array[i] = img_array
    
    return resized_array, h, w

def data_normalization(torch_array):
    torch_array = torch_array / 255.0
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    torch_array -= img_mean
    torch_array /= img_std

    return torch_array

class MedicalVolumeDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 objs: list, # list of object names
                 img_path_dict: dict, # dict of ['obj': img_path]
                 modality:str = 'MRI', # CT, MRI
                 z_axis:int = 0, # along which axis to run predictor
                 img_size:int = 512, # the image size of required input data
                 window_level:int = None, # CT modality requires window_level and window_width
                 window_width:int = None, # CT modality requires window_level and window_width
                 ):
        
        self.objs = objs
        self.img_path_dict = img_path_dict
        self.modality = modality
        self.window_level = window_level
        self.window_width = window_width
        self.z_axis = z_axis
        self.img_size = img_size
    
    def __len__(self):
        return len(self.objs)

    def __getitem__(self, idx):
        ## Load image
        obj = self.objs[idx]
        if obj not in self.img_path_dict:
            return None
        img_path = self.img_path_dict[obj]
        nii_image = sitk.ReadImage(img_path)
        nii_image_data = sitk.GetArrayFromImage(nii_image)
        # nii_mask = None
        # nii_mask_data = None
        # if obj in self.mask_path_dict:
        #     mask_path = self.mask_path_dict[obj]
        #     nii_mask = sitk.ReadImage(mask_path)
        #     nii_mask_data = sitk.GetArrayFromImage(nii_mask)
        #     assert len(nii_mask_data) == 3, 'the 3D mask data should have 3 dimensions'
            
        ## Preprocess and resize
        # ensure the shape is [d,h,w]
        assert len(nii_image_data.shape) == 3, 'the volume data should have 3 dimensions'
        if self.z_axis != 0:
            nii_image_data = np.moveaxis(nii_image_data, self.z_axis, 0)
            
        nii_image_data = preprocess(nii_image_data, modality=self.modality, window_level=self.window_level, window_width=self.window_width)
        nii_image_data, h, w = resize_grayscale_to_rgb_and_resize(nii_image_data, self.img_size)
        nii_image_data = torch.from_numpy(nii_image_data)
        nii_image_data = data_normalization(nii_image_data)

        assert len(nii_image_data.size()) == 4 and nii_image_data.size(1) == 3, 'the preprocessed volume data has wrong dimensions'
        sample={'obj':obj, 'img_data': nii_image_data, 
                'img_nii': nii_image, 'video_width': w, 'video_height':h}
        
        return sample
        
        
        
        
        
        