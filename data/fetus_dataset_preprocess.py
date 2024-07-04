# Code by Jiewen Yang - jyangcu@connect.ust.hk
# Dataset code for fetus object detection
# Do not distribute without premission
# Keep Dataset Confidential

import os
import random
import numpy as np
from PIL import Image

import cv2
import json

import SimpleITK as sitk
from shutil import copyfile

import torchvision.transforms as transforms
from collections import defaultdict

from torch.utils.data import Dataset

np.set_printoptions(threshold=np.inf)
random.seed(6666)
np.random.seed(6666)


class fetus_Dataset(Dataset):
    def __init__(self, args, slice='', is_train=True):
        self.dataset_path = args.dataset_path
        self.annotations_path = self.dataset_path + 'annotations/'
        self.annotations_dict = dict()
        raw_annotations = self.get_annotations()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def get_annotations(self):
        annotations_files = self.FindAllFile(self.annotations_path) 
        raw_annotations = self.read_json(annotations_files)
        return raw_annotations

    def read_json(self, paths):
        for path in paths:
            slice_name, file_type = path.split(".")
            if file_type == 'json':
                self.load_json_annotation(self.annotations_path + path, slice_name)
            else:
                pass

    # The Hierarchy of this dataset ->
    # L1: hospital -> Which hospital
    # L2: slice type -> The type of different slices
    # L3: image ID -> The name of different images
    # L4: 'GA', 'annotations', 'bodyPart', 'id', 'info', 'standard', 'subclass' -> Attributes of each image
    # L5: annotations -> Different parts(organs) of current slice
    # L6: Is a list if L5 annotations
    # L7: 'GA', 'alias', 'class', 'clear', 'color', 'name', 'rotation', 'type', 'vertex', 'zDepth' -> Attributes of each part(organ)
    # L8: 'GA'->?, 
        # 'alias'   -> Alias of organ, 
        # 'class'   -> ?, 
        # 'clear'   -> ?, 
        # 'name'    -> Name of organ, 
        # 'rotation'-> The Rotation of organ in image, 
        # 'type'    -> ?, 
        # 'vertex'  -> The proposal of organ,
        # 'zDepth'  -> ?

    def load_json_annotation(self, json_file, slice_name):
        self.annotations_dict[slice_name] = json.load(open(json_file))['annotations']

    def FindAllFile(self, path):
        file_list = list()
        for root, dirs, files in os.walk(path):
            for file in files:
                file_list.append(file)
        return file_list

if __name__ == '__main__':
    import argparse

    from tqdm import tqdm
    from einops import rearrange

    from torch.utils.data import DataLoader
    from torchvision import utils as vutils

    parser = argparse.ArgumentParser(description="Fetus Object Detection")
    parser.add_argument('--image-size', type=int, default=(256, 256), help='Image height and width (default: 256)')
    parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    args = parser.parse_args()
    args.dataset_path = r'/home/jyangcu/Dataset/Dataset_Fetus_Object_Detection/Hospital_1/'

    train_set = fetus_Dataset(args)
    #train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=1)

    #for img, bbox in tqdm(train_loader):
    #    pass