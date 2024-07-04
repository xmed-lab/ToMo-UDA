#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import random
import numpy as np
from PIL import Image
from PIL import ImageFilter

import cv2
import json

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as torch_transforms
from torchvision.transforms import functional as F

from skimage import transform as ski_transforms

import data.util as util
# from data.data_augment import build_strong_augmentation
from utils.boxlist import BoxList

seed_num = 6666
np.set_printoptions(threshold=np.inf)
np.random.seed(seed_num)
random.seed(seed_num)

def inverse_normalize(img):
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

def pytorch_normalize(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()

def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = ski_transforms.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    return pytorch_normalize(img)


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

class COCO_Dataset(Dataset):
    def __init__(self, opt, operation='train', domain='Source'):
        
        self.opt = opt
        self.operation = operation
        self.domain = domain

        if self.domain == "Source":
            self.selected_hospital = self.opt.selected_source_hospital
        elif self.domain == "Target":
            self.selected_hospital = self.opt.selected_target_hospital

        if opt.part == 'heart':
            anno_path = os.path.join(opt.dataset_path_heart, self.selected_hospital, 
                                     self.operation+'.json')
            self.img_path = os.path.join(opt.dataset_path_heart, self.selected_hospital, 'src')
        elif opt.part == 'head':
            anno_path = os.path.join(opt.dataset_path_head, self.selected_hospital, 
                                     self.operation+'.json')
            self.img_path = os.path.join(opt.dataset_path_head, self.selected_hospital, 'src')
        elif opt.part == 'cardiac':
            anno_path = os.path.join(opt.dataset_path_cardiac, self.selected_hospital, self.operation,
                                     'annotations.json')
            self.img_path = os.path.join(opt.dataset_path_cardiac, self.selected_hospital, self.operation)
        elif opt.part == 'mmwhs':
            pass
        else:
            raise ValueError("Dataset Error!")

        with open(anno_path, 'r') as f:
            self.data = json.load(f)

        if self.operation == 'train':
            self.transforms = preset_transform(opt)
            self.aug = build_augmentation(operation)

        else:
            self.transforms = preset_transform(opt, False)

    def __getitem__(self, idx):
        image_info = self.data['images'][idx]
        file_name = image_info['file_name']
        img_path = os.path.join(self.img_path, file_name)
        # info = self.images[self.used_dataset[index]]
        _img = util.read_image(img_path)
        annotations = [anno for anno in self.data['annotations'] if anno['image_id'] == image_info['id']]
        bboxes = [self.xywh_to_xyxy(anno['bbox']) for anno in annotations]
        labels = [anno['category_id'] for anno in annotations]
            
        bboxes = np.stack(bboxes, axis=0)
        labels = np.stack(labels, axis=0)
        # print(info['id'])

        if self.operation == 'train':
            target = BoxList(torch.as_tensor(bboxes.copy()), _img.size, mode='xyxy')
            target.fields['labels'] = torch.as_tensor(labels.copy())
        
            _img = self.aug(_img)
            img, target = self.transforms(_img, target)
            return img, target, idx
        else:
            target = BoxList(torch.as_tensor(bboxes.copy()), _img.size, mode='xyxy')
            target.fields['labels'] = torch.as_tensor(labels.copy())

            img, target = self.transforms(_img, target)
            return img, target, idx

    def __len__(self):
        return len(self.data['images'])

    # load json files
    def read_json(self, annotations_path):
        # Delear subfunction in advance
        def load_json_annotation(annotations_dict, json_file, slice_name):
            annotations_dict[slice_name] = json.load(open(json_file))['annotations']

        annotations_dict = dict()
        # Each path from paths indicate a slice of all slice in A hosptial
        for path in self.FindAllFile(annotations_path):
            slice_name, file_type = path.split(".")
            if file_type == 'json':
                load_json_annotation(annotations_dict, annotations_path + path, slice_name)
            else:
                pass
        
        return annotations_dict

    # The coordinates is : x_min, y_min, x_max, y_max
    def convert_bbox(self, bbox):
        coordinates = np.zeros(4)
        coordinates[0] = bbox[0][0]
        coordinates[1] = bbox[0][1]
        coordinates[2] = bbox[1][0]
        coordinates[3] = bbox[1][1]
        return coordinates.astype(np.float32)
        
    # Traverse all the file of A hospital
    def FindAllFile(self, path):
        file_list = list()
        for root, dirs, files in os.walk(path):
            for file in files:
                file_list.append(file)
        return file_list
    
    def xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h
        return [x_min, y_min, x_max, y_max]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)

        return img, target

    def __repr__(self):
        format_str = self.__class__.__name__ + '('
        for t in self.transforms:
            format_str += '\n'
            format_str += f'    {t}'
        format_str += '\n)'

        return format_str


class Resize:
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)

        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, img_size):
        w, h = img_size
        size = random.choice(self.min_size)
        max_size = self.max_size

        if max_size is not None:
            min_orig = float(min((w, h)))
            max_orig = float(max((w, h)))

            if max_orig / min_orig * size > max_size:
                size = int(round(max_size * min_orig / max_orig))

        if (w <= h and w == size) or (h <= w and h == size):
            return h, w

        if w < h:
            ow = size
            oh = int(size * h / w)

        else:
            oh = size
            ow = int(size * w / h)

        return oh, ow

    def __call__(self, img, target):
        size = self.get_size(img.size)
        img = F.resize(img, size)
        target = target.resize(img.size)

        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = F.hflip(img)
            target = target.transpose(0)

        return img, target


class ToTensor:
    def __call__(self, img, target):
        return F.to_tensor(img), target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, target
    
class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def preset_transform(opt, train=True):

    if train:
        if opt.train_min_size_range[0] == -1:
            min_size = opt.train_min_size

        else:
            min_size = list(
                range(
                    opt.train_min_size_range[0], opt.train_min_size_range[1] + 1
                )
            )

        max_size = opt.train_max_size
        flip = 0.5

    else:
        min_size = opt.test_min_size
        max_size = opt.test_max_size
        flip = 0

    # normalize = Normalize(mean=opt.pixel_mean, std=opt.pixel_std)
    transform = Compose(
        [Resize(min_size, max_size), RandomHorizontalFlip(flip), ToTensor()] #, normalize]
    )

    return transform


class ImageList:
    def __init__(self, tensors, sizes):
        self.tensors = tensors
        self.sizes = sizes

    def to(self, *args, **kwargs):
        tensor = self.tensors.to(*args, **kwargs)

        return ImageList(tensor, self.sizes)
    
def build_augmentation(is_train):

    augmentation = []
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        # augmentation.append(
        #     torch_transforms.RandomApply([torch_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8) # d:0.8
        # )
        # augmentation.append(torch_transforms.RandomGrayscale(p=0.2))
        # augmentation.append(torch_transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5))
        randcrop_transform = torch_transforms.Compose(
            [
                torch_transforms.ToTensor(),
                torch_transforms.RandomErasing(
                    p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"
                ),
                # torch_transforms.RandomErasing(
                #     p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value="random"
                # ),
                # torch_transforms.RandomErasing(
                #     p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value="random"
                # ),
                torch_transforms.ToPILImage(),
            ]
        )
        augmentation.append(randcrop_transform)

    return torch_transforms.Compose(augmentation)


def image_list(tensors, size_divisible=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = (max_size[1] | (stride - 1)) + 1
        max_size[2] = (max_size[2] | (stride - 1)) + 1
        max_size = tuple(max_size)

    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]

    return ImageList(batch, sizes)

def collate_fn(opt):
    def collate_data(batch):
        batch = list(zip(*batch))
        imgs = image_list(batch[0], opt.size_divisible)
        targets = batch[1]
        ids = batch[2]

        return imgs, targets, ids

    return collate_data



if __name__ == '__main__':
    import argparse

    from tqdm import tqdm
    from einops import rearrange

    from torch.utils.data import DataLoader
    from torchvision import utils as vutils

    parser = argparse.ArgumentParser(description="Fetus Object Detection")
    parser.add_argument('--min_size', type=int, default=600, help='Image min size of height and width (default: 600)')
    parser.add_argument('--max_size', type=int, default=1000, help='Image max size of height and width (default: 256)')
    parser.add_argument('--slices', type=list, default=['four_chamber_heart'], help='The selection of Slices, one or more')
    parser.add_argument('--selected-hospital', type=list, default=['Hospital_1', 'Hospital_2'], help='The selection of Hospital, one or more')
    parser.add_argument('--dataset-path', type=str, default='/home/jyangcu/Dataset/Dataset_Fetus_Object_Detection/', help='dataset path')
    #parser.add_argument('--mask-size', type=int, default=8, help='The size of mask patch (default: 16)')
    #parser.add_argument('--mask-ratio', type=float, default=0.6, help='The ratio of masking area in an image (default: 0.75)')
    opt = parser.parse_args()

    train_set = fetus_Dataset(opt)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1)

    for img, bboxs, labels, _ in tqdm(train_loader):
        print(img.shape)
        print(bboxs.shape)
        print(labels.shape)