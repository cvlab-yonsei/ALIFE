import os
import random
import copy
import torch
import numpy as np
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset, filter_images, group_images


VOC_CLASSES = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, image_set='train', is_aug=True, transform=None):

        self.root = os.path.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = image_set
        voc_root = self.root
        splits_dir = os.path.join(voc_root, 'list')

        if not os.path.isdir(voc_root):
            raise RuntimeError(
                'Dataset not found or corrupted.' + ' You can use download=True to download it'
                f'at location = {voc_root}'
            )

        if is_aug and image_set == 'train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found"
            split_f = os.path.join(splits_dir, 'train_aug.txt')
        else:
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" '
                f'{split_f}'
            )

        # remove leading \n
        with open(os.path.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [
            (
                os.path.join(voc_root, 'JPEGImages', x[0]+'.jpg'),
                os.path.join(voc_root, 'SegmentationClassAug', x[0]+'.png')
            ) for x in file_names
        ]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def viz_getter(self, index):
        image_path = self.images[index][0]
        raw_image = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(raw_image, target)
        else:
            img = copy.deepcopy(raw_image)
        return image_path, raw_image, img, target

    def __len__(self):
        return len(self.images)


class VOCSegmentationIncremental(data.Dataset):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        overlap=True,
    ):

        full_voc = VOCSegmentation(root, 'train' if train else 'val', is_aug=True, transform=None) 

        self.labels = []
        self.labels_old = []

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(l in labels_old for l in labels), "labels and labels_old must be disjoint sets"

            self.labels = [0] + labels
            self.labels_old = [0] + labels_old

            self.labels_cum = [0] + labels_old + labels
            
            # take index of images with at least one class in labels and all classes in labels+labels_old+[0,255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_voc, labels, labels_old, overlap=overlap)
                if idxs_path is not None:# and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            self.inverted_list = {gt_label: self.labels_cum.index(gt_label) for gt_label in self.labels_cum}
            if train:
                PRIOR = 0 # NOTE: prev (& future for the overlapped setting) classes will be labeled as Bg
                self.target_transform = torch.tensor([PRIOR]*256, dtype=torch.long)
                target_labels = self.labels               # Bg + current classes
                #target_labels = self.labels_old + labels # Bg + prev/current classes
                #target_labels = labels                   # current classes
                for cat, ind in self.inverted_list.items():
                    if cat in target_labels: 
                        self.target_transform[cat] = ind
                self.target_transform[255] = 255 # NOTE (PLOP-style): 'maintain 255' during training
                #self.target_transform[255] = 0  # NOTE ( MiB-style): 'convert 255 into Bg' during training
            else:
                masking_value = 0    # NOTE (PLOP-style): 'convert future classes into  Bg' at test time
                #masking_value = 255 # NOTE ( MiB-style): 'convert future classes into 255' at test time
                self.target_transform = torch.tensor([masking_value]*256, dtype=torch.long)
                target_labels = self.labels_old + labels ### Bg + prev/current classes
                for cat, ind in self.inverted_list.items():
                    if cat in target_labels: 
                        self.target_transform[cat] = ind
                self.target_transform[255] = 255
            self.dataset = Subset(full_voc, idxs, transform, self.target_transform)
        else:
            self.dataset = full_voc

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    def viz_getter(self, index):
        return self.dataset.viz_getter(index)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)
