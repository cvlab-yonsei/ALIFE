import os
import random
import numpy as np
import torch
import torch.utils.data as data
import torchvision as tv
from PIL import Image
from torch import distributed

from .utils import Subset, filter_images, group_images

ADE_CLASSES = [
    "void", "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane",
    "grass", "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant",
    "curtain", "chair", "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror",
    "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub",
    "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter",
    "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs",
    "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop",
    "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine",
    "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship",
    "fountain", "conveyer belt", "canopy", "washer", "plaything", "swimming pool", "stool",
    "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven", "ball", "food",
    "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan",
    "fan", "pier", "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator",
    "glass", "clock", "flag"
]


class AdeSegmentation(data.Dataset):

    def __init__(self, root, train=True, transform=None):

        ade_root = os.path.expanduser(root)
        if train:
            split = 'training'
        else:
            split = 'validation'
        annotation_folder = os.path.join(ade_root, 'annotations', split)
        image_folder = os.path.join(ade_root, 'images', split)

        self.images = []
        fnames = sorted(os.listdir(image_folder))
        self.images = [
            (os.path.join(image_folder, x), os.path.join(annotation_folder, x[:-3] + "png"))
            for x in fnames
        ]

        self.transform = transform

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

    def __len__(self):
        return len(self.images)


class AdeSegmentationIncremental(data.Dataset):

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        labels=None,
        labels_old=None,
        idxs_path=None,
        masking=True,
        overlap=True,
    ):

        full_data = AdeSegmentation(root, train)

        self.labels = []
        self.labels_old = []

        if labels is not None:
            # store the labels
            labels_old = labels_old if labels_old is not None else []

            self.__strip_zero(labels)
            self.__strip_zero(labels_old)

            assert not any(l in labels_old for l in labels), "labels and labels_old must be disjoint sets"

            self.labels = labels
            self.labels_old = labels_old

            self.order = [0] + labels_old + labels

            # take index of images with at least one class in labels and all classes in labels+labels_old+[255]
            if idxs_path is not None and os.path.exists(idxs_path):
                idxs = np.load(idxs_path).tolist()
            else:
                idxs = filter_images(full_data, labels, labels_old, overlap=overlap)
                if idxs_path is not None:# and distributed.get_rank() == 0:
                    np.save(idxs_path, np.array(idxs, dtype=int))

            self.inverted_order = {label: self.order.index(label) for label in self.order}
            
            if train:
                PRIOR = 0 # NOTE: prev (& future for the overlapped setting) classes will be labeled as Bg
                self.target_transform = torch.tensor([PRIOR]*256, dtype=torch.long)
                target_labels = self.labels               # Bg + current classes
                #target_labels = self.labels_old + labels # Bg + prev/current classes
                #target_labels = labels                   # current classes
                for cat, ind in self.inverted_order.items():
                    if cat in target_labels: 
                        self.target_transform[cat] = ind
                self.target_transform[255] = 255 # NOTE (PLOP-style): 'maintain 255' during training
                #self.target_transform[255] = 0  # NOTE ( MiB-style): 'convert 255 into Bg' during training 
            else:
                masking_value = 0    # NOTE (PLOP-style): 'convert future classes into  Bg' at test time
                #masking_value = 255 # NOTE ( MiB-style): 'convert future classes into 255' at test time 
                self.target_transform = torch.tensor([masking_value]*256, dtype=torch.long)
                target_labels = self.labels_old + labels ### Bg + prev/current classes
                for cat, ind in self.inverted_order.items():
                    if cat in target_labels: 
                        self.target_transform[cat] = ind
                    self.target_transform[255] = 255
            self.dataset = Subset(full_data, idxs, transform, self.target_transform)
        else:
            self.dataset = full_data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        return self.dataset[index]

    @staticmethod
    def __strip_zero(labels):
        while 0 in labels:
            labels.remove(0)

    def __len__(self):
        return len(self.dataset)
