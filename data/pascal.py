# Mostly from https://github.com/brandontrabucco/da-fusion/blob/main/semantic_aug/datasets/pascal.py
# from semantic_aug.few_shot_dataset import FewShotDataset
# from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torch
import os
import random
import glob

from PIL import Image
from collections import defaultdict


PASCAL_DIR = "/datastor1/vision_datasets/VOC2012"
SYNTH_IMAGE_DIR = "/datastor1/jiahuikchen/synth_fine_tune/pascal"

TRAIN_IMAGE_SET = os.path.join(
    PASCAL_DIR, "ImageSets/Segmentation/train.txt")
VAL_IMAGE_SET = os.path.join(
    PASCAL_DIR, "ImageSets/Segmentation/val.txt")

DEFAULT_IMAGE_DIR = os.path.join(PASCAL_DIR, "JPEGImages")
DEFAULT_LABEL_DIR = os.path.join(PASCAL_DIR, "SegmentationClass")
DEFAULT_INSTANCE_DIR = os.path.join(PASCAL_DIR, "SegmentationObject")


class PASCALDataset():

    class_names = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 
        'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 
        'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 
        'sofa', 'train', 'television']

    num_classes: int = len(class_names)

    def __init__(self, 
                 synth_image_dir = SYNTH_IMAGE_DIR,
                 cond_method = "embed_cutmix_dropout",
                 split: str = "train", seed: int = 0, 
                 train_image_set: str = TRAIN_IMAGE_SET, 
                 val_image_set: str = VAL_IMAGE_SET, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 label_dir: str = DEFAULT_LABEL_DIR, 
                 instance_dir: str = DEFAULT_INSTANCE_DIR, 
                 examples_per_class: int = None, 
                 synthetic_probability: float = 0.5,
                 image_size: Tuple[int] = (256, 256)):

        image_set = {"train": train_image_set, "val": val_image_set}[split]

        with open(image_set, "r") as f:
            image_set_lines = [x.strip() for x in f.readlines()]

        class_to_images = defaultdict(list)
        class_to_annotations = defaultdict(list)

        for image_id in image_set_lines:

            labels = os.path.join(label_dir, image_id + ".png")
            instances = os.path.join(instance_dir, image_id + ".png")

            labels = np.asarray(Image.open(labels))
            instances = np.asarray(Image.open(instances))

            instance_ids, pixel_loc, counts = np.unique(
                instances, return_index=True, return_counts=True)

            counts[0] = counts[-1] = 0  # remove background

            argmax_index = counts.argmax()

            mask = np.equal(instances, instance_ids[argmax_index])
            class_name = self.class_names[
                labels.flat[pixel_loc[argmax_index]] - 1]

            class_to_images[class_name].append(
                os.path.join(image_dir, image_id + ".jpg"))
            class_to_annotations[class_name].append(dict(mask=mask))

        rng = np.random.default_rng(seed)
        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.class_to_annotations = {
            key: [class_to_annotations[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.all_images = sum([
            self.class_to_images[key] 
            for key in self.class_names], [])

        self.all_annotations = sum([
            self.class_to_annotations[key] 
            for key in self.class_names], [])

        self.all_labels = [i for i, key in enumerate(
            self.class_names) for _ in self.class_to_images[key]]

        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15.0),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Lambda(lambda x: x.expand(3, *image_size)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                  std=[0.5, 0.5, 0.5])
        ])

        # loading synthetic image paths by class
        synth_img_dir = os.path.join(synth_image_dir, cond_method)
        synth_image_files = sorted(list(glob.glob(os.path.join(synth_img_dir, "*.jpg"))))
        synth_class_to_images = defaultdict(list)
        self.synth_label_to_images = {}
        for image_path in synth_image_files:
            class_name = image_path.split('/')[-1].split('_')[0]
            synth_class_to_images[class_name].append(image_path)
            # get int label from class name
            for i, key in enumerate(self.class_names):
                self.synth_label_to_images[i] = synth_class_to_images[key]

        self.transform = {"train": train_transform, "val": val_transform}[split]

        self.synthetic_probability = synthetic_probability

    def __len__(self):
        
        return len(self.all_images)

    def get_image_by_idx(self, idx: int) -> Image.Image:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> int:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> dict:

        return dict(name=self.class_names[self.all_labels[idx]], 
                    **self.all_annotations[idx])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.get_label_by_idx(idx)

        if np.random.uniform() < self.synthetic_probability:
            # randomly select synthetic image by label
            image = random.choice(self.synth_label_to_images[label])
            if isinstance(image, str): image = Image.open(image)

        else:
            image = self.get_image_by_idx(idx)

        return self.transform(image), label