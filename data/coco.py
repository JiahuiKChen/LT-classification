# Mostly from: https://github.com/brandontrabucco/da-fusion/blob/main/semantic_aug/datasets/coco.py

# from semantic_aug.few_shot_dataset import FewShotDataset
# from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import glob
import torch
import os
import random

from pycocotools.coco import COCO
from PIL import Image
from collections import defaultdict


COCO_DIR = "/home/karen/vision_datasets/coco2017"
SYNTH_IMAGE_DIR = "/home/karen/synth_fine_tune/COCO_synth_imgs"

TRAIN_IMAGE_DIR = os.path.join(COCO_DIR, "train2017")
VAL_IMAGE_DIR = os.path.join(COCO_DIR, "val2017")

DEFAULT_TRAIN_INSTANCES = os.path.join(
    COCO_DIR, "annotations/instances_train2017.json")
DEFAULT_VAL_INSTANCES = os.path.join(
    COCO_DIR, "annotations/instances_val2017.json")


class COCODataset():

    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    num_classes: int = len(class_names)

    def __init__(self, 
                 synth_image_dir = SYNTH_IMAGE_DIR,
                 cond_method = "embed_cutmix_dropout",
                 split: str = "train", seed: int = 0, 
                 train_image_dir: str = TRAIN_IMAGE_DIR, 
                 val_image_dir: str = VAL_IMAGE_DIR, 
                 train_instances_file: str = DEFAULT_TRAIN_INSTANCES, 
                 val_instances_file: str = DEFAULT_VAL_INSTANCES, 
                 examples_per_class: int = None, 
                 synthetic_probability: float = 0.5,
                 image_size: Tuple[int] = (256, 256)):

        image_dir = {"train": train_image_dir, "val": val_image_dir}[split]
        instances_file = {"train": train_instances_file, "val": val_instances_file}[split]

        class_to_images = defaultdict(list)
        class_to_annotations = defaultdict(list)

        self.cocoapi = COCO(instances_file)
        for image_id, x in self.cocoapi.imgs.items():

            annotations = self.cocoapi.imgToAnns[image_id]
            if len(annotations) == 0: continue

            maximal_ann = max(annotations, key=lambda x: x["area"])
            class_name = self.cocoapi.cats[maximal_ann["category_id"]]["name"]

            class_to_images[class_name].append(
                os.path.join(image_dir, x["file_name"]))
            class_to_annotations[class_name].append(maximal_ann)

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

    def get_image_by_idx(self, idx: int) -> torch.Tensor:

        return Image.open(self.all_images[idx]).convert('RGB')

    def get_label_by_idx(self, idx: int) -> torch.Tensor:

        return self.all_labels[idx]
    
    def get_metadata_by_idx(self, idx: int) -> Dict:

        annotation = self.all_annotations[idx]

        return dict(name=self.class_names[self.all_labels[idx]], 
                    mask=self.cocoapi.annToMask(annotation),
                    **annotation)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.get_label_by_idx(idx)

        if np.random.uniform() < self.synthetic_probability:
            # randomly select synthetic image by label
            image = random.choice(self.synth_label_to_images[label])
            if isinstance(image, str): image = Image.open(image)

        else:
            image = self.get_image_by_idx(idx)

        return self.transform(image), label