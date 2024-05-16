# Mostly from: https://github.com/brandontrabucco/da-fusion/blob/main/semantic_aug/datasets/caltech101.py
# from semantic_aug.few_shot_dataset import FewShotDataset
# from semantic_aug.generative_augmentation import GenerativeAugmentation
from typing import Any, Tuple, Dict

import numpy as np
import torchvision.transforms as transforms
import torch
import glob
import os
import random

from PIL import Image
from collections import defaultdict


DEFAULT_IMAGE_DIR = "/datastor1/vision_datasets/caltech-101/101_ObjectCategories"
SYNTH_IMAGE_DIR = "/datastor1/jiahuikchen/synth_fine_tune/caltech"


class CalTech101Dataset():

    class_names = ['accordion', 'airplanes', 'anchor', 'ant', 
        'background google', 'barrel', 'bass', 'beaver', 'binocular', 
        'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 
        'cannon', 'car side', 'ceiling fan', 'cellphone', 'chair', 
        'chandelier', 'cougar body', 'cougar face', 'crab', 'crayfish', 
        'crocodile', 'crocodile head', 'cup', 'dalmatian', 'dollar bill', 
        'dolphin', 'dragonfly', 'electric guitar', 'elephant', 'emu', 
        'euphonium', 'ewer', 'faces', 'faces easy', 'ferry', 'flamingo', 
        'flamingo head', 'garfield', 'gerenuk', 'gramophone', 'grand piano', 
        'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 
        'inline skate', 'joshua tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 
        'leopards', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 
        'menorah', 'metronome', 'minaret', 'motorbikes', 'nautilus', 
        'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 
        'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 
        'scissors', 'scorpion', 'sea horse', 'snoopy', 'soccer ball', 
        'stapler', 'starfish', 'stegosaurus', 'stop sign', 'strawberry', 
        'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water lilly', 
        'wheelchair', 'wild cat', 'windsor chair', 'wrench', 'yin yang']

    num_classes: int = len(class_names)

    def __init__(self, 
                 synth_image_dir = SYNTH_IMAGE_DIR,
                 cond_method = "embed_cutmix_dropout", 
                 split: str = "train", seed: int = 0, 
                 image_dir: str = DEFAULT_IMAGE_DIR, 
                 examples_per_class: int = None, 
                 synthetic_probability: float = 0.5,
                 image_size: Tuple[int] = (256, 256)):

        class_to_images = defaultdict(list)

        for image_path in glob.glob(os.path.join(image_dir, "*/*.jpg")):
            class_name = image_path.split("/")[-2].lower().replace("_", " ")
            class_to_images[class_name].append(image_path)

        rng = np.random.default_rng(seed)

        class_to_ids = {key: rng.permutation(
            len(class_to_images[key])) for key in self.class_names}
        
        class_to_ids = {key: np.array_split(class_to_ids[key], 2)[0 if split == "train" else 1] for key in self.class_names}

        if examples_per_class is not None:
            class_to_ids = {key: ids[:examples_per_class] 
                            for key, ids in class_to_ids.items()}

        self.class_to_images = {
            key: [class_to_images[key][i] for i in ids] 
            for key, ids in class_to_ids.items()}

        self.all_images = sum([
            self.class_to_images[key] 
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

        return dict(name=self.class_names[self.all_labels[idx]])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        label = self.get_label_by_idx(idx)

        if np.random.uniform() < self.synthetic_probability:
            # randomly select synthetic image by label
            image = random.choice(self.synth_label_to_images[label])
            if isinstance(image, str): image = Image.open(image)

        else:
            image = self.get_image_by_idx(idx)

        return self.transform(image), label