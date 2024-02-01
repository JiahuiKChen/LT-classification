"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""


import torch
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms
import os
import wandb
from PIL import Image

# Image statistics
RGB_statistics = {
    'iNaturalist18': {
        'mean': [0.466, 0.471, 0.380],
        'std': [0.195, 0.194, 0.192]
    },
    'default': {
        'mean': [0.485, 0.456, 0.406],
        'std':[0.229, 0.224, 0.225]
    }
}

CORRUPTED_COUNT = 0

# Data transformation with augmentation
def get_data_transform(split, rgb_mean, rbg_std, key='default'):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]) if key == 'iNaturalist18' else transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, phase, transform=None, synthetic=False, synthetic_root=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        # these are only used during training with synthetic data 
        self.real_img_inds = [] # needed to repeatedly randomly sample real images 
        self.synth_img_inds = []
        self.synth_data_count = 0

        # loading real data
        with open(txt) as f:
            for line in f:
                if phase == 'test':
                   # test data in imagenet data dirs is just val/*.JPEG, txt files have another dir
                   txt_path = line.split()[0].split("/")
                   local_path = os.path.join(txt_path[0], txt_path[-1])
                   self.img_path.append(os.path.join(root, local_path)) 
                else:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                
                # adding labels
                self.labels.append(int(line.split()[1]))

                # if synthetic data is used, track indices of real data
                if phase == 'train' and synthetic:
                    self.real_img_inds = list(range(len(self.labels)))
        
        # add synthetic training data if specified
        if phase == 'train' and synthetic:
            synth_dir = os.fsencode(synthetic_root)
            for file in os.listdir(synth_dir):
                img_name = os.fsdecode(file)
                self.img_path.append(os.path.join(synthetic_root, img_name))
                self.labels.append(int(img_name.split('_')[0]))
                self.synth_data_count += 1
            # track indices of synthetic data 
            self.synth_img_inds = list(range(len(self.real_img_inds), len(self.real_img_inds) + self.synth_data_count))

            if len(self.synth_img_inds) != self.synth_data_count:
                raise ValueError("Synthetic indices don't match synthetic data count")
            if (len(self.real_img_inds) + self.synth_data_count) != len(self.labels):
                raise ValueError("Total number of images doesn't match sum of real and synthetic counts")
            
        if len(self.labels) != len(self.img_path):
            raise ValueError("Number of labels and images doesn't match")
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            try:
                sample = Image.open(f).convert('RGB')
            except:
                CORRUPTED_COUNT += 1
                wandb.log({"corrupted_count": CORRUPTED_COUNT})
                return None

        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index
    

# Synthetic data Sampler
class HalfSynthHalfRealBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.ind_count = batch_size // 2
        self.real_img_inds_len = len(dataset.real_img_inds)
        self.synth_img_inds = dataset.synth_img_inds

    # return a mini batch of half real data and half synthetic data
    # amount of real data is notably less than synthetic, so randomly sample each batch
    # synthetic data is just sequentially accessed
    def __iter__ (self):
        batch = [0] * self.batch_size
        batch_ind = 0
        # add each synthetic index to the first half of the batch
        for synth_ind in self.synth_img_inds:
            batch[batch_ind] = synth_ind
            batch_ind += 1
            # once half the batch is filled with synthetic data,
            # randomly sample last half of batch from all real image indices
            if batch_ind == self.ind_count:
                real_inds = torch.randint(high=self.real_img_inds_len, size=(self.ind_count,)).tolist()
                batch[batch_ind:] = real_inds
                # shuffle indices so that synthetic and real images are mixed
                random.shuffle(batch)
                yield batch 
                batch_ind = 0
                batch = [0] * self.batch_size 
    
    def __len__ (self):
        return self.batch_size

# Load datasets
def load_data(data_root, dataset, phase, batch_size, synth_data, synth_root, data_subset=None, sampler_dic=None, num_workers=4, test_open=False, shuffle=True):

    if phase == 'train_plain':
        txt_split = 'train'
    elif phase == 'train_val':
        txt_split = 'val'
        phase = 'train'
    else:
        txt_split = phase

    if data_subset:
        txt = './data/%s/%s_%s_%s.txt'%(dataset, dataset, txt_split, data_subset)
    else:
        txt = './data/%s/%s_%s.txt'%(dataset, dataset, txt_split)
    # txt = './data/%s/%s_%s.txt'%(dataset, dataset, (phase if phase != 'train_plain' else 'train'))

    print('Loading data from %s' % (txt))

    if dataset == 'iNaturalist18':
        print('===> Loading iNaturalist18 statistics')
        key = 'iNaturalist18'
    else:
        key = 'default'
    rgb_mean, rgb_std = RGB_statistics[key]['mean'], RGB_statistics[key]['std']

    if phase not in ['train', 'val']:
        transform = get_data_transform('test', rgb_mean, rgb_std, key)
    else:
        transform = get_data_transform(phase, rgb_mean, rgb_std, key)

    print('Use data transformation:', transform)

    # Pass in synthetic data if specified, only during train
    if phase == 'train' and synth_data:
        set_ = LT_Dataset(data_root, txt, phase, transform, synth_data, synth_root)
    else: 
        set_ = LT_Dataset(data_root, txt, phase, transform)
    print(len(set_))

    if phase == 'test' and test_open:
        open_txt = './data/%s/%s_open.txt'%(dataset, dataset)
        print('Testing with opensets from %s'%(open_txt))
        open_set_ = LT_Dataset('./data/%s/%s_open'%(dataset, dataset), open_txt, transform)
        set_ = ConcatDataset([set_, open_set_])

    if sampler_dic and phase == 'train':
        print('Using sampler: ', sampler_dic['sampler'])
        # print('Sample %s samples per-class.' % sampler_dic['num_samples_cls'])
        print('Sampler parameters: ', sampler_dic['params'])
        
        return DataLoader(dataset=set_, batch_size=batch_size, shuffle=False, 
                           sampler=sampler_dic['sampler'](set_, **sampler_dic['params']),
                           num_workers=num_workers)
    elif synth_data and phase == 'train':
        # pass synthetic data minibatch sampler into Dataloader 
        # to ensure each minibatch has 50% synthetic images and 50% real images
        # batch sampler is mutually exclusive with batch_size and shuffle params 
       print("Using minibatch balancing HalfSynthHalfRealBatchSampler")
       return DataLoader(dataset=set_, 
                         batch_sampler=HalfSynthHalfRealBatchSampler(dataset=set_, batch_size=batch_size), 
                         num_workers=num_workers) 
    else:
        print('No sampler.')
        print('Shuffle is %s.' % (shuffle))
        return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
