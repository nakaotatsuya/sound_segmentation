#!/usr/bin/env python
# -*- coding:utf-8 -*-


import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
import numpy as np
import soundfile as sf
from scipy import signal
import pandas as pd
import cmath
import re

from torch.utils import data
import  os.path as osp
from torch.utils.data import DataLoader

class SoundSegmentationDataset(data.Dataset):
    def __init__(self, root, split="train", task="ssls", n_classes=1, spatial_type=None, mic_num=16, angular_resolution=40, input_dim=31):
        self.split = split
        self.task = task
        self.root = root

        self.datanum = len(os.listdir(osp.join(self.root, "save"))) // 3
        self.images = torch.from_numpy(np.array([])).float()
        self.labels = torch.from_numpy(np.array([])).float()
        self.phase = torch.from_numpy(np.array([])).float()
        
        
    def __len__(self):
        return self.datanum

    def __getitem__(self, i):
        images = np.load(osp.join(self.root, "save", "images_{:0=5d}.npy".format(i+1)))
        self.images = images[0]
        labels = np.load(osp.join(self.root, "save", "labels_{:0=5d}.npy".format(i+1)))
        self.labels = labels[0]
        phase = np.load(osp.join(self.root, "save", "phase_{:0=5d}.npy".format(i+1)))
        self.phase = phase[0]

        device = "cuda"
        self.images = torch.from_numpy(self.images).float()
        self.labels = torch.from_numpy(self.labels).float()
        self.phase = torch.from_numpy(self.phase).float()
        self.images.to(device)
        self.labels.to(device)
        self.phase.to(device)

        return self.images, self.labels, self.phase

# def collate_fn(batch):
#     images, labels, phase = list(zip(*batch))
#     images = torch.stack(images)
#     labels = torch.stack(labels)
#     phase = torch.stack(phase)
#     return images, labels, phase
                         
if __name__ == "__main__":
    from torch.utils.data.dataloader import default_collate
    root = "/home/jsk/nakao/sound_segmentation/house_audios"

    ssd = SoundSegmentationDataset(root, split="train", spatial_type="ipd")
    #loader = DataLoader(ssd, batch_size=1, num_workers=1, shuffle=True, pin_memory=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    
    #ssd.images.to(device)
    #ssd.labels.to(device)
    #ssd.phase.to(device)
    loader = DataLoader(ssd, batch_size=1, num_workers=1, shuffle=True, pin_memory=True)
    t = time.time()
    
    print("a")
    for i, (images, labels, phase) in enumerate(loader):
        #print(images.shape)
        print(time.time() - t)
        print(images.device)
        pass
    print("all", time.time() - t)
