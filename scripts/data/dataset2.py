#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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
    def __init__(self, root, split="train", task="ssls", n_classes=2, spatial_type=None, mic_num=16, angular_resolution=120, input_dim=31):
        self.split = split
        self.task = task

        self.spatial_type = spatial_type
        self.mic_num = mic_num
        self.angular_resolution = angular_resolution
        self.input_dim = input_dim

        #for esc50 : duration == 512 , for audios : duration = 96
        self.duration = 96 #93 to 96
        self.freq_bins = 256
        self.n_classes = n_classes

        #self.label_csv = #TODO

        if split == "train":
            mode_dir = osp.join(root, "train")
        elif split == "val":
            mode_dir = osp.join(root, "val")
        elif split == "real_val":
            mode_dir = osp.join(root, "real_val")
        else:
            raise ValueError("undefined")

        self.data_pair_folders = []

        datapair_dirs = os.listdir(mode_dir)
        datapair_dirs.sort(key=int)
        for datapair_dir in datapair_dirs:
            datapair_dir = osp.join(mode_dir, datapair_dir)
            if osp.isdir(datapair_dir):
                self.data_pair_folders.append(datapair_dir)

    def __len__(self):
        return len(self.data_pair_folders)

    def __getitem__(self, index):
        if osp.exists(osp.join(self.data_pair_folders[index], "sound_direction.txt")):
            with open(osp.join(self.data_pair_folders[index], "sound_direction.txt"), "r") as f:
                direction = f.read().split("\n")[:-1]
                #print(direction)

            c_angle_dict = {}
            for c_angle in direction:
                c, angle, ele_angle = c_angle.split(" ")
                c_angle_dict[c] = [float(angle), float(ele_angle)]
        else:
            c_angle_dict = {}
        mixture = np.zeros((self.input_dim, self.freq_bins, self.duration), dtype=np.float32)
        mixture_phase = np.zeros((self.freq_bins * 2, self.duration), dtype=np.float32)

        if self.task == "ssls":
            label = np.zeros((self.angular_resolution, self.freq_bins, self.duration), dtype=np.float32)

        direction_index = 0
        filelist = os.listdir(self.data_pair_folders[index])
        for filename in filelist:
            if filename[-4:] == ".wav":
                waveform, fs = sf.read(osp.join(self.data_pair_folders[index], filename))
                #print(waveform.shape) #24000
                #print(fs) #16000
                freq, t, stft = signal.stft(x=waveform.T, fs=fs, nperseg=512, return_onesided=False)
                #print(freq)
                #print(t)
                if "_" in filename:
                    if self.mic_num == 8 * 2:
                        #stft = stft[:, :, 1:len(stft.T) - 1]
                        #prepare 96
                        #stft = stft[:,:,1:len(stft.T)]
                        #stft = np.concatenate((stft, stft[:,:, len(stft.T)-4 : len(stft.T)-1]), axis=2)
                        #print(stft.shape)
                        stft = stft[:,:,:96]
                        ####
                        
                        #prepare 512
                        #stft = stft[:,:, :512]
                        ###
                        mixture_phase = np.angle(stft[0])
                        for nchan in range(self.mic_num):
                            if self.spatial_type == "ipd":
                                if nchan == 0:
                                    mixture[nchan] = abs(stft[nchan][:256])
                                else:
                                    mixture[nchan*2 - 1] = np.cos(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                                    mixture[nchan*2] = np.sin(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                            else:
                                raise ValueError("Please use spatial feature")

                else:
                    #stft = stft[:, 1:len(stft.T) - 1]
                    #print(stft.shape) #256, 93
                    #print(stft[:, len(stft.T) - 4 : len(stft.T) - 1].shape)
                    
                    #prepare 96
                    #stft = stft[:, 1:len(stft.T)]
                    ###stft = np.hstack((stft, stft[:, len(stft.T)-4 : len(stft.T)-1]))
                    stft = stft[:, :96]
                    #print(stft.shape)
                    ###

                    #prepare 512
                    #stft = stft[:, :512]
                    ###
                    
                    angle = c_angle_dict[filename[:-4]][0]
                    ele_angle = c_angle_dict[filename[:-4]][1]
                    #print(filename[:-4])
                    #print(angle)
                    #print(ele_angle)
                    angle = np.rad2deg(angle) + 0.1
                    ele_angle = np.rad2deg(ele_angle) + 0.1
                    #print(angle)
                    angle = int(angle) // (360 // (self.angular_resolution/5))
                    #print(angle)

                    #print(ele_angle)
                    ele_angle = (int(ele_angle) + 60) // 30
                    #print(ele_angle)
                    if self.task == "ssls":
                        #print(abs(stft[:256]))
                        #print(angle + ele_angle*24)
                        label[angle + ele_angle*8] += abs(stft[:256])
                    direction_index += 1


        mixture, label = self.normalize(mixture, label)
        mixture = torch.from_numpy(mixture).float()
        label = torch.from_numpy(label).float()

        return mixture, label, mixture_phase

    def normalize(self, mixture, label):
        if self.spatial_type == "ipd":
            mixture[0] += 10**-8
            mixture[0] = 20* np.log10(mixture[0])
            mixture[0] = np.nan_to_num(mixture[0])
            mixture[0] = (mixture[0] + 120) / 120

        label += 10**-8
        label = 20 * np.log10(label)
        label = np.nan_to_num(label)
        label = (label + 120) / 120

        mixture = np.clip(mixture, 0.0, 1.0)
        label = np.clip(label, 0.0, 1.0)
        #print(label.shape)
        #print(label[3])

        return mixture, label

if __name__ == "__main__":
    import rospkg
    import sys
    rospack = rospkg.RosPack()
    root = osp.join(rospack.get_path("sound_segmentation"), "esc50")
    
    ssd = SoundSegmentationDataset(root, split="val", spatial_type="ipd")

    loader = DataLoader(ssd, batch_size=1)
    for i, (images, labels, phase) in enumerate(loader):
        print(images.shape)
        #print(labels.shape)
        #print(phase.shape)
        sys.exit()
    #print(len(ssd))
