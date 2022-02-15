#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import rospkg
import os.path as osp
import wavio
import random
import os
import sys
import soundfile as sf
from scipy import signal
#import imageio
from PIL import Image as _Image

rospack = rospkg.RosPack()
file_path = osp.join(rospack.get_path("sound_segmentation"), "house_audios")
#wav_file_path = osp.join(file_path, "val", "00009")

#wav_file_path = osp.join(file_path, "noise_val2", "00004")
wav_file_path = osp.join(file_path, "noise")

#wav_file_path = osp.join(file_path, "noise_processed_real_val")
#nums = os.listdir(wav_file_path)
#nums.sort(key=int)

## spectrogram and phase
duration = 96 #512 or 96
freq_bins = 256
input_dim = 31
mixture = np.zeros((input_dim, freq_bins, duration), dtype=np.float32)
mixture_phase = np.zeros((freq_bins * 2, duration), dtype=np.float32)

def normalize(mixture):
    mixture[0] += 10**-8
    mixture[0] = 20*np.log10(mixture[0])
    mixture[0] = np.nan_to_num(mixture[0])
    mixture[0] = (mixture[0] + 120) / 120

#for num in nums:
#    print(num)
#wav_file_num_path = osp.join(wav_file_path, num)
#filelist = os.listdir(wav_file_num_path)
#filelist = os.listdir(wav_file_path)

filename = osp.join(wav_file_path, "Hz4RCEd6moo_40.wav")

## sound (time and amp)
# for filename in filelist:
#     if filename[-4:] == ".wav":
#         if "_" in filename:
#             waveform, fs = sf.read(osp.join(wav_file_path, filename))

#             x = np.arange(waveform.shape[0])
#             #print(waveform.shape)
#             x = x*1.0 / fs
#             print(x)

# plt.figure(figsize=(15,3))
# plt.plot(x, waveform.T[0])
# plt.show()


#for filename in filelist:
if filename[-4:] == ".wav":
    if "_" in filename:
        waveform, fs = sf.read(osp.join(wav_file_path, filename))

        x = np.arange(waveform.shape[0])
        print(x.shape)
        #print(waveform.shape)
        x = x*1.0 / fs
        print(x)

        plt.figure(figsize=(15,3))
        plt.plot(x, waveform.T[0])
        plt.ylim(-1,1)
        plt.show()
