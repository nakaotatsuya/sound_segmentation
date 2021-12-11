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
wav_file_path = osp.join(file_path, "visualize", "00029")

filelist = os.listdir(wav_file_path)

## sound (time and amp)
for filename in filelist:
    if filename[-4:] == ".wav":
        if "_" in filename:
            waveform, fs = sf.read(osp.join(wav_file_path, filename))

            x = np.arange(waveform.shape[0])
            #print(waveform.shape)
            x = x*1.0 / fs
            print(x)

plt.figure(figsize=(15,3))
plt.plot(x, waveform.T[0])
plt.show()

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
                            
for filename in filelist:
    if filename[-4:] == ".wav":
        if "_" in filename:
            filename="sin_removed_noise.wav"
            #filename="bottle_microwave.wav"
            #filename="filtered.wav"
            waveform, fs = sf.read(osp.join(wav_file_path, filename))
            _, _, stft = signal.stft(x=waveform.T, fs=fs, nperseg=512, return_onesided=False)
            #stft = stft[:, :, 1:len(stft.T)-1]

            print(stft.shape)
            stft = stft[:,:,:duration]
            print(stft.shape)
            mixture_phase = np.angle(stft[0])
            for nchan in range(16):
                if nchan == 0:
                    mixture[nchan] = abs(stft[nchan][:256])
                else:
                    mixture[nchan*2 - 1] = np.cos(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                    mixture[nchan*2] = np.sin(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
            normalize(mixture)

for i in range(input_dim):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.pcolormesh(mixture[i])
    #print(mixture[i].shape)
    #plt.show()
    fig.savefig(osp.join(wav_file_path, "file_{}.jpg".format(i)))


# plt.pcolormesh(mixture[1])
# plt.show()

# plt.pcolormesh(mixture[2])
# plt.show()

# plt.pcolormesh(mixture[15])
# plt.show()

# plt.pcolormesh(mixture[16])
# plt.show()

#plt.pcolormesh(mixture_phase)
#plt.show()

for filename in filelist:
    if filename[-4:] == ".wav":
        if not "_" in filename:
            waveform, fs = sf.read(osp.join(wav_file_path, filename))

            x = np.arange(waveform.shape[0])
            #print(waveform.shape)
            x = x*1.0 / fs
            print(x)
            
            plt.figure(figsize=(15,3))
            plt.plot(x, waveform)
            #plt.show()
