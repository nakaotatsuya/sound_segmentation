#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.ndimage import maximum_filter1d
import rospkg
import os
import os.path as osp
import wavio
import soundfile as sf
import librosa
import scipy

import matplotlib.pyplot as plt

from os import makedirs, listdir
import rospkg

from scipy import signal
import cmath
import soundfile as sf

import shutil
import noisereduce as nr
from scipy.io import wavfile
    
if __name__ == "__main__":
    rospack = rospkg.RosPack()
    # wav_file_path = osp.join(rospack.get_path(
    #     "sound_segmentation"), "audios")
    # wav_file = osp.join(wav_file_path, "wav", "sin")
    # waveform, fs = sf.read(osp.join(wav_file, "sin_00060.wav"))

    root_path = osp.join(rospack.get_path(
        "sound_segmentation"), "house_audios")
    #num = "00004"

    wav_file_path = osp.join(root_path, "real_val")
    nums = os.listdir(wav_file_path)
    nums.sort(key=int)

    #nums = nums[6900:]
    for num in nums:
        print(num)
        wav_file_num_path = osp.join(wav_file_path, num)
        filelist = os.listdir(wav_file_num_path)
        print(filelist)
        save_path = osp.join(root_path, "noise_processed_real_val", num)
        if not osp.exists(save_path):
            os.makedirs(save_path)
            
        for filename in filelist:
            if (filename[-4:] == ".wav") and ("_" in filename):
                #shutil.copy(osp.join(wav_file_num_path, filename), save_path)
                waveform, fs = sf.read(osp.join(wav_file_num_path, filename))

                #wav_noise, _ = sf.read(osp.join(root_path, "noise", "noise_00020.wav"))
                #fs, waveform = wavfile.read(osp.join(wav_file_num_path, filename))
                #wave = highpass(waveform, fs, 2000, 500, 3, 30)
        #recovered_signals = reduce_noise(waveform, fs)
        #recovered_signals_with_high = reduce_noise(wave, fs)
        print(waveform.shape)
        #waveform *= 32768
        n_fft = 2048 /2 
        hop_length =512 /2
        win_length = 2048 /2
        n_std_thresh = 1.0
        #recovered_signals = nr.reduce_noise(y=waveform.T, sr=fs, y_noise=wav_noise, n_jobs=-1, stationary=False, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        recovered_signals = nr.reduce_noise(y=waveform.T, sr=fs, n_jobs=1, stationary=True, n_std_thresh_stationary=n_std_thresh,  n_fft=n_fft, win_length=win_length, hop_length=hop_length)

        #wavio.write(osp.join(save_path, "highpass_and_ss.wav"), recovered_signals_with_high.T, 16000, sampwidth=3)
        wavio.write(osp.join(save_path, "noisereduce.wav"), recovered_signals.T, 16000, sampwidth=3)
        #wavio.write(osp.join(save_path, "highpass.wav"), wave, 16000, sampwidth=3)


