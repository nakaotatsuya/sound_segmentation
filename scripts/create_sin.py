#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from scipy import signal
import wavio

from os import makedirs, listdir
from os import path as osp
import rospkg
import  matplotlib.pyplot as plt
import struct

import wave

def create_sin():
    amp = 1
    fs = 16000
    f0 = 2000
    sec = 3

    point = np.arange(0, fs*sec)
    sin_wave = amp * np.sin(2*np.pi*f0*point/fs)
    #for n in np.arange(fs * sec):
    #    s = amp * np.sin(2.0 * np.pi * f0 * n / fs)
    #    wave.append(s)
    #print(wave)
    t = np.arange(0, len(sin_wave)) / 1.0 / fs
    #print(t)

    #plt.plot(t, sin_wave)
    #plt.xlim([0, 0.01])
    #plt.show()

    sin_wave = [int(x * 32767.0) for x in sin_wave] #16 bit signed-integer
    sin_wave = np.array(sin_wave)
    print(sin_wave.shape)
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)

    for i in [1,3,5]:
        sin_wave[i* 10000 : i * 10000 + 10000] = 0
    #print(len(binwave))
    #print(binwave[0])
    #binwave = np.array(binwave)
    #print(binwave.shape)
    
    rospack = rospkg.RosPack()
    target_class = "sin"
    root_dir = osp.join(rospack.get_path(
            "sound_segmentation"), "house_audios")
    wav_save_dir = osp.join(root_dir, "wav_house2")
    target_dir = osp.join(wav_save_dir, target_class)
    if not osp.exists(target_dir):
        makedirs(target_dir)

    file_num = len(
        listdir(target_dir)) + 1
    wav_file_name = osp.join(
        target_dir, "{}_{:0=5d}.wav".format(target_class, file_num))

    #w = wave.Wave_write(wav_file_name)
    #p = (1,2,fs,len(binwave), "NONE", "not compressed")
    #w.setparams(p)
    #w.writeframes(binwave)
    #w.close()
    wavio.write(wav_file_name, sin_wave, fs, sampwidth=2)
    
if __name__ == "__main__":
    create_sin()
