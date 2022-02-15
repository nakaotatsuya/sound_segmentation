#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
#import pyroomacoustics as pra
#import rospkg
import os.path as osp
#import wavio
#import math
#import random
import os
import sys
import wave
import soundfile as sf
import shutil

root_path = "/home/jsk/nakao/sound_segmentation/house_audios"
wav_file_path = osp.join(root_path, "noise_processed_real_val")

nums = os.listdir(wav_file_path)
nums.sort(key=int)
for num in nums:
    wav_file_num_path = osp.join(wav_file_path, num)
    filelist = os.listdir(wav_file_num_path)
    save_path = osp.join(root_path, "real_val", num)

    for filename in filelist:
        if filename == "ss.wav" or filename == "noisereduce.wav":
            print(filename)
            shutil.copy(osp.join(wav_file_num_path, filename), save_path)
    


