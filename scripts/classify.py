#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import rospkg
import os.path as osp
import wavio
import math
import random
import os
import sys
import wave
#from os import makedirs, listdir
import os
import shutil
import sys

def classify_ESC():
    rospack = rospkg.RosPack()
    file_path = osp.join(rospack.get_path("sound_segmentation"), "esc50")
    wav_file_path = osp.join(file_path, "wav")
    if not osp.exists(wav_file_path):
        os.makedirs(wav_file_path)

    filelist = os.listdir(file_path)
    sum = 0
    for filename in filelist:
        print(filename)
        if filename[-4:] == ".wav":
            class_label = int(filename[-6:-4])
            if class_label <= 0:
                class_label = abs(class_label)

            print(class_label)

            class_path = osp.join(wav_file_path, "{:0=2d}".format(class_label))
            if not osp.exists(class_path):
                os.makedirs(class_path)
                
            shutil.move(osp.join(file_path, filename), class_path)

            #sys.exit()
            # if filename[-6:-4] == "-0":
            #     print("Category 0")
            #     sum += 1
            # else:
            #     print(int(filename[-6:-4]))

if __name__ == "__main__":
    classify_ESC()
