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
import soundfile as sf
import cv2
from PIL import Image

rospack = rospkg.RosPack()
file_path = osp.join(rospack.get_path("sound_segmentation"), "scripts/results/multi")
file_path = osp.join(file_path, "2021_1212_supervised_ssls_UNet/real_prediction/0")

mixture_file = osp.join(file_path, "aa_mixture.png")

#im = np.array(Image.open(mixture_file))
im = cv2.imread(mixture_file, cv2.IMREAD_GRAYSCALE)
print(im.shape)
print(im)


#np.set_printoptions(threshold=np.inf)

#print(im[200])


for i in range(40):
    ele = i // 8
    azi = i % 8

    az0_el0 = osp.join(file_path, "aa" + str((360 // 8 ) * (azi % 8)) + "deg_" + str(ele * 30 - 60) + "deg_pred.png")
    #im2 = np.array(Image.open(az0_el0))
    im2 = cv2.imread(az0_el0)
    #print(im2)
    print(im2.shape)

    #im[200] = 0

    #pil_img = Image.fromarray(im)
    #print(pil_img.mode)
    
    #pil_img.save(osp.join(file_path, "test.png"))

    im_mask = im[None].transpose((1,2,0)) * im2

    cv2.imwrite(osp.join(file_path, "test" + str((360 // 8 ) * (azi % 8)) + "deg_" + str(ele * 30 - 60) + "deg_pred.png"), im_mask)
