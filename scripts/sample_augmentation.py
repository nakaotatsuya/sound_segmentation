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
from pydub import AudioSegment

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Normalize

rospack = rospkg.RosPack()
file_path = osp.join(rospack.get_path("sound_segmentation"), "house_audios")
wav_file_path = osp.join(file_path, "wav_house2")
class_names = os.listdir(wav_file_path)

augment = Compose([
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=0.5, max_fraction=0.5, p=0.5),
    Normalize(p=0.5),
    ])

class_names = ["sin"]

for c in class_names:
    print(c)
    class_path = osp.join(wav_file_path, c)
    filenames = os.listdir(class_path)
    for filename in filenames:
        print(filename)
        file_path = osp.join(class_path, filename)
        data, fs = sf.read(file_path)
        augmented_data = augment(samples=data, sample_rate=fs)
        wavio.write(osp.join(class_path, "{}_aug.wav".format(filename[:-4])), augmented_data, fs, sampwidth=2)

