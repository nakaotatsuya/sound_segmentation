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

rospack = rospkg.RosPack()
file_path = osp.join(rospack.get_path("sound_segmentation"), "house_audios")
wav_file_path = osp.join(file_path, "noise")

files = os.listdir(wav_file_path)

for f in files:
    print(f)
    noise_file = osp.join(wav_file_path, f)
    source = AudioSegment.from_wav(noise_file)

    for i in range(5):
        processed = source - np.random.uniform(5, 30)
        processed.export(osp.join(wav_file_path, f.rstrip(".wav") + "_convert{}.wav".format(i)), format="wav")

# file1 = "microwave_00002_converted.wav"
# file1 = osp.join(class1_path, file1)

# source = AudioSegment.from_wav(file1)
# print(source)
# processed = source - 40
# processed.export(osp.join(wav_file_path, "test.wav"), format="wav")

