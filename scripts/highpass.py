#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from audio_common_msgs.msg import AudioData
from sound_segmentation.msg import AudioHeaderData
from jsk_recognition_msgs.msg import Spectrum
from scipy import signal
import wavio

from os import makedirs, listdir
from os import path as osp
import rospkg
from sound_classification.msg import InSound

from scipy import signal
import cmath
import message_filters
import  matplotlib.pyplot as plt
import soundfile as sf

def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2.0
    wp = 1.0 * fp / fn
    ws = 1.0 * fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "low")
    y = signal.filtfilt(b, a, x)
    return y

def highpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2.0
    wp = 1.0 * fp / fn
    ws = 1.0 * fs / fn
    #print(wp)
    #print(ws)
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    #print(Wn)
    b, a = signal.butter(N, Wn, "high")
    #print(b)
    y = signal.filtfilt(b, a, x)
    return y

if __name__ == "__main__":
    rospack = rospkg.RosPack()
    wav_file_path = osp.join(rospack.get_path(
        "sound_segmentation"), "house_audios")
    wav_file = osp.join(wav_file_path, "visualize", "00026")
    waveform, fs = sf.read(osp.join(wav_file, "tap_sweep.wav"))
    print(waveform.shape)

    mic_sampling_rate = fs
    wave = highpass(waveform, mic_sampling_rate, 2000, 500, 3, 30)
    #wave = lowpass(wave, mic_sampling_rate, 3000, 6000, 3, 40)

    wavio.write(osp.join(wav_file, "filtered.wav"), wave, 16000, sampwidth=3)
