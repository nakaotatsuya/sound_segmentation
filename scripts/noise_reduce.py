#!/usr/bin/env python

import numpy as np
from scipy.ndimage import maximum_filter1d
import rospkg
import os
import os.path as osp
import wavio
import soundfile as sf

def envelope(y, rate, threshold):
    print(y[0:10])
    y_mean = maximum_filter1d(np.abs(y), mode="constant", size=rate//20)
    print(y_mean[0:10])
    mask = [mean > threshold for mean in y_mean]

    print(mask.shape)
    return mask, y_mean

def _stft(_y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

if __name__ == "__main__":
    rospack = rospkg.RosPack()
    wav_file_path = osp.join(rospack.get_path(
        "sound_segmentation"), "audios")
    
    wav_file = osp.join(wav_file_path, "wav", "bottle")
    waveform, fs = sf.read(osp.join(wav_file, "bottle_00001.wav"))

    #_, _, stft = signal.stft(x=waveform.T, fs=fs, nperseg=512, return_onesided=False)

    n_fft = 2048
    hop_length =512
    win_length = 2048
    n_std_thresh = 1.5

    mask, y_mean = envelope(waveform, rate=fs, threshold=0.5)
    #print(mask)
    #print(y_mean)
    
    # noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    # noise_stft_db = _amp_to_db(np.abs(noise_stft))

    # mean_freq_noise = np.mean(noise_stft_db, axis=1)
    # std_freq_noise = np.std(noise_stft_db, axis=1)
    # noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

