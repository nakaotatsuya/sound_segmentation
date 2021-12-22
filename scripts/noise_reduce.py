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

def envelope(y, rate, threshold):
    #print(y[0:10])
    #y_mean = maximum_filter1d(np.abs(y), mode="constant", size=rate//20)
    y_mean = maximum_filter1d(np.abs(y), mode="constant", size=rate//20)
    #print(y_mean[0:10])
    mask = np.array([mean > threshold for mean in y_mean])

    #print(mask.shape)
    return mask, y_mean

def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _amp_to_db(x):
    return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

def _db_to_amp(x):
    return librosa.core.db_to_amplitude(x, ref=1.0)

def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)

def reduce_noise(waveform, fs):
    n_fft = 2048
    hop_length =512
    win_length = 2048
    n_std_thresh = 1.5

    #print(waveform)
    mask, y_mean = envelope(waveform, rate=fs, threshold=0.2)
    #print(mask)
    #print(y_mean.shape)
    #print(y_mean)

    audio_clip = waveform * mask
    #print(audio_clip)
    #wavio.write(osp.join(wav_file, "audio_part.wav"), audio_clip, 16000, sampwidth=3)

    noise_clip = waveform * (1 - mask)
    #print(noise_clip)
    #wavio.write(osp.join(wav_file, "noise_part.wav"), noise_clip, 16000, sampwidth=3)

    for i in range(16):
        noise_stft = _stft(noise_clip.T[i], n_fft, hop_length, win_length)
        noise_stft_db = _amp_to_db(np.abs(noise_stft))
        #print(noise_stft_db.shape)

        mean_freq_noise = np.mean(noise_stft_db, axis=1)
        std_freq_noise = np.std(noise_stft_db, axis=1)
        noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
        #print(noise_thresh.shape)

        n_grad_freq = 2
        n_grad_time = 4
        prop_decrease = 1.0

        sig_stft = _stft(audio_clip.T[i], n_fft, hop_length, win_length)
        sig_stft_db = _amp_to_db(np.abs(sig_stft))

        # 時間と頻度でマスクの平滑化フィルターを作成
        smoothing_filter = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_freq + 2),
                ]
            )[1:-1],
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_time + 2),
                ]
            )[1:-1],
        )
        smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
        
        # 時間と周波数のしきい値の計算
        db_thresh = np.repeat(
            np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
            np.shape(sig_stft_db)[1],
            axis=0,
        ).T
        sig_mask = sig_stft_db < db_thresh
        sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
        sig_mask = sig_mask * prop_decrease
        
        #plt.figure(figsize=(15,3))
        #plt.pcolormesh(sig_mask)
        #plt.show()
        mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
        
        sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
        )

        sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
        sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (1j * sig_imag_masked)

        recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
        #print(recovered_signal.shape)

        if i==0:
            recovered_signals = recovered_signal
        else:
            recovered_signals = np.vstack((recovered_signals, recovered_signal))
            #print(recovered_signals.shape)

    return recovered_signals

    
if __name__ == "__main__":
    rospack = rospkg.RosPack()
    # wav_file_path = osp.join(rospack.get_path(
    #     "sound_segmentation"), "audios")
    # wav_file = osp.join(wav_file_path, "wav", "sin")
    # waveform, fs = sf.read(osp.join(wav_file, "sin_00060.wav"))

    root_path = osp.join(rospack.get_path(
        "sound_segmentation"), "house_audios")
    #num = "00004"

    wav_file_path = osp.join(root_path, "noise_val2")
    nums = os.listdir(wav_file_path)
    nums.sort(key=int)

    #nums = nums[17880:]
    for num in nums:
        print(num)
        wav_file_num_path = osp.join(wav_file_path, num)
        filelist = os.listdir(wav_file_num_path)
        print(filelist)
        save_path = osp.join(root_path, "noise_processed_val", num)
        if not osp.exists(save_path):
            os.makedirs(save_path)
            
        for filename in filelist:
            if (filename[-4:] == ".wav") and ("_" in filename):
                #shutil.copy(osp.join(wav_file_num_path, filename), save_path)
                waveform, fs = sf.read(osp.join(wav_file_num_path, filename))
                #wave = highpass(waveform, fs, 2000, 500, 3, 30)
        recovered_signals = reduce_noise(waveform, fs)
        #recovered_signals_with_high = reduce_noise(wave, fs)

        #wavio.write(osp.join(save_path, "highpass_and_ss.wav"), recovered_signals_with_high.T, 16000, sampwidth=3)
        wavio.write(osp.join(save_path, "ss.wav"), recovered_signals.T, 16000, sampwidth=3)
        #wavio.write(osp.join(save_path, "highpass.wav"), wave, 16000, sampwidth=3)


