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

def envelope(y, rate, threshold):
    #print(y[0:10])
    #y_mean = maximum_filter1d(np.abs(y), mode="constant", size=rate//20)
    y_mean = maximum_filter1d(np.abs(y), mode="constant", size=10)
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

if __name__ == "__main__":
    rospack = rospkg.RosPack()
    wav_file_path = osp.join(rospack.get_path(
        "sound_segmentation"), "audios")
    
    wav_file = osp.join(wav_file_path, "wav", "sin")
    waveform, fs = sf.read(osp.join(wav_file, "sin_00060.wav"))
    print(waveform.shape)

    #_, _, stft = signal.stft(x=waveform.T, fs=fs, nperseg=512, return_onesided=False)

    n_fft = 2048
    hop_length =512
    win_length = 2048
    n_std_thresh = 1.5

    #print(waveform)
    mask, y_mean = envelope(waveform, rate=fs, threshold=0.3)
    #print(mask)
    #print(y_mean.shape)
    #print(y_mean)

    audio_clip = waveform * mask
    #print(audio_clip)
    wavio.write(osp.join(wav_file, "sin_audio_part.wav"), audio_clip, 16000, sampwidth=3)

    noise_clip = waveform * (1 - mask)
    #print(noise_clip)
    wavio.write(osp.join(wav_file, "sin_noise_part.wav"), noise_clip, 16000, sampwidth=3)

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
        print(recovered_signal.shape)

        if i==0:
            recovered_signals = recovered_signal
        else:
            recovered_signals = np.vstack((recovered_signals, recovered_signal))
            print(recovered_signals.shape)

    wavio.write(osp.join(wav_file, "sin_removed_noise.wav"), recovered_signals.T, 16000, sampwidth=3)
