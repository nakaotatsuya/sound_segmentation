#!/usr/bin/env python

import wave
import numpy as np
import utils
import librosa
#from IPython import embed
import os
import os.path as osp
from sklearn import preprocessing

import rospkg
import rospy

def load_audio(filename, mono=True, fs=16000):
    file_base, file_extension = osp.splitext(filename)
    #print(file_base)
    #print(file_extension)
    if file_extension == ".wav":
        _audio_file = wave.open(filename)

        #audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        print("sample_rate:", sample_rate)
        print("sample_width:", sample_width)
        print("number_of_channels:", number_of_channels)
        print("number_of_frames:", number_of_frames)
        
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError("The length of data is not a multiple of sample size * number of channels.")
        if sample_width > 4:
            raise ValueError("Sample size cannot be bigger than 4 bytes.")

        if sample_width == 3:
            #24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1: sample_width] >> 7) * 255
            audio_data = a.view("<i4").reshape(a.shape[:-1]).T

        if mono:
            audio_data = np.mean(audio_data, axis=0)

        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None

def load_desc_file(_desc_file):
    _desc_dict = dict()
    for line in open(_desc_file):
        words = line.strip().split('\t')
        name = words[0].split('/')[-1]
        if name not in _desc_dict:
            _desc_dict[name] = list()
        _desc_dict[name].append([float(words[2]), float(words[3]), __class_labels[words[-1]]])
    return _desc_dict

def extract_mbe(_y, _sr, _nfft, _nb_mel):
    spec, n_fft = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_nfft//2, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel)
    print(mel_basis.shape) #40,1025
    print(spec.shape) #1025, 157
    #print(n_fft)
    return np.log(np.dot(mel_basis, spec))

#user set parameters
nfft = 2048
win_len = nfft
hop_len = win_len // 2
nb_mel_bands = 40
sr = 16000

is_mono = True

rospack = rospkg.RosPack()
audio_folder = osp.join(rospack.get_path("crnn"), "audios")
audio_filename = "a.wav"
feat_folder = osp.join(rospack.get_path("crnn"), "features")

audio_file = osp.join(audio_folder, audio_filename)
y, sr = load_audio(audio_file, mono=is_mono, fs=sr)
#print(y.shape)
#print(sr)
mbe = None

if is_mono:
    mbe = extract_mbe(y, sr, nfft, nb_mel_bands).T
    print(mbe.shape)
else:
    for ch in range(y.shape[0]):
        mbe_ch = extract_mbe(y[ch, :], sr, nfft, nb_mel_bands).T
        if mbe is None:
            mbe = mbe_ch
        else:
            mbe = np.concatenate((mbe, mbe_ch), 1)

# tmp_feat_file = osp.join(feat_folder, "{}_{}.npz".format(audio_filename, "mono" if is_mono else "bin"))
# np.savez(tmp_feat_file, mbe)
