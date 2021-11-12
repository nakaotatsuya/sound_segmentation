#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from audio_common_msgs.msg import AudioData
from jsk_recognition_msgs.msg import Spectrum
from scipy import signal
import wavio

from os import makedirs, listdir
from os import path as osp
import rospkg
from sound_classification.msg import InSound

from scipy import signal
import cmath

#やることとしてはros msg /audioからhigh pass filterをかけてwavファイルを保存する。また、stftをしてパスフィルタが通っているか確認する。

#highパスフィルタでノイズ除去したwavファイルを作成したら、学習してみる。

class CreateWave():
    def __init__(self):
        self.n_channel = rospy.get_param("~n_channel", 1)
        self.mic_sampling_rate = rospy.get_param("~mic_sampling_rate", 16000)
        bitdepth = rospy.get_param('~bitdepth', 16)
        if bitdepth == 16:
            self.dtype = 'int16'
        else:
            rospy.logerr("'~bitdepth' {} is unsupported.".format(bitdepth))
        self.audio_buffer = np.array([], dtype=self.dtype)
        self.audio_buffer_len = int(self.mic_sampling_rate * 1.536) #if the period is 1.536 seconds,  stft's image width is 96.
        self.audios_buffer = np.empty((0, self.n_channel), dtype=self.dtype)
        self.audios_buffer_len = int(self.mic_sampling_rate * 1.536) #if the period is 1.536 seconds,  stft's image width is 96.
        self.save_data_rate = rospy.get_param("~save_data_rate")
        self.save_when_sound = rospy.get_param("~save_when_sound")
        self.in_sound = False
        self.target_class = rospy.get_param("~target_class")

        high_cut_freq = rospy.get_param("~high_cut_freq", 6000)
        if high_cut_freq > self.mic_sampling_rate / 2:
            rospy.logerr('Set high_cut_freq lower than {} Hz'.format(
                self.mic_sampling_rate / 2))
        self.low_cut_freq = rospy.get_param("~low_cut_freq", 2000)

        rospack = rospkg.RosPack()
        self.root_dir = osp.join(rospack.get_path(
            "sound_segmentation"), "audios")
        if not osp.exists(self.root_dir):
            makedirs(self.root_dir)
        self.wav_save_dir = osp.join(self.root_dir, "wav")
        if not osp.exists(self.wav_save_dir):
            makedirs(self.wav_save_dir)
        self.target_dir = osp.join(self.wav_save_dir, self.target_class)
        if not osp.exists(self.target_dir):
            makedirs(self.target_dir)

        self.noise_mode = False
        self.save_noises = np.array([])
        rospy.Subscriber(
            "~in_sound", InSound, self.cb)
        rospy.Subscriber(
            "~audio", AudioData, self.audio_cb)
        rospy.Timer(rospy.Duration(1. / self.save_data_rate), self.timer_cb)

    def cb(self, msg):
        self.in_sound = msg.in_sound
        if self.save_when_sound is False:
            self.in_sound = True
        
    def audio_cb(self, msg):
        data = msg.data
        audio_buffer = np.frombuffer(data, dtype=self.dtype)
        audios_buffer = audio_buffer.reshape(-1, 8)
        audio_buffer = audio_buffer[0::self.n_channel]
        #save audio msg to audio buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_buffer)
        self.audio_buffer = self.audio_buffer[-self.audio_buffer_len:]
        #print(self.audio_buffer[0:10])

        #8ch audio_buffer
        self.audios_buffer = np.vstack((self.audios_buffer, audios_buffer))
        self.audios_buffer = self.audios_buffer[-self.audios_buffer_len:]
        #print(self.audios_buffer.T[0][0:10])

    def hpf(self, wave, fs, fe, n):
        nyq = fs/ 2.0
        b, a = signal.butter(1, fe/nyq, btype="high")
        for i in range(0, n):
            wave = signal.filtfilt(b, a, wave)
        return wave
    
    def timer_cb(self, timer):
        if len(self.audio_buffer) != self.audio_buffer_len:
            return

        #wave = self.hpf(self.audio_buffer, self.mic_sampling_rate, self.low_cut_freq, 5)

        wave = self.hpf(self.audios_buffer, self.mic_sampling_rate, self.low_cut_freq, 5)
        #print(wave.shape)

        if not self.in_sound:
            #if False:
            return
        else:
            file_num = len(
                listdir(self.target_dir)) + 1
            wav_file_name = osp.join(
                self.target_dir, "{}_{:0=5d}.wav".format(self.target_class, file_num))
            wavio.write(wav_file_name, wave.T[0], self.mic_sampling_rate, sampwidth=3)

            wav_raw_file_name = osp.join(
                self.target_dir, "{}_{:0=5d}_raw.wav".format(self.target_class, file_num))
            #wavio.write(wav_raw_file_name, self.audio_buffer, self.mic_sampling_rate, sampwidth=3)
            rospy.loginfo("save wav file:" + wav_file_name)

            nperseg = 256
            freq_bins = 128
            time_bins = 192
            _, _, stft = signal.stft(x=self.audios_buffer.T, fs=self.mic_sampling_rate, nperseg=nperseg, return_onesided=False)

            #print(stft.shape)
            stft = stft[:, :, 1:len(stft.T)]
            #print(stft.shape) #8,512,96
            phase = np.zeros((freq_bins*2, time_bins), dtype=np.float32)
            phase = np.angle(stft[0])
            #print(phase.shape) #512, 96

            Y_linear = abs(stft[0][:freq_bins])
            Y_linear = self.normalize(Y_linear)

            #noise subtraction
            #print(Y_linear.min(), Y_linear.max())
            if self.noise_mode:
                #print(Y_linear.shape)
                save_noise = np.average(Y_linear, axis=1)
                #print(save_noise.shape)
                if len(self.save_noises) == 0:
                    self.save_noises = save_noise[None]
                else:
                    self.save_noises = np.append(self.save_noises, save_noise[None], axis=0)
                #print(self.save_noises.shape)
                np.save(osp.join(self.wav_save_dir, "noise.npy"), self.save_noises)
                rospy.loginfo('Save {} noise samples.'.format(len(self.save_noises)))

            if osp.exists(osp.join(self.wav_save_dir, "noise.npy")):
                noise_data = np.load(osp.join(self.wav_save_dir, "noise.npy"))
                self.mean_noise = np.mean(noise_data, axis=0)
            else:
                self.mean_noise = 0

            #self.mean_noise = 0
            #noise_subtraction
            Y_linear = Y_linear.T - self.mean_noise
            Y_linear = Y_linear.T
            print(Y_linear.shape)
            #denormalize
            Y_linear = 10 ** ((Y_linear * 120 - 120) / 20)
            Y_linear = np.vstack((Y_linear, Y_linear[::-1]))
            
            Y_complex = np.zeros((freq_bins*2, time_bins), dtype=np.complex128)
            #print(Y_linear.shape) #512, 96
            for i in range(freq_bins*2):
                for j in range(time_bins):
                    Y_complex[i][j] = cmath.rect(Y_linear[i][j], phase[i][j])
            #print(Y_complex[10][10])

            _, converted_wave = signal.istft(Zxx=Y_complex, fs=self.mic_sampling_rate, nperseg=nperseg, input_onesided=False)
            converted_wave = converted_wave.real
            #print(converted_wave)
            
            wav_converted_file_name = osp.join(
                self.target_dir, "{}_{:0=5d}_converted.wav".format(self.target_class, file_num))
            #wavio.write(wav_converted_file_name, converted_wave.real, self.mic_sampling_rate, sampwidth=3)


    def normalize(self, stft_wave):
        wave = stft_wave + 10**-8
        wave = 20 * np.log10(wave)
        wave = np.nan_to_num(wave)
        wave = (wave + 120) / 120

        wave = np.clip(wave, 0.0, 1.0)
        return wave

if __name__ == "__main__":
    rospy.init_node("create_wav")
    a = CreateWave()
    rospy.spin()
