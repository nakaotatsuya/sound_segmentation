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

#やることとしてはros msg /audioからhigh pass filterをかけてwavファイルを保存する。また、stftをしてパスフィルタが通っているか確認する。

#highパスフィルタでノイズ除去したwavファイルを作成したら、学習してみる。

class Create16Wave():
    def __init__(self):
        self.n_channel = rospy.get_param("~n_channel", 1)
        self.mic_sampling_rate = rospy.get_param("~mic_sampling_rate", 16000)
        bitdepth = rospy.get_param('~bitdepth', 16)
        if bitdepth == 16:
            self.dtype = 'int16'
        else:
            rospy.logerr("'~bitdepth' {} is unsupported.".format(bitdepth))
        self.audio_buffer = np.array([], dtype=self.dtype)
        self.audio_buffer2 = np.array([], dtype=self.dtype)
        self.audio_buffer_len = int(self.mic_sampling_rate * 1.536) #if the period is 1.536 seconds,  stft's image width is 96.
        self.audios_buffer = np.empty((0, self.n_channel), dtype=self.dtype)
        self.audios_buffer2 = np.empty((0, self.n_channel), dtype=self.dtype)
        self.combined_audios_buffer = np.empty((0, self.n_channel*2), dtype=self.dtype)
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

        window_function = np.arange(
            0.0, 1.0, 1.0 / self.audio_buffer_len)
        self.window_function = 0.54 - 0.46 * np.cos(
            2 * np.pi * window_function)

        self.fig = plt.figure(figsize=(8, 5))
        self.fig.suptitle('Spectrum plot', size=12)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.90, bottom=0.1,
                                 wspace=0.2, hspace=0.6)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.grid(True)
        self.ax.set_xlabel('Frequency [Hz]', fontsize=12)
        self.ax.set_ylabel('Amplitude', fontsize=12)
        self.line, = self.ax.plot([0, 0], label='Amplitude of Spectrum')
        
        self.noise_mode = False
        self.save_noises = np.array([])
        rospy.Subscriber(
            "~in_sound", InSound, self.cb)
        #sub1 = message_filters.Subscriber("~audio1", AudioData)
        #sub2 = message_filters.Subscriber("~audio2", AudioData)
        sub1 = message_filters.Subscriber("~audio1", AudioHeaderData, queue_size=1000)
        sub2 = message_filters.Subscriber("~audio2", AudioHeaderData, queue_size=1000)
        subs = [sub1, sub2]
        #ts = message_filters.TimeSynchronizer(subs, 100000)
        ts = message_filters.ApproximateTimeSynchronizer(subs, 100000, slop=0.01)
        ts.registerCallback(self.audio_cb)
        # rospy.Subscriber(
        #     "~audio", AudioData, self.audio_cb)
        rospy.Timer(rospy.Duration(1. / self.save_data_rate), self.timer_cb)

    def cb(self, msg):
        self.in_sound = msg.in_sound
        if self.save_when_sound is False:
            self.in_sound = True
        
    def audio_cb(self, msg1, msg2):
        data1 = msg1.data
        data2 = msg2.data
        
        audio_buffer = np.frombuffer(data1, dtype=self.dtype)
        audios_buffer = audio_buffer.reshape(-1, 8)
        audio_buffer = audio_buffer[0::self.n_channel]

        audio_buffer2 = np.frombuffer(data2, dtype=self.dtype)
        audios_buffer2 = audio_buffer2.reshape(-1, 8)
        audio_buffer2 = audio_buffer2[0::self.n_channel]
        
        #save audio msg to audio buffer
        self.audio_buffer = np.append(self.audio_buffer, audio_buffer)
        self.audio_buffer = self.audio_buffer[-self.audio_buffer_len:]

        self.audio_buffer2 = np.append(self.audio_buffer2, audio_buffer2)
        self.audio_buffer2 = self.audio_buffer2[-self.audio_buffer_len:]
        #print(self.audio_buffer[0:10])

        #8ch audio_buffer
        self.audios_buffer = np.vstack((self.audios_buffer, audios_buffer))
        self.audios_buffer = self.audios_buffer[-self.audios_buffer_len:]

        self.audios_buffer2 = np.vstack((self.audios_buffer2, audios_buffer2))
        self.audios_buffer2 = self.audios_buffer2[-self.audios_buffer_len:]
        #print(self.audios_buffer.T[0][0:10])

        self.combined_audios_buffer = np.hstack((self.audios_buffer, self.audios_buffer2))

    def hpf(self, wave, fs, fe, n):
        nyq = fs/ 2.0
        b, a = signal.butter(1, fe/nyq, btype="high")
        for i in range(0, n):
            wave = signal.filtfilt(b, a, wave)
        return wave
    
    def timer_cb(self, timer):
        if len(self.audio_buffer) != self.audio_buffer_len:
            return

        #print(self.combined_audios_buffer.shape)
        
        #wave = self.hpf(self.audios_buffer2, self.mic_sampling_rate, self.low_cut_freq, 5)

        amplitude = np.fft.fft(self.combined_audios_buffer.T * self.window_function)
        amplitude = np.log(np.abs(amplitude))

        # self.ax.set_xlim((0, 6000))
        # self.ax.set_ylim((0.0, 20))
        # time = np.linspace(0, 6000, self.audios_buffer_len // 2)
        # self.line.set_data(time, amplitude[0][:self.audios_buffer_len//2])
        #plt.plot(time, amplitude[0][:self.audios_buffer_len//2])
        
        #wave = self.hpf(self.combined_audios_buffer, self.mic_sampling_rate, self.low_cut_freq, 5)
        wave = self.combined_audios_buffer
        print(wave.T[0].shape)

        if not self.in_sound:
            #if False:
            return
        else:
            file_num = len(
                listdir(self.target_dir)) + 1
            wav_file_name = osp.join(
                self.target_dir, "{}_{:0=5d}.wav".format(self.target_class, file_num))
            wavio.write(wav_file_name, wave, self.mic_sampling_rate, sampwidth=2)

            wav_mono_file_name = osp.join(
                self.target_dir, "{}_{:0=5d}_mono.wav".format(self.target_class, file_num))
            #wavio.write(wav_mono_file_name, wave.T[0], self.mic_sampling_rate, sampwidth=3)

            wav_raw_file_name = osp.join(
                self.target_dir, "{}_{:0=5d}_raw.wav".format(self.target_class, file_num))
            #wavio.write(wav_raw_file_name, self.combined_audios_buffer, self.mic_sampling_rate, sampwidth=3)
            rospy.loginfo("save wav file:" + wav_file_name)

if __name__ == "__main__":
    rospy.init_node("create_16_wav")
    a = Create16Wave()
    rospy.spin()
    #while not rospy.is_shutdown():
    #    plt.pause(.1)  # real-time plotting
