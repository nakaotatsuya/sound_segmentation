#!/usr/bin/env python

import rospy
from audio_common_msgs.msg import AudioData
from hark_msgs.msg import HarkWave, HarkWaveVal, HarkSource, HarkSourceVal
from jsk_hark_msgs.msg import HarkPower
import numpy as np
import matplotlib.pyplot as plt

import glob
import wavio
import os
import os.path as osp

import rospkg
import scipy.signal as sig

from sklearn.decomposition import PCA
import pyroomacoustics as pr

class AudioToRec():
    def __init__(self):
        rospack = rospkg.RosPack()
        self.mean_flag = False
        self.audio_prefetch = rospy.get_param("~audio_prefetch", 0.5)
        self.rate = rospy.get_param("~rate", 16000)
        self.bitwidth = rospy.get_param("~bitwidth", 2)
        self.bitdepth = rospy.get_param("~bitdepth", 16)

        self.sampling_rate = rospy.get_param("~sampling_rate", 16000)
        self.audio_prefetch_bytes = int(
            self.audio_prefetch * self.rate * self.bitdepth/8)
        self.audio_prefetch_buffer = str()
        self.audio_prefetch_buffer_sparse = np.array([], dtype='int16')
        self.audio_prefetch_sparse_bytes = int(
            self.audio_prefetch_bytes * self.sampling_rate // self.rate // 2)
        self.freq = np.linspace(0, self.sampling_rate, self.audio_prefetch_sparse_bytes)
        self.window_function = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0.0, 1.0, 1.0 / (self.audio_prefetch/8 * self.sampling_rate)))
        #self.window_function = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0.0, 1.0, 1.0 / 160))

        self.first_100 = 0
        self.first = True
        self.a_lis = np.empty((8, 0))
        self.mean_amplitude = 0
        
        #self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        #self.save_dir = os.path.join(self.cur_dir , "../data/")

        self.save_dir = osp.join(rospack.get_path("sound_segmentation"), "audios")
        self.sub = rospy.Subscriber("~input", AudioData, self._callback, queue_size=1000, buff_size=2**24)
        #self.pub = rospy.Publisher("~output", HarkWave, queue_size=1)

        self.save_data_rate = 1
        rospy.Timer(rospy.Duration(1. / self.save_data_rate), self.timer_cb)

    def _callback(self, msg):
        data = msg.data
        data16 = np.frombuffer(data, dtype="int16")
        data16 = np.array(data16)

        data_reshaped = data16.reshape(-1, 8)
        data_reshaped = data_reshaped.T
        
        self.audio_prefetch_buffer_sparse = np.append(
            self.audio_prefetch_buffer_sparse,
            data16)
        self.audio_prefetch_buffer_sparse = self.audio_prefetch_buffer_sparse[-self.audio_prefetch_sparse_bytes:]

        #print(self.audio_prefetch_buffer_sparse.shape)

        if len(self.audio_prefetch_buffer_sparse) != self.audio_prefetch * self.sampling_rate:
            return

        self.a_lis = np.append(self.a_lis, data_reshaped, axis=1)
        self.a_lis = self.a_lis[:, :80000]
        #print(self.a_lis.shape) #(8, 80000)

        #print(self.a_lis[0].max())
        #print(self.a_lis[0].min())

    def timer_cb(self, timer):
        fs=16000
        nperseg = 2048
        print(self.a_lis[0].shape)

        #self.a_lis[0] is the reference mic information.
        if self.a_lis.shape[1] == 80000:
            f, t, Zxx = sig.stft(self.a_lis[0], fs=fs, nperseg=nperseg)
            #print(t.shape) #80
            #print(f.shape) #1025
            print(Zxx[100][30]) #1025*80

            #self.a_lis[1:] is the others'.
            sin_theta_vec = []
            cos_theta_vec = []
            Zxx2_vec = []
            for i in range(7):
                f2, t2, Zxx2 = sig.stft(self.a_lis[i+1], fs=fs, nperseg=nperseg)
                print(Zxx2[100][30])
                #calculate angle diff
                theta1 = np.arctan2(Zxx.imag, Zxx.real)
                theta2 = np.arctan2(Zxx2.imag, Zxx2.real)
                #theta = np.arctan2((Zxx2 - Zxx).imag, (Zxx2 - Zxx).real)
                #print(theta)
                theta = theta2 - theta1
                #print(theta[100][30])
                sin_theta = np.sin(theta)
                #print(sin_theta[0][0])
                cos_theta = np.cos(theta)

                sin_theta_vec.append(sin_theta)
                cos_theta_vec.append(cos_theta)
                Zxx2_vec.append(Zxx2)

            if self.first:
                self.first = False
                fig = plt.figure()
                #print(self.a_lis.T)
                wavio.write(osp.join(self.save_dir, "out.wav"), self.a_lis.T, 16000, sampwidth=3)
                print("Audio wav file was saved")

                #self.ilrma()
                #self.fastmnmf()
                for i in range(7):
                    ax1 = fig.add_subplot(3, 6, i+1)
                    ax2 = fig.add_subplot(3, 6, i+8)

                    ax1.pcolormesh(t, f, sin_theta_vec[i], vmin=-1, vmax=1)
                    #ax2.pcolormesh(t, f, cos_theta_vec[i], vmin=-1, vmax=1)
                    ax2.pcolormesh(t, f, cos_theta_vec[i], vmin=-1, vmax=1)
                    ax1.set_ylim([f[1], f[-1]])
                    ax1.set_title("STFT")
                    ax1.set_ylabel("frequency")
                    ax1.set_xlabel("Time")
                ax = fig.add_subplot(3, 6, 15)
                ax.pcolormesh(t, f, np.abs(Zxx))
                ax.set_ylim([f[1], f[-1]])
                ax.set_title("STFT")
                ax.set_ylabel("frequency")
                ax.set_xlabel("Time")
                #plt.yscale("log")

                # ax = fig.add_subplot(3, 6, 16)
                # ax.pcolormesh(t, f, np.abs(self.sep1_Zxx))
                # ax.set_ylim([f[1], f[-1]])
                # ax.set_title("STFT")
                # ax.set_ylabel("frequency")
                # ax.set_xlabel("Time")

                # ax = fig.add_subplot(3, 6, 17)
                # ax.pcolormesh(t, f, np.abs(self.sep2_Zxx))
                # ax.set_ylim([f[1], f[-1]])
                # ax.set_title("STFT")
                # ax.set_ylabel("frequency")
                # ax.set_xlabel("Time")
                plt.show()

        # if self.first_100 <= 300:
        #     self.a_lis = np.append(self.a_lis, data_reshaped, axis=1)
        #     print(self.a_lis.shape)
        #     #self.a_lis = self.a_lis[:, -80000:]
        #     #print(self.a_lis.shape)
        # else:
        #     if self.first:
        #         wavio.write(osp.join(self.save_dir, "out.wav"), self.a_lis.T, 16000, sampwidth=3)
        #         self.first = False
        #         print("Audio wav file was saved.")
        #     pass
        #self.first_100 += 1
        
    def ilrma(self):
        pca = PCA(n_components=2, whiten=True)
        H = pca.fit_transform(self.a_lis.T)
        #print(H.shape)
        # n = 0.3*16000
        # i = 1
        # while(i*2 <= n):
        #     i *= 2
        # seg = i*2
        fs = 16000
        nperseg = 2048
        stft_list = []
        for i in range(2):
            _,_,Z = sig.stft(H[:,i], fs=fs, nperseg = nperseg)
            stft_list.append(Z.T)
        f_data = np.stack(stft_list, axis=2)
        #print(f_data.shape)
        Array, W_matrix = pr.bss.ilrma(f_data, n_src=None, n_iter=200, proj_back=False, W0=None, n_components=2, return_filters=True, callback=None)
        #print("W:", W_matrix)
        #print(W_matrix.shape)
        print(Array.shape)
        sep = []
        for i in range(2):
            x = sig.istft(Array[:,:,-(i+1)].T, nperseg = nperseg)
            sep.append(x[1])

        sep_1_array = (sep[0]).astype(np.int16)
        wavio.write(os.path.join(self.save_dir, "separate_0.wav"), (sep[0]).astype(np.int16), 16000, sampwidth=3)
        sep_2_array = (sep[1]).astype(np.int16)
        wavio.write(os.path.join(self.save_dir, "separate_1.wav"), (sep[1]).astype(np.int16), 16000, sampwidth=3)

        print(sep_1_array.shape)
        f, t, self.sep1_Zxx = sig.stft(sep_1_array, fs=fs, nperseg=nperseg)
        f, t, self.sep2_Zxx = sig.stft(sep_2_array, fs=fs, nperseg=nperseg)

    def fastmnmf(self):
        fftSize = 2048
        shiftSize = 1024
        win_a = pr.hamming(fftSize)
        win_s = pr.transform.compute_synthesis_window(win_a, shiftSize)

        print("aaa")
        print(self.a_lis.T.shape)
        X = pr.transform.analysis(self.a_lis.T, fftSize, shiftSize, win=win_a)
        print(X.shape)
        print("aaa")
        Y = pr.bss.fastmnmf(X, n_src=2, n_iter=30, n_components=4)
        print("bbb")
        y = pr.transform.synthesis(Y, fftSize, shiftSize, win=win_s)
        print("ccc")
        wavio.write(os.path.join(self.save_dir, "separate_0.wav"), y[:, 0], 16000, sampwidth=3)
        wavio.write(os.path.join(self.save_dir, "separate_1.wav"), y[:, 1], 16000, sampwidth=3)
        print(y[:,0].shape)

        fs = 16000
        nperseg = 2048
        f, t, self.sep1_Zxx = sig.stft(y[:, 0], fs=fs, nperseg=nperseg)
        f, t, self.sep2_Zxx = sig.stft(y[:, 1], fs=fs, nperseg=nperseg)
if __name__ == "__main__":
    rospy.init_node("audio_to_rec")
    ator = AudioToRec()
    rospy.spin()
