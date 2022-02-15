#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rospy
from audio_common_msgs.msg import AudioData
from sound_segmentation.msg import AudioHeaderData
from jsk_recognition_msgs.msg import Spectrum
from sensor_msgs.msg import Image
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
import torch
from models import read_model, UNet
from utils import scores, rmse, save_score_array
from utils import plot_class_stft, plot_input
from utils import restore

import cv2
from cv_bridge import CvBridge
#オンラインで方向、分離が可能なようにする。

class Create16Wave():
    def __init__(self):
        self.n_channel = rospy.get_param("~n_channel", 1)
        self.mic_num = self.n_channel * 2
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

        #stft
        self.angular_resolution = 40
        self.input_dim = 31
        self.freq_bins = 256
        self.duration = 96
        self.mixture = np.zeros((self.input_dim, self.freq_bins, self.duration), dtype=np.float32)
        self.mixture_phase = np.zeros((self.freq_bins * 2, self.duration), dtype=np.float32)
        self.labels = np.zeros((self.angular_resolution, self.freq_bins, self.duration), dtype=np.float32)
        
        #rospack
        rospack = rospkg.RosPack()
        self.root = osp.join(rospack.get_path(
            "sound_segmentation"))
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

        #estimate (load model)
        self.model = read_model("UNet", n_classes=1, angular_resolution=40, input_dim=16 * 2 - 1)
        pth_saved_dir = osp.join(self.root, "scripts")
        self.pth_saved_dir = osp.join(pth_saved_dir, "results", "multi", "2021_1218_supervised_ssls_UNet")
        self.model.load(osp.join(self.pth_saved_dir, "UNet.pth"))
        self.model.cuda()
        self.model.eval()

        #sub pub settings
        rospy.Subscriber(
            "~in_sound", InSound, self.cb)
        #sub1 = message_filters.Subscriber("~audio1", AudioData)
        #sub2 = message_filters.Subscriber("~audio2", AudioData)
        sub1 = message_filters.Subscriber("~audio1", AudioHeaderData, queue_size=1000)
        sub2 = message_filters.Subscriber("~audio2", AudioHeaderData, queue_size=1000)
        subs = [sub1, sub2]
        #ts = message_filters.TimeSynchronizer(subs, 100000)
        ts = message_filters.ApproximateTimeSynchronizer(subs, 1, slop=0.01)
        ts.registerCallback(self.audio_cb)
        # rospy.Subscriber(
        #     "~audio", AudioData, self.audio_cb)

        rospy.Timer(rospy.Duration(1. / self.save_data_rate), self.timer_cb)
        self.pub_spectrogram = rospy.Publisher("~output", Image, queue_size=1)
        self.pub_pred_spectrogram = rospy.Publisher("~output_pred", Image, queue_size=1)
        publish_rate = 96.0
        rospy.Timer(rospy.Duration(1.0 / publish_rate), self.timer2_cb)
        self.bridge = CvBridge()

    def cb(self, msg):
        self.in_sound = msg.in_sound
        if self.save_when_sound is False:
            self.in_sound = True

    def audio_cb(self, msg1, msg2):
        data1 = msg1.data
        data2 = msg2.data
        
        audio_buffer = np.frombuffer(data1, dtype=self.dtype)
        audios_buffer = audio_buffer.reshape(-1, 8)
        #print("audios_bufer:", audios_buffer.shape) #160,8
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
        #print("self.audios_buffer:", self.audios_buffer.shape) #24576,8

        self.audios_buffer2 = np.vstack((self.audios_buffer2, audios_buffer2))
        self.audios_buffer2 = self.audios_buffer2[-self.audios_buffer_len:]
        #print(self.audios_buffer.T[0][0:10])

        self.combined_audios_buffer = np.hstack((self.audios_buffer, self.audios_buffer2))
        #print(self.combined_audios_buffer.shape) #24576,16

        self.header = msg1.header

    def timer2_cb(self, timer2):
        if len(self.audio_buffer) != self.audio_buffer_len:
            return

        #input_shape = (samples, channels)
        freq, t, stft = signal.stft(x=self.combined_audios_buffer.T, fs=16000, nperseg=512, return_onesided=False)
        #print(stft.shape)

        stft = stft[:,:,:96]

        self.mixture_phase = np.angle(stft[0])
        for nchan in range(self.mic_num):
            if nchan == 0:
                self.mixture[nchan] = abs(stft[nchan][:256])
            else:
                self.mixture[nchan*2 - 1] = np.cos(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                self.mixture[nchan*2] = np.sin(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
        self.mixture = self.normalize(self.mixture)

        # self.mixture[0] を画像表示　loop でまわす.
        #mixture_torch = torch.from_numpy(self.mixture).float()
        #labels_torch = torch.from_numpy(self.labels).float()
        #print(self.mixture.shape)
        #print(mixture_torch.shape)
        #estimate
        #self.estimate(mixture_torch, labels_torch)
        
        cv_mixture = self.mixture[0][::-1,:]
        #print(cv_mixture.shape)
        cv_mixture = cv2.resize(
            cv_mixture, (96, 256))
        stft_spectrogram = self.bridge.cv2_to_imgmsg(cv_mixture, "32FC1")
        stft_spectrogram.header = self.header
        self.pub_spectrogram.publish(stft_spectrogram)

    def timer_cb(self, timer):
        if len(self.audio_buffer) != self.audio_buffer_len:
            return
        if not self.in_sound:
            #if False:
            return
        else:
            print("--------------")
            mixture_torch = torch.from_numpy(self.mixture).float()
            labels_torch = torch.from_numpy(self.labels).float()
            #print(self.mixture.shape)
            #print(mixture_torch.shape)
            #estimate
            pred = self.estimate(mixture_torch, labels_torch)

            max_power_per_pixel = 0
            cv_pred_list = []
            ppp_list = []
            for i in range(self.angular_resolution):
                cv_pred = pred[0][i][::-1,:]
                power_per_pixel = cv_pred.sum() / cv_pred.size
                print(cv_pred.shape)
                cv_pred = cv2.resize(
                    cv_pred, (96, 256))
                cv_pred_list.append(cv_pred)
                #power_per_pixel = cv_pred.sum() / cv_pred.size
                ppp_list.append(power_per_pixel)

            print(ppp_list)
            for i in range(self.angular_resolution):
                average_ppp = 0
                if i <= 7 or i >= 32:
                    pass
                elif i % 8 == 0:
                    average_ppp = ppp_list[i-8] + ppp_list[i-7] + ppp_list[i-1] + ppp_list[i] + ppp_list[i+1] + ppp_list[i+7] + ppp_list[i+8] + ppp_list[i+9] + ppp_list[i+15]
                elif i % 8 == 7:
                    average_ppp = ppp_list[i+8] + ppp_list[i+7] + ppp_list[i+1] + ppp_list[i] + ppp_list[i-1] + ppp_list[i-7] + ppp_list[i-8] + ppp_list[i-9] + ppp_list[i-15]
                elif i >= 8 and i <= 31:
                    average_ppp = ppp_list[i-9] + ppp_list[i-8] + ppp_list[i-7] + ppp_list[i-1] + ppp_list[i] + ppp_list[i+1] + ppp_list[i+7] + ppp_list[i+8] + ppp_list[i+9]

                if average_ppp > max_power_per_pixel:
                    max_power_per_pixel = average_ppp
                    max_idx = i
            print(max_idx)
            cv_perd = pred[0][max_idx]
            pred_spectrogram = self.bridge.cv2_to_imgmsg(cv_pred, "32FC1")
            self.pub_pred_spectrogram.publish(pred_spectrogram)

    def normalize(self, mixture):
        mixture[0] += 10**-8
        mixture[0] = 20 * np.log10(mixture[0])
        mixture[0] = np.nan_to_num(mixture[0])
        mixture[0] = (mixture[0] + 120) / 120

        mixture = np.clip(mixture, 0.0, 1.0)
        return mixture

    def estimate(self, images, labels):
        with torch.no_grad():
            images = images[None]
            images = images.cuda()
            outputs = self.model(images)
            labels = labels[None]
            labels = labels.cuda()

            X_in = images.data.cpu().numpy()
            pred = outputs.data.cpu().numpy()
            gt = labels.data.cpu().numpy()
        #print("a")
        print(pred.shape)

        #plot_mixture_stft(X_in, no=1, save_dir=pth_saved_dir, pred="real_prediction")
        #plot_input(X_in, no=0, save_dir=self.pth_saved_dir, pred="real_prediction")
        #plot_class_stft(gt, pred, no=0, save_dir=self.pth_saved_dir, classes=1, ang_reso=40, pred="real_prediction")

        return pred

if __name__ == "__main__":
    rospy.init_node("create_16_wav")
    a = Create16Wave()
    rospy.spin()
    #while not rospy.is_shutdown():
    #    plt.pause(.1)  # real-time plotting
