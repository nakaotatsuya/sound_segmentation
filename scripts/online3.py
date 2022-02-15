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
from distutils.version import LooseVersion
import noisereduce as nr
import soundfile as sf

from std_msgs.msg import Header
from hark_msgs.msg import HarkSource, HarkWave
from PIL import Image as Image_

### from sound_classification
#import torch
#import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import DataLoader, Dataset
#import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from jsk_recognition_msgs.msg import ClassificationResult
from lstm.lstm import LSTM, LSTM_torch

import skimage.transform

from process_gray_image import img_jet
from train_torch import PreprocessedDataset
#オンラインで方向、分離が可能なようにする。

class Create16Wave():
    def __init__(self):
        self.n_channel = rospy.get_param("~n_channel", 1)
        self.mic_num = self.n_channel
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

        #in_sound setting
        self.ins_buffer = np.empty((self.n_channel, 0))
        self.time = Header()
        self.threshold = rospy.get_param("~threshold", 0.2)
        #stft
        self.angular_resolution = 8
        self.input_dim = 15
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
        self.model = read_model("UNet", n_classes=1, angular_resolution=self.angular_resolution, input_dim=self.input_dim)
        pth_saved_dir = osp.join(self.root, "scripts")
        #self.pth_saved_dir = osp.join(pth_saved_dir, "results", "multi", "2021_1222_supervised_ssls_UNet")
        self.pth_saved_dir = osp.join(pth_saved_dir, "results", "multi", "2022_0104_supervised_ssls_UNet")
        #self.pth_saved_dir = osp.join(pth_saved_dir, "results", "multi", "2022_0127_supervised_ssls_UNet")
        self.model.load(osp.join(self.pth_saved_dir, "UNet.pth"))
        self.model.cuda()
        self.model.eval()

        #classification (load model)
        # self.train_data = "experiment_data"
        # self.dataset = PreprocessedDataset(transform=transforms.ToTensor(), train_data=self.train_data)
        # self.target_names_ordered = self.dataset.target_classes
        # self.target_names = self.target_names_ordered
        # for i, name in enumerate(self.target_names):
        #     if not name.endswith("\n"):
        #         self.target_names[i] = names + "\n"
        # self.model_name = "lstm"
        # self.insize = 227
        # self.c_model = LSTM_torch(n_class=len(self.target_names))
        # ckp_path = osp.join(self.dataset.root, "result_torch", self.model_name, "best_model.pt")
        # optimizer = optim.SGD(self.c_model.parameters(), lr=0.01, momentum=0.9)
        # self.c_model, self.optimizer = self.load_ckp(ckp_path, self.c_model, optimizer)
        # self.device = "cuda"
        # self.c_model.to(self.device)
        
        #sub pub settings
        # rospy.Subscriber(
        #     "~in_sound", InSound, self.cb)
        self.subscribe_in_sound()
        #sub1 = message_filters.Subscriber("~audio1", AudioData)
        #sub2 = message_filters.Subscriber("~audio2", AudioData)
        sub1 = message_filters.Subscriber("~audio1", AudioHeaderData, queue_size=1000)
        sub2 = message_filters.Subscriber("~audio2", AudioHeaderData, queue_size=1000)
        subs = [sub1, sub2]
        #ts = message_filters.TimeSynchronizer(subs, 100000)
        ts = message_filters.ApproximateTimeSynchronizer(subs, 100, slop=0.01)
        ts.registerCallback(self.audio_cb)
        # rospy.Subscriber(
        #     "~audio", AudioData, self.audio_cb)

        rospy.Timer(rospy.Duration(1. / self.save_data_rate), self.timer_cb)
        self.pub_spectrogram = rospy.Publisher("~output", Image, queue_size=1)
        self.pub_pred_spectrogram0 = rospy.Publisher("~output_pred0", Image, queue_size=1)
        self.pub_pred_spectrogram1 = rospy.Publisher("~output_pred1", Image, queue_size=1)
        self.pub_pred_spectrogram2 = rospy.Publisher("~output_pred2", Image, queue_size=1)
        self.pub_pred_spectrogram3 = rospy.Publisher("~output_pred3", Image, queue_size=1)
        self.pub_pred_spectrogram4 = rospy.Publisher("~output_pred4", Image, queue_size=1)
        self.pub_pred_spectrogram5 = rospy.Publisher("~output_pred5", Image, queue_size=1)
        self.pub_pred_spectrogram6 = rospy.Publisher("~output_pred6", Image, queue_size=1)
        self.pub_pred_spectrogram7 = rospy.Publisher("~output_pred7", Image, queue_size=1)

        self.pub_cls0 = rospy.Publisher("~output_cls0", ClassificationResult, queue_size=1)
        self.pub_cls1 = rospy.Publisher("~output_cls1", ClassificationResult, queue_size=1)
        self.pub_cls2 = rospy.Publisher("~output_cls2", ClassificationResult, queue_size=1)
        self.pub_cls3 = rospy.Publisher("~output_cls3", ClassificationResult, queue_size=1)
        self.pub_cls4 = rospy.Publisher("~output_cls4", ClassificationResult, queue_size=1)
        self.pub_cls5 = rospy.Publisher("~output_cls5", ClassificationResult, queue_size=1)
        self.pub_cls6 = rospy.Publisher("~output_cls6", ClassificationResult, queue_size=1)
        self.pub_cls7 = rospy.Publisher("~output_cls7", ClassificationResult, queue_size=1)

        self.pub_max_cls = rospy.Publisher("~output_max_cls", ClassificationResult, queue_size=1)
        self.pub_max_cls_kettle = rospy.Publisher("~output_max_cls_kettle", ClassificationResult, queue_size=1)
        publish_rate = 5.0
        rospy.Timer(rospy.Duration(1.0 / publish_rate), self.timer2_cb)
        self.bridge = CvBridge()

    # def cb(self, msg):
    #     self.in_sound = msg.in_sound
    #     #self.in_sound = True
    #     if self.save_when_sound is False:
    #         self.in_sound = True

    def load_ckp(self, checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return model, optimizer
    
    def subscribe_in_sound(self):
        self.in_sound_sub = rospy.Subscriber("/tamago1/harkwave", HarkWave, self.callback_in_sound, queue_size=1, buff_size=2**24)

    def unsubscribe_in_sound(self):
        self.in_sound_sub.unregister()

    def callback_in_sound(self, ins_msg):
        tests = np.array([])
        for i in range(len(ins_msg.src)):
            test = np.array(ins_msg.src[i].wavedata)
            #print(test.max())
            tests = np.append(tests, test.max())
            #print(tests.shape)
        tests = tests.reshape(self.n_channel, -1)
        #print(tests.shape)
        self.ins_buffer = np.hstack((self.ins_buffer, tests))
        self.ins_buffer = self.ins_buffer[:, -50:]
        #print(self.ins_buffer[0:8])
        #print(self.ins_buffer)
        average = np.average(self.ins_buffer, axis=0)
        max_ = np.max(self.ins_buffer, axis=0)
        # print(average.shape)
        #print(average[0:2])
        #print(max_)
        if np.all(max_[0:2] >= self.threshold):
            self.time = ins_msg.header
            self.in_sound = True
            #print(self.in_sound)
        else:
            self.in_sound = False
            
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
        n_fft = 2048/2
        hop_length = 512/2
        win_length = 2048/2
        n_std_thresh = 1.0
        #print(self.combined_audios_buffer.shape)
        #recovered_signals = nr.reduce_noise(y=self.combined_audios_buffer.T, sr=16000, n_jobs=1, stationary=True, n_std_thresh_stationary=n_std_thresh,  n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        #print(recovered_signals.shape)
        #exit()

        #save
        wav_file_name = osp.join(
            self.target_dir, "test.wav")
        wavio.write(wav_file_name, self.combined_audios_buffer, self.mic_sampling_rate, sampwidth=2)
        waveform, fs = sf.read(wav_file_name)
        #freq, t, stft = signal.stft(x=self.combined_audios_buffer.T, fs=16000, nperseg=512, return_onesided=False)
        freq, t, stft = signal.stft(x=waveform.T, fs=16000, nperseg=512, return_onesided=False)
        #freq, t, stft = signal.stft(x=recovered_signals, fs=16000, nperseg=512, return_onesided=False)
        #print(freq)
        #print(t.shape)
        #print(stft.shape)

        stft = stft[:,:,:96]

        self.mixture_phase = np.angle(stft[0])
        for nchan in range(self.mic_num):
            if nchan == 0:
                self.mixture[nchan] = abs(stft[nchan][:256])
            else:
                self.mixture[nchan*2 - 1] = np.cos(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
                self.mixture[nchan*2] = np.sin(np.angle(stft[0][:256]) - np.angle(stft[nchan][:256]))
        #print("self.mixture:",self.mixture.shape)
        #print(self.mixture[0].max())
        self.mixture = self.normalize(self.mixture)

        # self.mixture[0] を画像表示　loop でまわす.
        #mixture_torch = torch.from_numpy(self.mixture).float()
        #labels_torch = torch.from_numpy(self.labels).float()
        #print(self.mixture.shape)
        #print(mixture_torch.shape)
        #estimate
        #self.estimate(mixture_torch, labels_torch)
        
        cv_mixture = self.mixture[0][::-1,:]
        cv_mixture = (cv_mixture * 255).astype(np.uint8)
        title = "mixture"
        (text_w, text_h), baseline = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        if LooseVersion(cv2.__version__).version[0] < 3:
            line_type = cv2.CV_AA
        else:  # for opencv version > 3
            line_type = cv2.LINE_AA
        cv_mixture = cv2.putText(cv_mixture, title, (0, 15),
                                 cv2.FONT_HERSHEY_PLAIN,
                                 1.4, (255, 255, 255), 2,
                                 line_type)
        cv_mixture = cv2.applyColorMap(cv_mixture, cv2.COLORMAP_JET)
        stft_spectrogram = self.bridge.cv2_to_imgmsg(cv_mixture, "bgr8")
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
            #print(mixture_torch.shape)
            #estimate
            pred = self.estimate(mixture_torch, labels_torch)
            pred_spectrograms = []
            max_cv_preds = []
            cls_msgs = []
            class_labels = []
            
            clap_probs = np.array([])
            fridge_probs = np.array([])

            check_fridges = np.array([])
            check_alls = np.array([])
            
            if np.all(pred[0] <= 0.1):
                print("zero!!")
                return

            for i in range(8):
                max_cv_pred = pred[0][i][::-1,:]
                #print(max_cv_pred.shape)
                #print(max_cv_pred[0][0:10])
                max_cv_pred_ = (max_cv_pred * 255).astype(np.uint8)
                #print(max_cv_pred_)

                #if i==5:
                #    print(np.unravel_index(np.argmax(max_cv_pred_), max_cv_pred_.shape))
                #print(max_cv_pred_.shape) #256,96
                #print(i)
                #print(np.average(max_cv_pred_[176]))

                check_fridge = np.average(max_cv_pred_[174:176])
                check_fridges = np.append(check_fridges, check_fridge)
                check_all = (np.sum(max_cv_pred_[0:160]) + np.sum(max_cv_pred_[180:200])) / (180 * max_cv_pred.shape[1])
                check_alls = np.append(check_alls, check_all)

                #add text
                title = "{} deg".format(i * 45)
                (text_w, text_h), baseline = cv2.getTextSize(
                    title, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                if LooseVersion(cv2.__version__).version[0] < 3:
                    line_type = cv2.CV_AA
                else:  # for opencv version > 3
                    line_type = cv2.LINE_AA
                max_cv_pred_ = cv2.putText(max_cv_pred_, title, (0, 15),
                                           cv2.FONT_HERSHEY_PLAIN,
                                           1.4, (255, 255, 255), 2,
                                           line_type)
                max_cv_pred_ = cv2.applyColorMap(max_cv_pred_, cv2.COLORMAP_JET)
                pred_spectrogram = self.bridge.cv2_to_imgmsg(max_cv_pred_, "bgr8")
                pred_spectrogram.header = self.header
                #print("aa")
                max_cv_preds.append(max_cv_pred_)
                pred_spectrograms.append(pred_spectrogram)

            if pred_spectrograms:
                #check fridge max
                fridge_max_idx = np.argmax(check_fridges)
                fridge_max = np.max(check_fridges)

                #fridge_max_idx = np.argmax(fridge_probs)
                #fridge_max = np.max(fridge_probs)
                #if fridge_max > 0.99:

                #check_fridges[np.where(check_alls >= 20)] = 0
                #check_fridges[kettle_max_idx] = 0
                #print(check_fridges)
                #fridge_max_idx = np.argmax(check_fridges)
                #fridge_max = np.max(check_fridges)
                print(fridge_max)
                print(check_alls[fridge_max_idx])
                if fridge_max > 55 and check_alls[fridge_max_idx] < 15:
                    print(fridge_max_idx)

                    cls_title = "fridge"
                    (text_w, text_h), baseline = cv2.getTextSize(
                        cls_title, cv2.FONT_HERSHEY_PLAIN, 1, 1)
                    if LooseVersion(cv2.__version__).version[0] < 3:
                        line_type = cv2.CV_AA
                    else:  # for opencv version > 3
                        line_type = cv2.LINE_AA
                    max_cv_preds[fridge_max_idx] = cv2.putText(max_cv_preds[fridge_max_idx], cls_title, (0, 35),
                                                               cv2.FONT_HERSHEY_PLAIN,
                                                               1.4, (255, 255, 255), 2,
                                                               line_type)
                    pred_spectrograms[fridge_max_idx] = self.bridge.cv2_to_imgmsg(max_cv_preds[fridge_max_idx], "bgr8")

                    max_cls_msg = ClassificationResult(
                        header = self.header,
                        label_names = ["fridge"],
                        label_proba = [fridge_max_idx * 45])
                    self.pub_max_cls.publish(max_cls_msg)

                self.pub_pred_spectrogram0.publish(pred_spectrograms[0])
                self.pub_pred_spectrogram1.publish(pred_spectrograms[1])
                self.pub_pred_spectrogram2.publish(pred_spectrograms[2])
                self.pub_pred_spectrogram3.publish(pred_spectrograms[3])
                self.pub_pred_spectrogram4.publish(pred_spectrograms[4])
                self.pub_pred_spectrogram5.publish(pred_spectrograms[5])
                self.pub_pred_spectrogram6.publish(pred_spectrograms[6])
                self.pub_pred_spectrogram7.publish(pred_spectrograms[7])

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
        return pred

if __name__ == "__main__":
    rospy.init_node("create_16_wav")
    a = Create16Wave()
    rospy.spin()
    #while not rospy.is_shutdown():
    #    plt.pause(.1)  # real-time plotting
