#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import pyroomacoustics as pra
import rospkg
import os.path as osp
import wavio
import math
import random
import os
import sys
import wave

def create_mixed_file(data="audios", sr=16000, val=False):
    rospack = rospkg.RosPack()
    file_path = osp.join(rospack.get_path("sound_segmentation"), data)
    wav_file_path = osp.join(file_path, "wav")
    class_names = os.listdir(wav_file_path)
    chosen_classes = random.sample(class_names, 2)

    class1_path = osp.join(wav_file_path, chosen_classes[0])
    class2_path = osp.join(wav_file_path, chosen_classes[1])
    
    #choose wav file randomly
    file1 = random.choice(os.listdir(class1_path))
    file2 = random.choice(os.listdir(class2_path))
    file1 = osp.join(class1_path, file1)
    file2 = osp.join(class2_path, file2)

    print(file1)
    print(file2)
    wav_data = wavio.read(file1)
    wav_data2 = wavio.read(file2)
    #wav_data = wave.open(osp.join(wav_file_path, "tap/output.wav"), "r")
    #wav_data2 = wavio.read(file2, "r")
    #wav_data = wavio.read(osp.join(wav_file_path, "clap/output1.wav"))
    #wav_data2 = wavio.read(osp.join(wav_file_path, "kettle/output1.wav"))

    #create a simulate room #############################
    corners = np.array([[0,0],[0,6],[6,6],[6,0]]).T
    room = pra.Room.from_corners(corners, fs=wav_data.rate, max_order=3, absorption=0.2)
    room.extrude(3.)

    deg = np.arange(360)
    deg = deg[::15]
    theta = [np.deg2rad(i) for i in deg]
    chosen_theta = random.sample(theta, 2)
    print(np.rad2deg(chosen_theta))
    
    room.add_source([3.+ np.cos(chosen_theta[0]), 3.+ np.sin(chosen_theta[0]), 1.], signal=wav_data.data.T[0])
    room.add_source([3.+ np.cos(chosen_theta[1]), 3.+ np.sin(chosen_theta[1]), 1.], signal=wav_data2.data.T[0])

    R = pra.circular_2D_array(center=[3.,3.], M=8, phi0=0, radius=0.04)
    b = np.array([1,1,1,1,1,1,1,1])[None]
    R = np.concatenate([R, b], axis=0)
    room.add_microphone_array(pra.MicrophoneArray(R, room.fs))

    #room.plot_rir()
    #fig = plt.gcf()
    #fig.set_size_inches(20,10)

    #print(room.rir[0][0].shape)
    room.simulate()
    #print(room.mic_array.signals.shape)
    ############################################3
    
    #save files
    #train dir
    train_path = osp.join(file_path, "train")
    val_path = osp.join(file_path, "val")

    #train or val
    if val:
        train_path = val_path
    if not osp.exists(train_path):
        os.makedirs(train_path)
    file_num = len(os.listdir(train_path)) + 1
    train_path = osp.join(train_path, "{:0=5d}".format(file_num))
    if not osp.exists(train_path):
        os.makedirs(train_path)

    #print(room.mic_array.signals.T.shape)
    #print(wav_data.data.shape)

    # 24000, 220500
    save_wave = room.mic_array.signals.T[:220500]
    #print(save_wave.shape)

    wavio.write(osp.join(train_path, "{}_{}.wav".format(chosen_classes[0], chosen_classes[1])), save_wave, sr, sampwidth=3)

    wavio.write(osp.join(train_path, "{}.wav".format(chosen_classes[0])), wav_data.data, sr, sampwidth=3)
    wavio.write(osp.join(train_path, "{}.wav".format(chosen_classes[1])), wav_data2.data, sr, sampwidth=3)

    with open(osp.join(train_path, "sound_direction.txt"), mode="w") as f:
        for c, t in zip(chosen_classes, chosen_theta):
            f.write(c)
            f.write(" ")
            f.write("{}\n".format(str(t)))

    #wavio.write(osp.join(file_path, "output_dammy_00081.wav"), room.mic_array.signals[0,:], 16000, sampwidth=3)

    #wavio.write(osp.join(file_path, "output_dammy_00081_multi.wav"), room.mic_array.signals.T, 16000, sampwidth=3)

    #fig, ax = room.plot()
    #ax.set_xlim([-1, 7])
    #ax.set_ylim([-1, 7])
    #ax.set_zlim([0, 3])
    #plt.show()

if __name__ == "__main__":
    for i in range(400):
        create_mixed_file(data="esc50", sr=44100, val=False)
