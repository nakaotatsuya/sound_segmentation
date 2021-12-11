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
import soundfile as sf

def create_mixed_file(data="audios", sr=16000, train_type="train", sep=True, noise_flag=True):
    rospack = rospkg.RosPack()
    file_path = osp.join(rospack.get_path("sound_segmentation"), data)
    wav_file_path = osp.join(file_path, "wav_house2")
    class_names = os.listdir(wav_file_path)
    chosen_classes = random.sample(class_names, 2)

    class1_path = osp.join(wav_file_path, chosen_classes[0])
    #class1_path = osp.join(wav_file_path, "sin")
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

    #test_path = "/home/nakaotatsuya/ros/kinetic/src/sound_segmentation/audios/real_val/00001/test_00086.wav"
    #wav_data_test = wavio.read(test_path)
    #print(wav_data_test.data.shape)

    if data=="sep_esc50" or data=="esc50":
        test1 = wav_data.data.copy()
        test1[100000:] = 0

        test2 = wav_data2.data.copy()
        test2[:110000] = 0
    else:
        test1 = wav_data.data.copy()
        test1[12000:] = 0

        test2 = wav_data2.data.copy()
        test2[:12000] = 0

    #create a simulate room #############################
    #corners = np.array([[0,0],[0,6],[6,6],[6,0]]).T

    #room_dim = [6, 6, 3]
    #e_absorption, max_order = pra.inverse_sabine(0.5, room_dim)
    #room = pra.Room.from_corners(corners, fs=wav_data.rate, max_order=3, absorption=0.7)
    
    #room = pra.Room.from_corners(corners, fs=wav_data.rate, materials=pra.Material(0.24), max_order=3)
    #room.extrude(3.)

    #make room simulation x, y, z
    x = 8.3
    y = 6.5
    z = 2.8
    room_dim = [x, y, z]
    e_absorption, max_order = pra.inverse_sabine(0.5, room_dim)
    #print(e_absorption, max_order)

    room = pra.ShoeBox(
        room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=20)

    #set the mic arrays.
    mic_x = random.uniform(1 , x-1)
    mic_y = random.uniform(1 , y-1)
    R = pra.circular_2D_array(center=[mic_x, mic_y], M=8, phi0=0, radius=0.04)
    R2 = pra.circular_2D_array(center=[mic_x,1.02], M=8, phi0=0, radius=0.04)
    b = np.array([0.91]*8)[None]
    c = np.array([mic_y]*8)[None]
    R = np.concatenate([R, b], axis=0)
    R2 = np.concatenate([R2[0][None], c, R2[1][None]], axis=0)

    #print(R.shape)
    #print(R2.shape)
    R_R = np.hstack((R, R2))
    #print(R_R)
    #room.add_microphone_array(pra.MicrophoneArray(R, room.fs))
    #room.add_microphone_array(pra.MicrophoneArray(R2, room.fs))

    room.add_microphone_array(pra.MicrophoneArray(R_R, room.fs))

    #set the sound sources
    deg = np.arange(360)
    deg = deg[::45]
    theta = [np.deg2rad(i) for i in deg]
    chosen_theta = random.sample(theta, 2)
    print(np.rad2deg(chosen_theta))

    ele_deg = [-60, -30, 0, 30, 60]
    ele_theta = [np.deg2rad(i) for i in ele_deg]
    chosen_ele_theta = random.sample(ele_theta, 2)
    print(np.rad2deg(chosen_ele_theta))

    #sep or not sep (for train, not sep  : for val, sep)
    if not sep:
        room.add_source([mic_x + np.cos(chosen_theta[0]) * np.cos(chosen_ele_theta[0]), mic_y + np.sin(chosen_theta[0]) * np.cos(chosen_ele_theta[0]), 0.91 + np.sin(chosen_ele_theta[0])], signal=wav_data.data.T[0])
        room.add_source([mic_x + np.cos(chosen_theta[1]) * np.cos(chosen_ele_theta[1]), mic_y + np.sin(chosen_theta[1]) * np.cos(chosen_ele_theta[1]), 0.91 + np.sin(chosen_ele_theta[1])], signal=wav_data2.data.T[0])
    else:
        radius = 1
        radius2 = 1
        room.add_source([mic_x + radius * np.cos(chosen_theta[0]) * np.cos(chosen_ele_theta[0]), mic_y + radius * np.sin(chosen_theta[0]) * np.cos(chosen_ele_theta[0]), 0.91 + radius * np.sin(chosen_ele_theta[0])], signal=test1.T[0])
        room.add_source([mic_x + radius2 * np.cos(chosen_theta[1]) * np.cos(chosen_ele_theta[1]), mic_y+ radius2 * np.sin(chosen_theta[1]) * np.cos(chosen_ele_theta[1]), 0.91 + radius2 * np.sin(chosen_ele_theta[1])], signal=test2.T[0])

    #add noise
    random_noise_num = random.randint(1,4)
    noise_path = osp.join(file_path, "noise")

    for i in range(random_noise_num):
        noise_file = random.choice(os.listdir(noise_path))
        #noise_file = "EWYlONtV2Tk_0.wav"
        noise_file = osp.join(noise_path, noise_file)
        print(noise_file)

        noise = wavio.read(noise_file)
        #print(noise.data)

        chosen_x = random.uniform(0, x)
        chosen_y = random.uniform(0, y)
        chosen_z = random.uniform(0, z)
        print(chosen_x, chosen_y, chosen_z)

        noise_wave = noise.data.T[0].copy()
        #print(noise_wave.shape)
        noise_wave = noise_wave[:wav_data.data.shape[0]]
        print(noise_wave.shape)
        #print(noise_wave)
        if not noise_wave.shape[0]:
            continue

        if noise_flag:
            print("aaaa")
            room.add_source([chosen_x, chosen_y, chosen_z], signal=noise_wave)
    
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
    visualize_path = osp.join(file_path, "visualize")

    #train or val
    if train_type=="val":
        train_path = val_path
    if train_type=="visualize":
        train_path = visualize_path
    if not osp.exists(train_path):
        os.makedirs(train_path)
    file_num = len(os.listdir(train_path)) + 1
    train_path = osp.join(train_path, "{:0=5d}".format(file_num))
    if not osp.exists(train_path):
        os.makedirs(train_path)

    #print(room.mic_array.signals.T.shape)
    #print(wav_data.data.shape)

    # 24000, 220500
    #save_wave = room.mic_array.signals.T[:220500] #for esc50

    #save_wave = room.mic_array.signals.T[:24000]
    save_wave = room.mic_array.signals.T
    print(save_wave.shape)

    if not sep:
        wavio.write(osp.join(train_path, "{}_{}.wav".format(chosen_classes[0], chosen_classes[1])), save_wave, sr, sampwidth=3)
        wavio.write(osp.join(train_path, "{}.wav".format(chosen_classes[0])), wav_data.data, sr, sampwidth=3)
        wavio.write(osp.join(train_path, "{}.wav".format(chosen_classes[1])), wav_data2.data, sr, sampwidth=3)
    else:
        wavio.write(osp.join(train_path, "{}_{}.wav".format(chosen_classes[0], chosen_classes[1])), save_wave, sr, sampwidth=3)
        wavio.write(osp.join(train_path, "{}.wav".format(chosen_classes[0])), test1, sr, sampwidth=3)
        wavio.write(osp.join(train_path, "{}.wav".format(chosen_classes[1])), test2, sr, sampwidth=3)

    with open(osp.join(train_path, "sound_direction.txt"), mode="w") as f:
        for c, t, ele in zip(chosen_classes, chosen_theta, chosen_ele_theta):
            f.write(c)
            f.write(" ")
            f.write(str(t))
            f.write(" ")
            f.write("{}\n".format(str(ele)))

    # rt60 = room.measure_rt60()
    # print(rt60)
    
    # impulse_responses = room.compute_rir()
    # RIR = np.array(room.rir)
    # print(RIR[0][0].shape)
    # rt60 = pra.experimental.measure_rt60(RIR[0][0], fs=16000)

    # #plt.plot(room.rir[0][0])
    # #plt.show()
    # print(rt60)

    #wavio.write(osp.join(file_path, "output_dammy_00081.wav"), room.mic_array.signals[0,:], 16000, sampwidth=3)

    #wavio.write(osp.join(file_path, "output_dammy_00081_multi.wav"), room.mic_array.signals.T, 16000, sampwidth=3)

    # fig, ax = room.plot()
    # ax.set_xlim([-1, 9])
    # ax.set_ylim([-1, 7])
    # ax.set_zlim([0, 3])
    # plt.show()

if __name__ == "__main__":
    # for i in range(1):
    #     create_mixed_file(data="sep_esc50", sr=44100, val=True, sep=True)

    # for i in range(30000):
    #    create_mixed_file(data="house_audios", sr=16000, train_type="train", sep=False, noise_flag=False)

    # for i in range(1000):
    #    create_mixed_file(data="house_audios", sr=16000, train_type="val", sep=False, noise_flag=False)
    # for i in range(1000):
    #    create_mixed_file(data="house_audios", sr=16000, train_type="val", sep=True, noise_flag=False)

    for i in range(1):
        create_mixed_file(data="house_audios", sr=16000, train_type="visualize", sep=False, noise_flag=True)
    #create_mixed_file(data="audios", sr=16000, val=True)
