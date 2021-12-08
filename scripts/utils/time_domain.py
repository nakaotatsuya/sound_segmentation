import os
import glob
import numpy as np
import cmath
from mir_eval.separation import bss_eval_sources
import soundfile as sf
from scipy import signal
import re
import wavio
import sys

def _pred_dir_make(no, save_dir, pred="prediction"):
    pred_dir = os.path.join(save_dir, pred, str(no))
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    
    return pred_dir

def restore(Y_true, Y_pred, phase, no, save_dir, classes, ang_reso, dataset_dir, pred="prediction"):
    #print("aaa")
    plot_num = classes * ang_reso

    pred_dir = _pred_dir_make(no, save_dir, pred)
    #print(str(no))
    data_dir = os.path.join(dataset_dir, "val", "{:0=5d}".format(no+1))
    #print(data_dir)

    sdr_array = np.zeros((plot_num, 1))
    sir_array = np.zeros((plot_num, 1))
    sar_array = np.zeros((plot_num, 1))

    #wavefile = glob.glob(data_dir + '/0__*.wav')
    #X_wave, _ = sf.read(wavefile[0])

    for index_num in range(plot_num):
        if Y_true[no][index_num].max() > 0:
            print(index_num)
            Y_linear = 10 ** ((Y_pred[no][index_num] * 120 - 120) / 20)
            Y_linear = np.vstack((Y_linear, Y_linear[::-1]))

            Y_complex = np.zeros((Y_true.shape[-2] * 2, Y_true.shape[-1]), dtype=np.complex128)
            for i in range (Y_true.shape[-2] * 2):
                for j in range (Y_true.shape[-1]):
                    Y_complex[i][j] = cmath.rect(Y_linear[i][j], phase[no][i][j])

            if ang_reso == 1:
                #filename = label.index[index_num]+"_prediction.wav"
                filename = str(index_num) + "_prediction.wav"
            else:
                #filename = label.index[index_num // ang_reso] + "_" + str((360 // ang_reso) * (index_num % ang_reso)) + "deg_prediction.wav"
                #filename = str(index_num // ang_reso) + "_" + str((360 // ang_reso) * (index_num % ang_reso)) + "deg_prediction.wav"
                ele = index_num // 8
                azi = index_num % 8

                filename = str(index_num // ang_reso) + "_" + str((360 // 8) * (azi % 8)) + "deg_" + str(ele * 30 - 60) + "deg_prediction.wav"

            sr = 16000 #16000 or 44100
            _, Y_pred_wave = signal.istft(Zxx=Y_complex, fs=sr, nperseg=512, input_onesided=False)
            Y_pred_wave = Y_pred_wave.real
            sf.write(pred_dir + "/" + filename, Y_pred_wave.real, sr, subtype="PCM_16")

            # calculate SDR
            if classes == 1:
                with open(os.path.join(data_dir, "sound_direction.txt"), "r") as f:
                    #directions = f.read().split("\n")[:-1]
                    direction = f.read().split("\n")[:-1]
                c_angle_dict = {}
                for c_angle in direction:
                    c, angle, ele = c_angle.split(" ")
                    c_angle_dict[c] = float(angle)
                #print(c_angle_dict)
                for class_name, angle in c_angle_dict.items():
                    angle = np.rad2deg(angle) + 0.1
                    #print(int(angle))
                    if index_num == int(angle) // (360 // ang_reso):
                        Y_true_wave, _ = sf.read(data_dir + "/" + class_name + ".wav")
                        #print(Y_true_wave.shape)
                    #print(Y_true_wave.shape)
            # else:                
            #     Y_true_wave, _ = sf.read(data_dir + "/" + label.index[index_num // ang_reso] + ".wav")
            
            #Y_true_wave = Y_true_wave[:len(Y_pred_wave)]
            #X_wave = X_wave[:len(Y_pred_wave)]

            #sdr_base, sir_base, sar_base, per_base = bss_eval_sources(Y_true_wave[np.newaxis,:], X_wave[np.newaxis,:], compute_permutation=False)
            #sdr, sir, sar, per = bss_eval_sources(Y_true_wave[np.newaxis,:], Y_pred_wave[np.newaxis,:], compute_permutation=False)
            #print("No.", no, "Class", index_num // ang_reso, int(index_num // ang_reso), "SDR", round(sdr[0], 2), "SDR_Base", round(sdr_base[0], 2), "SDR improvement: ", round(sdr[0] - sdr_base[0], 2))
            
            #sdr_array[index_num] = sdr
            #sir_array[index_num] = sir
            #sar_array[index_num] = sar

    #return sdr_array, sir_array, sar_array

