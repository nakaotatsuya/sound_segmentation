#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
#from IPython.display import Audio 
import soundfile as sf
import math
import wavio
import os.path as osp

import sounddevice as sd
import pyroomacoustics as pra

from scipy.io.wavfile import read
from math import log10
from scipy.signal import sosfilt, butter
#import japanize_matplotlib

def normal_tsp(n, gain=100, repeat=1):
    N = 2**n
    m = N//4

    A = 50
    L = N//2 - m
    k = np.arange(0, N)

    tsp_freq = np.zeros(N, dtype=np.complex128)
    tsp_exp = np.exp(-1j*4*m*np.pi*(k/N)**2)

    tsp_freq[0:N//2+1] = tsp_exp[0:N//2+1]
    tsp_freq[N//2+1: N+1] = np.conj(tsp_exp[1 : N//2][::-1])

    tsp_inv_freq = 1 / tsp_freq

    tsp = np.real(np.fft.ifft(tsp_freq))
    tsp = gain * np.roll(tsp, L)

    tsp_repeat = np.r_[np.tile(tsp, repeat), np.zeros(N)]

    tsp_inv = np.real(np.fft.ifft(tsp_inv_freq))
    tsp_inv =  gain * np.roll(tsp_inv, -L)

    tsp_inv_repeat = np.r_[np.tile(tsp_inv, repeat), np.zeros(N)]

    return tsp_repeat, tsp_inv

def pink_tsp(n, gain=100, repeat=1):

    N = 2**n
    m = N//4

    L = N//2 - m
    k = np.arange(1, N)

    a = 4 * m * np.pi / (N * np.log(N/2))

    tsp_freq = np.zeros(N, dtype=np.complex128)
    tsp_exp = np.exp(1.j * a * k * np.log(k)) / np.sqrt(k)

    tsp_freq[0] = 1
    tsp_freq[1:N//2+1] = tsp_exp[1:N//2+1]
    tsp_freq[N//2+1: N+1] = np.conj(tsp_exp[1 : N//2][::-1])

    tsp_inv_freq = 1 / tsp_freq

    tsp = gain * np.real(np.fft.ifft(tsp_freq))[::-1]
    tsp =  gain * np.roll(tsp, L)
    tsp_repeat = np.r_[np.tile(tsp, repeat), np.zeros(N)]

    tsp_inv = np.real(np.fft.ifft(tsp_inv_freq))[::-1]
    tsp_inv =  gain * np.roll(tsp_inv, L)


    return tsp_repeat, tsp_inv


file_path = "/home/nakaotatsuya/ros/kinetic/src/sound_segmentation/tsp"
normal_tsp, normal_tsp_inv = pink_tsp(14, repeat=33)

sf.write(osp.join(file_path, "normal_tsp.wav"), normal_tsp, 16000)

def sychronous_addition(filename, repeat, N):
    '''
    filename : Name of wav file (str)
    repeat : Number of repetition (int)
    N : Length of input signal (int)
    '''
    data, fs = sf.read(filename)

    # add zeros if length is too short
    print(data.shape)
    data = data.T[4]
    print(data.shape)

    print(repeat *N)
    print(len(data))
    if len(data) < repeat * N:
        print("a")
        data = np.r_[data, np.zeros(repeat * N - len(data))]

    mean = np.zeros(N)
    for i in range(repeat+1):
        mean += data[i * N : (i + 1) * N]
    mean = mean / repeat

    return mean


n = 14
N = 2 ** n
fs = 16000
response_file = "/home/nakaotatsuya/ros/kinetic/src/sound_segmentation/tsp/response10.wav"
pink_mean = sychronous_addition(response_file, 8, N)

pink_tsp, pink_tsp_inv = pink_tsp(n)
pink_tsp_inv_freq = np.fft.fft(pink_tsp_inv)

H_pink = np.fft.fft(pink_mean) * pink_tsp_inv_freq
h_pink = np.fft.ifft(H_pink)

f = np.linspace(0, fs/2, N//2)

plt.plot(f[1:], 20*np.log10(np.abs(H_pink[1:N//2])/np.max(np.abs(H_pink))), color="blue", label="Pink TSP")
plt.xscale('log')
plt.xlim(20, 20000)
plt.title('Genelec 8050A')
plt.xlabel('freq[Hz]')
plt.ylabel('level[dB]')
plt.legend()
plt.show()

plt.plot(h_pink)
plt.title("Inpulse Response(with Pink TSP)")
plt.xlim(0, 20000)
plt.show()

fs = 16000

# analysis configure
fc = 3000
decay_lebel_dB = 30
times = 60 / decay_lebel_dB
 
# band pass filter
# nyquist = 0.5*fs
# low_cutoff = fc/(2**(1/2)) / nyquist
# high_cutoff = fc*(2**(1/2)) / nyquist
# bpf_coef = butter(4, [low_cutoff, high_cutoff], btype='bandpass', output='sos')
# ir_bpf = sosfilt(bpf_coef, IR)
# print(ir_bpf)

# plt.plot(ir_bpf)
# plt.show()

ir_bpf = h_pink

# decay curve
ir_bpf_square = ir_bpf ** 2.0
ir_bpf_square_sum = np.sum(ir_bpf_square)
temp = ir_bpf_square_sum
curve=[]
for i in range(len(h_pink)):
    temp = temp - ir_bpf_square[i]
    curve.append(temp)
curve_dB = 10.0 * np.log10(curve)
curve_offset = max(curve_dB)
decay_curve = curve_dB - curve_offset

print(decay_curve.shape)
plt.plot(decay_curve)
plt.show()
# find regression target
i = 0
while decay_curve[i] > -5.0:
    i += 1
start_sample = 1
while decay_curve[i] > -1.0*decay_lebel_dB - 5.0:
    i += 1
end_sample = i
regression_target = decay_curve[start_sample:end_sample]
 
# linear regression for T
x = np.linspace(start_sample, end_sample, end_sample-start_sample)
a, b = np.polyfit(x, regression_target, 1)
rt_sec = (-1.0*decay_lebel_dB/a)*times/fs
print(rt_sec)
