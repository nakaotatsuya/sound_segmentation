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

## Omura
# def up_tsp(N,J):
#     ret = np.arange(N, dtype=np.complex)
#     ret[:N//2] = np.exp(-2j*np.pi*J*(ret[:N//2]/float(N))**2)
#     ret[N//2:] = np.exp(-2j*np.pi*J*((N-ret[N//2:])/float(N))**2).conjugate()
#     return ret

# N = 2**16
# J = N/2
# sig = up_tsp(N,J)
# print(sig.shape)

# file_path = "/home/nakaotatsuya/ros/kinetic/src/sound_segmentation/tsp"
# wavio.write(osp.join(file_path, "sig.wav"), sig, 16000, sampwidth=4)

gain = 70
TSP = 16384#信号長

def Swept_Sine():
    m_exp = 1 
    m = TSP // (2 ** m_exp)  # (J=2m)実効長m=(3/4)N~(1/2)N程度
    a = ((m * np.pi) * (2.0 / TSP) ** 2)#α=4mπ/N^2
    ss_freqs = np.zeros(TSP, dtype=np.complex128)
    ss_freqs[:(TSP // 2) + 1] = np.exp(-1j * a * (np.arange((TSP // 2) + 1) ** 2))#exp(jαk^2)
    ss_freqs[(TSP // 2) + 1:] = np.conj(ss_freqs[1:(TSP // 2)][::-1])
    # ifft and real
    ss = np.real(ifft(ss_freqs))#逆フーリエ変換で信号作成
    # roll
    ss = gain * np.roll(ss, TSP//2 - m)#信号をシフトさせる
    return(ss)

ss = Swept_Sine()
print(ss)

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

def pink_tsp(n, gain=1000, repeat=1):

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
wavio.write(osp.join(file_path, "ss.wav"), ss, 16000, sampwidth=4)

ss_rep =  np.r_[np.tile(ss, 16), np.zeros(TSP)]

wavio.write(osp.join(file_path, "ss_rep.wav"), ss_rep, 16000, sampwidth=4)

normal_tsp, normal_tsp_inv = pink_tsp(15, repeat=8)

sf.write(osp.join(file_path, "normal_tsp.wav"), normal_tsp, 16000)

# inv_SS = np.conj(fft.fft(ss))

## rec and play
# fs = 16000
# a = 1
# f = 500
# sec = 1
# swav = []

# sd.default.samplerate = fs
# sd.default.channels = 1
# dev= [6,9]

# sd.default.device = dev
# print(sd.query_devices())

# if __name__ == "__main__":
#     for n in np.arange(fs * sec):
#         s = a * np.sin(2.0 * np.pi * f * n/fs)
#         swav.append(s)

#     print(len(swav))
#     record = sd.playrec(np.array(swav), fs)
#     sd.wait()

#     plt.subplot(2,1,1)
#     plt.plot(swav[0:10000], label="played sin wav")
#     plt.legend()
#     plt.show()

#     plt.subplot(2,1,2)
#     plt.plot(record[0:10000], label="recorded sin wav")
#     plt.legend()
#     plt.show()

inv_SS = np.conj(fft(ss))

response_file = "/home/nakaotatsuya/ros/kinetic/src/sound_segmentation/tsp/response.wav"
#wav = wavio.read(response_file)
#data = wav.data

fs, data = read(response_file)
print(data.shape)
plt.plot(data.T[0])
#plt.legend()
plt.show()


start = 0
output = data.T[0]
output_ave = np.zeros(TSP)
for i in range(17):
    output_ave += output[start:start+TSP]
    start += TSP
output_ave /= 16

TF = inv_SS*fft(output_ave)
IR = ifft(TF)

plt.plot(IR)
plt.show()

print(IR)
#rt60 = pra.experimental.measure_rt60(IR, fs=16000)
#print("zankyo time:{}".format(rt60))

# read data
#fs, ir = read(response_file)

#print(fs)
#print(ir.shape)

fs = 16000

# analysis configure
fc = 3000
decay_lebel_dB = 10
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

ir_bpf = IR

# decay curve
ir_bpf_square = ir_bpf ** 2.0
ir_bpf_square_sum = np.sum(ir_bpf_square)
temp = ir_bpf_square_sum
curve=[]
for i in range(len(IR)):
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



def sychronous_addition(filename, repeat, N):
    '''
    filename : Name of wav file (str)
    repeat : Number of repetition (int)
    N : Length of input signal (int)
    '''
    data, fs = sf.read(filename)

    # add zeros if length is too short
    data = data.T[0]
    if len(data) < repeat * N:
        data = np.r_[data, np.zeros(repeat * N - len(data))]

    mean = np.zeros(N)
    for i in range(repeat):
        mean += data[i * N : (i + 1) * N]
    mean = mean / repeat

    return mean


n = 15
N = 2 ** n
fs = 16000
response_file = "/home/nakaotatsuya/ros/kinetic/src/sound_segmentation/tsp/response4.wav"
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
plt.xlim(0, 10000)
plt.show()
