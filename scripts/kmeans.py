#!/usr/bin/env python

import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
#import rospkg
import os.path as osp
import wavio
import soundfile as sf
from scipy import signal
from sklearn.decomposition import PCA
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.manifold import TSNE
from umap import UMAP

import seaborn as sns
import pandas as pd
import sys

#rospack = rospkg.RosPack()
#file_path = osp.join(rospack.get_path("sound_segmentation"), "audios")

#cur_dir = os.getcwd()
#file_path = osp.join(cur_dir, "../audios")

os.chdir("../")

file_path = osp.join(osp.dirname(osp.abspath(__file__)), "audios")
print(file_path)
#sys.exit()

wav_file_path = osp.join(file_path, "wav")
class_names = os.listdir(wav_file_path)
print(class_names)
#original_dataset = np.empty((1,512,97))
original_dataset = np.empty((1,24832))
print(original_dataset.shape)

first=True
true_label = np.array([])
for i, c in enumerate(class_names):
    print(c)
    class_path = osp.join(wav_file_path, c)
    data_names = os.listdir(class_path)
    data_names.sort()
    
    for d in data_names:
        true_label = np.append(true_label, i)
        print(d)
        data_path = osp.join(class_path, d)
        #wav_data = wavio.read(data_path)
        #print(wav_data)

        waveform, fs = sf.read(data_path)
        _, _, stft = signal.stft(x=waveform.T, fs=fs, nperseg=512, return_onesided=False)
        #print(stft.shape)

        amp = abs(stft[:256])
        #print(amp)

        amp = np.ravel(amp)
        amp = amp[None]
        #print(amp.shape)
        if first:
            original_dataset = amp
            first=False
        else:
            original_dataset = np.append(original_dataset, amp, axis=0)

print(original_dataset[0])
x = original_dataset

print(true_label)

#TSNE
#arr_tsne = TSNE(n_components=2, random_state=0, verbose=1).fit_transform(x)
pca = PCA(n_components=50, random_state=0)
x = pca.fit_transform(x)
umap = UMAP(n_components=3, random_state=0, n_neighbors=4).fit_transform(x)
#print(arr_tsne.shape)
print(umap.shape)


# df_tsne = pd.DataFrame(umap, columns=["tsne_1", "tsne_2"])
# sns.scatterplot(x="tsne_1", y="tsne_2", data=df_tsne, palette="Set1")
# plt.show()

# model = cluster.KMeans(n_clusters=6, init="k-means++", n_init=10, max_iter=400, tol=1e-04, random_state=0)
# #model = cluster.AgglomerativeClustering(n_clusters=6, linkage="ward")
# #model = cluster.AgglomerativeClustering(n_clusters=6, linkage="single")
# #model = cluster.AffinityPropagation()

model = cluster.DBSCAN(eps=0.8, min_samples=5)
model.fit(umap)
print(model.labels_)

#dbscan result
# df_tsne["label"] = model.labels_
# sns.scatterplot(x="tsne_1", y="tsne_2", data=df_tsne, hue="label", palette="Set1")
# plt.show()

#true
# df_tsne["label"] = true_label
# sns.scatterplot(x="tsne_1", y="tsne_2", data=df_tsne, hue="label", palette="Set1")
# plt.show()

# xmeans
# xm_c = kmeans_plusplus_initializer(x, 2).initialize()
# xm_i = xmeans(data=x, initial_centers=xm_c, kmax=20, ccore=True)
# xm_i.process()

# z_xm = np.ones(original_dataset.shape[0])
# #print(z_xm.shape)
# print(xm_i._xmeans__clusters)
# for k in range(len(xm_i._xmeans__clusters)):
#     z_xm[xm_i._xmeans__clusters[k]] = k+1

# print(z_xm)
