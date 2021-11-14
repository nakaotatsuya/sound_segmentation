#!/usr/bin/env python

import os
import os.path as osp
import time
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch import nn 
from torch import optim
from torch.utils.data import DataLoader
from torch.backends import cudnn

#from model import read_model, FCN8s, UNet, CRNN, Deeplabv3plus
from models import read_model, UNet, Deeplabv3plus
from data import SoundSegmentationDataset
from utils import scores, rmse, save_score_array
from utils import plot_loss, plot_mixture_stft, plot_event, plot_class_stft
from utils import restore

from sklearn.metrics import f1_score

torch.manual_seed(1234)
np.random.seed(1234)

import rospkg
import sys

from scipy import signal
import soundfile as sf
import wavio

def train():
    train_dataset = SoundSegmentationDataset(dataset_dir, split="train", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    val_dataset = SoundSegmentationDataset(dataset_dir, split="val", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model = read_model(model_name, n_classes=n_classes, angular_resolution=angular_resolution, input_dim=input_dim)

    if task == "ssls":
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    #optimizer = optim.Adagrad(model.parameters(), lr=lr)
    #optimizer = optim.AdamW(model.parameters(), lr=lr)
    #optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95)

    losses, val_losses = [], []
    loss_temp, val_loss_temp = 0, 0
    best_val_loss = 99999

    print("Training Start")
    for epoch in range(epochs):
        model.train()
        for i, (images, labels, phase) in tqdm(enumerate(train_loader)):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)

            ##weakly
            if label_type == "weakly":
                weak_labels = (labels > 0.0).to(torch.float32)
                weak_labels = torch.mul(images[:, 0, :, :].unsqueeze(1), weak_labels)

                loss = criterion(outputs, weak_labels)

            else:
                loss = criterion(outputs, labels)
            loss_temp += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_temp = loss_temp / len(train_loader)
        losses.append(loss_temp)

        print("Train Epoch: {}/{} Loss: {:.6f} lr: {:.6f}".format(epoch+1, epochs, loss_temp, lr_scheduler.get_lr()[0]))

        model.eval()
        with torch.no_grad():
            for i, (images, labels, phase) in tqdm(enumerate(val_loader)):
                images = images.cuda()
                labels = labels.cuda()

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_temp += loss.item()

        val_loss_temp = val_loss_temp / len(val_loader)
        val_losses.append(val_loss_temp)

        print("Validation Epoch: {}/{} Loss: {:.6f}".format(epoch+1, epochs, val_loss_temp))

        lr_scheduler.step()

        if val_loss_temp < best_val_loss:
            print("Best loss, model saved")
            best_val_loss = val_loss_temp
            model.save(save_dir=save_dir)
            
        loss_temp, val_loss_temp = 0, 0
        #plot_loss(losses, val_losses, save_dir)
        shutil.copy("main.py", save_dir)
        #if osp.exists(os.getcwd() + "nohup.out"):
        #    shutil.copy("nohup.out", save_dir)

def val():
    val_dataset = SoundSegmentationDataset(dataset_dir, split="val", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model = read_model(model_name, n_classes=n_classes, angular_resolution=angular_resolution, input_dim=input_dim)
    model.load(osp.join(save_dir, model_name + ".pth"))
    model.cuda()

    print("Eval Start")
    model.eval()
    with torch.no_grad():
        for i, (images, labels, phase) in tqdm(enumerate(val_loader)):
            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)

            X_in = images.data.cpu().numpy()
            pred = outputs.data.cpu().numpy()
            gt = labels.data.cpu().numpy()

            if i == 0:
                X_ins = X_in
                phases = phase
                preds = pred
                gts = gt
            else:
                X_ins = np.concatenate((X_ins, X_in), axis=0)
                phases = np.concatenate((phases, phase), axis=0)
                preds = np.concatenate((preds, pred), axis=0)
                gts = np.concatenate((gts, gt), axis=0)

    #print(gts[3][18:24])
    #print(preds[3][18:24])

    #print(phases.shape)
    #print(X_ins.shape)

    #print(gts.shape)

    #if task == "ssls":
    #    scores_array = rmse(gts, preds, classes=n_classes)
    #    save_score_array(scores_array, save_dir)

    for n in range(len(preds)):
        plot_mixture_stft(X_ins, no=n, save_dir=save_dir)
        plot_class_stft(gts, preds, no=n, save_dir=save_dir, classes=n_classes, ang_reso=angular_resolution)
        restore(gts, preds, phases, no=n, save_dir=save_dir, classes=n_classes, ang_reso=angular_resolution, dataset_dir=dataset_dir)
        #sys.exit()

    # for i in range(angular_resolution):
    #     if gts[0][i].max() > 0:
    #         #print(gts[0][i].shape)
    #         #deleted_preds = preds[0][i][1:255]
    #         added_gts = np.concatenate((gts[0][i], np.zeros((1, 96))))
    #         #added_gts = np.concatenate((np.zeros((1, 96)), gts[0][i]))
    #         #print(added_gts.shape)
    #         time, data = signal.istft(Zxx = added_gts, fs=16000,
    #                                   nperseg = 512)
    #         #print(time)
    #         print(data)

    #         #sf.write("./new_file.wav", data, 16000)
    #         #data = data * 120 - 120
    #         #data = 10**(data / 20.0)
    #         #print(data)
    #         #wavio.write("./new_fie.wav", data, 16000, sampwidth=3)

    #         sys.exit()

if __name__ == "__main__":
    rospack = rospkg.RosPack()
    #root = osp.join(rospack.get_path("sound_segmentation"), "audios")
    root = osp.join(rospack.get_path("sound_segmentation"), "esc50")

    epochs = 300
    batch_size = 4
    lr = 0.001

    n_classes = 1

    label_type = "supervised"
    #label_type = "weakly"
    task = "ssls"
    model_name = "UNet"

    # TODO label_csv
    dataset_name = "multi"
    dataset_dir = root
    
    date = time.strftime("%Y_%m%d")
    dirname = date + "_" + label_type + "_" + task + "_" + model_name
    save_dir = osp.join("results", dataset_name, dirname)

    #
    #save_dir = osp.join("results", dataset_name, "2021_1106_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1112_supervised_ssls_UNet")
    save_dir = osp.join("results", dataset_name, "2021_1113_supervised_ssls_UNet")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    spatial_type = "ipd"
    mic_num = 8
    angular_resolution = 24
    if mic_num == 8:
        input_dim = mic_num * 2 - 1
    else:
        raise ValueError("mic num should be 8")

    #train()
    val()
