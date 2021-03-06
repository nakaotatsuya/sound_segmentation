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
    train_dataset = SoundSegmentationDataset(dataset_dir, split="noise_train2", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)

    val_dataset = SoundSegmentationDataset(dataset_dir, split="noise_val2", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = read_model(model_name, n_classes=n_classes, angular_resolution=angular_resolution, input_dim=input_dim)
    print(device)
    model = model.to(device)

    if device == "cuda":
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
        cudnn.benchmark = True

    if task == "ssls":
        criterion = nn.MSELoss()
        criterion.to(device)

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
            #images = images.cuda()
            images = images.to(device)
            #labels = labels.cuda()
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.to(device)

            ##weakly
            # if label_type == "weakly":
            #     weak_labels = (labels > 0.0).to(torch.float32)
            #     weak_labels = torch.mul(images[:, 0, :, :].unsqueeze(1), weak_labels)

            #     loss = criterion(outputs, weak_labels)

            # else:
            loss = criterion(outputs, labels)
            loss = loss.to(device)

            loss_temp += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_temp = loss_temp / len(train_loader)
        #losses.append(loss_temp)

        print("Train Epoch: {}/{} Loss: {:.6f} lr: {:.6f}".format(epoch+1, epochs, loss_temp, lr_scheduler.get_lr()[0]))
        print("a")

        model.eval()
        with torch.no_grad():
            for i, (images, labels, phase) in tqdm(enumerate(val_loader)):
                #images = images.cuda()
                images = images.to(device)
                #labels = labels.cuda()
                labels = labels.to(device)

                outputs = model(images)
                outputs = outputs.to(device)
                
                loss = criterion(outputs, labels)
                loss = loss.to(device)
                val_loss_temp += loss.item()

        val_loss_temp = val_loss_temp / len(val_loader)
        #val_losses.append(val_loss_temp)

        print("Validation Epoch: {}/{} Loss: {:.6f}".format(epoch+1, epochs, val_loss_temp))

        lr_scheduler.step()

        if val_loss_temp < best_val_loss:
            print("Best loss, model saved")
            best_val_loss = val_loss_temp
            #model.save(save_dir=save_dir)
            torch.save(model.module.state_dict(), osp.join(save_dir, "UNet.pth"))
            
        loss_temp, val_loss_temp = 0, 0
        #plot_loss(losses, val_losses, save_dir)
        shutil.copy("main.py", save_dir)
        #if osp.exists(os.getcwd() + "nohup.out"):
        #    shutil.copy("nohup.out", save_dir)

def val():
    val_dataset = SoundSegmentationDataset(dataset_dir, split="noise_val2", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False)

    device = "cuda"
    model = read_model(model_name, n_classes=n_classes, angular_resolution=angular_resolution, input_dim=input_dim)
    model.load(osp.join(save_dir, model_name + ".pth"))
    model.cuda()

    if device == "cuda":
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
        cudnn.benchmark = True

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
            #break

    if task == "ssls":
        scores_array = rmse(gts, preds, classes=n_classes)
        save_score_array(scores_array, save_dir)

    for n in range(len(preds)):
        plot_mixture_stft(X_ins, no=n, save_dir=save_dir, pred="prediction")
        plot_class_stft(gts, preds, no=n, save_dir=save_dir, classes=n_classes, ang_reso=angular_resolution, pred="prediction")
        restore(gts, preds, phases, no=n, save_dir=save_dir, classes=n_classes, ang_reso=angular_resolution, dataset_dir=dataset_dir, pred="prediction")
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

def real_val():
    val_dataset = SoundSegmentationDataset(dataset_dir, split="real_val", task=task, n_classes=n_classes, spatial_type=spatial_type, mic_num=mic_num, angular_resolution=angular_resolution, input_dim=input_dim)
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

    for n in range(len(preds)):
        max_power_per_pixel = 0
        max_idx = 0
        for an in range(angular_resolution):
            cv_pred = preds[n][an][::-1,:]
            power_per_pixel = cv_pred.sum() / cv_pred.size
            #print(power_per_pixel)
            #print(max_power_per_pixel)
            if power_per_pixel > max_power_per_pixel:
                max_power_per_pixel = power_per_pixel
                max_idx = an
        print("max_idx:", max_idx)

    for n in range(len(preds)):
        plot_mixture_stft(X_ins, no=n, save_dir=save_dir, pred="real_prediction")
        plot_class_stft(gts, preds, no=n, save_dir=save_dir, classes=n_classes, ang_reso=angular_resolution, pred="real_prediction")
        #restore(gts, preds, phases, no=n, save_dir=save_dir, classes=n_classes, ang_reso=angular_resolution, dataset_dir=dataset_dir, pred="real_prediction")
        
if __name__ == "__main__":
    rospack = rospkg.RosPack()
    root = osp.join(rospack.get_path("sound_segmentation"), "house_audios")
    #root = "/home/jsk/nakao/sound_segmentation/house_audios"
    #root = osp.join(rospack.get_path("sound_segmentation"), "esc50")
    #root = osp.join(rospack.get_path("sound_segmentation"), "sep_esc50")

    epochs = 600
    batch_size = 32
    lr = 0.002

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
    #save_dir = osp.join("results", dataset_name, "2021_1113_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1122_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1124_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1125_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1126_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1129_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1130_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1202_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1203_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1205_supervised_ssls_UNet")
    save_dir = osp.join("results", dataset_name, "2021_1210_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1212_supervised_ssls_UNet")
    #save_dir = osp.join("results", dataset_name, "2021_1218_supervised_ssls_UNet")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    spatial_type = "ipd"
    mic_num = 8 * 2
    angular_resolution = 8 * 5
    if mic_num == 8 * 2:
        input_dim = mic_num * 2 - 1
    else:
        raise ValueError("mic num should be 8")

    #train()
    #val()
    real_val()
