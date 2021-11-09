#!/usr/bin/env python

#it may be not used.
import torch 
import torch.nn as nn
import torch.nn.functional as F

class CustomMSE(nn.Module):
    def __init__(self):
        super(CustomMSE, self).__init__()

    def forward(self, outputs, targets):
        # outputs: prediction result
        # targets: seikai data

        class_num = 75
        angle_num = 8
        freq_bin = 256
        time_bin = 256
        outputs_new = torch.reshape(outputs, (-1, class_num, angle_num, freq_bin, time_bin)) # data, class, angle , freq, time
        targets_new = torch.reshape(targets, (-1, class_num, angle_num, freq_bin, time_bin))

        angle = torch.argmax(torch.mean(torch.mean(torch.mean(targets_new, 4), 3), 1), 1) # angle index
        print(angle)
        print(torch.sin(angle))
        weight = torch.zeros(12, 8)
        for i in range(8):
            print(weight[:,i].size())
            weight[:,i] = torch.sin(angle)
        print(weight)
        loss = torch.mean(torch.mean(torch.mean(((targets_new - outputs_new) ** 2), 1), 2), 2) * weight

        # 損失の計算
        alpha = 0.05
        loss = ((targets - outputs) ** 2) * (torch.tanh(targets) + alpha)
        return loss.mean()

def CrossEntropyLoss2d(outputs, targets, weight=None):
    n, c, h, w = outputs.size()
    log_p = F.log_softmax(outputs, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c) #(n*h*w, c)
    log_p = log_p[targets.view(n*h*w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = targets >= 0
    targets = targets[mask]
    loss = F.nll_loss(log_p, targets, weight=weight)
    return loss

