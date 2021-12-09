import torch


class stftm_loss(object):
    def __init__(self, frame_size=320, frame_shift=160, loss_type='mae'):
        super(stftm_loss, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type

    def __call__(self, est, data_info):
        # est : est waveform
        mask = data_info[3].cuda()
        est_spec1 = est[0]
        raw_spec = torch.stft(data_info[1].cuda(), n_fft=1024, hop_length=128, win_length=1024,
                             window=torch.hann_window(1024).cuda(), onesided=True,
                             return_complex=False).permute([0,2,1,3])


        if self.loss_type == 'mse':
            loss1 = torch.sum((((est_spec1 - raw_spec) ** 2)*mask)) / torch.sum(mask)

        return loss1


class mag_loss(object):
    def __init__(self, frame_size=320, frame_shift=160, loss_type='mae'):
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.loss_type = loss_type
        self.mag_stft = mag_STFT(self.frame_size, self.frame_shift)

    def __call__(self, est, data_info):
        # est : [est_mag,noisy_phase]
        # data_info : [mixture,speech,noise,mask,nframe,len_speech]
        mask = data_info[3].cuda()

        raw_mag = self.mag_stft.transform(data_info[1])[0].permute(0, 2, 1).cuda()
        est_mag = est[0]
        if self.loss_type == 'mse':
            loss = torch.sum((est_mag - raw_mag) ** 2) / torch.sum(mask)
        elif self.loss_type == 'mae':
            loss = torch.sum(torch.abs(est_mag - raw_mag)) / torch.sum(mask)
        return loss


class wavemse_loss(object):
    def __call__(self, est, data_info):
        mask = data_info[3]
        raw = data_info[1]
        loss = torch.sum((est - raw) ** 2) / torch.sum(mask)
        return loss


class sisdr_loss(object):
    def __init__(self):
        self.EPSILON = 1e-7

    def __call__(self, est, data_info):
        raw = data_info[1]
        batch = raw.size(0)
        raw = raw.contiguous().view(-1).unsqueeze(-1)
        est = est.contiguous().view(-1).unsqueeze(-1)
        Rss = torch.mm(raw.T, raw)
        a = (self.EPSILON + torch.mm(raw.T, est)) / (Rss + self.EPSILON)
        e_true = a * raw
        e_res = est - e_true
        Sss = (e_true ** 2).sum()
        Snn = (e_res ** 2).sum()
        sisdr = 10 * torch.log10((self.EPSILON + Sss) / (self.EPSILON + Snn))
        return sisdr


'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import *


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, m, s):
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 192), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1