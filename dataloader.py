import scipy.io as sio
import numpy as np
import struct
import mmap
import os

from torch.nn.utils.rnn import *
from torch.autograd.variable import *
from torch.utils.data import Dataset, DataLoader
from utils.util import gen_list, read_list
import soundfile as sf

class SpeechMixDataset(Dataset):
    def __init__(self, config,wav_dir):
        self.wav_dir = wav_dir
        self.wav_list = gen_list(self.wav_dir + '/mix', '.wav')
        self.config = config
    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        wav_name = self.wav_list[idx]
        s1, _ = sf.read(self.wav_dir + '/s1/' + wav_name)
        anchor, _ = sf.read(self.wav_dir + '/aux/' + wav_name)
        mixture, _ = sf.read(self.wav_dir + '/mix/' + wav_name)
        s1 = s1[:self.config['MAX_LEN']]
        anchor = anchor[:self.config['MAX_LEN']]
        mixture = mixture[:self.config['MAX_LEN']]

        alpha_pow_anchor = 1 / (np.sqrt(np.sum(anchor ** 2) / len(anchor)) + self.config['EPSILON'])

        len_speech = len(s1)
        nframe = ((len_speech) // self.config['WIN_OFFSET']-1)
        len_speech = (nframe-1) * self.config['WIN_OFFSET']
        s1 = s1[:len_speech]
        mixture = mixture[0:len_speech]
        alpha_pow = 1 / (np.sqrt(np.sum(mixture ** 2) / len_speech) + self.config['EPSILON'])

        mixture = alpha_pow * mixture
        s1 = alpha_pow * s1
        anchor = alpha_pow_anchor * anchor

        mask_for_loss = np.ones((nframe, self.config['WIN_LEN'] // 2 + 1, 2), dtype=np.float32)

        sample = (Variable(torch.FloatTensor(mixture.astype('float32'))),
                  Variable(torch.FloatTensor(s1.astype('float32'))),
                  Variable(torch.FloatTensor(anchor.astype('float32'))),
                  Variable(torch.FloatTensor(mask_for_loss)),
                  )
        return sample


class BatchDataLoader(object):
    def __init__(self, s_mix_dataset, batch_size, is_shuffle=True, workers_num=16):
        self.dataloader = DataLoader(s_mix_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=workers_num,
                                     collate_fn=self.collate_fn)

    def get_dataloader(self):
        return self.dataloader

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: x[0].size()[0], reverse=True)
        mixture, s1, anchor, mask_for_loss = zip(*batch)
        mixture = pad_sequence(mixture, batch_first=True)
        s1 = pad_sequence(s1, batch_first=True)
        anchor = pad_sequence(anchor, batch_first=True)
        mask_for_loss = pad_sequence(mask_for_loss, batch_first=True)

        return [mixture, s1, anchor, mask_for_loss]
