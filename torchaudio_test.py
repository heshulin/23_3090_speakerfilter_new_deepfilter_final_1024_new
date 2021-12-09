import soundfile as sf
import torch
from torch.autograd.variable import *
import scipy.io as sio
wav_path = '/data_zkh/ORIGIN_WSJ0/wsj0_2mix_extr/wav8k/max/tr/s1/026c0211_1.7768_20va010a_-1.7768_026a010n.wav'

if __name__ == '__main__':
    wav,sample = sf.read(wav_path)
    input = Variable(torch.FloatTensor(wav.astype('float32')))
    sf.write('testinput.wav',input,sample)
    spec = torch.stft(input,n_fft=1024,hop_length=128,win_length=1024,window=torch.hann_window(1024),onesided=True,return_complex=False)
    spec_mag = torch.sqrt(spec[:,:,0] ** 2 + spec[:,:,1] ** 2)
    output = torch.istft(spec,n_fft=1024,hop_length=128,win_length=1024,window=torch.hann_window(1024),onesided=True,return_complex=False)
    sf.write('test.wav',output.cpu().detach().numpy(),sample)
    sio.savemat("test.mat",{'spec_mag':spec_mag.cpu().detach().numpy()})

    print('1')
