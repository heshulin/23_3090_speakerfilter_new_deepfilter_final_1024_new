import torch.nn as nn
import torch.autograd.variable
from torch.autograd.variable import *
import torch
from thop import profile
from thop import clever_format
from components.deep_filter import DeepFilter

class SpeechConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super(SpeechConv, self).__init__()
        self.ln = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                            padding=padding, bias=True)
        self.ln_bn = nn.BatchNorm2d(out_channels)
        self.gate = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=True)
        self.gate_bn = nn.BatchNorm2d(out_channels)
        self.elu = torch.nn.ELU()
    def forward(self, in_feat):
        ln = self.elu(self.ln_bn(self.ln(in_feat)))
        gate = torch.sigmoid(self.gate_bn(self.gate(in_feat)))
        res = ln * gate
        return res


class SpeechDeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, is_bn=True,output_padding=0):
        super(SpeechDeConv, self).__init__()
        self.ln = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, bias=True,output_padding=output_padding)
        self.gate = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=True,output_padding=output_padding)
        self.is_bn = is_bn
        if is_bn:
            self.ln_bn = nn.BatchNorm2d(out_channels)
            self.gate_bn = nn.BatchNorm2d(out_channels)
        self.elu = torch.nn.ELU()

    def forward(self, in_feat):
        ln = self.elu(self.ln(in_feat))
        if self.is_bn:
            ln = self.elu(self.ln_bn(ln))
            gate = torch.sigmoid(self.gate_bn(self.gate(in_feat)))
        else:
            gate = torch.sigmoid(self.gate(in_feat))
        res = ln * gate
        return res
class NetLstm(nn.Module):
    def __init__(self):
        super(NetLstm, self).__init__()
        self.L = 2
        self.I = 1
        self.Len = (2 * self.L + 1) * (2 * self.I + 1)
        # speaker extractor
        self.GRU_input_size_an = 129
        self.GRU_output_size_an = 129
        self.GRU_layers_an = 2
        self.GRU_an = nn.GRU(input_size=self.GRU_input_size_an,
                             hidden_size=self.GRU_output_size_an,
                             num_layers=self.GRU_layers_an,
                             batch_first=True,
                             bidirectional=True)
        self.transition_linear = nn.Linear(self.GRU_output_size_an * 2, 513)
        self.conv1_an = SpeechConv(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv2_an = SpeechConv(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv3_an = SpeechConv(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv4_an = SpeechConv(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv5_an = SpeechConv(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv6_an = SpeechConv(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv7_an = SpeechConv(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))


        # speech separator
        self.lstm_input_size = 256 * 3
        self.lstm_output_size = 128 * 3
        self.lstm_layers = 2
        self.conv1 = SpeechConv(in_channels=2, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv2 = SpeechConv(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv3 = SpeechConv(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv4 = SpeechConv(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv5 = SpeechConv(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv6 = SpeechConv(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv7 = SpeechConv(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))

        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_output_size,
                            num_layers=self.lstm_layers,
                            batch_first=True,
                            bidirectional=True)

        self.conv7_t = SpeechDeConv(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv6_t = SpeechDeConv(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv5_t = SpeechDeConv(in_channels=512, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv4_t = SpeechDeConv(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv3_t = SpeechDeConv(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv2_t = SpeechDeConv(in_channels=64, out_channels=16, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0),output_padding=(0,1))
        self.conv1_t_ac = nn.ConvTranspose2d(in_channels=32, out_channels=30, kernel_size=(3, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv1_t_sigmoid = nn.ConvTranspose2d(in_channels=32, out_channels=30, kernel_size=(3, 3), stride=(1, 2),
                                          padding=(1, 0))


        self.df = DeepFilter(L=2, I=1, is_look_forward=True)

    def forward(self, input_mix,input_anchor):
        spec_an = torch.stft(input_anchor, n_fft=256,hop_length=128, win_length=256, window=torch.hann_window(256).cuda(), onesided=True,
                          return_complex=False)
        mag_an = torch.sqrt(spec_an[:,:,:,0] ** 2 + spec_an[:,:,:,1] ** 2).permute([0,2,1])
        spec_mix = torch.stft(input_mix, n_fft=1024, hop_length=128, win_length=1024, window=torch.hann_window(1024).cuda(),
                             onesided=True,
                             return_complex=False)
        mag_mix = torch.sqrt(spec_mix[:,:,:,0] ** 2 + spec_mix[:,:,:,1] ** 2).permute([0,2,1])

        GRU_out_an, _ = self.GRU_an(mag_an)
        GRU_real_out = self.transition_linear(GRU_out_an)
        real_anchor_in = torch.mean(GRU_real_out, dim=1, keepdim=True)
        real_anchor_in = real_anchor_in.repeat(1, mag_mix.size()[1], 1)

        # sio.savemat('test2.mat',{'real_anchor_in':real_anchor_in.detach().cpu().numpy(),'mixture':mixture.detach().cpu().numpy(),'lstm_out_an':lstm_out_an.detach().cpu().numpy(),'lstm_an_in':lstm_an_in.detach().cpu().numpy()})
        # in anchor
        e1_an = self.conv1_an(torch.stack([real_anchor_in], 1))
        e2_an = self.conv2_an(e1_an)
        e3_an = self.conv3_an(e2_an)
        e4_an = self.conv4_an(e3_an)
        e5_an = self.conv5_an(e4_an)
        e6_an = self.conv6_an(e5_an)
        e7_an = self.conv7_an(e6_an)

        # out_real_an = e5_an.contiguous().transpose(1, 2)
        # out_real_an = out_real_an.contiguous().view(out_real_an.size(0), out_real_an.size(1), -1)

        # GCNN in mix
        e1 = self.conv1(torch.cat([torch.stack([mag_mix], 1), torch.stack([real_anchor_in], 1)], 1))
        e1 = e1*torch.mean(e1_an,[2,3],keepdim=True)
        e2 = self.conv2(e1)
        e2 = e2*torch.mean(e2_an,[2,3],keepdim=True)
        e3 = self.conv3(e2)
        e3 = e3*torch.mean(e3_an,[2,3],keepdim=True)
        e4 = self.conv4(e3)
        e4 = e4*torch.mean(e4_an,[2,3],keepdim=True)
        e5 = self.conv5(e4)
        e5 = e5*torch.mean(e5_an,[2,3],keepdim=True)
        e6 = self.conv6(e5)
        e6 = e6 * torch.mean(e6_an, [2, 3], keepdim=True)
        e7 = self.conv7(e6)
        e7 = e7 * torch.mean(e7_an, [2, 3], keepdim=True)

        out_real = e7.contiguous().transpose(1, 2)
        out_real = out_real.contiguous().view(out_real.size(0), out_real.size(1), -1)
        lstm_out, _ = self.lstm(out_real)
        lstm_out_real = lstm_out.contiguous()
        lstm_out_real = lstm_out_real.view(lstm_out.size(0), lstm_out.size(1), 256, 3)
        lstm_out_real = lstm_out_real.contiguous().transpose(1, 2)
        t7 = self.conv7_t(torch.cat((lstm_out_real, e7), dim=1))
        t6 = self.conv6_t(torch.cat((t7, e6), dim=1))
        t5 = self.conv5_t(torch.cat((t6, e5), dim=1))
        t4 = self.conv4_t(torch.cat((t5, e4), dim=1))
        t3 = self.conv3_t(torch.cat((t4, e3), dim=1))
        t2 = self.conv2_t(torch.cat((t3, e2), dim=1))
        t1_ac = torch.tanh(self.conv1_t_ac(torch.cat((t2, e1), dim=1)))
        t1_sigmoid = torch.sigmoid(self.conv1_t_sigmoid(torch.cat((t2, e1), dim=1)))
        t1 = t1_ac*t1_sigmoid
        deep_filter = t1.reshape(t1.size(0), 5, 3, 2, t1.size(2), t1.size(3))
        stack_res = self.df.deep_filter_data(spec_mix.permute([0,2,1,3]), spec_mix.permute([0,2,1,3]).device)

        a = stack_res[:, :, :, 0]
        b = stack_res[:, :, :, 1]
        c = deep_filter[:, :, :, 0]
        d = deep_filter[:, :, :, 1]
        est_real = (a * c - b * d).sum(1).sum(1)
        est_img = (a * d + b * c).sum(1).sum(1)
        est_spec = torch.stack([est_real, est_img], dim=-1)
        # out = torch.cat([out[:,:,:,0],out[:,:,:,1]],-1)
        return [est_spec]



if __name__ == '__main__':
    input = Variable(torch.FloatTensor(torch.rand(1, 8000))).cuda(0)
    input2= Variable(torch.FloatTensor(torch.rand(1, 8000))).cuda(0)

    net = NetLstm().cuda()
    macs, params = profile(net, inputs=(input, input2))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs)
    print(params)
    # print("%s | %.2f | %.2f" % ('elephantstudent', params, macs))
