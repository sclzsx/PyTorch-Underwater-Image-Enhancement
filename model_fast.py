'''
Author:Xuelei Chen(chenxuelei@hotmail.com)
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class AConvBlock_fast(nn.Module):
    def __init__(self):
        super(AConvBlock_fast,self).__init__()

        block = [nn.Conv2d(3,3,3,padding = 1)]
        block += [Mish()]

        block += [nn.Conv2d(3,3,3,padding = 1)]
        block += [Mish()]

        block += [nn.AdaptiveAvgPool2d((1,1))]
        block += [nn.Conv2d(3,3,1)]
        block += [Mish()]

        self.block = nn.Sequential(*block)

    def forward(self,x):
        return self.block(x)

class tConvBlock_fast(nn.Module):
    def __init__(self):
        super(tConvBlock_fast,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, padding=0, bias=False)

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False), Mish())

        self.score = nn.Tanh()
        
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, padding=0, bias=False)

    def forward(self,x):
        x1 = self.conv1(x)

        x2 = self.conv2(x1)
        
        x3 = torch.cat([x1, x2], 1)

        x3 = self.score(x3)

        x3 = self.conv3(x3)

        x3 = x3 * x1
        
        return x3

class PhysicalNN_fast(nn.Module):
    def __init__(self):
        super(PhysicalNN_fast,self).__init__()

        self.ANet = AConvBlock_fast()
        self.tNet = tConvBlock_fast()

    def forward(self,x):

        A = self.ANet(x)

        B = torch.cat((x*0+A,x),1)

        t = self.tNet(B)

        out = ((x-A)*t + A)

        return torch.clamp(out,0.,1.)

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    import time

    with torch.no_grad():
        net = PhysicalNN_fast().cuda()

        f, p = get_model_complexity_info(net, (3, 2048, 2048), as_strings=True, print_per_layer_stat=False, verbose=False)
        print('FLOPs:', f, 'Parms:', p)

        x = torch.randn(1, 3, 2048, 2048).cuda()
        s = time.clock()
        y = net(x)
        print(y.shape, 1 / (time.clock() - s))