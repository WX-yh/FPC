import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from thop import profile
cfg1 = [12]*36
start1 = 24
trans1_channel = 168
trans2_channel = 312

c = torch.load('./pdes2/pruned0.1_10.pth.tar')
cfg1 = c['cfg']
start1 = c['start']
trans1_channel = c['trans1_channel']
trans2_channel = c['trans2_channel']


model = densenet(start=start1, trans1_channel=trans1_channel, trans2_channel=trans2_channel, cfg = cfg1, dataset='cifar10')

#print('cfg',c['cfg'])
model.cuda()
print(model)

chanels=0
for m in model.modules():
    if isinstance(m,nn.Conv2d):
        chanels += m.weight.data.shape[0]


input = torch.randn(1, 3, 32, 32).cuda().float()
flops, params = profile(model, inputs=(input,))
print("params=",params)
print("flops=",flops)

num_parameters = sum([param.nelement() for param in model.parameters()])
print('num_parameters=',num_parameters)
print("chanels=",chanels)
print(cfg1)
