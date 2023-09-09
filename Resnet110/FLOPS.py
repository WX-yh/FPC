import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from thop import profile

cfg=[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
         32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
         64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64]

c = torch.load('./pres3/pruned0.1_23.pth.tar')
cfg = c['cfg']

model = resnet110_cifar(cfg=cfg)

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
#print(c['cfg'])
