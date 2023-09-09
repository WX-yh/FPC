import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *
from thop import profile

c = torch.load('./pvgg2/pruned0.1_22.pth.tar')
#print('cfg',c['cfg'])
model = vgg(dataset='cifar10', depth=16,cfg=c['cfg'])
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
print(c['cfg'])
