import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from torch.autograd import Variable

norm_mean, norm_var = 0.0, 1.0

cov_cfg=[(3*i+1) for i in range(12*3+2+1)]


class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, filters, outplanes, expansion=1, dropRate=0):
        super(DenseBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3,
                               padding=1, bias=False)
        #self.conv1.cp_rate = compress_rate
        #self.conv1.tmp_name = tmp_name

        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, filters, trans_channel):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, trans_channel, kernel_size=1,bias=False)
        
        #self.conv1.cp_rate = compress_rate
        #self.conv1.tmp_name = tmp_name
        #self.conv1.last_prune_num=last_prune_num

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, start=None, trans1_channel=168,trans2_channel=312,
                 depth=40, cfg=None, block=DenseBasicBlock,dropRate=0, dataset='cifar10', filters=None):
        super(DenseNet, self).__init__()

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
            
        
        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6
        #print('n=',n)
        transition = Transition

        dense1_channel = cfg[0:n]
        dense2_channel = cfg[(n):(2*n)]
        dense3_channel = cfg[(2*n):(3*n)]
        if filters == None:
            #filters = []
            filters_1 = []
            #start = growthRate*2
            #print('start = ',start)
            channel = start
            #filters_1.append(channel)
            filters_1.append(channel)
            for j in dense1_channel:
                channel += j
                filters_1.append(channel)
                #print('j=',j)

            filters_1.append(channel)
            for j in dense2_channel:
                channel += j
                filters_1.append(channel)
                #print('j=',j)

            filters_1.append(channel)
            for j in dense3_channel:
                channel += j
                filters_1.append(channel)
                #print('j=',j)
            
            #for i in range(3):
            #    filters.append([start + growthRate*i for i in range(n+1)])
            #    start = (start + growthRate*n) // compressionRate
            #print('filters = ',filters)
            #print('filters_1 = ',filters_1)
            #filters = [item for sub_list in filters for item in sub_list]
            #print('filters = ',filters)
            #print('filters_1 = ',filters_1)

            #indexes = []
            #for f in filters_1:
                #print('f=',f)
                #indexes.append(np.arange(f))
                #print('indexes=',indexes)
            
        #self.covcfg=cov_cfg
        #print('cov_cfg=',cov_cfg)
        #self.compress_rate=compress_rate

        #self.growthRate = growthRate
        self.dropRate = dropRate
        print('start = ',start)
        self.inplanes = start#growthRate * 2
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        #self.conv1.cp_rate=compress_rate[0]
        #self.conv1.tmp_name = 'conv1'
        #self.last_prune_num=self.inplanes*compress_rate[0]
        #print('filters[0:n] = ',filters[0:n])
        self.dense1 = self._make_denseblock(dense1_channel,block, n, filters_1[0:n], 'dense1')
        self.trans1 = self._make_transition(transition, filters_1[n], 'trans1', trans1_channel)
        self.inplanes = trans1_channel
        #print('self.inplanes2 = ',self.inplanes)
        self.dense2 = self._make_denseblock(dense2_channel, block, n, filters_1[n+1:2*n+1],  'dense2')
        self.trans2 = self._make_transition(transition, filters_1[2*n+1],  'trans2', trans2_channel)
        self.inplanes = trans2_channel
        self.dense3 = self._make_denseblock(dense3_channel, block, n, filters_1[2*n+2:3*n+2], 'dense3')
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        print("11111111111111111")

        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                

    def _make_denseblock(self,dense_cfg, block, blocks, filters, tmp_name):
        layers = []
        #print('self.inplanes1 = ',self.inplanes)
        #print(dense_cfg)
        assert blocks == len(filters), 'Length of the filters parameter is not right.'
        #assert blocks == len(indexes), 'Length of the indexes parameter is not right.'
        for i in range(blocks):
        
            #print('filters[i] = ',filters[i])
            layers.append(block(self.inplanes, filters=filters[i], outplanes=dense_cfg[i],
                                dropRate=self.dropRate))
            self.inplanes += dense_cfg[i]#self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, transition, filters, index, trans_channel):
        inplanes = self.inplanes
        outplanes = filters#int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        #self.last_prune_num=int(compress_rate*filters)
        return transition(inplanes, outplanes, filters, trans_channel)

    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def densenet(start=None, trans1_channel=168, trans2_channel=312, cfg=None, dataset='cifar10'):
    return DenseNet(start=start, trans1_channel=trans1_channel,trans2_channel=trans2_channel,depth=40, cfg=cfg, dataset=dataset, block=DenseBasicBlock)

'''
#cfg = [12]*36
start1 = 24
cfg =  [1, 2, 2, 4, 5, 6, 7, 7, 9, 10, 11, 11,
        1, 2, 3, 4, 5, 6, 6, 8, 9, 10, 11, 12,
        1, 2, 3, 4, 5, 6, 7, 7, 9, 10, 11, 11]
cfg_start =  11

x = Variable(torch.FloatTensor(24, 3, 40, 40))
newmodel = densenet(start=cfg_start, cfg = cfg)
print(newmodel)
y = newmodel(x)
print(y.data.shape)
'''

