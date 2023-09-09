import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=164,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.1,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='./logs2/model_best0.1_25.pth.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./pres2', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)
cfg=[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
         32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
         64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64]
c = torch.load('./pres2/pruned0.1_25.pth.tar')
cfg =c['cfg']
model = resnet110_cifar(cfg=cfg)
print(model)
if args.cuda:
    model.cuda()
if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

kwargs = {'num_workers': 0, 'pin_memory': False} if args.cuda else {}
#data
if args.dataset == 'cifar10':
    print("d10")
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    print("d100")
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print("d100")

'''
total = 0

for name,para in model.named_parameters():
    if name in BN_layers:
        print(name)
        #print(para.size())
        total += para.size(0)
        #print(total)
'''
"""
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]
"""
feature_result = []
#feature_result.append(torch.zeros(64))
total = []
#total.append(0)
f = 0
f1 = -1

def get_feature_hook(self,input ,output):
    print('hook')

    
    global f1
    print('f1= ',f1)
    a = output.shape[0]
    b = output.shape[1]
    if (f1 != f):
        feature_result.append(torch.zeros(b))
        total.append(0)
        f1 = f1 + 1
    
    print(a)
    print(b)
    c = torch.zeros(b)
    print('c.shape:',c.size())
    for i in range(a):
        for j in range(b):
            u,s,v = torch.svd(output[i,j,:,:])
            c[j] = c[j]+s.sum()

    #print('feature_result = ',feature_result)
    #print('c = ',c)
    
    feature_result[f] = feature_result[f] * total[f] +c
    #print('feature_result = ',feature_result)
    total[f] = total[f] + a
    feature_result[f] = feature_result[f]/total[f]
    
    #print('feature_result = ',feature_result)
    #print('total = ',total)
    
            
def get_feature_hook1(self,input ,output):
    print("hook1")

def rank_svd(model):
    global f
    print('f= ',f)
    limit = 1
    d = 0
    
    model.eval()
    correct = 0
    for data, target in train_loader:
        if d >= limit:
            break
        d = d+1
        print('d=',d)
        if args.cuda:
             data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            #print("111")
            output = model(data)
            #print("222")
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    f = f + 1
    #print('\nTrain: Accuracy: {}/{} ({:.1f}%)\n'.format(
    #    correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))
    #return correct / float(len(train_loader.dataset))


for i in range(3):
    block = eval('model.layer%d' % (i + 1))
    #print('block:',block)
    for j in range(18):
        cov_layer = block[j].relu1
        if __name__ == '__main__':
            handler = cov_layer.register_forward_hook(get_feature_hook)
            rank_svd(model)
            handler.remove()

total_channel = 0
index = 0

for n in feature_result:
    #print('n shape:',n.shape[0])
    total_channel = total_channel + n.shape[0]

    
print('total_channel:',total_channel)
feature_s = torch.zeros(total_channel)
for n in feature_result:
    size = n.shape[0]
    feature_s[index:(index+size)] = n
    index = index +size



"""
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size
"""


y,i = torch.sort(feature_s)
thre_index = int(total_channel * args.percent)
thre = y[thre_index]

"""
for m in model.modules():
    if isinstance(m,nn.Conv2d):
        print('m=',m)
"""

pruned = 0
cfg1 = []
cfg_mask = []
#i = 0
for i in range(54):
    print('i=',i)
    feature_copy = feature_result[i]
    mask = feature_copy.gt(thre).float()#.cuda()
    if torch.sum(mask) == 0:
        cfg1.append(len(feature_copy))
        cfg_mask.append(torch.ones(len(feature_copy)).float())#.cuda())
        print('total channel: {:d} \t remaining channel: {:d}'.
            format(len(feature_copy), int(len(feature_copy))))
    else:
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('total channel: {:d} \t remaining channel: {:d}'.
            format(mask.shape[0], int(torch.sum(mask))))
        
        #i+=1
            

#print("cfg1 = ", cfg1)
#print('cfg_mask',cfg_mask)

cfg_mask1 = []
j = 0
out = torch.ones(16).float().cuda()#1
cfg_mask1.append(out)

cfg_mask1.append(cfg_mask[j])
j = j+1
cfg_mask1.append(torch.ones(16).float().cuda())#3

cfg_mask1.append(cfg_mask[j])
j = j+1
cfg_mask1.append(torch.ones(16).float().cuda())#5

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#7

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#9

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#11

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#13

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#15

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#17

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#19

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#21

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#23

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#25

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#27

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#29

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#31

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#31

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#35

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(16).float().cuda())#37

cfg_mask1.append(cfg_mask[j])
j = j+1
cfg_mask1.append(torch.ones(32).float().cuda())#39
cfg_mask1.append(40)#40

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#42

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#44

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#46

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#48

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#50

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#52

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#54

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#56

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#58

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#60

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#62

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#64

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#66

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#68

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#70

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#72

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(32).float().cuda())#74

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#76
cfg_mask1.append(77)#77

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#79

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#81

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#83

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#85

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#87

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#89

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#91

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#93

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#95

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#97

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#99

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#101

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#103

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#105

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#107

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#109

cfg_mask1.append(cfg_mask[j])
j=j+1
cfg_mask1.append(torch.ones(64).float().cuda())#111


#print('j=',j)
#print('out=',out)
#print('cfg1=',cfg_mask1)
        
"""
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')
"""


pruned_ratio = pruned/total_channel
print('pruned_ratio=',pruned_ratio)

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

#acc = test(model)

print("Cfg1:")
print(cfg1)

newmodel = resnet110_cifar(cfg=cfg1)
if args.cuda:
    newmodel.cuda()
print("newmodel:")
print(newmodel)

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune0.1_26.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg1)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    #fp.write("Test accuracy: \n"+str(acc)+"\n")
    fp.write("pruned: \n"+str(pruned)+"\n")
    fp.write("pruned_ratio: \n"+str(pruned_ratio)+"\n")

#for i in cfg_mask1:
#    print('i=',i)
#old_modules = list(model.modules())
#new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask1[layer_id_in_cfg]
t = 1
#conv_count = 0
down = [40,77]
#print("down=",down)
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if t in down:
        #print('end_mask=',end_mask)
        if isinstance(m0,nn.Conv2d):
            #print("11")
            m1.weight.data = m0.weight.data
            #print("111")
        elif isinstance(m0,nn.BatchNorm2d):
            #print("22")
            m1.weight.data = m0.weight.data
            m1.bias.data = m0.bias.data
            m1.running_mean = m0.running_mean
            m1.running_var = m0.running_var
            layer_id_in_cfg += 1
            t+=1
            #print("t=",t)
            if layer_id_in_cfg<len(cfg_mask1):
                end_mask = cfg_mask1[layer_id_in_cfg]
            #print("222")
    elif isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            print("bn,resize")
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        t+=1
        #print("t=",t)
        start_mask = end_mask.clone()
        #print(len(cfg_mask1))
        if layer_id_in_cfg < len(cfg_mask1):  # do not change in Final FC
            end_mask = cfg_mask1[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        #print(idx0)
        #print(idx1)
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            print("in,resize")
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            print("out,resize")
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()




"""
for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))

        if isinstance(old_modules[layer_id + 1], channel_selection):
            # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            # We need to set the channel selection layer.
            m2 = new_modules[layer_id + 1]
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if conv_count == 0:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
            # This convers the convolutions in the residual block.
            # The convolutions are either after the channel selection layer or after the batch normalization layer.
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

            # If the current convolution is not the last convolution in the residual block, then we can change the 
            # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
            if conv_count % 3 != 1:
                w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            continue

        # We need to consider the case where there are downsampling convolutions. 
        # For these convolutions, we just copy the weights.
        m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()
"""


torch.save({'cfg': cfg1, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned0.1_26.pth.tar'))

print(newmodel)
model = newmodel
test(model)
