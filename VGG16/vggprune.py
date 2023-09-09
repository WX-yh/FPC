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
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.1,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='./logs3/model_best0.1_22.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./pvgg3', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#torch.cuda.set_device("1")
if not os.path.exists(args.save):
    os.makedirs(args.save)

c = torch.load('./pvgg3/pruned0.1_22.pth.tar')
#print('cfg',c['cfg'])
model = vgg(dataset=args.dataset, depth=args.depth,cfg=c['cfg'])
if args.cuda:
    model.cuda()

print(model)
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

#data
kwargs = {'num_workers': 0, 'pin_memory': False} if args.cuda else {}
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
elif args.dataset == 'CINIC':
    print('CINIC')
    train = datasets.ImageFolder('./data/train',
                                 transform=transforms.Compose([
                                     transforms.Pad(4),
                                     transforms.RandomCrop(32),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.47889522, 0.47227842, 0.43047404),(0.24205776, 0.23828046, 0.25874835))
                                     ]))
    test = datasets.ImageFolder('data/test',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                     transforms.Normalize((0.47889522, 0.47227842, 0.43047404),(0.24205776, 0.23828046, 0.25874835))
                                     ]))
    print(train)
    train_loader = torch.utils.data.DataLoader(train,batch_size=args.batch_size, shuffle=True, **kwargs)

    train1,val = torch.utils.data.random_split(test,[75000,15000])
    print(val)
    print(len(val))
    #train_loader = torch.utils.data.DataLoader(train1,batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(val,batch_size=args.test_batch_size, shuffle=True, **kwargs)
    print(len(train_loader))
    print(len(test_loader))
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

#[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
feature_result = []
#feature_result.append(torch.zeros(64))
total = []
#total.append(0)
f = 0
f1 = -1

def get_feature_hook(self,input ,output):
    #print('hook')  
    global f1
    a = output.shape[0]
    b = output.shape[1]
    if (f1 != f):
        feature_result.append(torch.zeros(b))
        total.append(0)
        f1 = f1 + 1
    
    #print(a)
    #print(b)
    c = torch.zeros(b)
    #print('c.shape:',c.size())
    for i in range(a):
        for j in range(b):
            u,s,v = torch.svd(output[i,j,:,:])
            c[j] = c[j]+s.sum()

    #print('c = ',c)
    
    feature_result[f] = feature_result[f] * total[f] +c
    #print('feature_result = ',feature_result)
    total[f] = total[f] + a
    feature_result[f] = feature_result[f]/total[f]
    
    #print('feature_result = ',feature_result)
    #print('total = ',total)
            


def rank_svd(model):
    global f
    #print('f= ',f)
    limit = 1
    i = 0
    
    model.eval()
    correct = 0
    for data, target in train_loader:
        if i >= limit:
            break
        i = i+1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    f = f + 1
    #print('\nTrain: Accuracy: {}/{} ({:.1f}%)\n'.format(
    #    correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))
    #return correct / float(len(train_loader.dataset))

for m in model.modules():
    if isinstance(m,nn.ReLU):
        handler = m.register_forward_hook(get_feature_hook)
        rank_svd(model)
        handler.remove()

total_channel = 0
for n in feature_result:
    #print(n.size())
    total_channel = total_channel + n.shape[0]
print('total_channel:',total_channel)

index = 0
feature_s = torch.zeros(total_channel)
for n in feature_result:
    size = n.shape[0]
    feature_s[index:(index+size)] = n
    index = index +size

y, i = torch.sort(feature_s)
thre_index = int(total_channel * args.percent)
thre = y[thre_index]
#print(model)
#total = 0
#for m in model.modules():
#    print(m)
#    if isinstance(m, nn.BatchNorm2d):
#        total += m.weight.data.shape[0]

#bn = torch.zeros(total)
#c = torch.zeros(total)
#c_bn = torch.zeros(total)
#index = 0
#index_c = 0

#bn_j = 0
#c_j = 0
#w = []
#for m in model.modules():
#    if isinstance(m, nn.BatchNorm2d):
#        size = m.weight.data.shape[0]
#        bn[index:(index+size)] = m.weight.data.abs().clone()
#        bnn = m.weight.data.abs().clone()
#        index += size
#        bn_j = 1
#    if isinstance(m,nn.Conv2d):
#        size_c = m.weight.data.shape[0]
#        c[index_c:(index_c+size_c)] = torch.sum(m.weight.data.abs(),dim=(1,2,3)).clone()
#        cc = torch.sum(m.weight.data.abs(),dim=(1,2,3)).clone()
#        index_c += size_c
#        c_j = 1
#    if (bn_j==1) and (c_j == 1):
#        w.append(cc.mul(bnn))
#        bn_j = 0
#        cj = 0

#c_bn = bn.mul(c)

#y, i = torch.sort(feature_s)
#y, i = torch.sort(bn)
#thre_index = int(total_channel * args.percent)
#thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
i = 0
for k, m in enumerate(model.modules()):
    
    if isinstance(m, nn.ReLU):
        #weight_copy = m.weight.data.abs().clone()
        #mask = weight_copy.gt(thre).float()#.cuda()
        feature_copy = feature_result[i]
        #print("i=",i)
        mask = feature_copy.gt(thre).float()#.cuda()
        if torch.sum(mask) == 0:
            cfg.append(len(feature_copy))
            cfg_mask.append(torch.ones(len(feature_copy)))
            i +=1
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, len(feature_copy), int(len(feature_copy))))
        else:
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            #m.weight.data.mul_(mask)
            #m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            i+=1
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total_channel

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    

    
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

acc = test(model)

# Make real prune
print(cfg)
newmodel = vgg(dataset=args.dataset, cfg=cfg)
if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
print(num_parameters)
savepath = os.path.join(args.save, "prune0.1_23.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc)+"\n")
    fp.write("pruned: \n"+str(pruned)+"\n")
    fp.write("pruned_ratio: \n"+str(pruned_ratio)+"\n")

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            print("bn,resize")
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        #print(len(cfg_mask))
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
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

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned0.1_23.pth.tar'))

print(newmodel)
model = newmodel
test(model)
