from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--sparsity-regularization', '-sr',dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.00001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='./pres2/pruned0.1_26.pth.tar', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs2', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='resnet110_cifar', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=164, type=int,
                    help='depth of the neural network')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


print('args.sr=',args.sr)
#torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
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
        batch_size=args.batch_size, shuffle=True, **kwargs)
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



if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch](cfg=checkpoint['cfg'])
    model.load_state_dict(checkpoint['state_dict'])
else:
    cfg=[16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,
         32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,
         64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64,64]
    model = models.__dict__[args.arch](cfg=cfg)

if args.cuda:
    model.cuda()

print(model)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

channel_SL = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,
              38,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,
              75,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110]

"""
# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1
"""

#for m in model.modules():
#    print(m)

'''
def updateBN():
    i=1
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if i in channel_SL:
                m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1
                i+=1
            else:
                i+=1

def L1_loss():
    i=1
    l1_loss = torch.tensor(0.,requires_grad=True).cuda()#float()
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            if i in channel_SL:
                l1_loss +=torch.sum(m.weight.data.abs(),dim=(0,1,2,3))
                i +=1
            else:
                i+=1
    return l1_loss
'''

"""
def L1_loss():
    l1_loss = torch.tensor(0.,requires_grad=True).cuda()#float()
    for m in model.modules():
        if isinstance(m,nn.Conv2d):
            l1_loss +=torch.sum(m.weight.data.abs(),dim=(0,1,2,3))
    return l1_loss
"""        

def train(epoch):
    print("train")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        #L1 = L1_loss()
        loss = F.cross_entropy(output, target)#+1e-7*L1
        pred = output.data.max(1, keepdim=True)[1]
        loss.backward()
        
        #if args.sr:
        #    updateBN()
        
                
        optimizer.step()
        if batch_idx % args.log_interval == 0:
        
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    #print("L1=",L1)
    

def test():
    print("test")
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint0.1_26.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint0.1_26.pth.tar'), os.path.join(filepath, 'model_best0.1_26.pth.tar'))

best_prec1 = 0.

for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.4, args.epochs*0.6,args.epochs*0.8]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    print('epoch=',epoch)
    train(epoch)
    prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save)

print("Best accuracy: "+str(best_prec1))

"""
for m in model.modules():
    if isinstance(m,nn.Conv2d):
        print("c = ",m.weight.data)
    elif isinstance(m,nn.BatchNorm2d):
        print("b = ",m.weight.data)
"""




