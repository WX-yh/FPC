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
parser.add_argument('--depth', type=int, default=40,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.1,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='./logs4/model_best0.1_14.tar', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='pdes4', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)
cfg1 = [12]*36
start1 = 24
trans1_channel=168
trans2_channel=312

c = torch.load('./pdes4/pruned0.1_14.pth.tar')
cfg1 = c['cfg']
start1 = c['start']
trans1_channel_o = c['trans1_channel']
trans2_channel_o = c['trans2_channel']

dense1_channel_o = cfg1[0:12]
dense2_channel_o = cfg1[12:24]
dense1_channel_sum_o = 0
dense2_channel_sum_o = 0
for i in dense1_channel_o:
    dense1_channel_sum_o += i
dense1_channel_sum_o += start1
#print('dense1_channel_sum_o = ',dense1_channel_sum_o)
for i in dense2_channel_o:
    dense2_channel_sum_o  += i
dense2_channel_sum_o += dense1_channel_sum_o
#print('dense2_channel_sum_o = ',dense2_channel_sum_o)
model = densenet(start=start1, trans1_channel=trans1_channel_o, trans2_channel=trans2_channel_o, cfg = cfg1, dataset=args.dataset)
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


feature_result = []
total = []
feature_result_start =[]
total_start = []
feature_result_trans = []
total_trans = []

f = 0
f1 = -1
f_start = 0
f1_start = -1
f_trans = 0
f1_trans = -1

cfg_id = 0
start = 0

def get_feature_hook(self,input ,output):
    global f1
    global f1_start
    global f_start
    global f1_trans
    global f_trans
    global start
    global dense1_channel_sum_o, dense2_channel_sum_o
    #print('start = ',start)
    if start == 0:
        a = output.shape[0]
        b = output.shape[1]
        #dense1_channel_sum_o += b
        #dense2_channel_sum_o += dense1_channel_sum_o
        #print('a = ',a)
        #print('b = ',b)
        if (f1_start != f_start):
            feature_result_start.append(torch.zeros(b))
            total_start.append(0)
            f1_start = f1_start + 1
        
        c = torch.zeros(b)
        #print('f_start = ',f_start)
        for i in range(a):
            for j in range(b):
                u,s,v = torch.svd(output[i,j,:,:])
                c[j] = c[j]+s.sum()

        feature_result_start[f_start] = feature_result_start[f_start] * total_start[f_start] + c
        total_start[f_start] = total_start[f_start] + a
        feature_result_start[f_start] = feature_result_start[f_start]/total_start[f_start]
        start += 1
        #print('total_start = ',total_start)
        f_start = f_start + 1
        
    
    else:
        a = output.shape[0]
        b = output.shape[1]
        if (f1_trans != f_trans):
            feature_result_trans.append(torch.zeros(b))
            total_trans.append(0)
            f1_trans = f1_trans + 1
        
        c = torch.zeros(b)
    
        for i in range(a):
            for j in range(b):
                u,s,v = torch.svd(output[i,j,:,:])
                c[j] = c[j]+s.sum()

        feature_result_trans[f_trans] = feature_result_trans[f_trans] * total_trans[f_trans] +c
        total_trans[f_trans] = total_trans[f_trans] + a
        feature_result_trans[f_trans] = feature_result_trans[f_trans]/total_trans[f_trans]
        f_trans = f_trans +1
        print('total_trans = ',total_trans)

def get_feature_hook_densenet(self,input ,output):
    global f1
    global f
    global cfg_id
    a = output.shape[0]
    b = output.shape[1]
    #print('a = ',a)
    #print('b = ',b)
    if (f1 != f):
        #print('cfg_id = ',cfg_id)
        #print('cfg[cfg_id] = ',cfg1[cfg_id])
        feature_result.append(torch.zeros(cfg1[cfg_id]))
        total.append(0)
        f1 = f1 + 1

    c = torch.zeros(cfg1[cfg_id])

    for i in range(a):
        for j in range(b-cfg1[cfg_id],b):
            u,s,v = torch.svd(output[i,j,:,:])
            c[j-(b-cfg1[cfg_id])] = c[j-(b-cfg1[cfg_id])]+s.sum()
    #print('c = ',c)
    
    feature_result[f] = feature_result[f] * total[f] +c
    #print('feature_result = ',feature_result)
    total[f] = total[f] + a
    #print('total = ',total)
    feature_result[f] = feature_result[f]/total[f]
    cfg_id = cfg_id + 1
    f = f + 1
    #print('total = ',total)
    
def rank_svd(model):
    global f
    global f1_start
    global f1_trans
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
    #print('\nTrain: Accuracy: {}/{} ({:.1f}%)\n'.format(
    #    correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))
    #return correct / float(len(train_loader.dataset))

for i in range(3):
    dense = eval('model.dense%d' % (i + 1))
    for j in range(12):
        cov_layer = dense[j].relu
        if j==0:
            handler = cov_layer.register_forward_hook(get_feature_hook)
        else:
            handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
        rank_svd(model)
        handler.remove()

    if i<2:
        trans=eval('model.trans%d' % (i + 1))
        cov_layer = trans.relu
        handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
        rank_svd(model)
        handler.remove()
cov_layer = model.relu
handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
rank_svd(model)
handler.remove()

total_channel = 0
index = 0

for n in feature_result:
    #print('n shape:',n.shape[0])
    total_channel = total_channel + n.shape[0]

for n in feature_result_start:
    #print('n shape:',n.shape[0])
    total_channel = total_channel + n.shape[0]


print('total_channel:',total_channel)
feature_s = torch.zeros(total_channel)

for n in feature_result:
    size = n.shape[0]
    feature_s[index:(index+size)] = n
    index = index +size

for n in feature_result_start:
    size = n.shape[0]
    feature_s[index:(index+size)] = n
    index = index +size


y, i = torch.sort(feature_s)
thre_index = int(total_channel * args.percent)
thre = y[thre_index]

pruned = 0

cfg = []
cfg_mask = []

for i in range(36):
    #print('i=',i)
    feature_copy = feature_result[i]
    mask = feature_copy.gt(thre).float()#.cuda()
    if torch.sum(mask) <= 3:
        cfg.append(len(feature_copy))
        cfg_mask.append(torch.ones(len(feature_copy)).float())#.cuda())
        print('total channel: {:d} \t remaining channel: {:d}'.
            format(len(feature_copy), int(len(feature_copy))))
    else:
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('total channel: {:d} \t remaining channel: {:d}'.
            format(mask.shape[0], int(torch.sum(mask))))

cfg_start = 0
cfg_mask_start = []
for i in range(1):
    #print('i=',i)
    feature_copy = feature_result_start[i]
    mask = feature_copy.gt(thre).float()#.cuda()
    if torch.sum(mask) == 0:
        cfg_start = (len(feature_copy))
        cfg_mask_start.append(torch.ones(len(feature_copy)).float())#.cuda())
        print('start total channel: {:d} \t remaining channel: {:d}'.
            format(len(feature_copy), int(len(feature_copy))))
    else:
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        cfg_start = int(torch.sum(mask))
        cfg_mask_start.append(mask.clone())
        print('start total channel: {:d} \t remaining channel: {:d}'.
            format(mask.shape[0], int(torch.sum(mask))))

print("Cfg:")
print(cfg)
print("Cfg_start:")
print(cfg_start)


dense1_channel = cfg[0:12]
#print('dense1_channel = ',dense1_channel)
dense2_channel = cfg[12:24]
#print('dense2_channel = ',dense2_channel)
dense1_channel_sum = 0
dense2_channel_sum = 0
for i in dense1_channel:
    #print('1_i = ',i)
    dense1_channel_sum += i

for i in dense2_channel:
    #print('2_i = ',i)
    dense2_channel_sum += i
trans1_channel = dense1_channel_sum + cfg_start
trans2_channel = dense2_channel_sum + dense1_channel_sum + cfg_start
#print('trans1_channel = ',trans1_channel)
#print('trans2_channel = ',trans2_channel)

y1,i1 = torch.sort(feature_result_trans[0])
thre1 = y1[dense1_channel_sum_o - trans1_channel - 1]
 
y2,i2 = torch.sort(feature_result_trans[1])
thre2 = y2[dense2_channel_sum_o - trans2_channel - 1]

#print('dense2_channel_sum_o = ',dense2_channel_sum_o)
#print('thre2 = ',thre2)
#print('len(y2)',len(y2))
#print('y2 = ',y2)

mask_trans1 = feature_result_trans[0].gt(thre1).float().clone()#.cuda()
mask_trans2 = feature_result_trans[1].gt(thre2).float().clone()#.cuda()

#mask_trans1 = torch.ones(trans1_channel)#.cuda()
#mask_trans2 = torch.ones(trans2_channel)#.cuda()

pruned = pruned + mask_trans1.shape[0] - torch.sum(mask_trans1)
pruned = pruned + mask_trans2.shape[0] - torch.sum(mask_trans2)

print("Cfg_trans1:")
print(torch.sum(mask_trans1))
if (torch.sum(mask_trans1) != trans1_channel):
    trans1_channel = int(torch.sum(mask_trans1))

print("Cfg_trans2:")
print(torch.sum(mask_trans2))
print("Cfg_trans2_t = ",(trans2_channel))
if (torch.sum(mask_trans2) != trans2_channel):
    trans2_channel = int(torch.sum(mask_trans2))

if torch.sum(mask_trans1) == 0:
    trans1_channel = trans1_channel_o
    mask_trans1 = torch.ones(trans1_channel).float()#.cuda()

if torch.sum(mask_trans2) == 0:
    trans2_channel = trans2_channel_o
    mask_trans2 = torch.ones(trans2_channel).float()#.cuda()
    
print("Cfg_trans2_t = ",(trans2_channel))
total_channel = total_channel + dense1_channel_sum_o + dense2_channel_sum_o
pruned_ratio = pruned/total_channel
print('pruned_ratio=',pruned_ratio)

print('Pre-processing Successful!')


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    '''
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
    '''
    
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

#acc = test(model)


print('cfg = ',cfg)
print('cfg_start = ',cfg_start)
newmodel = densenet(start= cfg_start, trans1_channel=trans1_channel,trans2_channel=trans2_channel,cfg = cfg, dataset=args.dataset)
print(newmodel)
#print('newmodel',newmodel)



if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune0.1_15.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    #fp.write("Test accuracy: \n"+str(acc))
    fp.write("pruned: \n"+str(pruned)+"\n")
    fp.write("pruned_ratio: \n"+str(pruned_ratio)+"\n")

#old_modules = list(model.modules())
#new_modules = list(newmodel.modules())
#print("########################################")
#test(newmodel)

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
layer_id = 1
t =1
#print("########################################")
#test(newmodel)
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    #print('t = ',t)
    if t == 1:
        if isinstance(m0,nn.Conv2d):
            #print('t = ',t)
            #print('m0 = ',m0)
            
            #print('卷积层1')
            #print('start_mask = ',start_mask)
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask_start[0].cpu().numpy())))
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
        elif isinstance(m0,nn.BatchNorm2d):
            #print('t = ',t)
            #print('m0 = ',m0)
            #print('BN层1')
            idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask_start[0].cpu().numpy())))
            if idx1.size == 1:
                print("bn,resize")
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            
            #layer_id += 1
            start_mask = cfg_mask_start[0]
            t += 1
    elif t == 14:
        #print('3')
        #print('t = ',t)
        #print('m0 = ',m0)
        if isinstance(m0, nn.BatchNorm2d):
            #print('5')
            idx1 = np.squeeze(np.argwhere(np.asarray(mask_trans1.cpu().numpy())))
            if idx1.size == 1:
                print("bn,resize")
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            start_mask = mask_trans1
            t += 1
            
        elif isinstance(m0, nn.Conv2d):
            #print('t= ',t)
            #print('m0 = ',m0)
            #print('6')
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(mask_trans1.cpu().numpy())))
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

    elif t == 27:
        #print('3')
        #print('27')
        if isinstance(m0, nn.BatchNorm2d):
            #print('t = ',t)
            #print('m0 = ',m0)
            #print('5')
            idx1 = np.squeeze(np.argwhere(np.asarray(mask_trans2.cpu().numpy())))
            if idx1.size == 1:
                print("bn,resize")
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            start_mask = mask_trans2
            t += 1
        elif isinstance(m0, nn.Conv2d):
            #print('6')
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(mask_trans2.cpu().numpy())))
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
        
    elif isinstance(m0, nn.BatchNorm2d):
        #print('2')
        #print('t = ',t)
        #print('m0 = ',m0)
        #print('start_mask = ',start_mask)
        #print('end_mask = ',end_mask)
        con = torch.cat((start_mask,end_mask),)
        #print('con = ',con)
        idx1 = np.squeeze(np.argwhere(np.asarray(con.cpu().numpy())))
        if idx1.size == 1:
            print("bn,resize")
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        t+=1
        start_mask = con.clone()
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]
            
    elif isinstance(m0, nn.Conv2d):
        #print('1')
        #print('t = ',t)
        #print('m0 = ',m0)
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
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
        #print('4')
            
#print("########################################")
#test(newmodel)

'''
first_conv = True

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))

        if isinstance(old_modules[layer_id + 1], channel_selection):
            # If the next layer is the channel selection layer, then the current batch normalization layer won't be pruned.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            # We need to set the mask parameter `indexes` for the channel selection layer.
            m2 = new_modules[layer_id + 1]
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
            continue

    elif isinstance(m0, nn.Conv2d):
        if first_conv:
            # We don't change the first convolution layer.
            m1.weight.data = m0.weight.data.clone()
            first_conv = False
            continue
        if isinstance(old_modules[layer_id - 1], channel_selection):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            # If the last layer is channel selection layer, then we don't change the number of output channels of the current
            # convolutional layer.
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            m1.weight.data = w1.clone()
            continue

    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhe re(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))

        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()
'''

torch.save({'trans1_channel':trans1_channel,'trans2_channel':trans2_channel,'start':cfg_start, 'cfg': cfg, 'state_dict': newmodel.state_dict()},
           os.path.join(args.save, 'pruned0.1_15.pth.tar'))

#print(newmodel)
model = newmodel
test(newmodel)
