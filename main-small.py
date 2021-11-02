'''Train Office31 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets

import os
import argparse

from apn_small import APN

import random
from PIL import Image
import numpy as np
from itertools import cycle
import math
import pickle

from util import ImageFolder2

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

parser = argparse.ArgumentParser(description='PyTorch SpineMets Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch_decay_start', type=int, default=60)
parser.add_argument('--kth', default=0, help='the kth run of the algorithm for the same seed')

parser.add_argument('--fs', default=512, type=int)
parser.add_argument('--lamb', default=0.5, type=float)
parser.add_argument('--temp', default=10.0, type=float)


args = parser.parse_args()
store_weights = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
last_acc = 0
best_avg_acc = 0
last_avg_acc = 0

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

nb_classes = 6
nb_epochs = args.n_epoch
batch_size = 128

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.2)),
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.19123026,0.19123026,0.19123026),  (0.15038502,0.15038502,0.15038502)),
])

transform_test = transforms.Compose([
    # transforms.Resize((256, 256)),
    # transforms.CenterCrop((224, 224)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.19123026,0.19123026,0.19123026),  (0.15038502,0.15038502,0.15038502)),    
])


train_dir = '/hdd8/zhulei/spine-mets/1-199_TRAINING_T2-centroid-300scale-splitV1/train'
val_dir = '/hdd8/zhulei/spine-mets/1-199_TRAINING_T2-centroid-300scale-splitV1/val'

folders = ['b1a', 'b1b', 'b1c', 'b2', 'b3', 'normal']

class_indexes_train = {}
for folder in folders:
    for file in os.listdir(os.path.join(train_dir, folder)):
        if folders.index(folder) in class_indexes_train:
            class_indexes_train[folders.index(folder)].append(os.path.join(train_dir, folder, file))
        else:
            class_indexes_train[folders.index(folder)] = [os.path.join(train_dir, folder, file)]

for k, v in class_indexes_train.items():
    v.sort()

class_indexes_val = {}
for folder in folders:
    for file in os.listdir(os.path.join(val_dir, folder)):
        if folders.index(folder) in class_indexes_val:
            class_indexes_val[folders.index(folder)].append(os.path.join(val_dir, folder, file))
        else:
            class_indexes_val[folders.index(folder)] = [os.path.join(val_dir, folder, file)]

for k, v in class_indexes_val.items():
    v.sort()

random.seed(args.seed)
for i in range(0, nb_classes):
    random.shuffle(class_indexes_train[i])

image_list_train = []
for k,v in class_indexes_train.items():
    for e in v:
        image_list_train.append((k,e))
random.shuffle(image_list_train)

for i in range(0, nb_classes):
    random.shuffle(class_indexes_val[i])

image_list_val = []
for k,v in class_indexes_val.items():
    for e in v:
        image_list_val.append((k,e))
random.shuffle(image_list_val)


class_size = []
for folder in folders:
    class_size.append(len(class_indexes_train[folders.index(folder)])+len(class_indexes_val[folders.index(folder)]))

class_weights = [1-float(e)/sum(class_size) for e in class_size]
class_weights = torch.FloatTensor(class_weights).cuda()
# print(class_size)
# print(class_weights)
# exit(0)

train_loader = torch.utils.data.DataLoader(ImageFolder2(transform_train, image_list_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
val_loader = torch.utils.data.DataLoader(ImageFolder2(transform_test, image_list_val), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

learning_rate = args.lr

# Model
print('==> Building model ...')

model = APN(feat_size=args.fs, nb_prototypes=nb_classes, lamb=args.lamb, temp=args.temp)
model.cuda()
print(model.parameters)

# state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)

# # load imagenet weight
# for name, m in model.state_dict().items():
#     if name[5:] in state_dict:
#         m.data.copy_(state_dict[name[5:]])


# # initialize the weight for the center, covariance matrix and fc layer
# for name, m in model.named_parameters():    
#     if 'predictor' in name:
#         continue
#     if 'layer4' in name:
#         continue
#     m.requires_grad = False

for name, m in model.named_parameters():
    if m.requires_grad:
        print(name)


criterion = nn.CrossEntropyLoss(weight=class_weights)

# Adjust learning rate and betas for Adam Optimizer
lr_plan = [learning_rate] * args.n_epoch
for i in range(0, args.n_epoch):
    # lr_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
    if i < 40:
        lr_plan[i] = learning_rate
    elif i < 60:
        lr_plan[i] = learning_rate / 10.
    elif i < 80:
        lr_plan[i] = learning_rate / 100.
    else:
        lr_plan[i] = learning_rate / 1000.


# Training
def train(epoch, lr_scheduler=None):
    print('\nEpoch: %d' % epoch)

    optimizer = torch.optim.SGD([
            {'params': model.predictor.parameters(), 'lr': lr_plan[epoch]},
            {'params': model.feat.parameters(), 'lr': lr_plan[epoch]},
            ], weight_decay=0.0005)

    train_loss = 0
    correct = 0
    total = 0

    model.train()
    loader_train = iter(train_loader)
    iteration = len(train_loader)

    confusion_matrix = np.zeros((nb_classes, nb_classes))

    print('start training ...')
    # count_pix_value = 0
    # count_number = 0
    for batch_idx in range(0, iteration):

        inputs, targets = next(loader_train)
        inputs, targets = inputs.to(device), targets.to(device)

        # count_pix_value += np.sum(np.square(inputs.detach().cpu().numpy()-0.19123026), (0,2,3))
        # count_number += 224 * 224 * inputs.detach().cpu().numpy().shape[0]
        # print(inputs.shape)
        # print(np.max(inputs.detach().cpu().numpy()), np.min(inputs.detach().cpu().numpy()))
        # continue
        
        optimizer.zero_grad()

        logits, reg = model(inputs, targets, epoch, class_weights)
        ce_loss = criterion(logits, targets)
        loss = ce_loss + reg
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        targets = list(targets.detach().cpu().numpy())
        predicted = list(predicted.detach().cpu().numpy())

        for i in range(0, len(targets)):
            confusion_matrix[targets[i]][predicted[i]] += 1

    # print('std:', np.sqrt(count_pix_value/count_number))
    # exit(0)

    accs_per_class = []
    for i in range(0, nb_classes):
        accs_per_class.append(confusion_matrix[i, i] / np.sum(confusion_matrix[i]))

    accs_per_class = np.array(accs_per_class)
    avg_acc_per_class = 100. * np.mean(accs_per_class)        

    train_acc = 100.*float(correct)/total

    print ('Epoch [%d/%d], Lr: %F, Training Accuracy: %.2F, Avg Acc Per Class: %.2F, Loss: %.2f, reg: %.2f.' % (epoch+1, args.n_epoch, lr_plan[epoch], train_acc, avg_acc_per_class, loss.item(), reg.item()))

    return train_acc


def val(epoch):
    global best_acc
    global last_acc
    global best_avg_acc
    global last_avg_acc
    
    model.eval()

    correct = 0
    total = 0
    confusion_matrix = np.zeros((nb_classes, nb_classes))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, _ = model(inputs, None, epoch)

            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            targets = list(targets.detach().cpu().numpy())
            predicted = list(predicted.detach().cpu().numpy())

            for i in range(0, len(targets)):
                confusion_matrix[targets[i]][predicted[i]] += 1

    accs_per_class = []
    for i in range(0, nb_classes):
        accs_per_class.append(confusion_matrix[i, i] / np.sum(confusion_matrix[i]))

    accs_per_class = np.array(accs_per_class)
    avg_acc_per_class = 100. * np.mean(accs_per_class)

    last_avg_acc = avg_acc_per_class

    if avg_acc_per_class > best_avg_acc:
        best_avg_acc = avg_acc_per_class

    # Save checkpoint.
    acc = 100.*correct/total
    last_acc = acc
    
    if acc > best_acc:
        best_acc = acc
    
    # Save checkpoint.
    if store_weights and epoch == args.n_epoch-1:
        print('Saving..')
        state1 = {
            'net': model.state_dict(),
            'acc': last_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('/hdd8/zhulei/spine-mets/checkpoint'):
            os.mkdir('/hdd8/zhulei/spine-mets/checkpoint')

        torch.save(state1, '/hdd8/zhulei/spine-mets/checkpoint/small-model-'+str(args.n_epoch)+'-'+str(args.kth)+'.t7')

    print ('Epoch [%d/%d], Acc: %.2F, Best Acc: %.2F, Avg Acc Per Class: %.2F, Best Avg Acc Per Class: %.2F.' % (epoch+1, args.n_epoch, acc, best_acc, avg_acc_per_class, best_avg_acc))
    
    return acc


for epoch in range(start_epoch, args.n_epoch):
    train_acc = train(epoch)
    test_acc = val(epoch)

    with open('./record-sgd-'+str(args.lr)+'-'+str(args.kth)+'.txt', "a") as myfile:
        myfile.write(str(args.kth) +'-'+ str(int(epoch)) + '-' + str(batch_size) + ': '  + str(train_acc) + ' ' + str(test_acc) + "\n")

with open('./record-all.txt', 'a') as f:
    f.write('small-model-'+str(args.kth)+'-sgd-'+str(last_acc)+'-'+str(best_acc)+'-'+str(last_avg_acc)+'-'+str(best_avg_acc)+'\n')
