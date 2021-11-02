from __future__ import print_function
import os 

'''Train Office31 with PyTorch.'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from adv import VATLoss
from torchvision import datasets


import argparse

from apn import APN

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
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epoch_decay_start', type=int, default=60)
parser.add_argument('--kth', default=0, help='the kth run of the algorithm for the same seed')
parser.add_argument('--note', default='', type=str)

parser.add_argument('--fs', default=512, type=int)
parser.add_argument('--lamb', default=0.5, type=float)
parser.add_argument('--temp', default=10.0, type=float)
batch_size = 64

args = parser.parse_args()
store_weights = True


writer_train_loss = SummaryWriter("runs/train/loss/")
writer_train_acc = SummaryWriter("runs/train/acc/")
writer_val_test_acc = SummaryWriter("runs/val_test/acc/")
writer_val_test_loss = SummaryWriter("runs/val_test/loss/")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
last_acc = 0
best_avg_acc = 0
last_avg_acc = 0

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

nb_classes = 2
nb_epochs = args.n_epoch


transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.19374922,0.19374922,0.19374922),  (0.14204098,0.14204098,0.14204098)),

])

transform_test = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.19374922,0.19374922,0.19374922),  (0.14204098,0.14204098,0.14204098)),   
])



train_dir = '/hdd8/zhulei/spine-mets/Jul012021_UpdatedTrainTestSplitandLabels/PreprocessedVersion_MedoidPoint10_Jul062021_combine_mild_normal_binary/train'
val_dir = '/hdd8/zhulei/spine-mets/Jul012021_UpdatedTrainTestSplitandLabels/PreprocessedVersion_MedoidPoint10_Jul062021_combine_mild_normal_binary/val'
test_dir='/hdd8/zhulei/spine-mets/Jul012021_UpdatedTrainTestSplitandLabels/PreprocessedVersion_MedoidPoint10_Jul062021_combine_mild_normal_binary/test'

folders = ['normal', 'abnormal']

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

class_indexes_test = {}
for folder in folders:
    for file in os.listdir(os.path.join(test_dir, folder)):
        if folders.index(folder) in class_indexes_test:
            class_indexes_test[folders.index(folder)].append(os.path.join(test_dir, folder, file))
        else:
            class_indexes_test[folders.index(folder)] = [os.path.join(test_dir, folder, file)]


# for k, v in class_indexes_val.items():
#     v.sort()

for k, v in class_indexes_test.items():
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

image_list_test = []
for k,v in class_indexes_test.items():
    for e in v:
        image_list_test.append((k,e))
random.shuffle(image_list_test)

class_size = []
for folder in folders:
    class_size.append(len(class_indexes_train[folders.index(folder)]))

class_weights = [1-float(e)/sum(class_size) for e in class_size]
class_weights = torch.FloatTensor(class_weights).cuda()
# print(class_size)
# print(class_weights)
# exit(0)


train_loader = torch.utils.data.DataLoader(ImageFolder2(transform_train, image_list_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
val_loader = torch.utils.data.DataLoader(ImageFolder2(transform_test, image_list_val), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
test_loader = torch.utils.data.DataLoader(ImageFolder2(transform_test, image_list_test), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
learning_rate = args.lr

# Model
print('==> Building model ...')

model = APN(feat_size=args.fs, nb_prototypes=nb_classes, lamb=args.lamb, temp=args.temp)
model.cuda()
print(model.parameters)

state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)

# load imagenet weight
for name, m in model.state_dict().items():
    if name[5:] in state_dict:
        m.data.copy_(state_dict[name[5:]])


for name, m in model.named_parameters():
    if m.requires_grad:
        print(name)


criterion = nn.CrossEntropyLoss(weight=class_weights)

# Adjust learning rate and betas for Adam Optimizer
lr_plan = [learning_rate] * args.n_epoch
# for i in range(0, args.n_epoch):
#     # lr_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
#     if i < 60:
#         lr_plan[i] = learning_rate
#     # elif i < 60:
#     #     lr_plan[i] = learning_rate / 10.
#     # elif i < 80:
#     #     lr_plan[i] = learning_rate / 100.
#     else:
#         lr_plan[i] = learning_rate / 10.


# Training
def train(epoch, lr_scheduler=None):
    print('\nEpoch: %d' % epoch)

    optimizer = torch.optim.SGD([
            {'params': model.predictor.parameters(), 'lr': lr_plan[epoch]},
            {'params': model.feat.parameters(), 'lr': lr_plan[epoch]/10.},
            ], weight_decay=0.0005)

    train_loss = 0
    correct = 0
    total = 0

    model.train()
    loader_train = iter(train_loader)
    iteration = len(train_loader)

    confusion_matrix = np.zeros((nb_classes, nb_classes))
    vat_loss = VATLoss(0.1, eps=0.1, ip=1)
    print('start training ...')
    # count_pix_value = 0
    # count_number = 0

    for batch_idx in range(0, iteration):

        inputs, targets = next(loader_train)
        inputs, targets = inputs.to(device), targets.to(device)  


        optimizer.zero_grad()

        logits, reg = model(inputs, targets, epoch)
        ce_loss = criterion(logits, targets)
        loss = ce_loss + reg
        loss.backward()
        optimizer.step()
        lds, lds_each = vat_loss(model.predictor, logits)

        train_loss += loss.item()

        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        targets = list(targets.detach().cpu().numpy())
        predicted = list(predicted.detach().cpu().numpy())

        for i in range(0, len(targets)):
            confusion_matrix[targets[i]][predicted[i]] += 1

    # print('mean:', count_pix_value/count_number)
    # print('std:', np.sqrt(count_pix_value/count_number))
    # exit(0)
    

    accs_per_class = []
    for i in range(0, nb_classes):
        accs_per_class.append(confusion_matrix[i, i] / np.sum(confusion_matrix[i]))

    accs_per_class = np.array(accs_per_class)
    avg_acc_per_class = 100. * np.mean(accs_per_class)        

    train_acc = 100.*float(correct)/total


    for i in range(0, nb_classes):
        pps = ''
        for j in range(0, nb_classes):
            pps += str(confusion_matrix[i][j]) + ', '
        pps += str(round(100.*accs_per_class[i],2))
        print(pps)


    writer_train_acc.add_scalar('train_acc', train_acc, epoch+1)
    writer_train_loss.add_scalar('train_loss', train_loss/batch_size, epoch+1)
    print ('Epoch [%d/%d], Lr: %F, Training Accuracy: %.2F, Avg Acc Per Class: %.2F, Loss: %.2f, reg: %.2f.' % (epoch+1, args.n_epoch, lr_plan[epoch], train_acc, avg_acc_per_class, loss.item(), reg.item()))

    return train_acc, avg_acc_per_class


def val(epoch):
    global best_acc
    global last_acc
    global best_avg_acc
    global last_avg_acc
    
    model.eval()

    correct = 0
    total = 0
    confusion_matrix = np.zeros((nb_classes, nb_classes))
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
        # for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, reg = model(inputs, targets, epoch)
            ce_loss = criterion(logits, targets)
            loss = ce_loss + reg
            val_loss += loss.item()


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


    for i in range(0, nb_classes):
        pps = ''
        for j in range(0, nb_classes):
            pps += str(confusion_matrix[i][j]) + '    '
        pps += str(round(100.*accs_per_class[i],2))
        print(pps)

    print('acc   ', 100.*(np.trace(confusion_matrix)/np.sum(confusion_matrix)),  ' ', avg_acc_per_class)

    if avg_acc_per_class > best_avg_acc:
        best_avg_acc = avg_acc_per_class

    # Save checkpoint.
    acc = 100.*correct/total
    last_acc = acc
    
    if acc > best_acc:
        best_acc = acc
    
    # Save checkpoint.
    if store_weights:
        print('Saving..')
        state1 = {
            'net': model.state_dict(),
            'acc': last_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('./checkpoint-july-2021/'):
            os.makedirs('./checkpoint-july-2021/')

        torch.save(state1, './checkpoint-july-2021/data-july-2021-orig-binary-model-'+str(args.kth)+'-'+str(epoch)+'.t7')
    # print ('Epoch [%d/%d], Acc: %.2F, Best Acc: %.2F, Avg Acc Per Class: %.2F, Best Avg Acc Per Class: %.2F.' % (epoch+1, args.n_epoch, acc, best_acc, avg_acc_per_class, best_avg_acc))
    
    return acc,val_loss/batch_size


def test(epoch):
    global best_acc
    global last_acc
    global best_avg_acc
    global last_avg_acc
    save_model=False
    model.eval()
    test_loss=0
    correct = 0
    total = 0
    confusion_matrix = np.zeros((nb_classes, nb_classes))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
        # for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            

            logits, reg = model(inputs, targets, epoch)
            ce_loss = criterion(logits, targets)
            loss = ce_loss + reg
            test_loss += loss.item()

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


    for i in range(0, nb_classes):
        pps = ''
        for j in range(0, nb_classes):
            pps += str(confusion_matrix[i][j]) + '    '
        pps += str(round(100.*accs_per_class[i],2))
        print('test results:',pps)

    # print('acc   ', 100.*(np.trace(confusion_matrix)/np.sum(confusion_matrix)), '  ', avg_acc_per_class)

    if avg_acc_per_class > best_avg_acc:
        best_avg_acc = avg_acc_per_class
        save_model=True
    # Save checkpoint.
    acc = 100.*correct/total
    last_acc = acc
    
    if acc > best_acc:
        best_acc = acc
    
    # Save checkpoint.
        # Save checkpoint.
    if store_weights :
        print('Saving..')
        state1 = {
            'net': model.state_dict(),
            'acc': last_acc,
            'epoch': epoch,
        }
        torch.save(state1, '/hdd8/wenqiao/checkpoint_5/'+str(epoch)+'.t7')
    print ('test Epoch [%d/%d], Acc: %.2F, Best Acc: %.2F, Avg Acc Per Class: %.2F, Best Avg Acc Per Class: %.2F.' % (epoch+1, args.n_epoch, acc, best_acc, avg_acc_per_class, best_avg_acc))
    return acc,test_loss/batch_size


# code for train
for epoch in range(start_epoch, args.n_epoch):
    train_acc, train_avg_acc = train(epoch)
    # val_acc,val_loss = val(epoch)
    test_acc,test_loss=test(epoch)

    # writer_val_test_acc.add_scalars('acc', {'val': val_acc, 'test': test_acc}, epoch+1)
    # writer_val_test_loss.add_scalars('loss', {'val': val_loss, 'test': test_loss}, epoch+1)
    with open('./records/data-july-2021-orig-binary-record-sgd-'+str(args.lr)+'-'+str(args.kth)+'-'+str(args.lamb)+'-'+str(args.temp)+'-'+str(args.n_epoch)+'-'+args.note+'.txt', "a") as myfile:
        myfile.write(str(args.kth) +'-'+ str(int(epoch)) + '-' + str(batch_size) + ': '  + str(train_acc) + ' ' + str(train_avg_acc) + ' ' + str(test_acc) + ' ' + str(best_acc) + ' ' + str(last_avg_acc) + ' ' + str(best_avg_acc) + "\n")

with open('./record-all.txt', 'a') as f:
    f.write('data-july-2021-orig-binary-'+str(args.kth)+'-sgd-'+str(args.lr)+'-'+args.note+'-'+str(args.lamb)+'-'+str(args.temp)+'-'+str(last_acc)+'-'+str(best_acc)+'-'+str(last_avg_acc)+'-'+str(best_avg_acc)+'\n')
