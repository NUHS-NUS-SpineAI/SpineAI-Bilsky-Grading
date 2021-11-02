from __future__ import print_function
import os 

'''Train Office31 with PyTorch.'''
import os,shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
import matplotlib.pyplot as plt

import argparse

from apn import APN

import random
from PIL import Image
import numpy as np
from itertools import cycle
import math
import pickle

import sklearn.metrics as metrics 
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import recall_score, accuracy_score,precision_score, f1_score
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
    # transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.19123026,0.19123026,0.19123026),  (0.15038502,0.15038502,0.15038502)),
    # transforms.Normalize((0.19374922,0.19374922,0.19374922),  (0.14204098,0.14204098,0.14204098)),
    # transforms.Normalize((0.24309957,0.24309957,0.24309957),  (0.15961831,0.15961831,0.15961831)),
    # transforms.Normalize((0.21285992,0.21285992,0.21285992),  (0.13401703,0.13401703,0.13401703)),
    # transforms.Normalize((0.19374922,0.19374922,0.19374922),  (0.14204098,0.14204098,0.14204098)),
    # transforms.Normalize((0.23393838,0.23393838,0.23393838),  (0.15613543,0.15613543,0.15613543)),
    

    # [0.24309957,0.24309957,0.24309957] [0.15961831,0.15961831,0.15961831]
    # [0.21285992,0.21285992,0.21285992] [0.13401703,0.13401703,0.13401703]
    # [0.19374922,0.19374922,0.19374922] [0.14204098,0.14204098,0.14204098]
    # Orig: [0.23393838,0.23393838,0.23393838] [0.15613543,0.15613543,0.15613543]
])

transform_test = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.19374922,0.19374922,0.19374922),  (0.14204098,0.14204098,0.14204098)),    
    transforms.Normalize((0.19374922,0.19374922,0.19374922),  (0.14204098,0.14204098,0.14204098)),  
])

test_dir='/hdd8/zhulei/spine-mets/Jul012021_UpdatedTrainTestSplitandLabels/PreprocessedVersion_OriginalCrop_Jul062021_combine_mild_normal_binary/test'

folders = ['normal', 'abnormal']




class_indexes_test = {}
for folder in folders:
    for file in os.listdir(os.path.join(test_dir, folder)):
        if folders.index(folder) in class_indexes_test:
            class_indexes_test[folders.index(folder)].append(os.path.join(test_dir, folder, file))
        else:
            class_indexes_test[folders.index(folder)] = [os.path.join(test_dir, folder, file)]



for k, v in class_indexes_test.items():
    v.sort()

random.seed(args.seed)



image_list_test = []
for k,v in class_indexes_test.items():
    for e in v:
        image_list_test.append((k,e))
random.shuffle(image_list_test)

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




def mycopyfile(srcfile,dstfile):
    fpath,fname=os.path.split(dstfile)    
    if not os.path.exists(fpath):
        os.makedirs(fpath)                
    shutil.copyfile(srcfile,dstfile)      


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
    Y_pred=[]
    Y_valid=[]
    file_paths=[]
    TP=0
    FN=0
    FP=0
    with torch.no_grad():
        for batch_idx, (inputs, targets, paths) in enumerate(test_loader):
        # for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits, reg = model(inputs, targets, epoch)
            _, predicted = logits.max(1)
            for logit in logits:
                Y_pred.append(logit.cuda().data.cpu().numpy())
            for target in targets:
                Y_valid.append(target.cuda().data.cpu().numpy())
            # for path in paths:
            #     file_paths.append(path.cuda().data.cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            targets = list(targets.detach().cpu().numpy())
            predicted = list(predicted.detach().cpu().numpy())

            # for i in range(len(paths)):
            #     fpath,fname=os.path.split(paths[i])
            #     des_path="/hdd8/wenqiao/spinemet/results/external"
            #     if int(predicted[i])==0:
            #         des_path="/hdd8/wenqiao/spinemet/results/external"+"/normal/"+fname
            #     if int(predicted[i])==1:
            #         des_path="/hdd8/wenqiao/spinemet/results/external"+"/abnormal/"+fname
            #     mycopyfile(paths[i],des_path)


            

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
    # if store_weights :
    #     print('Saving..')
    #     state1 = {
    #         'net': model.state_dict(),
    #         'acc': last_acc,
    #         'epoch': epoch,
    #     }

        
        # torch.save(state1, '/hdd8/wenqiao/checkpoint_2/'+str(epoch)+'.t7')
    # Y_pred=Y_pred.cuda(().data.cpu().numpy() 
    # Y_valid=Y_valid.cuda().data.cpu().numpy()
    Yre=Y_pred
    Yva=Y_valid
    Y_pred = [np.argmax(y) for y in Y_pred] 
    precision = metrics.precision_score(Y_valid, Y_pred, average='weighted')
    recall = metrics.recall_score(Y_valid, Y_pred, average='weighted')
    f1_score = metrics.f1_score(Y_valid, Y_pred, average='weighted')
    accuracy_score = metrics.accuracy_score(Y_valid, Y_pred)
    print("Precision_score:",precision)
    print("Recall_score:",recall)
    print("F1_score:",f1_score)
    print("Accuracy_score:",accuracy_score)
    fpr, tpr, thresholds_keras = metrics.roc_curve(Y_valid, Y_pred)
    auc = metrics.auc(fpr, tpr)
    print("AUC : ", auc)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=' (area = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig("ROC_2分类.png")



    print ('Acc: %.2F, Best Acc: %.2F, Avg Acc Per Class: %.2F, Best Avg Acc Per Class: %.2F.' % (acc, best_acc, avg_acc_per_class, best_avg_acc))
    return acc,test_loss/batch_size



# code for test
store_weights = False
checkpoint = torch.load('/hdd8/wenqiao/checkpoint_1/80.t7')
model.load_state_dict(checkpoint['net'])
#test_dir='/hdd8/zhulei/spine-mets/Jul012021_UpdatedTrainTestSplitandLabels/PreprocessedVersion_MedoidPoint10_Jul062021_combine_mild_normal_binary/test'
test_dir='/hdd8/zhulei/spine-mets/Jul012021_UpdatedTrainTestSplitandLabels/AKremoved_JH_GT_medoidP10_Oct162021'

class_indexes_test = {}
for folder in folders:
    for file in os.listdir(os.path.join(test_dir, folder)):
        if folders.index(folder) in class_indexes_test:
            class_indexes_test[folders.index(folder)].append(os.path.join(test_dir, folder, file))
        else:
            class_indexes_test[folders.index(folder)] = [os.path.join(test_dir, folder, file)]

for k, v in class_indexes_test.items():
    v.sort()

for i in range(0, nb_classes):
    random.shuffle(class_indexes_test[i])

image_list_test = []
for k,v in class_indexes_test.items():
    for e in v:
        image_list_test.append((k,e))
random.shuffle(image_list_test)

test_loader = torch.utils.data.DataLoader(ImageFolder2(transform_test, image_list_test), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)

test(1)