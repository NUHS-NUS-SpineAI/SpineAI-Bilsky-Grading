import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from resnet import resnet50
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)

        return x




class Predictor(nn.Module):
    def __init__(self, nb_prototypes=6, lamb=0.5, temp=10.0):
        super(Predictor, self).__init__()

        self.nb_prototypes = nb_prototypes
        self.lamb = lamb
        self.temp = temp

        self.prototypes = Parameter(torch.rand((nb_prototypes, 256)).cuda())

    def forward(self, x, xlabel=None, class_weights=None):

        out = x

        XY = torch.matmul(out, torch.t(self.prototypes))
        XX = torch.sum(torch.pow(out,2), dim=1, keepdim=True)
        YY = torch.sum(torch.pow(torch.t(self.prototypes),2), dim=0, keepdim=True)
        neg_sqr_dist = XX - 2.0 * XY + YY

        logits = (-1.0/self.temp) * neg_sqr_dist

        if xlabel is not None:
            prot_batch = torch.index_select(self.prototypes, 0, xlabel)
            instance_weights = torch.index_select(class_weights, 0, xlabel)

            if class_weights is not None:
                # print(torch.sum(torch.pow(out-prot_batch,2), dim=1).shape)
                compact_reg_loss = self.lamb * torch.sum(instance_weights*torch.sum(torch.pow(out-prot_batch,2), dim=1)) * (1. / out.size(0))
            else:
                compact_reg_loss = self.lamb * torch.sum(torch.pow(out-prot_batch,2)) * (1. / out.size(0))

            return logits, compact_reg_loss

        return logits, torch.tensor(0).cuda()


class APN(nn.Module):
    def __init__(self, feat_size=512, nb_prototypes=6, lamb=0.5, temp=10.0):
        super(APN, self).__init__()

        self.feat_size = feat_size
        self.nb_prototypes = nb_prototypes
        self.lamb = lamb
        self.temp = temp

        self.feat = Net()
        self.predictor = Predictor(nb_prototypes=nb_prototypes, lamb=lamb, temp=temp)

    def forward(self, x, xlabel, epoch, class_weights=None):
        out = self.feat(x)
        logits, compact_reg_loss = self.predictor(out, xlabel, class_weights)
        return logits, compact_reg_loss
        