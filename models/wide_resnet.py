import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, use_bn=False):
        super(wide_basic, self).__init__()
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        if use_bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        if dropout_rate:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        if self.use_bn:
            out = self.conv1(F.relu(self.bn1(x)))
        else:
            out = self.conv1(F.relu(x))
        if self.dropout_rate:
            out = self.dropout(out)
        out = self.conv2(F.relu(out))
        out += self.shortcut(x)
        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, 
            use_bn=False, use_pooling=True, in_ch=3, in_dim=32, bn_after_dense=False):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        self.use_bn = use_bn
        self.use_pooling = use_pooling
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [self.in_planes, self.in_planes*2*k, self.in_planes*4*k, self.in_planes*8*k]

        self.conv1 = conv3x3(in_ch,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)

        dummy_in = torch.randn(1, in_ch, in_dim, in_dim)
        dummy_out = self.layer3(self.layer2(self.layer1(self.conv1(dummy_in))))
        self.pool_kernel = dummy_out.shape[-1]
        if self.use_pooling:
            dummy_out = F.avg_pool2d(dummy_out, self.pool_kernel)

        self.linear1 = nn.Linear(dummy_out.numel(), 512)
        self.bn_dense = nn.BatchNorm1d(512) if bn_after_dense else None
        self.linear2 = nn.Linear(dummy_out.numel(), num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, self.use_bn))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(out)
        if self.use_pooling:
            out = F.avg_pool2d(out, self.pool_kernel)
        out = out.view(out.size(0), -1)
        out = self.linear2(out)
        return out

def wide_resnet_cifar_bn(in_ch=3, in_dim=32):
    return Wide_ResNet(10, 4, None, 10, use_bn=True)

def wide_resnet_cifar_10(in_ch=3, in_dim=32):
    return Wide_ResNet(10, 4, 0.3, 10)

def wide_resnet_cifar(in_ch=3, in_dim=32):
    return Wide_ResNet(16, 4, 0.3, 10)

def wide_resnet_mnist_bn(in_ch=1, in_dim=28):
    return Wide_ResNet(10, 4, None, 10, use_bn=True, use_pooling=True, in_ch=1, in_dim=28)    

def wide_resnet_cifar_bn_wo_pooling(): # 1113M, 21M
    return Wide_ResNet(10, 4, None, 10, use_bn=True, use_pooling=False)

def wide_resnet_cifar_bn_wo_pooling_dropout(): # 1113M, 21M
    return Wide_ResNet(10, 4, 0.3, 10, use_bn=True, use_pooling=False)
