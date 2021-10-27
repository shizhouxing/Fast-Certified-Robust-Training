import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import sys
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, downsample, residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(identity)
            x += identity

        return self.relu(x)


class Wide_ResNet(nn.Module):
    def __init__(
        self, block, layers, num_classes=1000,
        zero_init_residual=False, dense=False, residual=True,
        widen_factor=1, base_width=16, pool=True,
        in_ch=3, in_dim=32
    ):
        super(Wide_ResNet, self).__init__()
        self.residual = residual

        self.inplanes = base_width
        self.conv1 = conv3x3(in_ch, self.inplanes)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, base_width * widen_factor, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2 * widen_factor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4 * widen_factor, layers[2], stride=2)

        self.pool = pool
        if pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            if dense:
                dim_dense = 512
                self.dense = nn.Linear(self.inplanes, dim_dense)
                self.dense_bn = nn.BatchNorm1d(dim_dense)
                self.fc = nn.Linear(dim_dense, num_classes)    
            else:
                self.dense = None
                self.fc = nn.Linear(self.inplanes, num_classes)

        else:
            assert dense
            dim_dense = 512
            self.dense = nn.Linear(self.inplanes*((in_dim//4)**2), dim_dense)
            self.dense_bn = nn.BatchNorm1d(dim_dense)
            self.fc = nn.Linear(dim_dense, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        if blocks == 0:
            return nn.Sequential()

        downsample = None
        if self.residual and (stride != 1 or self.inplanes != planes):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, residual=self.residual)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes, residual=self.residual))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)        
        if self.pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dense:
            x = F.relu(self.dense_bn(self.dense(x)))
        return self.fc(x)

def wide_resnet_bn_imagenet64(in_ch=3, in_dim=56):
    model = Wide_ResNet(BasicBlock, [1, 1, 1], num_classes=200, widen_factor=10, base_width=16, pool=False, dense=True, in_ch=in_ch, in_dim=in_dim)
    return model

def count_params(model):
    cnt = 0
    for p in model.parameters():
        cnt += p.numel()
    return cnt

if __name__ == '__main__':
    model = wide_resnet_bn_imagenet64(in_ch=3, in_dim=64)
    dummy_in = torch.zeros(2, 3, 64, 64)
    print(model(dummy_in).shape)
    print(model)
    print('wideresnet', count_params(model)/1e6)
    exit(0)