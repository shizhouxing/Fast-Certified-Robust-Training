import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from .utils import Flatten

def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, downsample, residual=True, bn=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(
        self, block, layers, num_classes=1000,
        zero_init_residual=False, dense=False, residual=True,
        widen_factor=1, base_width=16, pool=True,
        in_ch=3, in_dim=32, bn=True
    ):
        super(ResNet, self).__init__()

        self.residual = residual

        self.inplanes = base_width
        self.conv1 = conv3x3(in_ch, self.inplanes)
        self.bn = bn
        self.bn1 = nn.BatchNorm2d(self.inplanes) if bn else nn.Identity()
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
                self.dense_bn = nn.BatchNorm1d(dim_dense) if bn else nn.Identity()
                self.fc = nn.Linear(dim_dense, num_classes)    
            else:
                self.dense = None
                self.fc = nn.Linear(self.inplanes, num_classes)

        else:
            assert dense
            dim_dense = 512
            self.dense = nn.Linear(self.inplanes*((in_dim//4)**2), dim_dense)
            self.dense_bn = nn.BatchNorm1d(dim_dense) if bn else nn.Identity()
            self.fc = nn.Linear(dim_dense, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        if blocks == 0:
            return nn.Sequential()

        downsample = None
        if self.residual and (stride != 1 or self.inplanes != planes):
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes) if self.bn else nn.Identity(),
            )

        layers = [block(self.inplanes, planes, stride, downsample, residual=self.residual, bn=self.bn)]
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

        x = Flatten()(x)

        if self.dense:
            x = F.relu(self.dense_bn(self.dense(x)))

        x = self.fc(x)

        return x


def wide_resnet_8_pool(in_ch=3, in_dim=32):
    model = ResNet(BasicBlock, [1,1,1], num_classes=10, widen_factor=8, pool=True, dense=False)
    return model    

def wide_resnet_8(in_ch=3, in_dim=32):
    model = ResNet(BasicBlock, [1,1,1], num_classes=10, widen_factor=8, 
        dense=True, pool=False, in_ch=in_ch, in_dim=in_dim)
    return model   

def wide_resnet_12(in_ch=3, in_dim=32):
    model = ResNet(BasicBlock, [1,1,1], num_classes=10, widen_factor=12, dense=True, pool=False)
    return model      

def wide_resnet_8_dense_no_pool(in_ch=3, in_dim=32):
    model = ResNet(BasicBlock, [1,1,1], num_classes=10, widen_factor=8, dense=True, pool=False)
    return model   

def wide_resnet_12_dense_no_pool(in_ch=3, in_dim=32):
    model = ResNet(BasicBlock, [1,1,1], num_classes=10, widen_factor=12, dense=True, pool=False)
    return model     

def wide_resnet(in_ch=3, in_dim=32):
    return wide_resnet_8(in_ch, in_dim)

def wide_resnet_no_bn(in_ch=3, in_dim=32):
    model = ResNet(BasicBlock, [1,1,1], num_classes=10, widen_factor=8, 
        dense=True, pool=False, in_ch=in_ch, in_dim=in_dim, bn=False)
    return model

def count_params(model):
    cnt = 0
    for p in model.parameters():
        cnt += p.numel()
    return cnt

if __name__ == '__main__':
    dummy_in = torch.randn(1,3,32,32)

    model = wide_resnet_8_dense_no_pool()
    print(model)
    print('wideresnet', count_params(model)/1e6)

    from .wide_resnet import wide_resnet_cifar_bn
    model = wide_resnet_cifar_bn()
    print(model)
    print('wideresnet kaidi', count_params(model)/1e6)
