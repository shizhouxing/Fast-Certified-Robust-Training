import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Flatten


class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1, bn=True):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(group_width) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(group_width) if bn else nn.Identity()
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width) if bn else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*group_width) if bn else nn.Identity()
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, 
            num_classes=10, in_ch=3, in_dim=32, bn=True):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.bn = bn

        self.conv1 = nn.Conv2d(in_ch, self.in_planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_planes) if bn else nn.Identity()
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)

        self.linear1 = nn.Linear(self.in_planes * ((in_dim // 4)**2), 512)
        self.bn_dense = nn.BatchNorm1d(512) if bn else nn.Identity()
        self.linear2 = nn.Linear(512, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride, bn=self.bn))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= Block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = Flatten()(out)
        out = F.relu(self.bn_dense(self.linear1(out)))
        out = self.linear2(out)
        return out

def ResNeXt_2_32(in_ch=3, in_dim=32, bn=True):
    return ResNeXt(num_blocks=[1,1,1], cardinality=2, bottleneck_width=32, 
        in_ch=in_ch, in_dim=in_dim, bn=bn)

def ResNeXt_4_20(in_ch=3, in_dim=32):
    return ResNeXt(num_blocks=[1,1,1], cardinality=4, bottleneck_width=20, in_ch=in_ch, in_dim=in_dim)

def resnext(in_ch=3, in_dim=32):
    return ResNeXt_2_32(in_ch, in_dim)

def resnext_no_bn(in_ch=3, in_dim=32):
    return ResNeXt_2_32(in_ch, in_dim, bn=False)    
