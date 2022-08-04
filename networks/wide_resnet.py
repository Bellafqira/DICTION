import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as func
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, no_relu=False):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=True)

        self.no_relu = no_relu
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=(1, 1), stride=(stride, stride), bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(func.relu(self.bn1(x))))
        if self.no_relu:
            out = self.conv2(self.bn2(out))
        else:
            out = self.conv2(func.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResnet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResnet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        print('| Wide-Resnet %dx%d' % (depth, k))
        n_stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, n_stages[0])
        self.layer1 = self._wide_layer(WideBasic, n_stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, n_stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, n_stages[3], n, dropout_rate, stride=2, no_relu=True)
        self.bn1 = nn.BatchNorm2d(n_stages[3], momentum=0.9)
        self.linear = nn.Linear(n_stages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, no_relu=False):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides[:-1]:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        layers.append(block(self.in_planes, planes, dropout_rate, strides[-1], no_relu))
        self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn1(out)
        out = func.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = func.relu(out)
        out = self.linear(out)
        return out

def wide_resnet28():
    return WideResnet(28, 10, 0.3, 10)
