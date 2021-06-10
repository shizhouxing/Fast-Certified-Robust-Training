import torch
import torch.nn as nn
import math
import pdb
from auto_LiRPA.bound_ops import *
import numpy as np
from auto_LiRPA import BoundedModule, BoundDataParallel, BoundedTensor, CrossEntropyWrapper
from auto_LiRPA.perturbations import *

def get_params(model):
    weights = []
    biases = []

    for p in model.named_parameters():
        if 'weight' in p[0]:
            weights.append(p)
        elif 'bias' in p[0]:
            biases.append(p)
        else:
            print('Skipping parameter {}'.format(p[0]))
        
    return weights, biases

def manual_init(args, model_ori, model, train_data, mode=1):
    mode =  args.manual_init_mode

    # Main
    if mode == 1:
        weights = []
        biases = []

        for p in model_ori.named_parameters():
            if 'weight' in p[0]:
                weights.append(p)
            elif 'bias' in p[0]:
                biases.append(p)
            else:
                raise ValueError

        for i in range(len(weights)-1):
            if weights[i][1].ndim == 1:
                continue
            weight = weights[i][1]
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            std = math.sqrt(2 * math.pi / (fan_in**2))     
            std_before = weight.std().item()
            torch.nn.init.normal_(weight, mean=0, std=std)
            print('Reinitialize {}, std before {:.5f}, std now {:.5f}'.format(
                weights[i][0], std_before, weight.std()))
    else:
        raise ValueError(mode)


def kaiming_init(model):
    for p in model.named_parameters():
        if p[0].find('.weight') != -1:
            if p[0].find('bn') != -1 or p[1].ndim == 1:
                continue
            torch.nn.init.kaiming_normal_(p[1].data)
