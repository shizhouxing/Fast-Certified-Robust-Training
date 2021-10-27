import torch
import torch.nn as nn
import math
import pdb
from auto_LiRPA.bound_ops import *
import numpy as np
from auto_LiRPA import BoundedModule, BoundDataParallel, BoundedTensor, CrossEntropyWrapper
from auto_LiRPA.perturbations import *
from utils import logger

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

def ibp_init(model_ori, model):
    weights, biases = get_params(model_ori)
    for i in range(len(weights)-1):
        if weights[i][1].ndim == 1:
            continue
        weight = weights[i][1]
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2 * math.pi / (fan_in**2))     
        std_before = weight.std().item()
        torch.nn.init.normal_(weight, mean=0, std=std)
        print(f'Reinitialize {weights[i][0]}, std before {std_before:.5f}, std now {weight.std():.5f}')
    for node in model._modules.values():
        if isinstance(node, BoundConv) or isinstance(node, BoundLinear):
            if len(node.inputs[0].inputs) > 0 and isinstance(node.inputs[0].inputs[0], BoundAdd):
                print(f'Adjust weights for node {node.name} due to residual connection')
                node.inputs[1].param.data /= 2

def kaiming_init(model):
    for p in model.named_parameters():
        if p[0].find('.weight') != -1:
            if p[0].find('bn') != -1 or p[1].ndim == 1:
                continue
            torch.nn.init.kaiming_normal_(p[1].data)

def orthogonal_init(model):
    params = []
    bns = []
    for p in model_ori.named_parameters():
        if p[0].find('.weight') != -1:
            if p[0].find('bn') != -1 or p[1].ndim == 1:
                bns.append(p)
            else:
                params.append(p)
    for p in params[:-1]: 
        std_before = p[1].std().item()
        print('before mean abs', p[1].abs().mean())
        torch.nn.init.orthogonal_(p[1])
        print('Reinitialize {} with orthogonal matrix, std before {:.5f}, std now {:.5f}'.format(
            p[0], std_before, p[1].std()))
        print('after mean abs', p[1].abs().mean())

def manual_init(args, model_ori, model, train_data, mode=1):
    if args.init_method == 'ibp':
        ibp_init(model_ori, model)
    elif args.init_method == 'orthogonal': 
        orthogonal_init(model_ori)
    elif args.init_method == 'kaiming':
        logger.info('Initialization: Kaiming normal')
        kaiming_init(model_ori)   
    elif args.init_method == 'identity':
        raise NotImplementedError         
    else:
        raise ValueError(args.init_method)
