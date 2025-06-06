import random
import os
import pdb
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from auto_LiRPA.bound_ops import BoundExp, BoundRelu
from auto_LiRPA.utils import logger
from auto_LiRPA.eps_scheduler import *
from models import *
import wandb

ce_loss = nn.CrossEntropyLoss()

def set_file_handler(logger, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    file_handler = logging.FileHandler(os.path.join(dir, 'train.log'))
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
    logger.addHandler(file_handler) 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_weight_norm(model):
    # Skip param_mean and param_std
    return torch.norm(torch.stack([
        torch.norm(p[1].detach()) for p in model.named_parameters() if 'weight' in p[0]]))

def get_exp_module(bounded_module):
    for node in bounded_module._modules.values():
        # Find the Exp neuron in computational graph
        if isinstance(node, BoundExp):
            return node
    return None

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)  

"""In loss fusion, update the state_dict of `model` from the loss fusion version 
`model_loss`. This is necessary when BatchNorm is involved."""
def update_state_dict(model, model_loss):
    state_dict_loss = model_loss.state_dict()
    state_dict = model.state_dict()
    keys = model.state_dict().keys()
    for name in state_dict_loss:
        v = state_dict_loss[name]
    #     for prefix in ['model.', '/w.', '/b.', '/running_mean.']:
    #         if name.startswith(prefix):
    #             name = name[len(prefix):]
    #             break
    #     if not name in keys:
    #         raise KeyError(name)
        name = model_loss.node_name_map[name]
        state_dict[name] = v
    model.load_state_dict(state_dict)

def update_meter(meter, regular_ce, robust_loss, regular_err, robust_err, batch_size):
    meter.update('CE', regular_ce, batch_size)
    if robust_loss is not None:
        meter.update('Rob_Loss', robust_loss, batch_size)
    if regular_err is not None:
        meter.update('Err', regular_err, batch_size)
    if robust_err is not None:
        meter.update('Rob_Err', robust_err, batch_size)
        
def update_log_writer(args, meter, epoch, train, robust):
    prefix = 'train' if train else 'test'
    metrics = {
        f'loss/{prefix}': meter.avg("CE"),
        f'err/{prefix}': meter.avg("Err"),
        f'active/{prefix}': meter.avg("active"),
        f'inactive/{prefix}': meter.avg("inactive"),
        f'loss_reg/{prefix}': meter.avg("loss_reg"),
        f'L_tightness/{prefix}': meter.avg("L_tightness"),
        f'L_relu/{prefix}': meter.avg("L_relu"),
        f'L_std/{prefix}': meter.avg("L_std"),
        f'wnorm/{prefix}': meter.avg("wnorm"),
        f'time/{prefix}': meter.avg("Time")
    }
    
    if robust:
        metrics.update({
            f'loss/robust_{prefix}': meter.avg("Rob_Loss"),
            f'err/robust_{prefix}': meter.avg("Rob_Err")
        })
    
    # Log to wandb
    wandb.log(metrics, step=epoch)
    
    if not train and robust:
        eps_metric = {'eps': meter.avg('eps')}
        wandb.log(eps_metric, step=epoch)

def update_log_reg(meter, epoch, train, model):
    set = 'train' if train else 'test'
    metrics = {
        f'loss/pre_{set}': meter.avg("loss_reg")
    }
    
    if not train:
        for item in ['std', 'relu', 'tightness']:
            key = f'L_{item}'
            if key in meter.lasts:
                metrics[f'loss/{key}'] = meter.avg(key)
    
    # Log to wandb
    wandb.log(metrics, step=epoch)

def parse_opts(s):
    opts = s.split(',')
    params = {}
    for o in opts:
        if o.strip():
            key, val = o.split('=')
            try:
                v = eval(val)
            except:
                v = val
            if type(v) not in [int, float, bool]:
                v = val
            params[key] = v
    return params

def prepare_model(args, logger, config):
    model = args.model

    if config['data'] == 'MNIST':
        input_shape = (1, 28, 28)
    elif config['data'] == 'CIFAR':
        input_shape = (3, 32, 32)
    elif config['data'] == 'tinyimagenet':
        input_shape = (3, 64, 64)
    else:
        raise NotImplementedError(config['data'])

    model_ori = eval(model)(in_ch=input_shape[0], in_dim=input_shape[1], **parse_opts(args.model_params))
        
    checkpoint = None
    if args.auto_load:
        path_last = os.path.join(args.dir, 'ckpt_last')
        if os.path.exists(path_last):
            args.load = path_last
            logger.info('Use last checkpoint {}'.format(path_last))
        else:
            latest = -1
            for filename in os.listdir(args.dir):
                if filename.startswith('ckpt_'):
                    latest = max(latest, int(filename[5:]))
            if latest != -1:
                args.load = os.path.join(args.dir, 'ckpt_{}'.format(latest))
                try:
                    checkpoint = torch.load(args.load)
                except:
                    logger.warning('Cannot load {}'.format(args.load))    
                    args.load = os.path.join(args.dir, 'ckpt_{}'.format(latest-1))
                    logger.warning('Trying {}'.format(args.load))
    if checkpoint is None and args.load:
        checkpoint = torch.load(args.load)
    if checkpoint is not None:
        epoch, state_dict = checkpoint['epoch'], checkpoint['state_dict']
        best = checkpoint.get('best', (100., 100., -1))
        model_ori.load_state_dict(state_dict, strict=False)
        logger.info(f'Checkpoint loaded: {args.load}, epoch {epoch}')
    else:
        epoch = 0
        best = (100., 100., -1)

    return model_ori, checkpoint, epoch, best

def save(args, epoch, best, model, opt, is_best=False):
    ckpt = {
        'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 
        'epoch': epoch, 'best': best
    }
    path_last = os.path.join(args.dir, 'ckpt_last')
    if os.path.exists(path_last):
        os.system('mv {path} {path}.bak'.format(path=path_last))
    torch.save(ckpt, path_last)
    if is_best:
        path_best = os.path.join(args.dir, 'ckpt_best')
        if os.path.exists(path_best):
            os.system('mv {path} {path}.bak'.format(path=path_best))                
        torch.save(ckpt, path_best)
    if args.save_all:
        torch.save(ckpt, os.path.join(args.dir, 'ckpt_{}'.format(epoch)))   
    logger.info('')

def get_eps_scheduler(args, max_eps, train_data):
    eps_scheduler = eval(args.scheduler_name)(max_eps, args.scheduler_opts)
    epoch_length = int((len(train_data.dataset) + train_data.batch_size - 1) / train_data.batch_size)
    eps_scheduler.set_epoch_length(epoch_length)
    return eps_scheduler

def get_lr_scheduler(args, opt):
    for pg in opt.param_groups: 
        pg['lr'] = args.lr
    return optim.lr_scheduler.MultiStepLR(opt, 
        milestones=map(int, args.lr_decay_milestones.split(',')), gamma=args.lr_decay_factor)        

def get_optimizer(args, params, checkpoint=None):
    if args.opt == 'SGD':
        opt = optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        opt = eval('optim.' + args.opt)(params, lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f'Optimizer {opt}')   
    if checkpoint:
        if 'optimizer' not in checkpoint:
            logger.error('Cannot find optimzier checkpoint')
        else:
            opt.load_state_dict(checkpoint['optimizer'])
    return opt

def get_bound_opts_lf(bound_opts):
    bound_opts = copy.deepcopy(bound_opts)
    bound_opts['loss_fusion'] = True
    return bound_opts    

def update_relu_stat(model, meter):
    for node in model._modules.values():
        if isinstance(node, BoundRelu):
            l, u = node.inputs[0].lower, node.inputs[0].upper
            meter.update('active', (l>0).float().sum()/l.numel())
            meter.update('inactive', (u<0).float().sum()/l.numel())


# def map_to_binary_labels(labels):
#     for i in range(labels.size(0)):
#         if labels[i] in [0, 1, 8, 9]
#             labels[i] = 0
#         else:
#             labels[i] = 1

#     for i, l in enumerate(labels):
#         if l in [0, 1, 8, 9]:
#             labels[i] = 0
#         else:
#             labels[i] = 1
#     return labels
