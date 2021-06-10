import random
import os
import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA.bound_ops import BoundExp, BoundRelu
from auto_LiRPA.utils import logger
from models import *

ce_loss = nn.CrossEntropyLoss()

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
    state_dict = {}
    for name in state_dict_loss:
        assert (name.startswith('model.'))
        state_dict[name[6:]] = state_dict_loss[name]
    model.load_state_dict(state_dict)

def update_meter(meter, regular_ce, robust_ce, regular_err, robust_err, batch_size):
    meter.update('CE', regular_ce, batch_size)
    if robust_ce is not None:
        meter.update('Robust_CE', robust_ce, batch_size)
    if regular_err is not None:
        meter.update('Err', regular_err, batch_size)
    if robust_err is not None:
        meter.update('Rob_Err', robust_err, batch_size)
        
def update_log_writer(args, writer, meter, epoch, train, robust):
    if train:
        writer.add_scalar('loss/train', meter.avg("CE"), epoch)
        writer.add_scalar('err/train', meter.avg("Err"), epoch)
        if robust:
            writer.add_scalar('loss/robust_train', meter.avg("Robust_CE"), epoch)
            writer.add_scalar('err/robust_train', meter.avg("Rob_Err"), epoch)
    else:
        writer.add_scalar('loss/test', meter.avg("CE"), epoch)
        writer.add_scalar('err/test', meter.avg("Err"), epoch)
        if robust:
            writer.add_scalar('loss/robust_test', meter.avg("Robust_CE"), epoch)
            writer.add_scalar('err/robust_test', meter.avg("Rob_Err"), epoch)   
            writer.add_scalar('eps', meter.avg('eps'), epoch)

def update_log_reg(writer, meter, epoch, train, model):
    set = 'train' if train else 'test'
    writer.add_scalar('loss/pre_{}'.format(set), meter.avg("loss_reg"), epoch)

    if not train:
        for item in ['std', 'relu', 'tightness']:
            key = 'L_{}'.format(item)
            if key in meter.lasts:
                writer.add_scalar('loss/{}'.format(key), meter.avg(key), epoch)

def prepare_model(args, logger, config):
    model = args.model
    if config['data'] == 'MNIST':
        model_ori = eval(model)(in_ch=1, in_dim=28)
    elif config['data'] == 'CIFAR':
        model_ori = eval(model)(in_ch=3, in_dim=32)
    elif config['data'] == 'tinyimagenet':
        model_ori = eval(model)(in_ch=3, in_dim=64)
    else:
        raise NotImplementedError

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
        logger.info('Checkpoint loaded: {}'.format(args.load))
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