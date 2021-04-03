import random
import os
import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA.bound_ops import BoundExp, BoundRelu
from models import *
from manual_init import manual_init, kaiming_init

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_weight_norm(model):
    # Skip param_mean and param_std
    return torch.norm(torch.stack([
        torch.norm(p[1].detach()) for p in model.named_parameters() if not p[0].startswith('param')]))

def get_exp_module(bounded_module):
    for node in bounded_module._modules.values():
        # Find the Exp neuron in computational graph
        if isinstance(node, BoundExp):
            return node
    return None

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
        meter.update('Verified_Err', robust_err, batch_size)
        
def update_log_writer(writer, meter, epoch, train, robust):
    if train:
        writer.add_scalar('loss/train', meter.avg("CE"), epoch)
        writer.add_scalar('err/train', meter.avg("Err"), epoch)
        if robust:
            writer.add_scalar('loss/robust_train', meter.avg("Robust_CE"), epoch)
            writer.add_scalar('err/robust_train', meter.avg("Verified_Err"), epoch)
    else:
        writer.add_scalar('loss/test', meter.avg("CE"), epoch)
        writer.add_scalar('err/test', meter.avg("Err"), epoch)
        if robust:
            writer.add_scalar('loss/robust_test', meter.avg("Robust_CE"), epoch)
            writer.add_scalar('err/robust_test', meter.avg("Verified_Err"), epoch)   
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
    model = args.model or config['model']
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
        if args.manual_init:
            manual_init(model_ori, args.manual_init_mode)
        if args.kaiming_init:
            kaiming_init(model_ori)

    return model_ori, checkpoint, epoch, best