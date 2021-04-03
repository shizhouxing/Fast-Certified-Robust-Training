import time
import json
import argparse
import pdb
import logging
import copy
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from auto_LiRPA import BoundedModule, BoundDataParallel, BoundedTensor, CrossEntropyWrapper
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.bound_ops import *
from auto_LiRPA.eps_scheduler import *
from config import load_config
from datasets import load_data
from utils import (get_exp_module, update_log_writer, update_log_reg, 
                get_weight_norm, set_seed, update_state_dict, update_meter, prepare_model)
from parser import parse_args
from bounds import get_bound_loss
from regularization import compute_reg, compute_stab_reg, compute_L1_reg

args = parse_args()

if args.debug:
    args.num_reg_epochs += 1

writer = SummaryWriter(os.path.join(args.dir, 'log'), flush_secs=10)
file_handler = logging.FileHandler(os.path.join(args.dir, 'train.log'))
file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
logger.addHandler(file_handler) 
logger.info('Arguments: {}'.format(args))

def Train(model, model_ori, t, loader, eps_scheduler, opt, train_config, bound_config, loss_fusion=False, valid=False):
    train = opt is not None
    method = train_config['method']
    meter = MultiAverageMeter()
    meter_layer = []

    exp_module = get_exp_module(model)
    norm = float(bound_config['norm'])

    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))
        eps_scheduler.step_epoch()
    else:
        model.eval()
        eps_scheduler.eval()

    data_max, data_min, std = loader.data_max, loader.data_min, loader.std
    if args.device == 'cuda':
        data_min, data_max, std = data_min.cuda(), data_max.cuda(), std.cuda()

    def compute(model, data, labels, eps, robust=False, reg=False):
        if not robust:
            eps = max(eps, args.min_eps_reg)

        data_ub = torch.min(data + (eps / std).view(1,-1,1,1), data_max)
        data_lb = torch.max(data - (eps / std).view(1,-1,1,1), data_min) 

        ptb = PerturbationLpNorm(
            norm=norm, eps=(eps / std).view(1,-1,1,1), x_L=data_lb, x_U=data_ub)
        x = BoundedTensor(data, ptb)

        if loss_fusion:
            x = (x, labels)
            if robust and train and bound_config['bound_opts'].get('bn') == 'ibp':
                regular_ce = regular_err = torch.tensor(0., device=args.device)
                regular = False
            else:
                output = model(*x)
                regular_ce = torch.mean(torch.log(output) + exp_module.max_input)
                regular_err = None
                regular = True
        else:
            if robust and train and bound_config['bound_opts'].get('bn') == 'ibp':
                regular_ce = regular_err = torch.tensor(0., device=args.device)     
                regular = False
            else:
                output = model(x)
                regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
                regular_err = torch.sum(torch.argmax(output, dim=1) != labels).item() / x.size(0)
                regular = True
            x = (x, )

        if robust or reg or args.xiao_reg:
            b_res, robust_ce = get_bound_loss(args, model, loss_fusion, eps_scheduler, 
                x=(x if not regular else None), data=data, labels=labels, exp_module=exp_module)
            if loss_fusion:
                robust_err = None
            else:
                robust_err = torch.sum((b_res < 0).any(dim=1)).item() / data.size(0)
        else:
            robust_ce = robust_err = None

        return regular_ce, robust_ce, regular_err, robust_err            

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()     

        if train:
            eps *= args.train_eps_mul

        reg = t <= args.num_reg_epochs

        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = 'natural' if (eps < 1e-50) else method
        robust = batch_method == 'robust'

        grad_acc = args.grad_acc_steps

        labels = labels.to(torch.long)
        if args.device == 'cuda':
            data, labels = data.cuda().detach().requires_grad_(), labels.cuda()

        data_batch, labels_batch = data, labels
        assert data.shape[0] % grad_acc == 0
        bsz = data.shape[0] // grad_acc
        last_batch_idx = 0

        for k in range(grad_acc):
            if grad_acc > 1:
                data = data_batch[bsz*k:bsz*(k+1)]
                labels = labels_batch[bsz*k:bsz*(k+1)]

            regular_ce, robust_ce, regular_err, robust_err = compute(
                model, data, labels, eps=eps, robust=robust, reg=reg)
            update_meter(meter, regular_ce, robust_ce, regular_err, robust_err, data.size(0))

            if reg:
                loss = compute_reg(args, model, meter, eps, eps_scheduler)
            elif args.xiao_reg:
                loss = compute_stab_reg(args, model, meter, eps, eps_scheduler) + compute_L1_reg(args, model_ori, meter, eps, eps_scheduler)
            else:
                loss = torch.tensor(0.).to(args.device)
            loss += (robust_ce if robust else regular_ce)

            if args.stat:
                cnt = 0
                for m in model._modules.values():
                    if isinstance(m, BoundRelu):
                        cnt += 1
                        inactive = (m.inputs[0].upper<0).float().sum() / m.upper.numel()
                        active = (m.inputs[0].lower>0).float().sum() / m.upper.numel()
                        meter.update('inactive_{}'.format(cnt), inactive)
                        meter.update('active_{}'.format(cnt), active)

            meter.update('Loss', loss.item(), data.size(0)) 
            if train:
                loss /= grad_acc
                loss.backward()

        if train:
            if args.clip_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                meter.update('grad_norm', grad_norm)
            opt.step() 
            opt.zero_grad()
                
        meter.update('wnorm', get_weight_norm(model))
        meter.update('Time' , time.time() - start)

        if (i + 1) % args.log_interval == 0 and train:
            logger.info('[{:2d}:{:4d}/{:4d}]: eps={:.8f} {}'.format(t, i + 1, len(loader), eps, meter))
    logger.info('[{:2d}]: eps={:.8f} {}'.format(t, eps, meter))

    if batch_method != 'natural':
        meter.update('eps', eps_scheduler.get_eps())
    
    if t <= args.num_reg_epochs:
        update_log_reg(writer, meter, t, train, model)
    update_log_writer(writer, meter, t, train, robust=(batch_method != 'natural'))
       
    return meter


def main(args):
    config = load_config(args.config)
    logger.info('config: {}'.format(json.dumps(config)))
    set_seed(args.seed or config['seed'])
    model_ori, checkpoint, epoch, best = prepare_model(args, logger, config)

    train_config = config['training_params']
    bound_config = config['bound_params']

    batch_size = (args.batch_size or train_config['batch_size'])
    test_batch_size = args.test_batch_size or batch_size
    dummy_input, train_data, test_data = load_data(
        args, config['data'], batch_size, test_batch_size, aug=not args.no_data_aug)        

    bound_opts = bound_config['bound_opts']
    bound_opts['ibp_relative'] = False #args.relative
    bound_opts_lf = copy.deepcopy(bound_opts)
    bound_opts_lf['loss_fusion'] = True
    bound_config_test = copy.deepcopy(bound_config)
    if args.bound_type == 'CROWN':
        if args.loss_fusion: 
            bound_config_test['bound_opts']['relu'] = 'zero-lb'    
    logger.info('bound_config_test {}'.format(bound_config_test))

    model_ori.train()
    model = BoundedModule(model_ori, dummy_input, bound_opts=bound_config_test['bound_opts'], device=args.device)
    model_loss = BoundedModule(
        CrossEntropyWrapper(model_ori), 
        (dummy_input.cuda(), torch.zeros(1, dtype=torch.long).cuda()), 
        bound_opts=bound_opts_lf, device=args.device)
    if args.multi_gpu:
        model = BoundDataParallel(model)
        model_loss = BoundDataParallel(model_loss)

    lr = args.lr or train_config['lr']        
    opt = optim.Adam(model_loss.parameters(), lr=lr, weight_decay=args.weight_decay)
    if checkpoint and not args.verify:
        if 'optimizer' not in checkpoint:
            logger.warning('Cannot find optimzier checkpoint')
        else:
            opt.load_state_dict(checkpoint['optimizer'])
    loss_fusion = args.loss_fusion

    scheduler_opts = args.scheduler_opts or train_config['scheduler_opts']
    max_eps = args.eps or bound_config['eps']
    scheduler_name = args.scheduler_name or train_config['scheduler_name']
    eps_scheduler = eval(scheduler_name)(max_eps, scheduler_opts)
    logger.info('Model structure: \n {}'.format(str(model_ori)))

    if args.verify:
        eps_scheduler = FixedScheduler(max_eps)
        logger.info('Test without loss fusion')
        meter = Train(model, model_ori, 10000, test_data, eps_scheduler, None, loss_fusion=False,
            train_config=train_config, bound_config=bound_config_test)
        logger.info(meter)
        return

    lr_decay_milestones = map(int, (args.lr_decay_milestones or train_config['lr_decay_milestones']).split(','))   
    lr_decay_factor = args.lr_decay_factor or train_config['lr_decay_factor']
    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_decay_milestones, gamma=lr_decay_factor)
    opt.param_groups[0]['lr'] = lr

    if epoch > 0 and not args.plot:
        assert(train_config['scheduler_name'] != 'AdaptiveScheduler')
        # skip epochs
        epoch_length = int((len(train_data.dataset) + train_data.batch_size - 1) / train_data.batch_size)
        eps_scheduler.set_epoch_length(epoch_length)
        eps_scheduler.train()
        for i in range(epoch):
            # FIXME Can use `last_epoch` argument of lr_scheduler
            lr_scheduler.step()
            eps_scheduler.step_epoch(verbose=False)

    if args.stat:
        pass
    elif args.plot:
        eps_scheduler.set_epoch_length(100)
        meters = []
        for t in range(1, args.num_epochs + 1):
            print('Testing on epoch {}'.format(t))
            eps_scheduler.train()
            eps_scheduler.step_epoch(verbose=False)
            # load checkpoint
            checkpoint = torch.load(os.path.join(args.dir, 'ckpt_{}'.format(t)))
            epoch, state_dict = checkpoint['epoch'], checkpoint['state_dict']
            model_ori.load_state_dict(state_dict, strict=False)
            model = BoundedModule(model_ori, dummy_input.cuda(), bound_opts=bound_config_test['bound_opts'], device=args.device)

            with torch.no_grad():
                meter = Train(model, model_ori, t, test_data, eps_scheduler, None, loss_fusion=False,
                    train_config=train_config, bound_config=bound_config_test)
            meters.append(meter)
            print()
        torch.save(meters, os.path.join(args.dir, 'plot'))
    else:
        timer = 0.0
        num_epochs = args.num_epochs or train_config['num_epochs']
        for t in range(epoch + 1, num_epochs + 1):
            logger.info('Epoch {}, learning rate {}, dir {}'.format(
                t, lr_scheduler.get_last_lr(), args.dir))

            lf = loss_fusion and args.bound_type == 'CROWN-IBP'

            start_time = time.time()
            if lf:
                Train(model_loss, model_ori, t, train_data, eps_scheduler, opt, loss_fusion=True,
                    train_config=train_config, bound_config=bound_config)
                update_state_dict(model, model_loss)
            else:
                Train(model, model_ori, t, train_data, eps_scheduler, opt, 
                    train_config=train_config, bound_config=bound_config)

            epoch_time = time.time() - start_time
            timer += epoch_time

            lr_scheduler.step()
            logger.info('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))

            if t % args.test_interval == 0:
                logger.info('Test without loss fusion')
                with torch.no_grad():
                    meter = Train(model, model_ori, t, test_data, eps_scheduler, None, loss_fusion=False,
                        train_config=train_config, bound_config=bound_config_test)

                if eps_scheduler.get_eps() == eps_scheduler.get_max_eps():
                    if meter.avg('Verified_Err') < best[1]:
                        best = (meter.avg('Err'), meter.avg('Verified_Err'), t)
                    logger.info('Best epoch {}, error {:.4f}, verified error {:.4f}'.format(
                        best[-1], best[0], best[1]))

            torch.save({
                'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 
                'epoch': t, 'best': best
            }, os.path.join(args.dir, 'ckpt_{}'.format(t)))   
            logger.info('')

            if eps_scheduler.get_eps() == eps_scheduler.get_max_eps():
                if meter.avg('Verified_Err') < best[1]:
                    best = (meter.avg('Err'), meter.avg('Verified_Err'), t)
                logger.info('Best epoch {}, error {:.4f}, verified error {:.4f}'.format(
                    best[-1], best[0], best[1]))

if __name__ == '__main__':
    main(args)
