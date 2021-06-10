import time
import json
import argparse
import pdb
import logging
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundDataParallel, BoundedTensor, CrossEntropyWrapper
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.bound_ops import *
from auto_LiRPA.eps_scheduler import *
from config import load_config
from datasets import load_data
from utils import *
from manual_init import manual_init, kaiming_init
from parser import parse_args
from certified import cert
from adversarial import adv
from regularization import compute_reg, compute_stab_reg, compute_L1_reg

args = parse_args()

writer = SummaryWriter(os.path.join(args.dir, 'log'), flush_secs=10)
file_handler = logging.FileHandler(os.path.join(args.dir, 'train.log'))
file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
logger.addHandler(file_handler) 
logger.info('Arguments: {}'.format(args))

def Train(model, model_ori, t, loader, eps_scheduler, opt, bound_config, loss_fusion=False, valid=False):
    train = opt is not None
    meter = MultiAverageMeter()
    meter_layer = []

    exp_module = get_exp_module(model)

    data_max, data_min, std = loader.data_max, loader.data_min, loader.std
    if args.device == 'cuda':
        data_min, data_max, std = data_min.cuda(), data_max.cuda(), std.cuda()

    if train:
        model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
    else:
        model.eval()
        eps_scheduler.eval()        

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps()     

        if train:
            eps *= args.train_eps_mul
        if eps < args.min_eps:
            eps = args.min_eps
        if args.natural_scheduling or args.fix_eps:
            eps = eps_scheduler.get_max_eps()
        if args.natural:
            eps = 0.

        reg = t <= args.num_reg_epochs

        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = 'natural' if (eps < 1e-50) else 'robust'
        robust = batch_method == 'robust'

        labels = labels.to(torch.long)
        if args.device == 'cuda':
            data, labels = data.cuda().detach().requires_grad_(), labels.cuda()

        data_batch, labels_batch = data, labels
        grad_acc = args.grad_acc_steps
        assert data.shape[0] % grad_acc == 0
        bsz = data.shape[0] // grad_acc

        for k in range(grad_acc):
            if grad_acc > 1:
                data, labels = data_batch[bsz*k:bsz*(k+1)], labels_batch[bsz*k:bsz*(k+1)]

            if args.mode == 'cert':
                regular_ce, robust_ce, regular_err, robust_err = cert(
                    args, model, data, labels, eps=eps, 
                    data_max=data_max, data_min=data_min, std=std, robust=robust, reg=reg,
                    loss_fusion=loss_fusion, exp_module=exp_module, eps_scheduler=eps_scheduler, 
                    train=train, meter=meter)
            elif args.mode == 'adv':
                method = args.method if train else 'pgd'
                regular_ce, robust_ce, regular_err, robust_err = adv(
                    args, model, data, labels, eps=eps, 
                    data_max=data_max, data_min=data_min, std=std, train=train)
            update_meter(meter, regular_ce, robust_ce, regular_err, robust_err, data.size(0))

            if reg:
                loss = compute_reg(args, model, meter, eps, eps_scheduler)
            elif args.xiao_reg:
                loss = compute_stab_reg(args, model, meter, eps, eps_scheduler) + compute_L1_reg(args, model_ori, meter, eps, eps_scheduler)
            else:
                loss = torch.tensor(0.).to(args.device)
            loss += robust_ce if robust else regular_ce
            meter.update('Loss', loss.item(), data.size(0)) 

            if train:
                loss /= grad_acc
                loss.backward()

        if train:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            meter.update('grad_norm', grad_norm)

            model.named_parameters()

            opt.step() 
            opt.zero_grad()
                
        meter.update('wnorm', get_weight_norm(model))
        meter.update('Time' , time.time() - start)

        if (i + 1) % args.log_interval == 0 and (train or args.eval):
            logger.info('[{:2d}:{:4d}/{:4d}]: eps={:.8f} {}'.format(t, i + 1, len(loader), eps, meter))
    logger.info('[{:2d}]: eps={:.8f} {}'.format(t, eps, meter))

    if batch_method != 'natural':
        meter.update('eps', eps_scheduler.get_eps())
    
    if t <= args.num_reg_epochs:
        update_log_reg(writer, meter, t, train, model)
    update_log_writer(args, writer, meter, t, train, robust=(batch_method != 'natural'))
       
    return meter

def main(args):
    config = load_config(args.config)
    logger.info('config: {}'.format(json.dumps(config)))
    set_seed(args.seed or config['seed'])
    model_ori, checkpoint, epoch, best = prepare_model(args, logger, config)

    # params = list(model_ori.named_parameters())
    # pdb.set_trace()
    
    bound_config = config['bound_params']

    batch_size = (args.batch_size or config['batch_size'])
    test_batch_size = args.test_batch_size or batch_size
    dummy_input, train_data, test_data = load_data(
        args, config['data'], batch_size, test_batch_size, aug=not args.no_data_aug)        

    bound_opts = bound_config['bound_opts']
    bound_opts_lf = copy.deepcopy(bound_opts)
    bound_opts_lf['loss_fusion'] = True
    bound_config_test = copy.deepcopy(bound_config)
    if args.bound_type == 'CROWN':
        if args.loss_fusion: 
            bound_config_test['bound_opts']['relu'] = 'zero-lb'    
    logger.info('bound_config_test {}'.format(bound_config_test))

    model_ori.train()
    if args.mode == 'cert':
        model = BoundedModule(model_ori, dummy_input, bound_opts=bound_config_test['bound_opts'], device=args.device)
    else:
        model = model_ori.cuda()

    if checkpoint is None:
        if args.manual_init:
            manual_init(args, model_ori, model, train_data)
        if args.kaiming_init:
            kaiming_init(model_ori)     

    lf = args.loss_fusion and args.bound_type == 'CROWN-IBP'
    if lf:
        model_loss = BoundedModule(
            CrossEntropyWrapper(model_ori), 
            (dummy_input.cuda(), torch.zeros(1, dtype=torch.long).cuda()), 
            bound_opts=bound_opts_lf, device=args.device)
    else:
        model_loss = model

    if args.opt == 'SGD':
        opt = optim.SGD(
            model_loss.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        opt = eval('optim.' + args.opt)(
            model_loss.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if checkpoint and not args.verify:
        if 'optimizer' not in checkpoint:
            logger.error('Cannot find optimzier checkpoint')
        else:
            opt.load_state_dict(checkpoint['optimizer'])

    max_eps = args.eps or bound_config['eps']
    eps_scheduler = eval(args.scheduler_name)(max_eps, args.scheduler_opts)
    epoch_length = int((len(train_data.dataset) + train_data.batch_size - 1) / train_data.batch_size)
    eps_scheduler.set_epoch_length(epoch_length)
    logger.info('Model structure: \n {}'.format(str(model_ori)))

    if args.verify:
        eps_scheduler = FixedScheduler(max_eps)
        logger.info('Test without loss fusion')
        meter = Train(model, model_ori, 10000, test_data, eps_scheduler, None, loss_fusion=False,
            bound_config=bound_config_test)
        logger.info(meter)
        return

    lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, 
        milestones=map(int, args.lr_decay_milestones.split(',')), gamma=args.lr_decay_factor)
    opt.param_groups[0]['lr'] = args.lr

    if epoch > 0 and not args.plot:
        # skip epochs
        eps_scheduler.train()
        for i in range(epoch):
            lr_scheduler.step()
            eps_scheduler.step_epoch(verbose=False)

    if args.plot:
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
                    bound_config=bound_config_test)
            meters.append(meter)
            print()
        torch.save(meters, os.path.join(args.dir, 'plot'))
    elif args.eval:
        with torch.no_grad():
            meter = Train(model, model_ori, 1000, test_data, eps_scheduler, None, loss_fusion=False,
                bound_config=bound_config_test)
        print(meter)
    else:
        timer = 0.0
        for t in range(epoch + 1, args.num_epochs + 1):
            logger.info('Epoch {}, learning rate {}, dir {}'.format(
                t, lr_scheduler.get_last_lr(), args.dir))

            start_time = time.time()
            if lf:
                Train(model_loss, model_ori, t, train_data, eps_scheduler, opt, loss_fusion=True, bound_config=bound_config)
                update_state_dict(model, model_loss)
            else:
                Train(model, model_ori, t, train_data, eps_scheduler, opt, bound_config=bound_config)
            epoch_time = time.time() - start_time
            timer += epoch_time
            lr_scheduler.step()
            logger.info('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))

            is_best = False
            if t % args.test_interval == 0:
                logger.info('Test without loss fusion')
                with torch.no_grad():
                    meter = Train(model, model_ori, t, test_data, eps_scheduler, None, loss_fusion=False,
                        bound_config=bound_config_test)

                if eps_scheduler.get_eps() == eps_scheduler.get_max_eps():
                    if meter.avg('Rob_Err') < best[1]:
                        is_best, best = True, (meter.avg('Err'), meter.avg('Rob_Err'), t)
                    logger.info('Best epoch {}, error {:.4f}, verified error {:.4f}'.format(
                        best[-1], best[0], best[1]))

            save(args, epoch=t, best=best, model=model, opt=opt, is_best=is_best)

if __name__ == '__main__':
    main(args)
