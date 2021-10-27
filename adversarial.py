import torch
import torch.nn as nn
import numpy as np
from utils import *
from auto_LiRPA.bound_ops import *
from datasets import cifar10_mean, cifar10_std
from torch.autograd import Variable
import pdb
import random

criterion = ce_loss

def init_delta(X, epsilon, data_min, data_max):
    delta = torch.zeros_like(X)
    for i in range(len(epsilon)):
        delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
    delta.data = clamp(delta, data_min - X, data_max - X)
    return delta

def fgsm(args, model_bound, model, epoch, epoch_progress, 
        X, y, data_min, data_max, epsilon, alpha, train, meter):
    delta = init_delta(X, epsilon, data_min, data_max)
    if args.bn_eval:
        model.eval()    
    with torch.enable_grad():
        delta.requires_grad = True
        output = model(X + delta[:X.size(0)])
        loss = F.cross_entropy(output, y)
        grad = torch.autograd.grad(loss, delta)[0].detach()
        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
        delta.data[:X.size(0)] = clamp(delta[:X.size(0)], data_min - X, data_max - X)
        delta = delta.detach()
        output = model(X + delta[:X.size(0)])
        loss = criterion(output, y)
    delta = delta.detach()
    if train:
        model.train()    
    output = model(X + delta)
    loss = criterion(output, y)   
    return loss, output

def get_relus(model_bound, x):
    model_bound(x)
    relus = []
    for node in model_bound._modules.values():
        if isinstance(node, BoundRelu):
            fv = node.inputs[0].fv
            status = fv > 0
            relus.append((fv, status))
    return relus    

def pgd(args, model_bound, model, epoch, epoch_progress,
        X, y, data_min, data_max, epsilon, alpha, attack_iters, train, meter):
    delta = init_delta(X, epsilon, data_min, data_max)
    if args.bn_eval:
        model.eval()
    with torch.enable_grad():
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            loss = criterion(output, y)
            grad = torch.autograd.grad(loss, delta)[0].detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, data_min - X, data_max - X)

    delta = delta.detach()
    if train:
        model.train()
    output = model(X + delta)
    loss = criterion(output, y)  

    if args.pgd_relu:
        loss_relu, unsafe_relu, delta_relu = pgd_relu(
            args, model_bound, model, X, y, data_min, data_max, epsilon, 
            alpha, attack_iters, train, meter=meter, adv_out=output)
        meter.update('loss_relu', loss_relu)
        loss += loss_relu * args.pgd_relu_w
    
    return loss, output  

def trades(args, model_bound, model, epoch, epoch_progress, 
        X, y, data_min, data_max, epsilon, alpha, attack_iters, train, meter):
    beta = 6.0
    batch_size = len(X)
    delta = torch.zeros_like(X)
    for i in range(len(epsilon)):
        delta[:, i, :, :].uniform_(-0.001/cifar10_std[i], 0.001/cifar10_std[i])
    delta.data = clamp(delta, data_min - X, data_max - X)
    x_adv = X.detach() + delta.detach()
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()

    relus_0 = get_relus(model_bound, X)

    def get_relu_loss(x_adv, relus_0):
        loss = 0
        relus = get_relus(model_bound, x_adv)
        unsafe_count = 0
        for i in range(len(relus)):
            changed = relus[i][1] != relus_0[i][1]
            loss += (changed.int() * (relus_0[i][0] - relus[i][0])**2 )\
                .view(x_adv.size(0), -1).sum(dim=-1).mean()
            unsafe_count += changed.int().sum()
        unsafe_count = float(unsafe_count) / x_adv.size(0)
        return loss, unsafe_count
        
    with torch.enable_grad():
        for _ in range(attack_iters):
            x_adv.requires_grad_(True)
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                F.softmax(model(X), dim=1))
            loss = loss_kl
            if args.trades_relu:
                loss_relu, unsafe_count = get_relu_loss(x_adv, relus_0)
                loss += loss_relu * batch_size
            grad = torch.autograd.grad(loss, [x_adv])[0]
            with torch.no_grad():
                x_adv = x_adv.detach() + alpha * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
                x_adv = torch.min(torch.max(x_adv, data_min), data_max)
    if train:
        model.train()
    x_adv.requires_grad_(False) # x_adv = Variable(x_adv, requires_grad=False)
    output = logits = model(X)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
        F.softmax(model(X), dim=1))
    loss = loss_natural + beta * loss_robust
    if args.trades_relu:
        loss_relu, unsafe_count = get_relu_loss(x_adv, relus_0)
        relu_w = args.trades_relu_w
        if args.trades_relu_scheduler > 0 and epoch <= args.trades_relu_scheduler:
            w_per_epoch = args.trades_relu_w / args.trades_relu_scheduler
            relu_w = w_per_epoch * (epoch - 1 + batch)
        loss_relu *= relu_w
        loss += loss_relu
        meter.update('loss_relu', loss_relu)
        meter.update('unsafe_count', unsafe_count)

    if args.pgd_relu:
        loss_relu, unsafe_relu, delta_relu = pgd_relu(
            args, model_bound, model, X, y, data_min, data_max, epsilon, 
            alpha, attack_iters, train, meter=meter, adv_out=output)
        meter.update('loss_relu', loss_relu)

        pgd_relu_w = args.pgd_relu_w
        if args.pgd_relu_scheduler > 0 and epoch <= args.pgd_relu_scheduler:
            w_per_epoch = args.pgd_relu_w / args.pgd_relu_scheduler
            pgd_relu_w = w_per_epoch * (epoch - 1 + epoch_progress)
        loss_relu *= pgd_relu_w
        loss += loss_relu

    return loss, output

def adv(args, model_bound, model, epoch, epoch_progress, 
        X, y, eps, data_max, data_min, std, meter, train=False):
    std = std.view(1, -1, 1, 1)
    epsilon = eps / std
    alpha = args.alpha / std
    # FIXME hardcoded
    pgd_alpha = args.pgd_alpha / std if train else args.test_pgd_alpha / std
    attack_iters = args.attack_iters if train else args.attack_iters_eval
    method = args.method
    
    if not train or method == 'pgd':
        loss, output = pgd(args, model_bound, model, epoch, epoch_progress, 
            X, y, data_min, data_max, epsilon, pgd_alpha, attack_iters, train, meter)
    elif method == 'trades':
        loss, output = trades(args, model_bound, model, epoch, epoch_progress,
            X, y, data_min, data_max, epsilon, pgd_alpha, attack_iters, train, meter)
    elif method == 'fgsm':
        loss, output = fgsm(args, model_bound, model, epoch, epoch_progress,
            X, y, data_min, data_max, epsilon, alpha, train, meter)
    else:
        raise ValueError(args.method)

    acc = (torch.argmax(output, dim=1) == y).float().mean()
    robust_ce = loss
    robust_err = 1 - acc

    if not train:
        output = model(X)
        regular_ce = ce_loss(output, y)
        regular_err = 1 - (torch.argmax(output, dim=1) == y).float().mean()     
    else:
        regular_ce = regular_err = torch.zeros_like(loss)   

    return regular_ce, robust_ce, regular_err, robust_err
