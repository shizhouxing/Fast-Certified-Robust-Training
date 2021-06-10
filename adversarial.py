import torch
import torch.nn as nn
from utils import *
from auto_LiRPA.bound_ops import *
import pdb
import random

criterion = ce_loss

def fgsm(args, model, X, y, data_min, data_max, epsilon, alpha, train):
    delta = torch.zeros_like(X).cuda()   
    for j in range(len(epsilon)):
        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
    delta.data = clamp(delta, data_min - X, data_max - X)
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
    print(torch.max( ((X+delta) - data_min).view(X.shape[0], -1), dim=-1).values)
    print(torch.max( (data_max-(X+delta)).view(X.shape[0], -1), dim=-1).values)
    pdb.set_trace()
    loss = criterion(output, y)   
    return loss, output

def pgd(args, model, X, y, data_min, data_max, epsilon, alpha, attack_iters, train):
    delta = torch.zeros_like(X).cuda()   
    for i in range(len(epsilon)):
        delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
    delta.data = clamp(delta, data_min - X, data_max - X)
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
    return loss, output  

def trades(args, model, X, y, data_min, data_max, epsilon, alpha, train):
    beta = 6.0
    batch_size = len(X)
    for i in range(len(epsilon)):
        delta[:, i, :, :].uniform_(-0.001/cifar10_std[i], 0.001/cifar10_std[i])
    delta.data = clamp(delta, data_min - X, data_max - X)
    x_adv = X.detach() + delta.detach()
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    for _ in range(attack_iters):
        x_adv.requires_grad_()
        loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
            F.softmax(model(X), dim=1))
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        with torch.no_grad():
            x_adv = x_adv.detach() + pgd_alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
            x_adv = torch.min(torch.max(x_adv, data_min), data_max)
    if train:
        model.train()
    x_adv = Variable(x_adv, requires_grad=False)
    output = logits = model(X)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
        F.softmax(model(X), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, ouptut


def adv(args, model, X, y, eps, data_max, data_min, std, train=False):
    std = std.view(1, -1, 1, 1)
    epsilon = eps / std
    alpha = args.alpha / std
    pgd_alpha = (2 / 255.) / std   

    attack_iters = args.attack_iters if train else args.attack_iters_eval
    method = args.method
    
    if method == 'pgd':
        loss, output = pgd(args, model, X, y, data_min, data_max, epsilon, pgd_alpha, attack_iters, train)
    elif method == 'trades':
        loss, output = trades(args, model, X, y, data_min, data_max, epsilon, pgd_alpha, train)
    elif method == 'fgsm':
        loss, output = fgsm(args, model, X, y, data_min, data_max, epsilon, alpha, train)
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
