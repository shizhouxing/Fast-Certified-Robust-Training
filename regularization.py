import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from auto_LiRPA import BoundedModule, BoundDataParallel, BoundedTensor, CrossEntropyWrapper
from auto_LiRPA.bound_ops import *
from collections import namedtuple

Node = namedtuple('Node', 'node lower upper')

def compute_stab_reg(args, model, meter, eps, eps_scheduler):
    loss = torch.zeros(()).to(args.device)
    if isinstance(model, BoundDataParallel):
        modules = list(model._modules.values())[0]._modules
    else:
        modules = model._modules

    nodes = {}
    for m in modules.values():
        if isinstance(m, BoundRelu):
            if isinstance(model, BoundDataParallel):
                lower = model(get_property=True, node_name=m.name, att_name='lower')
                upper = model(get_property=True, node_name=m.name, att_name='upper')
            else:
                lower, upper = m.lower, m.upper
            nodes[m.name] = Node(m, lower, upper)

    
    for k, v in nodes.items():
        loss += - torch.tanh(1 + v.lower * v.upper).view(v.lower.size(0), -1).sum(dim=-1).mean()

    meter.update('relu_stab_Loss', loss) 

    return loss * args.xiao_coeff


def compute_vol_reg(args, model, meter, eps, eps_scheduler):
    loss = torch.zeros(()).to(args.device)
    nodes = {}
    for m in model._modules.values():
        if isinstance(m, BoundRelu):
            l, u = m.inputs[0].lower, m.inputs[0].upper
            loss += ((-l).clamp(min=0) * u.clamp(min=0)).mean()
    meter.update('relu_vol_Loss', loss) 
    return loss * args.colt_coeff


def compute_L1_reg(args, model, meter, eps, eps_scheduler):
    loss = torch.zeros(()).to(args.device)
    for module in model._modules.values():
        if isinstance(module, nn.Linear):
            loss += torch.abs(module.weight).sum()
        elif isinstance(module, nn.Conv2d):
            loss += torch.abs(module.weight).sum()
    meter.update('L1_loss', loss) 
    return loss * args.l1_coeff


def compute_reg(args, model, meter, eps, eps_scheduler):
    loss = torch.zeros(()).to(args.device)

    # Handle the non-feedforward case
    l0 = torch.zeros_like(loss)
    loss_tightness, loss_std, loss_relu, loss_ratio = (l0.clone() for i in range(4))

    if isinstance(model, BoundDataParallel):
        modules = list(model._modules.values())[0]._modules
    else:
        modules = model._modules
    node_inp = modules['/input.1']
    tightness_0 = ((node_inp.upper - node_inp.lower) / 2).mean()
    ratio_init = tightness_0 / ((node_inp.upper + node_inp.lower) / 2).std()
    cnt_layers = 0
    cnt = 0
    for m in model._modules.values():
        if isinstance(m, BoundRelu):
            lower, upper = m.inputs[0].lower, m.inputs[0].upper
            center = (upper + lower) / 2
            diff = ((upper - lower) / 2)
            tightness = diff.mean()
            mean_ = center.mean()
            std_ = center.std()            

            loss_tightness += F.relu(args.tol - tightness_0 / tightness.clamp(min=1e-12)) / args.tol
            loss_std += F.relu(args.tol - std_) / args.tol
            cnt += 1

            # L_{relu}
            mask_act, mask_inact = lower>0, upper<0
            mean_act = (center * mask_act).mean()
            mean_inact = (center * mask_inact).mean()
            delta = (center - mean_)**2
            var_act = (delta * mask_act).sum()# / center.numel()
            var_inact = (delta * mask_inact).sum()# / center.numel()                        

            mean_ratio = mean_act / -mean_inact
            var_ratio = var_act / var_inact
            mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
            var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
            loss_relu_ = ((
                F.relu(args.tol - mean_ratio) + F.relu(args.tol - var_ratio)) 
                / args.tol)       
            if not torch.isnan(loss_relu_) and not torch.isinf(loss_relu_):
                loss_relu += loss_relu_ 

            if args.debug:
                bn_mean = (lower.mean() + upper.mean()) / 2
                bn_var = ((upper**2 + lower**2) / 2).mean() - bn_mean**2
                print(m.name, m, 
                    'tightness {:.4f} gain {:.4f} std {:.4f}'.format(
                        tightness.item(), (tightness/tightness_0).item(), std_.item()),
                    'input', m.inputs[0], m.inputs[0].name,
                    'active {:.4f} inactive {:.4f}'.format(
                        (lower>0).float().sum()/lower.numel(),
                        (upper<0).float().sum()/lower.numel()),
                    'bnv2_mean {:.5f} bnv2_var {:.5f}'.format(bn_mean.item(), bn_var.item())
                )
                # pre-bn
                lower, upper = m.inputs[0].inputs[0].lower, m.inputs[0].inputs[0].upper
                bn_mean = (lower.mean() + upper.mean()) / 2
                bn_var = ((upper**2 + lower**2) / 2).mean() - bn_mean**2
                print('pre-bn',
                    'bnv2_mean {:.5f} bnv2_var {:.5f}'.format(bn_mean.item(), bn_var.item()))

    loss_tightness /= cnt
    loss_std /= cnt
    loss_relu /= cnt

    if args.debug:
        pdb.set_trace()

    for item in ['tightness', 'relu', 'std']:
        loss_ = eval('loss_{}'.format(item))
        if item in args.reg_obj:
            loss += loss_
        meter.update('L_{}'.format(item), loss_)                

    meter.update('loss_reg', loss)

    if args.no_reg_dec:
        intensity = args.reg_lambda
    else:
        intensity = args.reg_lambda * (1 - eps_scheduler.get_eps() / eps_scheduler.get_max_eps())
    loss *= intensity

    return loss