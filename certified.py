import torch
import torch.nn as nn
from utils import *
from auto_LiRPA import BoundedTensor, BoundDataParallel
from auto_LiRPA.perturbations import *
from auto_LiRPA.bound_ops import *
import pdb
import math

# eps is normalized by max_eps
def get_crown_loss(args, lb, eps=None):
    lb_padded = torch.cat([torch.zeros_like(lb[:, :1]), lb], dim=1)
    fake_labels = torch.zeros(lb.size(0), dtype=torch.long, device=lb.device)
    if args.ls > 0:
        threshold = 1 - eps * args.ls
        prob = nn.Softmax(dim=-1)(-lb_padded)[:, 0]
        robust_loss_ = (-torch.log(prob[:]) * (prob < threshold).float()).mean()
        return robust_loss_
    robust_loss_ = ce_loss(-lb_padded, fake_labels)
    return robust_loss_    

def get_C(args, data, labels):
    return get_spec_matrix(data, labels, args.num_class)

def get_bound_loss(args, model, loss_fusion, eps_scheduler, 
                    x=None, data=None, labels=None, eps=None, 
                    meter=None, train=False):
    if loss_fusion:
        c, bound_lower, bound_upper = None, False, True
    else:
        c, bound_lower, bound_upper = get_C(args, data, labels), True, False
    if args.bound_type == 'IBP':
        # FIXME remove `x=x`???
        lb, ub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None,
                        no_replicas=True) 
    elif args.bound_type == 'CROWN-IBP':
        factor = (eps_scheduler.get_max_eps() - eps_scheduler.get_eps()) / eps_scheduler.get_max_eps()
        ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
        if factor < 1e-5:
            lb, ub = ilb, iub
        else:
            clb, cub = model.compute_bounds(IBP=False, C=c, method='backward', 
                bound_lower=bound_lower, bound_upper=bound_upper)
            # clb, cub, A_dict = model.compute_bounds(IBP=False, C=c, method='backward', 
            #     bound_lower=bound_lower, bound_upper=bound_upper, return_A=True)
            if loss_fusion:
                ub = cub * factor + iub * (1 - factor)
            else:
                lb = clb * factor + ilb * (1 - factor)
    else:
        raise ValueError
    update_relu_stat(model, meter)
    if loss_fusion:
        if isinstance(model, BoundDataParallel):
            raise NotImplementedError
        return None, torch.mean(torch.log(ub) + get_exp_module(model).max_input)
    else:
        # Pad zero at the beginning for each example, and use fake label '0' for all examples
        robust_loss = get_crown_loss(args, lb, 
            eps=eps_scheduler.get_eps()/eps_scheduler.get_max_eps())
        return lb, robust_loss  

def cert(args, model, model_ori, epoch, epoch_progress, data, labels, eps, data_max, data_min, std, 
        robust=False, reg=False, loss_fusion=False, eps_scheduler=None, 
        train=False, meter=None):
    if not robust and reg:
        eps = max(eps, args.min_eps_reg)
    if type(eps) == float:
        eps = (eps / std).view(1,-1,1,1)
    else: # [batch_size, channels]
        eps = (eps.view(*eps.shape, 1, 1) / std.view(1, -1, 1, 1))

    data_ub = torch.min(data + eps, data_max)
    data_lb = torch.max(data - eps, data_min) 
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=data_lb, x_U=data_ub)
    x = BoundedTensor(data, ptb)

    if loss_fusion:
        x = (x, labels)
        output = model(*x)
        regular_ce = torch.mean(torch.log(output) + get_exp_module(model).max_input)
        regular_err = None
    else:
        output = model(x)
        regular_ce = ce_loss(output, labels)  # regular CrossEntropyLoss used for warming up
        regular_err = torch.sum(torch.argmax(output, dim=1) != labels).item() / x.size(0)
        x = (x, )

    if robust or reg or args.xiao_reg or args.vol_reg:
        b_res, robust_loss = get_bound_loss(args, model, loss_fusion, eps_scheduler, 
            x=(x if loss_fusion else None), data=data, labels=labels, 
            eps=eps, meter=meter, train=train)
        robust_err = torch.sum((b_res < 0).any(dim=1)).item() / data.size(0) if not loss_fusion else None
    else:
        robust_loss = robust_err = None

    if robust_loss is not None and torch.isnan(robust_loss):
        robust_err = 100.

    return regular_ce, robust_loss, regular_err, robust_err 