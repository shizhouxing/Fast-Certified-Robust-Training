import torch
import torch.nn as nn
from utils import ce_loss
from auto_LiRPA import BoundedTensor, BoundDataParallel
from auto_LiRPA.perturbations import *
import pdb

# eps is normalized by max_eps
def get_crown_loss(args, lb, labels, eps):
    lb_padded = torch.cat([torch.zeros_like(lb[:, :1]), lb], dim=1)
    fake_labels = torch.zeros_like(labels)
    if args.ls > 0:
        threshold = 1 - eps * args.ls
        prob = nn.Softmax(dim=-1)(-lb_padded)[:, 0]
        robust_ce_ = (-torch.log(prob[:]) * (prob < threshold).float()).mean()
        return robust_ce_
    robust_ce_ = ce_loss(-lb_padded, fake_labels)
    return robust_ce_    

def get_bound_loss(args, model, loss_fusion, eps_scheduler, 
                    x=None, data=None, labels=None, eps=None, 
                    exp_module=None, meter=None, train=False):
    if loss_fusion:
        bound_lower, bound_upper = False, True
        c = None
    else:
        bound_lower, bound_upper = True, False
        c = torch.eye(args.num_class).type_as(data)[labels].unsqueeze(1) - torch.eye(args.num_class).type_as(data).unsqueeze(0)
        I = (~(labels.data.unsqueeze(1) == torch.arange(args.num_class).type_as(labels.data).unsqueeze(0)))
        c = (c[I].view(data.size(0), args.num_class - 1, args.num_class))  
        
    if args.bound_type == 'IBP':
        lb, ub = model(method_opt="compute_bounds", x=x, IBP=True, C=c, method=None,
                        no_replicas=True)        
    elif args.bound_type == 'CROWN-IBP':
        factor = (eps_scheduler.get_max_eps() - eps_scheduler.get_eps()) / eps_scheduler.get_max_eps()
        ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
        if factor < 1e-5:
            lb, ub = ilb, iub
        else:
            clb, cub, A_dict = model.compute_bounds(IBP=False, C=c, method='backward', 
                bound_lower=bound_lower, bound_upper=bound_upper, return_A=True)
            if train:
                Ax0 = (A_dict['/77']['/input.1'][1][0]*x[0]).sum(dim=[1,2,3]).unsqueeze(-1)
                meter.update('A_impact', ( (cub - (A_dict['/77']['bias'][1]+Ax0)) / cub).mean()) 
            if loss_fusion:
                ub = cub * factor + iub * (1 - factor)
            else:
                lb = clb * factor + ilb * (1 - factor)
    elif args.bound_type == 'CROWN':
        ilb, iub = model.compute_bounds(IBP=True, C=c, method=None)
        factor = 1.0 
        clb, cub = model.compute_bounds(IBP=False, C=c, method='backward', 
            bound_lower=bound_lower, bound_upper=bound_upper)
        if loss_fusion:
            ub = cub
        else:
            lb = clb
    else:
        raise ValueError

    if loss_fusion:
        if isinstance(model, BoundDataParallel):
            raise NotImplementedError
        return None, torch.mean(torch.log(ub) + exp_module.max_input)
    else:
        # Pad zero at the beginning for each example, and use fake label '0' for all examples
        robust_ce = get_crown_loss(args, lb, labels, 
            eps=eps_scheduler.get_eps()/eps_scheduler.get_max_eps())
        return lb, robust_ce  

def cert(args, model, data, labels, eps, data_max, data_min, std, 
        robust=False, reg=False, loss_fusion=False, exp_module=None, eps_scheduler=None, 
        train=False, meter=None):
    if not robust:
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
        if False and robust and train:
            regular_ce = regular_err = torch.tensor(0., device=args.device)
        else:
            output = model(*x)
            regular_ce = torch.mean(torch.log(output) + exp_module.max_input)
            regular_err = None
    else:
        output = model(x)
        regular_ce = ce_loss(output, labels)  # regular CrossEntropyLoss used for warming up
        regular_err = torch.sum(torch.argmax(output, dim=1) != labels).item() / x.size(0)
        x = (x, )

    if robust or reg or args.xiao_reg:
        b_res, robust_ce = get_bound_loss(args, model, loss_fusion, eps_scheduler, 
            x=(x if loss_fusion else None), data=data, labels=labels, 
            eps=eps, exp_module=exp_module, meter=meter, train=train)
        if loss_fusion:
            robust_err = None
        else:
            robust_err = torch.sum((b_res < 0).any(dim=1)).item() / data.size(0)
    else:
        robust_ce = robust_err = None

    return regular_ce, robust_ce, regular_err, robust_err 