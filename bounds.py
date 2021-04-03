import torch
from auto_LiRPA import BoundDataParallel
from torch.nn import CrossEntropyLoss

def get_crown_loss(lb, labels):
    lb_padded = torch.cat([torch.zeros_like(lb[:, :1]), lb], dim=1)
    fake_labels = torch.zeros_like(labels)
    robust_ce_ = CrossEntropyLoss()(-lb_padded, fake_labels)
    return robust_ce_    

def get_bound_loss(args, model, loss_fusion, eps_scheduler, 
                    x=None, data=None, labels=None, exp_module=None):
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
            clb, cub = model.compute_bounds(IBP=False, C=c, method='backward', 
                bound_lower=bound_lower, bound_upper=bound_upper)
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
        robust_ce = get_crown_loss(lb, labels)
        return lb, robust_ce  