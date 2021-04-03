import torch
import math

def manual_init(model_ori, mode=1):
    if mode == 1:
        params = []
        bns = []
        for p in model_ori.named_parameters():
            if p[0].find('.weight') != -1:
                if p[0].find('bn') != -1 or p[1].ndim == 1:
                    bns.append(p)
                else:
                    params.append(p)

        for p in params[:-1]:
            weight = p[1]
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            std = math.sqrt(2 * math.pi / (fan_in**2))     
            std_before = p[1].std().item()
            torch.nn.init.normal_(p[1], mean=0, std=std)
            print('Reinitialize {}, std before {:.5f}, std now {:.5f}'.format(
                p[0], std_before, p[1].std()))

    # Orthogonal initialization
    elif mode == 3:
        params = []
        bns = []
        for p in model_ori.named_parameters():
            if p[0].find('.weight') != -1:
                if p[0].find('bn') != -1 or p[1].ndim == 1:
                    bns.append(p)
                else:
                    params.append(p)

        for p in params[:-1]:
            # weight = p[1]
            # fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
            # std = math.sqrt(2 * math.pi / (fan_in**2))     
            std_before = p[1].std().item()
            print('before mean abs', p[1].abs().mean())
            # torch.nn.init.normal_(p[1], mean=0, std=std)
            torch.nn.init.orthogonal_(p[1])
            print('Reinitialize {} with orthogonal matrix, std before {:.5f}, std now {:.5f}'.format(
                p[0], std_before, p[1].std()))
            print('after mean abs', p[1].abs().mean())
    
    else:
        raise ValueError(mode)


def kaiming_init(model):
    for p in model.named_parameters():
        if p[0].find('.weight') != -1:
            if p[0].find('bn') != -1 or p[1].ndim == 1:
                continue
            torch.nn.init.kaiming_normal_(p[1].data)
