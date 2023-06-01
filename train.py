import time
import json
import pdb
from torch.utils.tensorboard import SummaryWriter
from auto_LiRPA import BoundedModule, CrossEntropyWrapper
from auto_LiRPA.perturbations import *
from auto_LiRPA.utils import MultiAverageMeter
from auto_LiRPA.bound_ops import *
from config import load_config
from datasets import load_data
from utils import *
from manual_init import manual_init, kaiming_init
from argparser import parse_args
from certified import cert
from regularization import compute_reg, compute_stab_reg, compute_vol_reg, compute_L1_reg

args = parse_args()

writer = SummaryWriter(os.path.join(args.dir, 'log'), flush_secs=10)
if not args.verify:
    set_file_handler(logger, args.dir)
logger.info('Arguments: {}'.format(args))

def Train(model, model_ori, t, loader, eps_scheduler, opt, loss_fusion=False, valid=False):
    train = opt is not None
    meter = MultiAverageMeter()
    meter_layer = []

    data_max, data_min, std = loader.data_max, loader.data_min, loader.std
    if args.device == 'cuda':
        data_min, data_max, std = data_min.cuda(), data_max.cuda(), std.cuda()

    if train:
        model_ori.train(); model.train(); eps_scheduler.train()
        eps_scheduler.step_epoch()
    else:
        model_ori.eval(); model.eval(); eps_scheduler.eval()        

    for i, (data, labels) in enumerate(loader):
        start = time.time()
        eps_scheduler.step_batch()
        eps = eps_scheduler.get_eps() 
        epoch_progress =  (i+1) * 1. / len(loader) if train else 1.0

        if train:
            eps *= args.train_eps_mul
        if eps < args.min_eps:
            eps = args.min_eps
        if args.fix_eps:
            eps = eps_scheduler.get_max_eps()
        if args.natural:
            eps = 0.

        reg = t <= args.num_reg_epochs

        # For small eps just use natural training, no need to compute LiRPA bounds
        batch_method = 'natural' if (eps < 1e-50) else 'robust'
        robust = batch_method == 'robust'

        # labels = labels.to(torch.long)
        if args.device == 'cuda':
            data, labels = data.cuda().detach().requires_grad_(), labels.cuda()

        data_batch, labels_batch = data, labels
        grad_acc = args.grad_acc_steps
        assert data.shape[0] % grad_acc == 0
        bsz = data.shape[0] // grad_acc

        for k in range(grad_acc):
            if grad_acc > 1:
                data, labels = data_batch[bsz*k:bsz*(k+1)], labels_batch[bsz*k:bsz*(k+1)]

            regular_ce, robust_loss, regular_err, robust_err = cert(
                args, model, model_ori, t, epoch_progress, data, labels, eps=eps, 
                data_max=data_max, data_min=data_min, std=std, robust=robust, reg=reg,
                loss_fusion=loss_fusion, eps_scheduler=eps_scheduler, 
                train=train, meter=meter)
            update_meter(meter, regular_ce, robust_loss, regular_err, robust_err, data.size(0))

            if reg:
                loss = compute_reg(args, model, meter, eps, eps_scheduler)
            elif args.xiao_reg:
                loss = compute_stab_reg(args, model, meter, eps, eps_scheduler) + compute_L1_reg(args, model_ori, meter, eps, eps_scheduler)
            elif args.vol_reg: # by colt
                loss = compute_vol_reg(args, model, meter, eps, eps_scheduler)
            else:
                loss = torch.tensor(0.).to(args.device)
            if robust:
                loss += robust_loss
            else:
                loss += regular_ce
            meter.update('Loss', loss.item(), data.size(0)) 

            if train:
                loss /= grad_acc
                loss.backward()

                if args.check_nan:
                    for p in model.parameters():
                        if torch.isnan(p.grad).any():
                            pdb.set_trace()
                            ckpt = { 'model_ori': model_ori, 'args_cert': (t, epoch_progress, data, labels, eps, data_max, data_min, std, robust, reg, loss_fusion, eps_scheduler, train, meter) } 
                            torch.save(ckpt, 'nan_ckpt')
                            pdb.set_trace()

        if train:
            grad_norm = torch.nn.utils.clip_grad_norm_(model_ori.parameters(), max_norm=args.grad_norm)
            meter.update('grad_norm', grad_norm)
            opt.step() 
            opt.zero_grad()
                
        meter.update('wnorm', get_weight_norm(model_ori))
        meter.update('Time' , time.time() - start)

        if (i + 1) % args.log_interval == 0 and (train or args.eval or args.verify):
            logger.info('[{:2d}:{:4d}/{:4d}]: eps={:.8f} {}'.format(t, i + 1, len(loader), eps, meter))
            if args.debug:
                print()
                pdb.set_trace()            
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
    logger.info('Model structure: \n {}'.format(str(model_ori)))

    custom_ops = {}
    bound_config = config['bound_params']
    batch_size = (args.batch_size or config['batch_size'])
    test_batch_size = args.test_batch_size or batch_size
    dummy_input, train_data, test_data = load_data(
        args, config['data'], batch_size, test_batch_size, aug=not args.no_data_aug)        
    lf = args.loss_fusion and args.bound_type == 'CROWN-IBP'
    bound_opts = bound_config['bound_opts']

    model_ori.train()
    model = BoundedModule(model_ori, dummy_input, bound_opts=bound_opts, custom_ops=custom_ops, device=args.device)
    model_ori.to(args.device)
     
    if checkpoint is None:
        if args.manual_init:
            manual_init(args, model_ori, model, train_data)
        if args.kaiming_init:
            kaiming_init(model_ori)     

    if lf:
        model_loss = BoundedModule(
            CrossEntropyWrapper(model_ori), 
            (dummy_input.cuda(), torch.zeros(1, dtype=torch.long).cuda()), 
            bound_opts=get_bound_opts_lf(bound_opts), device=args.device)
        params = list(model_loss.parameters())
    else:
        model_loss = model
        params = list(model_ori.parameters())
    logger.info('Parameter shapes: {}'.format([p.shape for p in params]))
    if args.multi_gpu:
        raise NotImplementedError('Multi-GPU is not supported yet')

    opt = get_optimizer(args, params, checkpoint)
    max_eps = args.eps or bound_config['eps']
    eps_scheduler = get_eps_scheduler(args, max_eps, train_data)
    lr_scheduler = get_lr_scheduler(args, opt)
    if epoch > 0 and not args.plot:
        # skip epochs
        eps_scheduler.train()
        for i in range(epoch):
            # FIXME Can use `last_epoch` argument of lr_scheduler
            lr_scheduler.step()
            eps_scheduler.step_epoch(verbose=False)    

    if args.verify:
        logger.info('Inference')
        meter = Train(model, model_ori, 10000, test_data, eps_scheduler, None, loss_fusion=False)
        logger.info(meter)
    else:
        timer = 0.0
        for t in range(epoch + 1, args.num_epochs + 1):
            logger.info('Epoch {}, learning rate {}, dir {}'.format(
                t, lr_scheduler.get_last_lr(), args.dir))
            start_time = time.time()
            if lf:
                Train(model_loss, model_ori, t, train_data, eps_scheduler, opt, loss_fusion=True)
            else:
                Train(model, model_ori, t, train_data, eps_scheduler, opt)
            update_state_dict(model_ori, model_loss)
            epoch_time = time.time() - start_time
            timer += epoch_time
            lr_scheduler.step()
            logger.info('Epoch time: {:.4f}, Total time: {:.4f}'.format(epoch_time, timer))
            is_best = False
            if t % args.test_interval == 0:
                logger.info('Test without loss fusion')
                with torch.no_grad():
                    meter = Train(model, model_ori, t, test_data, eps_scheduler, None, loss_fusion=False)
                if eps_scheduler.get_eps() == eps_scheduler.get_max_eps():
                    if meter.avg('Rob_Err') < best[1]:
                        is_best, best = True, (meter.avg('Err'), meter.avg('Rob_Err'), t)
                    logger.info('Best epoch {}, error {:.4f}, robust error {:.4f}'.format(
                        best[-1], best[0], best[1]))
            save(args, epoch=t, best=best, model=model_ori, opt=opt, is_best=is_best)

if __name__ == '__main__':
    main(args)
