import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verify', action='store_true', help='verification mode, do not train')
    parser.add_argument('--load', type=str, default='', help='Load reged model')
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='use cpu or cuda')
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--num-class', type=int, default=10)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dir', type=str, default='model')
    parser.add_argument('--eps', type=float)
    parser.add_argument('--scheduler_opts', type=str, default='start=2,length=80')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--loss-fusion', action='store_true')
    parser.add_argument('--grad-acc-steps', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=160)
    parser.add_argument('--scheduler_name', type=str, default='SmoothedScheduler')
    parser.add_argument('--num-reg-epochs', type=int, default=0)
    parser.add_argument('--train-eps-mul', type=float, default=1.0)
    parser.add_argument('--lr-decay-factor', type=float, default=0.2)
    parser.add_argument('--lr-decay-milestones', type=str, default='120,140')
    parser.add_argument('--test-batch-size', type=int)
    parser.add_argument('--random-crop', type=int, default=2)
    parser.add_argument('--auto-load', action='store_true', help='Automatically load the latest checkpoint in the directory without specifying the checkpoint file')
    parser.add_argument('--method', type=str, default=None, 
                        choices=['vanilla', 'fast', 'crown-ibp', 'crown', 
                        'default', 'manual', 'reg', 'xiao_reg', 'pgd', 'fgsm', 'trades'])
    parser.add_argument('--test-interval', type=int, default=1)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--weight-decay', '--wd', type=float, default=0) 
    parser.add_argument('--no-reg-dec', action='store_true')
    parser.add_argument('--no-data-aug', action='store_true')
    parser.add_argument('--bound-type', type=str, default='IBP', choices=['IBP', 'CROWN-IBP'])
    parser.add_argument('--reg-concat', action='store_true')
    parser.add_argument('--init-for-bn', action='store_true')
    parser.add_argument('--add-natural', action='store_true')
    parser.add_argument('--natural-scheduling', action='store_true')
    parser.add_argument('--w-scale-cls', type=float, default=100, help='Weight scaling for the classification layer')
    parser.add_argument('--fix-eps', action='store_true', help='No epsilon scheduling')
    parser.add_argument('--no-fix-eps', action='store_true', help='No epsilon scheduling')
    parser.add_argument('--eps-aug', action='store_true')
    parser.add_argument('--eps-w-max', action='store_true')
    parser.add_argument('--natural', action='store_true', help='Natural training')
    parser.add_argument('--ls', type=float, default=0.0)

    parser.add_argument('--reg-obj', '--pre-obj', type=str, nargs='+', default=['tightness', 'relu'])
    parser.add_argument('--reg-lambda', '--pi', type=float, default=0.5)    
    parser.add_argument('--reg', action='store_true')
    parser.add_argument('--min-eps-reg', type=float, default=1e-6)
    parser.add_argument('--min-eps', type=float, default=0)
    parser.add_argument('--manual-init', action='store_true')
    parser.add_argument('--manual-init-mode', '--init-version', default=1, type=float)
    parser.add_argument('--tol', type=float, default=0.5)
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--length', type=int)
    parser.add_argument('--xiao_reg', action="store_true")
    parser.add_argument('--xiao_coeff', type=float, default=1e-5)
    parser.add_argument('--l1_coeff', type=float, default=1e-5)
    parser.add_argument('--kaiming_init', action='store_true')
    parser.add_argument('--no-loss-fusion', action='store_true')
    parser.add_argument('--init-eps', type=float)
    parser.add_argument('--no-init-class', action='store_true')
    parser.add_argument('--eps-add-max-debug', action='store_true')
    parser.add_argument('--probe-intermediate', action='store_true')
    parser.add_argument('--save-all', action='store_true', help='Save all the checkpoints')

    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--mode', type=str, default='cert', choices=['cert', 'adv', 'natural'])
    parser.add_argument('--alpha', type=float, default=10./255)
    parser.add_argument('--attack-iters', type=int, default=7, help='Attack iterations for pgd training')
    parser.add_argument('--attack-iters-eval', type=int, default=10, help='Attack iterations for pgd evaluation')
    parser.add_argument('--bn-eval', action='store_true', help='Use eval mode BN for adversarial attack')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--no-init', action='store_true', help='No initialization')

    args = parser.parse_args()

    # Rename for compatibility
    if args.method == 'default':
        args.method = 'vanilla'
    elif args.method == 'reg':
        args.method = 'fast'

    # Override arguments depending on `method`
    if args.method == 'reg-only' or args.method == 'fast' and args.no_init:
        args.reg = True
    elif args.method == 'fast':
        args.reg = args.manual_init = True
    elif args.method == 'manual':
        args.manual_init = True
    elif args.method == 'crown-ibp':
        args.bound_type = 'CROWN-IBP'
        # Enable loss fusion by default
        if not args.no_loss_fusion:
            args.loss_fusion = True
    elif args.method == 'xiao_reg':
        args.xiao_reg = True
    elif args.method == 'crown':
        args.bound_type = 'CROWN'
    elif args.method in ['pgd', 'fgsm', 'trades']:
        args.mode = 'adv'

    print('Regularizer:', args.reg)
    print('Manual initialization:', args.manual_init)

    if args.mode == 'adv' and not args.no_fix_eps:
        args.fix_eps = True
        args.scheduler_opts = 'start=1,length=0'
        args.scheduler_name = 'LinearScheduler'

    if args.length:
        if args.length == 20:
            args.scheduler_opts = 'start=2,length=20'
            args.lr_decay_milestones = '50,60'
            args.num_epochs = 70
        elif args.length == 80:
            args.scheduler_opts = 'start=2,length=80'
            args.lr_decay_milestones = '120,140'
            args.num_epochs = 160
        else:
            raise ValueError('Unknown length {}'.format(args.length))

    # Overridde the legacy num_reg_epochs 
    if args.reg:
        start, length = args.scheduler_opts.split(',')
        start = int(start.split('=')[1])
        length = int(length.split('=')[1])
        args.num_reg_epochs = start + length - 1

    return args