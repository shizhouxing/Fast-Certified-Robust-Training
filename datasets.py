import multiprocessing
import torch
from torch.utils import data
from functools import partial
import torchvision.transforms as transforms
import torchvision.datasets as datasets

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2023, 0.1994, 0.2010]

"""
MNIST and CIFAR10 datasets with `index` also returned in `__getitem__`
"""

class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, use_index=False):
        super().__init__(root, train, transform, target_transform, download)
        self.use_index = use_index        

    def __getitem__(self, index):
        img, target = super().__getitem__(index)         
        if self.use_index:
            return img, target, index
        else:
            return img, target

class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, use_index=False):
        super().__init__(root, train, transform, target_transform, download) 
        self.use_index = use_index
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)         
        if self.use_index:
            return img, target, index
        else:
            return img, target

def load_data(args, data, batch_size, test_batch_size, use_index=False, aug=True):
    if data == 'MNIST':
        """Fix 403 Forbidden error in downloading MNIST
        See https://github.com/pytorch/vision/issues/1938."""
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)  
              
        dummy_input = torch.randn(2, 1, 28, 28)
        mean, std = torch.tensor([0.0]), torch.tensor([1.0])
        train_data = MNIST('./data', train=True, download=True, transform=transforms.ToTensor(), use_index=use_index)
        test_data = MNIST('./data', train=False, download=True, transform=transforms.ToTensor(), use_index=use_index)
    elif data == 'CIFAR':
        mean = torch.tensor(cifar10_mean)
        std = torch.tensor([0.2, 0.2, 0.2] if args.lip or args.global_lip or 'lip' in args.model else cifar10_std)
        dummy_input = torch.randn(2, 3, 32, 32)
        normalize = transforms.Normalize(mean = mean, std = std)
        if aug:
            transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 2, padding_mode='edge'),
                    transforms.ToTensor(),
                    normalize])
        else:
            # No random cropping
            transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

        train_data = CIFAR10('./data', train=True, download=True, 
            transform=transform, use_index=use_index)
        test_data = CIFAR10('./data', train=False, download=True, 
                transform=transform_test, use_index=use_index)
    elif data == "tinyimagenet":
        mean = torch.tensor([0.4802, 0.4481, 0.3975])
        std = torch.tensor([0.22, 0.22, 0.22] if args.lip else [0.2302, 0.2265, 0.2262])
        dummy_input = torch.randn(2, 3, 64, 64)
        normalize = transforms.Normalize(mean=mean, std=std)
        data_dir = 'data/tinyImageNet/tiny-imagenet-200'
        train_data = datasets.ImageFolder(data_dir + '/train',
                                        transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(64, 4, padding_mode='edge'),
                                            transforms.ToTensor(),
                                            normalize,
                                        ]))
        test_data = datasets.ImageFolder(data_dir + '/val',
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize]))

    train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, pin_memory=True, num_workers=0)
        
    train_data.mean = test_data.mean = mean
    train_data.std = test_data.std = std  

    for loader in [train_data, test_data]:
        loader.mean, loader.std = mean, std
        loader.data_max = data_max = torch.reshape((1. - mean) / std, (1, -1, 1, 1))
        loader.data_min = torch.reshape((0. - mean) / std, (1, -1, 1, 1))

    dummy_input = dummy_input.to(args.device)

    return dummy_input, train_data, test_data
