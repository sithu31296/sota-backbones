import os
import torch.distributed as dist
from torch.utils.data import SequentialSampler, DistributedSampler, RandomSampler
from .imagenet import ImageNet
from torchvision import datasets, transforms as T


def get_sampler(ddp, train_dataset, val_dataset):
    if not ddp:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    return train_sampler, SequentialSampler(val_dataset)


def get_dataset(dataset_name, root, split, transform, num_classes):
    assert split in ['train', 'val']
    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root, split, transform)
        num_classes = 10
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root, split, transform)
        num_classes = 100
    elif dataset_name == 'imagenet':
        data_path = os.path.join(root, split)
        dataset = datasets.ImageFolder(data_path, transform)
        num_classes = 1000
    else:
        data_path = os.path.join(root, split)
        dataset = datasets.ImageFolder(data_path, transform)
        num_classes = num_classes
    return dataset, num_classes