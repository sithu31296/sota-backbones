import torch.distributed as dist
from torch.utils.data import SequentialSampler, DistributedSampler, RandomSampler
from .imagenet import ImageNet


def get_sampler(ddp, train_dataset, val_dataset):
    if not ddp:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    return train_sampler, SequentialSampler(val_dataset)