from .imagenet import ImageNet
from .samplers import *
from torch import distributed as dist

ds = {
    "imagenet": ImageNet
}


def get_dataset(cfg, train_transform=None, val_transform=None):
    dataset_name = cfg['DATASET']['NAME']
    assert dataset_name in ds.keys(), f"Unavailable dataset name >> {dataset_name}.\nList of available datasets: {list(ds.keys())}"
    trainset = ds[dataset_name](cfg['DATASET']['ROOT'], split='train', transform=train_transform)
    valset = ds[dataset_name](cfg['DATASET']['ROOT'], split='val', transform=val_transform)
    return trainset, valset


def get_sampler(cfg, train_dataset, val_dataset):
    if not cfg['TRAIN']['DDP']['ENABLE']:
        train_sampler = RandomSampler(train_dataset)
    else:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        if cfg['TRAIN']['DDP']['REPEATED_AUG']:
            train_sampler = RASampler(train_dataset, num_tasks, global_rank, shuffle=True)
        else:
            train_sampler = DistributedSampler(train_dataset, num_tasks, global_rank, shuffle=True)
    val_sampler = SequentialSampler(val_dataset)

    return train_sampler, val_sampler