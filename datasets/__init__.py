from torchvision import transforms as T
from .imagenet import ImageNet


__all__ = {
    "imagenet": ImageNet
}


def get_dataset(cfg, split, transform=None):
    dataset_name = cfg['DATASET']['NAME']
    assert dataset_name in __all__.keys(), f"Unavailable dataset name >> {dataset_name}.\nList of available datasets: {list(__all__.keys())}"
    if split == 'val': 
        transform = T.Compose(
            T.Resize(tuple(map(lambda x: int(x / 0.9), cfg['EVAL']['IMAGE_SIZE']))),    # to main aspect ratio
            T.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    return __all__[dataset_name](cfg['DATASET']['ROOT'], split=split, transform=transform)