"""torchvision builtin transforms
# shape transform
CenterCrop(size)
Resize(size)
RandomCrop(size, padding=None, pad_if_needed=False, fill=0)
RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.33))
RandomRotation(degrees)
Pad(padding, fill=0)

# spatial transform
ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
GaussianBlur(kernel_size, sigma=(0.1, 2.0))
RandomAffine(degrees, translate=None, scale=None, shear=None)
RandomGrayscale(p=0.1)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomPerspective(distortion_scale=0.5, p=0.5)
RandomInvert(p=0.5)
RandomPosterize(bits, p=0.5)
RandomSolarize(threshold, p=0.5)
RandomAdjustSharpness(sharpness_factor, p=0.5)
RandomAutocontrast(p=0.5)

# auto-augment
AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET)

# others
RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
RandomApply(transforms, p=0.5)      # apply randomly a list of transformations with a given probability
"""

from torchvision import transforms as T


def get_transforms(cfg):
    train_transform = T.Compose(
        T.RandomSizedCrop(cfg['TRAIN']['IMAGE_SIZE']),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1),
        T.AutoAugment(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(0.2),
    )

    val_transform = T.Compose(
        T.Resize(tuple(map(lambda x: int(x / 0.9), cfg['EVAL']['IMAGE_SIZE']))),
        T.CenterCrop(cfg['EVAL']['IMAGE_SIZE']),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    )

    return train_transform, val_transform