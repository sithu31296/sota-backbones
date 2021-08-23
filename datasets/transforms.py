import random
import torch
from torchvision import transforms as T


def get_augmentations(cfg):
    return T.Compose(
        T.RandomSizedCrop(cfg['TRAIN']['IMAGE_SIZE']),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1),
        T.AutoAugment(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(0.2),
    )

def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.shape[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


class MixUp:
    def __init__(self, alpha=0.8, p=1.0, n_classes=1000, label_smooth=0.1) -> None:
        self.alpha = alpha
        self.p = p
        self.n_classes = n_classes
        self.label_smooth = label_smooth

    def __call__(self, image, target):
        assert image.shape[0] == 0, "Batch Size should be even why using mixup augmentation"
        if random.random() < self.p:
            image_flipped = image.flip(0).mul_(1. - self.alpha)
            image.mul_(self.alpha).add_(image_flipped)
            off_value = self.label_smooth / self.n_classes
            on_value = 1. - self.label_smooth + off_value
            y1 = one_hot(target, self.n_classes, on_value, off_value, device=image.device)
            y2 = one_hot(target.flip(0), self.n_classes, on_value, off_value, device=image.device)
            target = y1 * self.alpha + y2 * (1. - self.alpha)
        return image, target
            



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

