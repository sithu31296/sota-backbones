from .hrnet import HRNet
from .resnet import ResNet
from .mlpmixer import MLPMixer
from .vit import ViT
from .deit import DeiT
from .gmlp import gMLP
from .mobilenetv2 import MobileNetV2
from .resmlp import ResMLP
from .lvvit import LVViT
from .cait import CaiT
from .xcit import XciT
from .vip import ViP
from pathlib import Path

__all__ = {
    "hrnet": HRNet,
    "resnet": ResNet,
    "mobilenetv2": MobileNetV2,
    
    "vit": ViT,
    "deit": DeiT,
    "lvvit": LVViT,
    "cait": CaiT,
    "xcit": XciT,

    "mixer": MLPMixer,
    "resmlp": ResMLP,
    "gmlp": gMLP,
    "vip": ViP,
}

def get_model(model_name: str, model_variant: str, pretrained: str = None, num_classes: int = 1000, image_size: int = 224):
    assert model_name in __all__.keys(), f"Unavailable model name >> {model_name}.\nList of available model names: {list(__all__.keys())}"
    if pretrained is not None:
        assert Path(pretrained).exists(), "Please set the correct pretrained model path"
    return __all__[model_name](model_variant, pretrained, num_classes, image_size)    