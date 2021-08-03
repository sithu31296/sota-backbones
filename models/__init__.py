from pathlib import Path
from .hrnet import HRNet
from .resnet import ResNet
from .mlpmixer import MLPMixer
from .vit import ViT
from .deit import DeiT
from .gmlp import gMLP
from .mobilenetv2 import MobileNetV2
from .mobilenetv3 import MobileNetV3
from .resmlp import ResMLP
from .lvvit import LVViT
from .cait import CaiT
from .xcit import XciT
from .vip import ViP
from .cyclemlp import CycleMLP
from .cswin import CSWin
from .volo import VOLO
from .gfnet import GFNet
from .pvt import PVTv2

__all__ = {
    "hrnet": HRNet,
    "resnet": ResNet,
    "mobilenetv2": MobileNetV2,
    "mobilenetv3": MobileNetV3,
    
    "vit": ViT,
    "deit": DeiT,
    "lvvit": LVViT,
    "cait": CaiT,
    "xcit": XciT,
    "cswin": CSWin,
    "volo": VOLO,
    "gfnet": GFNet,
    "pvtv2": PVTv2,

    "mixer": MLPMixer,
    "resmlp": ResMLP,
    "gmlp": gMLP,
    "vip": ViP,
    'cyclemlp': CycleMLP,
}

def get_model(model_name: str, model_variant: str, pretrained: str = None, num_classes: int = 1000, image_size: int = 224):
    assert model_name in __all__.keys(), f"Unavailable model name >> {model_name}.\nList of available model names: {list(__all__.keys())}"
    if pretrained is not None:
        assert Path(pretrained).exists(), "Please set the correct pretrained model path"
    return __all__[model_name](model_variant, pretrained, num_classes, image_size)    