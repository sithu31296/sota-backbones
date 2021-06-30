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

def choose_models(model_name: str):
    assert model_name in __all__.keys(), f"Error: Unsupported model name provided.\nOnly support models in {list(__all__.keys())}"
    return __all__[model_name]