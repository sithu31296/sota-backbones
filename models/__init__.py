from pathlib import Path
from .resnet import *
from .xcit import *
from .vip import *
from .cyclemlp import *
from .cswin import *
from .volo import *
from .gfnet import *
from .pvt import *
from .shuffle import *
from .rest import *
from .conformer import *


__all__ = {
    "resnet": ResNet,
     
    "xcit": XciT,
    "cswin": CSWin,
    "volo": VOLO,
    "gfnet": GFNet,
    "pvtv2": PVTv2,
    "shuffle": Shuffle,
    "rest": ResT,
    "conformer": Conformer,

    "vip": ViP,
    'cyclemlp': CycleMLP,
}

def get_model(model_name: str, model_variant: str, pretrained: str = None, num_classes: int = 1000, image_size: int = 224):
    assert model_name in __all__.keys(), f"Unavailable model name >> {model_name}.\nList of available model names: {list(__all__.keys())}"
    if pretrained is not None: assert Path(pretrained).exists(), "Please set the correct pretrained model path"
    return __all__[model_name](model_variant, pretrained, num_classes, image_size)    