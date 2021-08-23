from pathlib import Path
from .resnet import ResNet, resnet_settings
from .xcit import XciT, xcit_settings
from .vip import ViP, vip_settings
from .cyclemlp import CycleMLP, cyclemlp_settings
from .cswin import CSWin, cswin_settings
from .volo import VOLO, volo_settings
from .gfnet import GFNet, gfnet_settings
from .pvt import PVTv2, pvtv2_settings
from .shuffle import Shuffle, shuffle_settings
from .rest import ResT, rest_settings
from .conformer import Conformer, conformer_settings


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