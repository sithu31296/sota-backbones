from pathlib import Path
from .resnet import ResNet, resnet_settings
from .cswin import CSWin, cswin_settings
from .pvt import PVTv2, pvtv2_settings
from .rest import ResT, rest_settings
from .micronet import MicroNet, micronet_settings
from .poolformer import PoolFormer, poolformer_settings
from .patchconvnet import PatchConvnet, patchconvnet_settings
from .wavemlp import WaveMLP, wavemlp_settings
from .convnext import ConvNeXt, convnext_settings
from .uniformer import UniFormer, uniformer_settings
from .van import VAN, van_settings
from .focalnet import FocalNet, focalnet_settings


__all__ = [
    'ResNet', 'MicroNet', 'ConvNeXt', 'VAN',
    'PVTv2', 'ResT',
    'CSWin', 
    'WaveMLP',
    'PoolFormer', 'PatchConvnet', 'UniFormer', 'FocalNet',
]


def get_model(model_name: str, model_variant: str, pretrained: str = None, num_classes: int = 1000, image_size: int = 224):
    assert model_name in __all__, f"Unavailable model name >> {model_name}.\nList of available model names: {__all__}"
    if pretrained is not None: assert Path(pretrained).exists(), "Please set the correct pretrained model path"
    return eval(model_name)(model_variant, pretrained, num_classes, image_size)    