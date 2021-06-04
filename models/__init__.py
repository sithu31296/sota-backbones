from .hrnet import HRNet
from .resnet import ResNet
from .mlpmixer import MLPMixer

choose_models = {
    "hrnet": HRNet,
    "resnet": ResNet,
    "mixer": MLPMixer
}