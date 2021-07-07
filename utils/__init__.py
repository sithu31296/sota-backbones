from torch import nn
from .loss import *
from .schedulers import *

losses = {
    'vanilla': CrossEntropyLoss,
    'label_smooth': LabelSmoothCrossEntropy,
    'soft_target': SoftTargetCrossEntropy
}

kds = {
    'vanilla': VanillaKD,
}

schs = {
    "steplr": StepLR
}

def get_loss(cfg):
    if cfg['KD']['ENABLE']:
        method = cfg['KD']['METHOD']
        assert method in kds.keys(), f"Unavailable knowledge distillation method >> {method}.\nList of available methods: {list(kds.keys())}"
        return kds[method](cfg['KD']['ALPHA'], cfg['KD']['TEMP'])
    else:
        loss_fn_name = cfg['TRAIN']['LOSS']
        assert loss_fn_name in losses.keys(), f"Unavailable loss function name >> {loss_fn_name}.\nList of available loss functions: {list(losses.keys())}"
        return losses[loss_fn_name]

def get_scheduler(cfg, optimizer):
    scheduler_name = cfg['TRAIN']['SCHEDULER']['NAME']
    assert scheduler_name in schs.keys(), f"Unavailable scheduler name >> {scheduler_name}.\nList of available schedulers: {list(schs.keys())}"
    return schs[scheduler_name](optimizer, *cfg['TRAIN']['SCHEDULER']['PARAMS'])