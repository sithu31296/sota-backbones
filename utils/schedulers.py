from torch.optim.lr_scheduler import StepLR



__all__ = {
    "steplr": StepLR
}

def get_scheduler(cfg, optimizer):
    scheduler_name = cfg['NAME']
    assert scheduler_name in __all__.keys(), f"Unavailable scheduler name >> {scheduler_name}.\nList of available schedulers: {list(__all__.keys())}"
    return __all__[scheduler_name](optimizer, cfg['STEP_SIZE'], cfg['GAMMA'])