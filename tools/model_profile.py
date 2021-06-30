"""PyTorch Profiler
    -   To measure the time and memory consumption of the model's operators.
    -   Is useful when user needs to determine the most expensive operators in the model.

"""
import argparse
import yaml
import torch
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity

import sys
sys.path.insert(0, '.')
from models import choose_models
from utils.utils import setup_cudnn



def profile_model(model, inputs, device='cpu', profile_memory=True, profile_shape=False, sort_by='time_total', save_json=''):
    assert sort_by in ['time_total', 'memory_usage']
    assert device in ['cpu', 'cuda']

    if device == 'cpu':
        activities = [ProfilerActivity.CPU]
    else:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # note: the first use of CUDA profiling may bring an extra overhead
    with profile(activities=activities, record_shapes=True, profile_memory=profile_memory) as prof:
        with record_function("model_inference"):
            model(inputs)

    print(prof.key_averages(group_by_input_shape=profile_shape).table(sort_by=f"{device}_{sort_by}", row_limit=10))

    if save_json.endswith('.json'):
        prof.export_chrome_trace(save_json)


def main(cfg):
    setup_cudnn()
    model_name = cfg['MODEL']['NAME']
    model_sub_name = cfg['MODEL']['SUB_NAME']

    if cfg['PROFILE']['SAVE']:
        save_dir = Path(cfg['SAVE_DIR'])
        if not save_dir.exists(): save_dir.mkdir()
        save_json = save_dir / f"{model_name}_{model_sub_name}_trace.json"
    else:
        save_json = ''

    model = choose_models(model_name)(model_sub_name, pretrained=None, num_classes=cfg['DATASET']['NUM_CLASSES'], image_size=cfg['PROFILE']['IMAGE_SIZE'][0])    
    inputs = torch.randn(1, 3, *cfg['PROFILE']['IMAGE_SIZE'])

    profile_model(model, inputs, device=cfg['DEVICE'], profile_memory=cfg['PROFILE']['MEMORY'], profile_shape=cfg['PROFILE']['SHAPE'], sort_by=cfg['PROFILE']['SORT'], save_json=save_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)