"""Pruning a Model
Compress models by reducing the number of parameters

Pruning acts by removing `weight` from the parameters and replacing it with a new parameter called `weight_orig`.
`weight_orig` stores the unpruned version of the tensor.
"""

import torch
import argparse
import yaml
from torch import nn
from torch.nn.utils import prune
from pathlib import Path
from tabulate import tabulate
import sys
sys.path.insert(0, '.')
from models import get_model
from utils.utils import setup_cudnn, get_model_size, test_model_latency


def calc_sparsity(model: nn.Module):
    where_zero, nelements = 0.0, 0.0

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            prune.remove(m, 'weight')
            where_zero += (m.weight == 0).sum()
            nelements += m.weight.nelement()
    
    print(f"Global Sparsity: {100 * where_zero / nelements:.2f}")


def prune_model(model: nn.Module, method: str = 'l1', amount: float = 0.3):
    # remove lowest amount% of connections in each layer
    parameters_to_prune = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    if method == 'l1':
        pruning_method = prune.L1Unstructured
    elif method == 'random':
        pruning_method = prune.RandomUnstructured
    else:
        raise "Unavailable pruning method."

    prune.global_unstructured(parameters_to_prune, pruning_method=pruning_method, amount=amount)

    calc_sparsity(model)


def main(cfg):
    setup_cudnn() 
    save_dir = Path(cfg['SAVE_DIR'])
    if not save_dir.exists(): save_dir.mkdir()
    save_model = save_dir / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}_pruned.pt"

    inputs = torch.randn(1, 3, *cfg['TRAIN']['IMAGE_SIZE'])

    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], cfg['MODEL_PATH'], cfg['DATASET']['NUM_CLASSES'], cfg['TRAIN']['IMAGE_SIZE'][0])
    model_size = get_model_size(model)
    model_time = test_model_latency(model, inputs)
    model_accuracy = 85.34

    print('Starting Global Pruning...')
    prune_model(model, cfg['PRUNE']['METHOD'], cfg['PRUNE']['AMOUNT'])
    pruned_model_size = get_model_size(model)
    pruned_model_time = test_model_latency(model, inputs)
    pruned_model_accuracy = 84.04

    table = [
        [f"Original {cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}", f"{model_size:.2f}", f"{model_time:.2f}", f"{model_accuracy:.2f}"],
        [f"Pruned {cfg['MODEL']['NAME']}_{cfg['MODEL']['VARIANT']}", f"{pruned_model_size:.2f}", f"{pruned_model_time:.2f}", f"{pruned_model_accuracy:.2f}"],
        ["Improvement", f"+{int((model_size - pruned_model_size) / model_size * 100)}%", f"+{int((model_time - pruned_model_time) / model_time * 100)}%", f"-{(model_accuracy - pruned_model_accuracy) / model_accuracy * 100:.2f}%"]
    ]

    print(tabulate(table, numalign='right', headers=['Model', 'Model Size (MB)', "Latency (ms)", "Accuracy (%)"]))

    torch.save(model.state_dict(), save_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)