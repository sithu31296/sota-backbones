import torch
import argparse
import yaml
from torch import nn
from torch.utils import benchmark
import sys
sys.path.insert(0, '.')
from models import get_model
from utils.utils import setup_cudnn


def benchmark_model(model: nn.Module, inputs: torch.Tensor, times: int = 100, num_threads: int = None, wall_time: bool = False):
    if num_threads is None:
        num_threads = torch.get_num_threads()
    
    print(f'Benchmarking with {num_threads} threads for {times} times.')
    timer = benchmark.Timer(
        stmt=f"{model}(x)",             # computation which will be run in a loop and times
        setup= f"x = {inputs}",         # setup will be run before calling the measurement loop and is used to populate any state which is need by 'stmt'
        num_threads=num_threads
    )
    if wall_time:
        return timer.blocked_autorange(min_run_time=0.2)
    return timer.timeit(times)


def main(cfg):
    setup_cudnn()

    model = get_model(cfg['MODEL']['NAME'], cfg['MODEL']['VARIANT'], None, cfg['DATASET']['NUM_CLASSES'], cfg['PROFILE']['IMAGE_SIZE'][0])
    inputs = torch.randn(1, 3, *cfg['PROFILE']['IMAGE_SIZE'])

    time = benchmark_model(model, inputs)
    print(time)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg)