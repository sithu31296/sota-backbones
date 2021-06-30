import torch
from torch import nn
from torch.utils import benchmark


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