import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(pred: torch.Tensor, target:torch.Tensor, topk: tuple = (1,)):
    maxk = max(topk)
    batch_size = target.shape[0]
    pred = pred.topk(maxk, 1)[-1]
    pred = pred.t()
    correct = pred == target.view(1, -1).expand_as(pred)
    return [correct[:k].reshape(-1).float().sum(0).item()*100. /  batch_size for k in topk]