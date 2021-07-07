import torch


def accuracy(pred: torch.Tensor, target:torch.Tensor, topk: tuple = (1,)):
    maxk = max(topk)
    batch_size = target.shape[0]
    pred = pred.topk(maxk, 1)[-1]
    pred = pred.t()
    correct = pred == target.view(1, -1).expand_as(pred)

    return [correct[:k].reshape(-1).float().sum(0)*100. /  batch_size for k in topk]