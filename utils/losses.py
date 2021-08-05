from torch import nn, Tensor
from typing import Union
from torch.nn import CrossEntropyLoss


class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = self.log_softmax(pred)
        nll_loss = -pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -pred.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = self.log_softmax(pred)
        loss = (-target * pred).sum(dim=-1)
        return loss.mean()


class VanillaKD(nn.Module):
    """Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """
    def __init__(self, alpha: float = 0.95, temp: Union[float, int] = 6) -> None:
        super().__init__()
        self.alpha = alpha
        self.temp = temp
        self.kd_loss = nn.KLDivLoss()
        self.entropy_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pred_student: Tensor, pred_teacher: Tensor, target: Tensor) -> Tensor:
        loss = self.kd_loss(self.log_softmax(pred_student / self.temp), self.softmax(pred_teacher / self.temp)) * (self.alpha * self.temp * self.temp)
        loss += self.entropy_loss(pred_student, target) * (1. - self.alpha)
        return loss


losses = {
    'vanilla': CrossEntropyLoss,
    'label_smooth': LabelSmoothCrossEntropy,
    'soft_target': SoftTargetCrossEntropy
}

kds = {
    'vanilla': VanillaKD,
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
