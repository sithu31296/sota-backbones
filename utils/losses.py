import torch
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


class DistillationLoss(nn.Module):
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


def to_one_hot(labels: Tensor, num_classes: int) -> Tensor:
    B, C, H, W = labels.shape
    assert C == 1

    o = torch.zeros(B, num_classes, H, W, device=labels.device)
    labels = o.scatter_(1, labels.long(), value=1)
    return labels


class PolyLoss(nn.Module):
    """PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    https://arxiv.org/abs/2204.12511v1
    https://github.com/yiyixuxu/polyloss-pytorch
    """
    reduction: str

    def __init__(self, softmax=False, epsilon=1.0, weight=None, reduction: str = 'mean') -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.ce = CrossEntropyLoss(weight, reduction='none')

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred: shape [B, C, H, W]
        target: shape[B, C, H, W] or [B, 1, H, W]
        """
        if pred.shape[1] != target.shape[1]:
            ce_loss = self.ce(pred, target.squeeze(dim=1).long())
            target = to_one_hot(target, num_classes=pred.shape[1])
        else:
            # target is in one-hot format, convert to BH format to calculate ce loss
            ce_loss = self.ce(pred, target.argmax(dim=1))

        if self.softmax:
            pred = pred.softmax(dim=1)

        pt = (pred * target).sum(dim=1)
        poly_loss = ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            loss = poly_loss.mean()
        elif self.reduction == 'sum':
            loss = poly_loss.sum()
        else:
            loss = poly_loss.unsqueeze(1)
        return loss



if __name__ == '__main__':
    loss = PolyLoss(softmax=True)
    B, C, H, W = 2, 10, 224, 224

    pred = torch.rand(B, C, H, W, requires_grad=True)
    target = torch.randint(0, C-1, (B, H, W)).long()
    target = to_one_hot(target[:, None, ...], num_classes=C)
    print(target.shape)
    output = loss(pred, target)
    output.backward()
    print(output)