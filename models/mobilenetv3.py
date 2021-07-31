import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor



class Conv(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, g=1, d=1):
        p = (k - 1) // 2 * d
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6(True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, c1, c2, s, expand_ratio):
        super().__init__()
        hidden_dim = int(round(c1 * expand_ratio))
        self.use_res_connect = s == 1 and c1 == c2

        layers = [Conv(c1, hidden_dim, k=1)] if expand_ratio != 1 else []   # pw
        layers.extend([
            Conv(hidden_dim, hidden_dim, s=s, g=hidden_dim),        # dw
            nn.Conv2d(hidden_dim, c2, 1, 1, 0, bias=False),         # pw-linear
            nn.BatchNorm2d(c2)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(x) if self.use_res_connect else self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, model_name: str = None, pretrained: str = None, num_classes: int = 1000, *args, **kwargs):
        super().__init__()
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        input_channel = 32
        self.last_channel = 1280

        # building first layer
        features = [Conv(3, input_channel, s=2)]

        # building inverted residual blocks
        for t, output_channel, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel

        # building last several layers
        features.append(Conv(input_channel, self.last_channel, k=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'))
        else:
            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


if __name__ == '__main__':
    model = MobileNetV3('./mobilenet_v2.pth')
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)