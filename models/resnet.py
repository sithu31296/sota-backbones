import torch 
import torch.nn as nn
from torch import Tensor
from typing import Type, Optional, Union


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int, s: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_ch: int, out_ch: int, s: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, s, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.conv3 = nn.Conv2d(out_ch, out_ch * self.expansion, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion)

        self.relu = nn.ReLU(True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


resnet_settings = {
    '18': [BasicBlock, [2, 2, 2, 2]],
    '34': [BasicBlock, [3, 4, 6, 3]],
    '50': [Bottleneck, [3, 4, 6, 3]],
    '101': [Bottleneck, [3, 4, 23, 3]],
    '152': [Bottleneck, [3, 8, 36, 3]]
}


class ResNet(nn.Module):
    def __init__(self, model_name: str = '50', pretrained: str = None, num_classes: int = 1000) -> None:
        super().__init__()
        self.inplanes = 64

        assert model_name in ['18', '34', '50', '101', '152'], "ResNet model name should be '18' or '34' or '50' or '101' or '152'"
        block, layers = resnet_settings[model_name]

        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64, layers[0], s=1)
        self.layer2 = self._make_layer(block, 128, layers[1], s=2)
        self.layer3 = self._make_layer(block, 256, layers[2], s=2)
        self.layer4 = self._make_layer(block, 512, layers[3], s=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(pretrained)


    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int, s: int = 1) -> nn.Sequential:
        downsample = None
        if s != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, s),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, s, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x



if __name__ == '__main__':
    model = ResNet('50')
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)