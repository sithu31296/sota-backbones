import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.activation import ReLU


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)

        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HRModule(nn.Module):
    def __init__(self, num_branches, num_channels):
        super().__init__()
        self.num_branches = num_branches
        self.num_channels = num_channels

        self.branches = self._make_branches()
        self.fuse_layers = self._make_fuse_layers()

        self.relu = nn.ReLU(False)

    def _make_one_branch(self, branch_index):
        layers = [BasicBlock(self.num_channels[branch_index], self.num_channels[branch_index]) for _ in range(4)]
        return nn.Sequential(*layers)
    
    def _make_branches(self):
        branches = [self._make_one_branch(branch_index) for branch_index in range(self.num_branches)]
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        fuse_layers = []

        for i in range(self.num_branches):
            fuse_layer = []

            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(self.num_channels[j], self.num_channels[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(self.num_channels[i]),
                            nn.Upsample(scale_factor=2**(j-i))
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j -1:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(self.num_channels[j], self.num_channels[i], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(self.num_channels[i])
                                )
                            )
                        else:
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(self.num_channels[j], self.num_channels[j], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(self.num_channels[j]),
                                    nn.ReLU(False)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])

            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
                
            x_fuse.append(self.relu(y))

        return x_fuse


hrnet_settings = {
    "w18": [18, 36, 72, 144],
    "w32": [32, 64, 128, 256],
    "w48": [48, 96, 192, 384],
    "w64": [64, 128, 256, 512]
}


class HRNet(nn.Module):
    def __init__(self, model_name: str = 'w18', pretrained: str = None, num_classes: int = 1000) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)

        all_channels = hrnet_settings[model_name]

        # Stage 1
        self.layer1 = self._make_layer(64, 64, 4)
        stage1_out_channel = Bottleneck.expansion * 64

        # Stage 2
        stage2_channels = all_channels[:2]
        self.transition1 = self._make_transition_layer([stage1_out_channel], stage2_channels)
        self.stage2 = self._make_stage(1, 2, stage2_channels)

        # # Stage 3
        stage3_channels = all_channels[:3]
        self.transition2 = self._make_transition_layer(stage2_channels, stage3_channels)
        self.stage3 = self._make_stage(4, 3, stage3_channels)

        # # Stage 4
        self.transition3 = self._make_transition_layer(stage3_channels, all_channels)
        self.stage4 = self._make_stage(3, 4, all_channels)

        # # Classification head
        self.incre_modules, self.downsample_modules, self.final_layer = self._make_head(all_channels)

        self.classifier = nn.Linear(2048, num_classes)

        self._init_weights(pretrained)

    def _make_head(self, pre_stage_channels):
        head_channels = [32, 64, 128, 256]

        # Increasing the channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []

        for pre_channel, head_channel in zip(pre_stage_channels, head_channels):
            incre_modules.append(self._make_layer(pre_channel, head_channel, 1))

        # downsampling modules
        downsamp_modules = []

        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * Bottleneck.expansion
            out_channels = head_channels[i+1] * Bottleneck.expansion

            downsamp_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
        
        final_layer = nn.Sequential(
            nn.Conv2d(head_channels[-1] * Bottleneck.expansion, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )

        return nn.ModuleList(incre_modules), nn.ModuleList(downsamp_modules), final_layer

    def _make_layer(self, inplanes, planes, blocks):
        downsample = None

        if inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes*Bottleneck.expansion, 1, 1, 0, bias=False),
                nn.BatchNorm2d(planes*Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(inplanes, planes, downsample=downsample))
        inplanes = planes * Bottleneck.expansion

        for i in range(1, blocks):
            layers.append(Bottleneck(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_pre = len(num_channels_pre_layer)
        num_branches_cur = len(num_channels_cur_layer)
        
        transition_layers = []

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(True)
                    ))
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(True)
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        
        return nn.ModuleList(transition_layers)

    def _make_stage(self, num_modules, num_branches, num_channels):
        modules = []

        for i in range(num_modules):
            modules.append(HRModule(num_branches, num_channels))

        return nn.Sequential(*modules)

    def _init_weights(self, pretrained=None):
        if pretrained:
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
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


    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # Stage 1
        x = self.layer1(x)

        # Stage 2
        x_list = [trans(x) for trans in self.transition1]
        y_list = self.stage2(x_list)

        # Stage 3
        x_list = y_list + [trans(y_list[-1]) for trans in self.transition2]
        y_list = self.stage3(x_list)

        # # Stage 4
        x_list = y_list + [trans(y_list[-1]) for trans in self.transition3]
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])

        for i in range(len(self.downsample_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + self.downsample_modules[i](y)

        y = self.final_layer(y)

        y = torch.flatten(y, 2).mean(2)
        # y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        
        y = self.classifier(y)

        return y


if __name__ == '__main__':
    model = HRNet()
    x = torch.randn((1, 3, 224, 224))
    y = model(x)
    print(y.shape)




