import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .layers import MLP, DropPath, trunc_normal_


class Attention(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5

        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, head, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):
    def __init__(self, c1, c2, s=1, res_conv=False):
        super().__init__()
        ch = c2 // 4
        self.res_conv = res_conv
        self.conv1 = nn.Conv2d(c1, ch, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)

        self.conv2 = nn.Conv2d(ch, ch, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

        self.conv3 = nn.Conv2d(ch, c2, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

        if self.res_conv:
            self.residual_conv = nn.Conv2d(c1, c2, 1, s, 0, bias=False)
            self.residual_bn = nn.BatchNorm2d(c2)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x: Tensor, x_t: Tensor = None, return_x2=True):
        residual = x

        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv2(x) if x_t is None else self.conv2(x+x_t)
        x2 = self.act(self.bn2(x))

        x = self.bn3(self.conv3(x2))
        if self.res_conv:
            residual = self.residual_bn(self.residual_conv(residual))
        x += residual
        x = self.act(x)

        if return_x2:
            return x, x2
        return x


class FCUDown(nn.Module):
    def __init__(self, c1, c2, dw_stride):
        super().__init__()
        self.conv_project = nn.Conv2d(c1, c2, 1, 1, 0)
        self.sample_pooling = nn.AvgPool2d(dw_stride, dw_stride)
        self.ln = nn.LayerNorm(c2)
        self.act = nn.GELU()

    def forward(self, x, x_t):
        x = self.conv_project(x)
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)
        return x


class FCUUp(nn.Module):
    def __init__(self, c1, c2, up_stride):
        super().__init__()
        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(c1, c2, 1, 1, 0)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x = self.act(self.bn(self.conv_project(x)))
        x = F.interpolate(x, size=(H*self.up_stride, W*self.up_stride))
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, c1, c2, res_conv, stride, dw_stride, embed_dim, head=12, dpr=0., last_fusion=False):
        super().__init__()
        expansion = 4
        self.dw_stride = dw_stride
        self.cnn_block = ConvBlock(c1, c2, stride, res_conv)

        if last_fusion:
            self.fusion_block = ConvBlock(c2, c2, 2, True)
        else:
            self.fusion_block = ConvBlock(c2, c2)

        self.squeeze_block = FCUDown(c2//expansion, embed_dim, dw_stride)
        self.expand_block = FCUUp(embed_dim, c2//expansion, dw_stride)
        self.trans_block = Block(embed_dim, head, dpr)

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)
        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)
        x_t = self.trans_block(x_st+x_t)
        x_t_r = self.expand_block(x_t, H//self.dw_stride, W//self.dw_stride)

        x = self.fusion_block(x, x_t_r, return_x2=False)
        return x, x_t


conformer_settings = {
    'T': [1, 384, 6, 0.1],    # [channel_ratio, embed_dim, head, dpr]
    'S': [4, 384, 6, 0.2],
    'B': [6, 576, 9, 0.3]
}


class Conformer(nn.Module):  # this model works with any image size, even non-square image size
    def __init__(self, model_name: str = 'S', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in conformer_settings.keys(), f"Conformer model name should be in {list(conformer_settings.keys())}"
        channel_ratio, embed_dim, head, drop_path_rate = conformer_settings[model_name]
        depth = 12

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Stem
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Stage1
        stage1_channel = int(64*channel_ratio)
        self.conv_1 = ConvBlock(64, stage1_channel, res_conv=True)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, 4, 4, 0)
        self.trans_1 = Block(embed_dim, head, dpr[0])

        # Stage2-4
        self.conv_trans_2 = ConvTransBlock(stage1_channel, stage1_channel, False, 1, 4, embed_dim, head, dpr[1])
        self.conv_trans_3 = ConvTransBlock(stage1_channel, stage1_channel, False, 1, 4, embed_dim, head, dpr[2])
        self.conv_trans_4 = ConvTransBlock(stage1_channel, stage1_channel, False, 1, 4, embed_dim, head, dpr[3])

        # Stage5-8
        self.conv_trans_5 = ConvTransBlock(stage1_channel, stage1_channel*2, True, 2, 2, embed_dim, head, dpr[4])
        self.conv_trans_6 = ConvTransBlock(stage1_channel*2, stage1_channel*2, False, 1, 2, embed_dim, head, dpr[5])
        self.conv_trans_7 = ConvTransBlock(stage1_channel*2, stage1_channel*2, False, 1, 2, embed_dim, head, dpr[6])
        self.conv_trans_8 = ConvTransBlock(stage1_channel*2, stage1_channel*2, False, 1, 2, embed_dim, head, dpr[7])

        # Stage9-12
        self.conv_trans_9 = ConvTransBlock(stage1_channel*2, stage1_channel*4, True, 2, 1, embed_dim, head, dpr[8])
        self.conv_trans_10 = ConvTransBlock(stage1_channel*4, stage1_channel*4, False, 1, 1, embed_dim, head, dpr[9])
        self.conv_trans_11 = ConvTransBlock(stage1_channel*4, stage1_channel*4, False, 1, 1, embed_dim, head, dpr[10])
        self.conv_trans_12 = ConvTransBlock(stage1_channel*4, stage1_channel*4, False, 1, 1, embed_dim, head, dpr[11], True)

        self.depth = depth

        self.trans_norm = nn.LayerNorm(embed_dim)
        # self.pooling = nn.AdaptiveAvgPool2d(1)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes)
        self.conv_cls_head = nn.Linear(int(256*channel_ratio), num_classes)

        trunc_normal_(self.cls_token, std=.02)
        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'))
        else:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None: 
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # stem
        x_base = self.maxpool(self.act(self.bn1(self.conv1(x))))

        # stage 1
        x = self.conv_1(x_base, return_x2=False)
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)

        # stage 2-12
        for i in range(2, self.depth+1):
            x, x_t = eval(f'self.conv_trans_{i}')(x, x_t)

        # x_p = self.pooling(x).flatten(1)
        # conv_cls = self.conv_cls_head(x_p)

        x_t = self.trans_norm(x_t)
        trans_cls = self.trans_cls_head(x_t[:, 0])
        return trans_cls


if __name__ == '__main__':
    model = Conformer('S', 'checkpoints/conformer/Conformer_small_patch16.pth')
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)