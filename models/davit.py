import torch
import math
import itertools
from torch import nn, Tensor
from torch.nn import functional as F
from layers import DropPath, trunc_normal_


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H//window_size, W//window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for m in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = m(*inputs)
            else:
                inputs = m(inputs)
        return inputs


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim)
        self.activation = nn.GELU()

    def forward(self, x, H, W) -> Tensor:
        B, N, C = x.shape
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        k = k * self.scale
        attn = k.transpose(-1, -2) @ v
        attn = attn.softmax(dim=-1)
        x = (attn @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        print(x.shape)
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):
    def __init__(self, dim, num_heads, dpr=0.):
        super().__init__()
        self.cpe = nn.ModuleList([
            ConvPosEnc(dim, 3),
            ConvPosEnc(dim, 3)
        ])

        self.norm1 = nn.LayerNorm(dim)
        self.attn = ChannelAttention(dim, num_heads)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x, H, W) -> Tensor:
        x = self.cpe[0](x, H, W)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.cpe[1](x, H, W)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, H, W


class SpatialBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, dpr=0.):
        super().__init__()
        self.window_size = window_size
        self.cpe = nn.ModuleList([
            ConvPosEnc(dim, 3),
            ConvPosEnc(dim, 3)
        ])

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x, H, W) -> Tensor:
        B, L, C = x.shape
        assert L == H * W

        shortcut = self.cpe[0](x, H, W)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H*W, C)
        x = shortcut + self.drop_path(x)

        x = self.cpe[1](x, H, W)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_ch=3, embed_dim=768):
        super().__init__()
        if patch_size == 4:
            self.proj = nn.Conv2d(in_ch, embed_dim, 7, patch_size, 3)
            self.norm = nn.LayerNorm(embed_dim)
        elif patch_size == 2:
            self.proj = nn.Conv2d(in_ch, embed_dim, 2, patch_size)
            self.norm = nn.LayerNorm(in_ch)

    def forward(self, x, H, W):
        dim = len(x.shape)

        if dim == 3:
            B, HW, C = x.shape
            x = self.norm(x)
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        B, C, H, W = x.shape
        x = self.proj(x)
        Ho, Wo = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)

        if dim == 4:
            x = self.norm(x)
        return x, Ho, Wo


davit_settings = {
    'T': [[1, 1, 3, 1], [64, 128, 192, 256], [3, 6, 12, 24]],       # [depths, dims]
    'S': [[2, 2, 18, 2], 96],
    'B': [[2, 2, 18, 2], 128]
}


class DaViT(nn.Module):     
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in davit_settings.keys(), f"DaViT model name should be in {list(davit_settings.keys())}"
        depths, embed_dims, num_heads = davit_settings[model_name]
        drop_path_rate = 0.1
        window_size = 7
        arch = [[index] * item for index, item in enumerate(depths)]
        self.arch = arch
 
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(4 if i == 0 else 2, 3 if i == 0 else embed_dims[i-1], embed_dims[i])
        for i in range(len(depths))])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*arch))))]

        self.main_blocks = nn.ModuleList()
        for id, param in enumerate(arch):
            layer_offset_id = len(list(itertools.chain(*arch[:id])))

            block = nn.ModuleList([
                MySequential(
                    SpatialBlock(embed_dims[item], num_heads[item], window_size, dpr[2 * (layer_id + layer_offset_id)]),
                    ChannelBlock(embed_dims[item], num_heads[item], dpr[2 * (layer_id + layer_offset_id) + 1])
                )
            for layer_id, item in enumerate(param)])

            self.main_blocks.append(block)

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        # to use as a backbone, uncomment below
        # for i in range(4):
        #     self.add_module(f"norm{i}", nn.LayerNorm(embed_dims[i]))

        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            try:
                print(f"Loading imagenet pretrained weights from {pretrained}")
            except RuntimeError:
                pretrained_dict = torch.load(pretrained, map_location='cpu')['model']
                pretrained_dict.popitem()   # remove bias
                pretrained_dict.popitem()   # remove weight
                self.load_state_dict(pretrained_dict, strict=False)
            finally:
                print(f"Loaded imagenet pretrained from {pretrained}")
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()
                
    def return_features(self, x):
        x, Ho, Wo = self.patch_embeds[0]
        outs = []

        for i, layer in enumerate(self.layers):
            out, H, W, x, Ho, Wo = layer(x, Ho, Wo)
            out = getattr(self, f"norm{i}")(out)
            out = out.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(out)
        return outs
        
    def forward(self, x: torch.Tensor):
        x, *size = self.patch_embeds[0](x, x.shape[2], x.shape[3])
        features = [x]
        sizes = [size]
        branches = [0]
        
        for id, param in enumerate(self.arch):
            branch_ids = sorted(set(param))
            for branch_id in branch_ids:
                if branch_id not in branches:
                    x, *size = self.patch_embeds[branch_id](features[-1], *sizes[-1])
                    features.append(x)
                    sizes.append(size)
                    branches.append(branch_id)

            for layer_index, branch_id in enumerate(param):
                features[branch_id], _ = self.main_blocks[id][layer_index](features[branch_id], *sizes[branch_id])
        
        # x = self.head(x)
        return x

if __name__ == '__main__':
    model = DaViT('T')
    # , 'C:\\Users\\sithu\\Documents\\weights\\backbones\\focalnet\\focalnet_tiny_lrf.pth'
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)