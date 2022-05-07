import torch
import math
from torch import nn, Tensor
from .layers import DropPath, trunc_normal_


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=2) -> None:
        super().__init__()
        self.focal_level = focal_level
        
        self.f = nn.Linear(dim, 2 * dim + (focal_level + 1))
        self.h = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        
        self.focal_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, (focal_factor * k + focal_window), 1, (focal_factor * k + focal_window)//2, groups=dim, bias=False),
                nn.GELU()
            )
        for k in range(focal_level)])

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape

        # pre linear projection
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level + 1), 1)

        # context aggregation
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, l:l+1]

        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level:]

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q * self.modulator
        x_out = x_out.permute(0, 2, 3, 1).contiguous()

        # post linear projection
        x_out = self.proj(x_out)
        return x_out
    

class Block(nn.Module):
    def __init__(self, dim, focal_window, focal_level, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.modulation = FocalModulation(dim, focal_window, focal_level)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x, H, W) -> Tensor:
        B, _, C = x.shape
        shortcut = x

        # Focal modulation
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x = self.modulation(x).view(B, H*W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=7, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, depth, focal_window, focal_level, dpr, downsample=None) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(dim, focal_window, focal_level, dpr[i] if isinstance(dpr, list) else dpr)
        for i in range(depth)])

        self.downsample = None
        if downsample is not None:
            self.downsample = downsample(2, dim, out_dim)

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)

        if self.downsample is not None:
            x_down = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x_down, Ho, Wo = self.downsample(x_down)
            return x, H, W, x_down, Ho, Wo
        else:
            return x, H, W, x, H, W


focalnet_settings = {
    'T': [[2, 2, 6, 2], 96],       # [depths, dims]
    'S': [[2, 2, 18, 2], 96],
    'B': [[2, 2, 18, 2], 128]
}


class FocalNet(nn.Module):     
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in focalnet_settings.keys(), f"FocalNet model name should be in {list(focalnet_settings.keys())}"
        depths, embed_dim = focalnet_settings[model_name]
        drop_path_rate = 0.1
        focal_level = 3
        focal_window = 3
        embed_dims = [embed_dim * (2 ** i) for i in range(len(depths))]

        self.patch_embed = PatchEmbed(4, 3, embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = BasicLayer(embed_dims[i], embed_dims[i+1] if i < len(depths) - 1 else None, depths[i], focal_window, focal_level, dpr[sum(depths[:i]):sum(depths[:i+1])], PatchEmbed if i < len(depths) - 1 else None)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
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
        x, Ho, Wo = self.patch_embed(x)
        outs = []

        for i, layer in enumerate(self.layers):
            out, H, W, x, Ho, Wo = layer(x, Ho, Wo)
            out = getattr(self, f"norm{i}")(out)
            out = out.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(out)
        return outs
        
    def forward(self, x: torch.Tensor):
        x, H, W = self.patch_embed(x)

        for layer in self.layers:
            _, _, _, x, H, W = layer(x, H, W)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = x.flatten(1)
        x = self.head(x)
        return x

if __name__ == '__main__':
    model = FocalNet('T', 'C:\\Users\\sithu\\Documents\\weights\\backbones\\focalnet\\focalnet_tiny_lrf.pth')
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)