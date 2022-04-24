import torch
import math
from torch import nn, Tensor
from .layers import DropPath, trunc_normal_


class DWConv(nn.Module):
    def __init__(self, dim=768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.dwconv(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.dwconv(self.fc1(x))))


class LKA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, 1, 9, 3, dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        u = x.clone()
        attn = self.conv1(self.conv_spatial(self.conv0(x)))
        return u * attn


class Attention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x.clone()
        x = self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))
        x = x + shortcut
        return x
    

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dpr=0., init_value=1e-2):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio))
        
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=7, stride=4, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride, patch_size//2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


van_settings = {
    'S': [[2, 2, 4, 2], [64, 128, 320, 512]],       # [depths, dims]
    'B': [[3, 3, 12, 3], [64, 128, 320, 512]],
    'L': [[3, 5, 27, 3], [64, 128, 320, 512]]
}


class VAN(nn.Module):     
    def __init__(self, model_name: str = 'S', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in van_settings.keys(), f"VAN model name should be in {list(van_settings.keys())}"
        depths, embed_dims = van_settings[model_name]
        drop_path_rate = 0.
        mlp_ratios = [8, 8, 4, 4]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            if i == 0:
                patch_embed = OverlapPatchEmbed(7, 4, 3, embed_dims[i])
            else:
                patch_embed = OverlapPatchEmbed(3, 2, embed_dims[i-1], embed_dims[i])
            
            block = nn.Sequential(*[
                Block(embed_dims[i], mlp_ratios[i], dpr[cur+j])
            for j in range(depths[i])])

            norm = nn.LayerNorm(embed_dims[i], eps=1e-6)
            cur += depths[i]

            setattr(self, f"patch_embed{i+1}", patch_embed)
            setattr(self, f"block{i+1}", block)
            setattr(self, f"norm{i+1}", norm)

        self.head = nn.Linear(embed_dims[-1], num_classes)

        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['state_dict'])
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
        B = x.shape[0]
        outs = []
        
        for i in range(4):
            x, H, W = getattr(self, f"patch_embed{i+1}")(x)
            x = getattr(self, f"block{i+1}")(x)   
            x = x.flatten(2).transpose(1, 2)
            x = getattr(self, f"norm{i+1}")(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs
        
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        
        for i in range(4):
            x, H, W = getattr(self, f"patch_embed{i+1}")(x)
            x = getattr(self, f"block{i+1}")(x)
            x = x.flatten(2).transpose(1, 2)
            x = getattr(self, f"norm{i+1}")(x)

            if i != 3:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = x.mean(dim=1)
        x = self.head(x)
        return x

if __name__ == '__main__':
    model = VAN('S', 'C:\\Users\\sithu\\Documents\\weights\\backbones\\van\\van_small_811.pth.tar')
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)