import torch
import math
from torch import nn, Tensor
from .layers import MLP


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        x = x.view(B, H, W, C).to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x *= torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x


class Block(nn.Module):
    def __init__(self, dim, h=14, w=8, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, h, w)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x += self.mlp(self.norm2(self.filter(self.norm1(x)))) * self.gamma
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=4, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.num_patches = (img_size[1]//patch_size) * (img_size[0]//patch_size)
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.proj(x)                   # b x hidden_dim x 14 x 14
        x = x.flatten(2).swapaxes(1, 2)     # b x (14*14) x hidden_dim
        return x


class Downsample(nn.Module):
    def __init__(self, img_size=56, c1=64, c2=128):
        super().__init__()
        self.img_size = img_size
        self.c2 = c2
        self.proj = nn.Conv2d(c1, c2, 2, 2)
        
    def forward(self, x: torch.Tensor) -> Tensor:
        B, _, C = x.shape
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.c2)
        return x


gfnet_settings = {
    'T': [[64, 128, 256, 512], [3, 3, 10, 3], 1e-3],  # [embed_dims, depths, init_values]
    'S': [[96, 192, 384, 768], [3, 3, 10, 3], 1e-5],
    'B': [[96, 192, 384, 768], [3, 3, 27, 3], 1e-6]
}


class GFNet(nn.Module):
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in gfnet_settings.keys(), f"GFNet model name should be in {list(gfnet_settings.keys())}"
        embed_dims, depths, init_values = gfnet_settings[model_name]

        self.patch_embed = nn.ModuleList([
            PatchEmbed(image_size, 4, embed_dims[0]),
            Downsample(image_size//4, embed_dims[0], embed_dims[1]),
            Downsample(image_size//8, embed_dims[1], embed_dims[2]),
            Downsample(image_size//16, embed_dims[2], embed_dims[3])
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed[0].num_patches, embed_dims[0]))

        self.blocks = nn.ModuleList([
            nn.Sequential(*[
                Block(embed_dims[i], image_size//(2*2**(i+1)), (image_size//(4*2**(i+1)))+1, init_values)
            for _ in range(depths[i])])
        for i in range(4)])

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self._init_weights(pretrained)


    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'])
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if n.startswith('head'):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


    def forward(self, x: Tensor) -> Tensor:
        for i, (patch, blk) in enumerate(zip(self.patch_embed, self.blocks)):
            x = patch(x)
            if i == 0:
                x += self.pos_embed
            x = blk(x)
            
        x = self.norm(x).mean(dim=1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = GFNet('B', 'checkpoints/gfnet/gfnet-h-b.pth', image_size=224)
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)