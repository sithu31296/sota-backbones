import torch
import math
import torch.nn.functional as F
from torch import nn, Tensor

from .layers import MLP


class PositionalEncodingFourier(nn.Module):
    def __init__(self, dim: int = 768, temp: int = 10000):
        super().__init__()
        self.hidden_dim = 32
        self.token_projection = nn.Conv2d(self.hidden_dim * 2, dim, 1)
        self.scale = 2 * math.pi
        self.temperature = temp
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos


class Conv3x3(nn.Sequential):
    def __init__(self, c1, c2, s=1):
        super().__init__(
            nn.Conv2d(c1, c2, 3, s, 1, bias=False),
            nn.BatchNorm2d(c2)
        )


class ConvPatchEmbed(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers
    """
    def __init__(self, img_size=224, patch_size=8, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.proj = nn.Sequential(
            Conv3x3(3, embed_dim // 4, 2),
            nn.GELU(),
            Conv3x3(embed_dim // 4, embed_dim // 2, 2),
            nn.GELU(),
            Conv3x3(embed_dim // 2, embed_dim, 2),
        )

    def forward(self, x: Tensor):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        return x, (Hp, Wp)


class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim

        self.conv1 = nn.Conv2d(dim, out_dim, 3, 1, 1, groups=out_dim)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, out_dim, 3, 1, 1, groups=out_dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv2(self.bn(self.act(self.conv1(x))))
        x = x.reshape(B, C, N).permute(0, 2, 1)
        return x


class ClassAttention(nn.Module):
    """ClassAttention as in CaiT
    """
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qc = q[:, :, 0:1]  # CLS token

        attn_cls = (qc * k).sum(dim=-1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)

        cls_token = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C)
        cls_token = self.proj(cls_token)

        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        return x


class XCA(nn.Module):
    """ Cross-Covariance Attention (XCA) operation where the channels are updated using a weighted
     sum. The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0].transpose(-2, -1), qkv[1].transpose(-2, -1), qkv[2].transpose(-2, -1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class ClassAttentionBlock(nn.Module):
    def __init__(self, dim, heads, eta=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ClassAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))


    def forward(self, x: Tensor) -> Tensor:
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = self.norm2(x)

        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)

        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        x = x_res + x

        return x


class XCABlock(nn.Module):
    def __init__(self, dim, heads, eta=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = XCA(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))
        self.norm3 = nn.LayerNorm(dim)
        self.local_mp = LPI(dim)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim))
        self.gamma2 = nn.Parameter(eta * torch.ones(dim))
        self.gamma3 = nn.Parameter(eta * torch.ones(dim))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma3 * self.local_mp(self.norm3(x), H, W)
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


xcit_settings = {
    'T24': [8, 24, 192, 4],     #[patch_size, layers, embed dim, heads]
    'S24': [8, 24, 384, 8],
    'M24': [8, 24, 512, 8],
    'L24': [8, 24, 768, 16]
}


class XciT(nn.Module):
    def __init__(self, model_name: str = 'S24', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in xcit_settings.keys(), f"XciT model name should be in {list(xcit_settings.keys())}"
        patch_size, layers, embed_dim, heads = xcit_settings[model_name]
        
        self.patch_embed = ConvPatchEmbed(image_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embeder = PositionalEncodingFourier(dim=embed_dim)

        self.blocks = nn.ModuleList([
            XCABlock(embed_dim, heads)
        for _ in range(layers)])

        self.cls_attn_blocks = nn.ModuleList([
            ClassAttentionBlock(embed_dim, heads)
        for _ in range(2)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights(pretrained)


    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'))
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


    def forward(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)   
        pos_encoding = self.pos_embeder(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)  
        x = x + pos_encoding

        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.cls_attn_blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


if __name__ == '__main__':
    model = XciT('S24', 'checkpoints/xcit/xcit_small_24_p8_224_dist.pth')
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
