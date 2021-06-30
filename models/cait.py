import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .layers import MLP, PatchEmbedding


class ClassAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q *= self.scale
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)

        return x


class AttentionTalkingHead(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_l = nn.Linear(heads, heads)
        self.proj_w = nn.Linear(heads, heads)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class LayerScaleCABlock(nn.Module):
    def __init__(self, dim, heads, init_values=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ClassAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor, x_cls: Tensor) -> Tensor:
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.gamma_1 * self.attn(self.norm1(u))
        x_cls = x_cls +  self.gamma_2 * self.mlp(self.norm2(x_cls))

        return x_cls


class LayerScaleBlock(nn.Module):
    def __init__(self, dim, heads, init_values=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AttentionTalkingHead(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.mlp(self.norm2(x))

        return x


cait_settings = {
    'S24': [16, 24, 384, 8, 1e-5],     #[patch_size, number_of_layers, embed dim, heads, init scale]
    'S36': [16, 36, 384, 8, 1e-6],
    'M36': [16, 36, 768, 16, 1e-6]
}


class CaiT(nn.Module):
    def __init__(self, model_name: str = 'S24', pretrained: str = None, num_classes: int = 1000, image_size: int = 384) -> None:
        super().__init__()
        assert model_name in cait_settings.keys(), f"CaiT model name should be in {list(cait_settings.keys())}"
        assert image_size == 384, "Image size for CaiT models must be 384x384"
        patch_size, layers, embed_dim, heads, init_scale = cait_settings[model_name]
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            LayerScaleBlock(embed_dim, heads, init_scale)
        for _ in range(layers)])

        self.blocks_token_only = nn.ModuleList([
            LayerScaleCABlock(embed_dim, heads, init_scale)
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
        x = self.patch_embed(x)             
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x += self.pos_embed
        for blk in self.blocks:
            x = blk(x)

        for blk in self.blocks_token_only:
            cls_tokens = blk(x, cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


if __name__ == '__main__':
    model = CaiT('M36', 'checkpoints/cait/M36_384.pth', image_size=384)
    x = torch.zeros(1, 3, 384, 384)
    y = model(x)
    print(y.shape)
