import torch
import torch.nn.functional as F
from torch import nn

from .layers import MLP, PatchEmbedding


class Attention(nn.Module):
    def __init__(self, dim, heads=12):
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: torch.Tensor):
        x += self.attn(self.norm1(x))
        x += self.mlp(self.norm2(x))

        return x


deit_settings = {
    'T': [16, 12, 192, 3],     #[patch_size, number_of_layers, embed_dim, heads]
    'S': [16, 12, 384, 6],
    'B': [16, 12, 768, 12]
}


class DeiT(nn.Module):
    def __init__(self, model_name: str = 'S', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in deit_settings.keys(), f"DeiT model name should be in {list(deit_settings.keys())}"
        patch_size, layers, embed_dim, heads = deit_settings[model_name]

        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 2, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, heads)
        for _ in range(layers)])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

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
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        dist_token = self.dist_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_token, dist_token, x), dim=1)
        x += self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])

        if self.training:
            return x, x_dist
        else:
            return (x + x_dist) / 2



if __name__ == '__main__':
    model = DeiT('S', 'checkpoints/deit/deit_small_distilled_patch16_224.pth')
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    for g in y:
        print(g.shape)