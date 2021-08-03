import torch
import math
import numpy as np
from torch import nn, Tensor
from torch.nn import functional as F
from .layers import MLP


def rand_bbox(size, lam, scale=1):
    """get bbox as token labeling"""
    W, H = size[1]//scale, size[2]//scale
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = np.int(W * cut_rat), np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W) 
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape

        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        return x


class OutlookAttention(nn.Module):
    def __init__(self, dim, num_heads, k=3, s=1, p=1):
        super().__init__()
        self.s = s
        self.k = k
        self.p = p
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.v = nn.Linear(dim, dim, bias=False)
        self.attn = nn.Linear(dim, k**4 * num_heads)

        self.proj = nn.Linear(dim, dim)

        self.unfold = nn.Unfold(k, padding=p, stride=s)
        self.pool = nn.AvgPool2d(s, s, ceil_mode=True)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)   # B, C, H, W

        h, w = math.ceil(H / self.s), math.ceil(W / self.s)
        v = self.unfold(v).reshape(B, self.num_heads, C//self.num_heads, self.k*self.k, h*w).permute(0, 1, 4, 3, 2) # B, H, N, kxk, C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(B, h*w, self.num_heads, self.k*self.k, self.k*self.k).permute(0, 2, 1, 3, 4) # B, H, N, kxk, kxk
        attn *= self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C*self.k*self.k, h*w)
        x = F.fold(x, (H, W), self.k, padding=self.p, stride=self.s)

        x = self.proj(x.permute(0, 2, 3, 1))
        return x


class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim*2, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, C//self.num_heads)
        q *= self.scale

        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        cls_embed = self.proj(cls_embed)
        return cls_embed


class OutlookBlock(nn.Module):
    def __init__(self, dim, num_heads, k, s, p):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = OutlookAttention(dim, num_heads, k, s, p)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 3))

    def forward(self, x: Tensor) -> Tensor:
        x += self.attn(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*3))

    def forward(self, x: Tensor) -> Tensor:
        x += self.attn(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x


class CABlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ClassAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*3))

    def forward(self, x: Tensor) -> Tensor:
        cls_embed = x[:, :1]
        cls_embed += self.attn(self.norm1(x))
        cls_embed += self.mlp(self.norm2(cls_embed))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    """
    def __init__(self, hidden_dim=64, embed_dim=384, s=1, patch_size=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 7, s, 3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True)
        )
        self.proj = nn.Conv2d(hidden_dim, embed_dim, patch_size//s, patch_size//s)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.proj(x)        # B, C, H, W
        return x


class Downsample(nn.Module):
    def __init__(self, c1, c2, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


volo_settings = {
    'D1': [[192, 384, 384, 384], [4, 4, 8, 2], [6, 12, 12, 12]],     #[embed_dim, depths, heads]
    'D2': [[256, 512, 512, 512], [6, 4, 10, 4], [8, 16, 16, 16]],
    'D3': [[256, 512, 512, 512], [8, 8, 16, 4], [8, 16, 16, 16]],
    'D4': [[384, 768, 768, 768], [8, 8, 16, 4], [12, 16, 16, 16]]
}


class VOLO(nn.Module):
    def __init__(self, model_name: str = 'D1', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in volo_settings.keys(), f"VOLO model name should be in {list(volo_settings.keys())}"
        embed_dims, depths, heads = volo_settings[model_name]
        patch_size = 8
        self.pooling_scale = 2

        self.patch_embed = PatchEmbed(64, embed_dims[0], 2, patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, image_size//patch_size//self.pooling_scale, image_size//patch_size//self.pooling_scale, embed_dims[-1]))

        network = []

        for i in range(len(depths)):
            if i == 0:
                stage = nn.Sequential(*[
                    OutlookBlock(embed_dims[i], heads[i], 3, 2, 1)
                for _ in range(depths[i])])
                network.append(stage)
                network.append(Downsample(embed_dims[i], embed_dims[i+1], 2))
            else:
                stage = nn.Sequential(*[
                    AttentionBlock(embed_dims[i], heads[i])
                for _ in range(depths[i])])
                network.append(stage)

        self.network = nn.ModuleList(network)

        self.post_network = nn.ModuleList([
            CABlock(embed_dims[-1], heads[-1])
        for _ in range(2)])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[-1]))

        self.beta = 1.0
        self.aux_head = nn.Linear(embed_dims[-1], num_classes)

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

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


    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)  
        x = x.permute(0, 2, 3, 1)   # B, H, W, C          
        
        # token level mixtoken augmentation
        if self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[1]//self.pooling_scale, x.shape[2]//self.pooling_scale
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, self.pooling_scale)
            sbbx1, sbby1, sbbx2, sbby2 = self.pooling_scale*bbx1, self.pooling_scale*bby1, self.pooling_scale*bbx2, self.pooling_scale*bby2
            temp_x = x.clone()
            temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
            x = temp_x

        for i, blk in enumerate(self.network):
            if i == 2:  # add positional encoding after outlookblock
                x += self.pos_embed
            x = blk(x)
        
        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        for blk in self.post_network:
            x = blk(x)

        x = self.norm(x)
        x_cls = self.head(x[:, 0])
        x_aux = self.aux_head(x[:, 1:])

        # recover the mixed token
        if self.training:
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])
            temp_x  = x_aux.clone()
            temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
            x_aux = temp_x
            x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)
        return x_cls + 0.5 * x_aux.max(1)[0]


if __name__ == '__main__':
    model = VOLO('D2', 'checkpoints/volo/d2_224_85.2.pth.tar')
    x = torch.zeros(1, 3, 224, 224)
    y, y2, _ = model(x)
    print(y.shape, y2.shape)