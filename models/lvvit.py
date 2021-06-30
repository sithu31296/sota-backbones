import torch
import numpy as np
from torch import nn, Tensor

from .layers import MLP


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = np.int(W * cut_rat), np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding with 4 layer convolution
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768) -> None:
        super().__init__()
        new_patch_size = patch_size // 2
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.proj = nn.Conv2d(64, embed_dim, new_patch_size, new_patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.proj(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, heads=12):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale =  self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, self.head_dim * self.num_heads * 3, bias=False)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 3))
        self.skip_lam = 2.0

    def forward(self, x: torch.Tensor):
        x += self.attn(self.norm1(x)) / self.skip_lam
        x += self.mlp(self.norm2(x)) / self.skip_lam

        return x


lvvit_settings = {
    'S': [16, 16, 384, 6],     #[patch_size, number_of_layers, hidden_size, mlp_size, heads]
    'M': [16, 20, 512, 8]
}


class LVViT(nn.Module):
    def __init__(self, model_name: str = 'S', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in lvvit_settings.keys(), f"LV-ViT model name should be in {list(lvvit_settings.keys())}"
        patch_size, layers, embed_dim, heads = lvvit_settings[model_name]
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches+1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, heads)
        for _ in range(layers)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.aux_head = nn.Linear(embed_dim, num_classes)

        self.beta = 1.0

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

        # token level mixtoken augmentation
        if self.training:
            lam = np.random.beta(self.beta, self.beta)
            patch_h, patch_w = x.shape[2], x.shape[3]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            temp_x = x.clone()
            temp_x[:, :, bbx1:bbx2, bby1:bby2] = x.flip(0)[:, :, bbx1:bbx2, bby1:bby2]
            x = temp_x
        
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed
        for blk in self.blocks:
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
    model = LVViT('S', 'checkpoints/lvvit/lvvit_s-26M-224-83.3.pth')
    model.eval()
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    # for g in y[:-1]:
    #     print(g.shape)

