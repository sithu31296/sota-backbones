import torch
import math
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
from .layers import MLP, DropPath, trunc_normal_


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, head=8):
        super().__init__()
        self.scale = (dim // head) ** -0.5
        self.head = head
        self.resolution = resolution

        if idx == -1:
            self.H_sp, self.W_sp = resolution, resolution
        elif idx == 0:
            self.H_sp, self.W_sp = resolution, split_size
        elif idx == 1:
            self.W_sp, self.H_sp = resolution, split_size

        self.get_v = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def im2cswin(self, x: Tensor) -> Tensor:
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))
        x = rearrange(x, 'b c (h hsp) (w wsp) -> (b h w) (hsp wsp) c', hsp=self.H_sp, wsp=self.W_sp)
        x = rearrange(x, 'b n (H c) -> b H n c', H=self.head)
        return x

    def get_lepe(self, x: Tensor):
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))
        x = rearrange(x, 'b c (h hsp) (w wsp) -> (b h w) c hsp wsp', hsp=self.H_sp, wsp=self.W_sp)
        lepe = self.get_v(x)
        lepe = rearrange(lepe, 'b (H c) h w -> b H (h w) c', H=self.head)
        x = rearrange(x, 'b (H c) h w -> b H (h w) c', H=self.head)
        return x, lepe

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor: 
        B = q.shape[0]
        q = self.im2cswin(q) * self.scale
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v) + lepe
        x = rearrange(x, 'h b w c -> h w (b c)')
        x = x.view(B, self.resolution//self.H_sp, self.resolution//self.W_sp, self.H_sp, self.W_sp, -1)
        x = rearrange(x, 'b r1 h w r2 c -> b (r1 h w r2) c')
        return x


class CSWinBlock(nn.Module):
    def __init__(self, dim, resolution, head, split_size=7, last_stage=False, dpr=0.):
        super().__init__()
        self.resolution = resolution
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim*3, bias=True)

        if last_stage:
            branch_num = 1
            self.attns = nn.ModuleList([
                LePEAttention(dim, resolution, -1, split_size, head)
            for _ in range(branch_num)])
        else:
            branch_num = 2
            self.attns = nn.ModuleList([
                LePEAttention(dim//2, resolution, i, split_size, head//2)
            for i in range(branch_num)])

        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()

        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor) -> Tensor:
        C = x.shape[-1]
        qkv = self.qkv(self.norm1(x))
        qkv = rearrange(qkv, 'b n (l c) -> l b n c', c=C)
        
        if len(self.attns) > 1:
            x1 = self.attns[0](*qkv[..., :C//2])
            x2 = self.attns[1](*qkv[..., C//2:])
            x = torch.cat([x1, x2], dim=2)
        else:
            x = self.attns[0](*qkv)
        
        x = x + self.drop_path(self.proj(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MergeBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, 2, 1)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        H = W = int(math.sqrt(x.shape[1]))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x


cswin_settings = {
    'T': [64, [1, 2, 21, 1], [2, 4, 8, 16], 0.1],     #[embed_dim, depths, heads]
    'S': [64, [2, 4, 32, 2], [2, 4, 8, 16], 0.2],
    'B': [96, [2, 4, 32, 2], [4, 8, 16, 32], 0.3],
    'L': [144, [2, 4, 32, 2], [6, 12, 24, 24], 0.5]
}


class CSWin(nn.Module):
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in cswin_settings.keys(), f"CSWin Transformer model name should be in {list(cswin_settings.keys())}"
        embed_dim, depths, heads, drop_path_rate = cswin_settings[model_name]

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(3, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stage1 = nn.ModuleList([
            CSWinBlock(embed_dim, image_size//4, heads[0], 1, dpr=dpr[i])
        for i in range(depths[0])])

        self.merge1 = MergeBlock(embed_dim, embed_dim*2)
        embed_dim *= 2

        self.stage2 = nn.ModuleList([
            CSWinBlock(embed_dim, image_size//8, heads[1], 2, dpr=dpr[sum(depths[:1])+i])
        for i in range(depths[1])])

        self.merge2 = MergeBlock(embed_dim, embed_dim*2)
        embed_dim *= 2

        self.stage3 = nn.ModuleList([
            CSWinBlock(embed_dim, image_size//16, heads[2], 7, dpr=dpr[sum(depths[:2])+i])
        for i in range(depths[2])])

        self.merge3 = MergeBlock(embed_dim, embed_dim*2)
        embed_dim *= 2

        self.stage4 = nn.ModuleList([
            CSWinBlock(embed_dim, image_size//32, heads[3], 7, True, dpr[sum(depths[:-1])+i])
        for i in range(depths[3])])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.head.weight, std=.02)
        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            try:
                print(f"Loading imagenet pretrained weights from {pretrained}")
            except RuntimeError:
                pretrained_dict = torch.load(pretrained, map_location='cpu')['state_dict_ema']
                pretrained_dict.popitem()   # remove bias
                pretrained_dict.popitem()   # remove weight
                self.load_state_dict(pretrained_dict, strict=False)
            finally:
                print(f"Loaded imagenet pretrained from {pretrained}")
        else:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None: 
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1_conv_embed(x)  

        for blk in self.stage1:
            x = blk(x)
        
        for pre, blks in zip([self.merge1, self.merge2, self.merge3], [self.stage2, self.stage3, self.stage4]):
            x = pre(x)
            for blk in blks:
                x = blk(x)

        x = self.norm(x).mean(dim=1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = CSWin('T', 'checkpoints/cswin/cswin_tiny_224.pth', image_size=224)
    x = torch.zeros(2, 3, 224, 224)
    y = model(x)
    print(y.shape)