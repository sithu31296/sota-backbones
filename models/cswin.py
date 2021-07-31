import torch
import math
from torch import nn, Tensor
from .layers import MLP


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, num_heads=8):
        super().__init__()
        self.scale = (dim // num_heads) ** -0.5
        self.num_heads = num_heads
        self.resolution = resolution

        if idx == -1:
            self.H_sp, self.W_sp = resolution, resolution
        elif idx == 0:
            self.H_sp, self.W_sp = resolution, split_size
        elif idx == 1:
            self.W_sp, self.H_sp = resolution, split_size

        self.get_v = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def im2cswin(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = x.view(B, C, H//self.H_sp, self.H_sp, W//self.W_sp, self.W_sp)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.H_sp*self.W_sp, C)
        x = x.reshape(-1, self.H_sp*self.W_sp, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x: Tensor):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = x.view(B, C, H//self.H_sp, self.H_sp, W//self.W_sp, self.W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, self.H_sp, self.W_sp)

        lepe = self.get_v(x)
        lepe = lepe.reshape(-1, self.num_heads, C//self.num_heads, self.H_sp*self.W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C//self.num_heads, self.H_sp*self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor: 
        B, _, C = q.shape

        q = self.im2cswin(q) * self.scale
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp*self.W_sp, C)

        x = x.view(B, self.resolution//self.H_sp, self.resolution//self.W_sp, self.H_sp, self.W_sp, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, self.resolution, self.resolution, -1)
        x = x.view(B, -1, C)

        return x


class CSWinBlock(nn.Module):
    def __init__(self, dim, resolution, num_heads, split_size=7, last_stage=False):
        super().__init__()
        self.resolution = resolution
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim*3, bias=True)

        if last_stage:
            branch_num = 1
            self.attns = nn.ModuleList([
                LePEAttention(dim, resolution, -1, split_size, num_heads)
            for _ in range(branch_num)])
        else:
            branch_num = 2
            self.attns = nn.ModuleList([
                LePEAttention(dim//2, resolution, i, split_size, num_heads//2)
            for i in range(branch_num)])

        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))

    def forward(self, x: Tensor) -> Tensor:
        B, _, C = x.shape
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        if len(self.attns) > 1:
            x1 = self.attns[0](*qkv[..., :C//2])
            x2 = self.attns[0](*qkv[..., C//2:])
            attend_x = torch.cat([x1, x2], dim=2)
        else:
            attend_x = self.attns[0](*qkv)
        
        x += self.proj(attend_x)
        x += self.mlp(self.norm2(x))
        return x


class MergeBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, 2, 1)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        x = x.view(B, C*2, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with overlapping
    """
    def __init__(self, embed_dim=768, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, 7, patch_size, 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.proj(x)                   # b x hidden_dim x 14 x 14
        x = x.flatten(2).swapaxes(1, 2)     # b x (14*14) x hidden_dim
        x = self.norm(x)
        return x


cswin_settings = {
    'T': [64, [1, 2, 21, 1], [2, 4, 8, 16]],     #[embed_dim, depths, heads]
    'S': [64, [2, 4, 32, 2], [2, 4, 8, 16]],
    'B': [96, [2, 4, 32, 2], [4, 8, 16, 32]],
    'L': [144, [2, 4, 32, 2], [6, 12, 24, 24]]
}


class CSWin(nn.Module):
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in cswin_settings.keys(), f"CSWin Transformer model name should be in {list(cswin_settings.keys())}"
        embed_dim, depths, heads = cswin_settings[model_name]

        self.patch_embed = PatchEmbed(embed_dim, 4)

        self.stage1 = nn.ModuleList([
            CSWinBlock(embed_dim, image_size//4, heads[0], 1)
        for _ in range(depths[0])])

        self.merge1 = MergeBlock(embed_dim, embed_dim*2)
        embed_dim *= 2

        self.stage2 = nn.ModuleList([
            CSWinBlock(embed_dim, image_size//8, heads[1], 2)
        for _ in range(depths[1])])

        self.merge2 = MergeBlock(embed_dim, embed_dim*2)
        embed_dim *= 2

        self.stage3 = nn.ModuleList([
            CSWinBlock(embed_dim, image_size//16, heads[2], 7)
        for _ in range(depths[2])])

        self.merge3 = MergeBlock(embed_dim, embed_dim*2)
        embed_dim *= 2

        self.stage4 = nn.ModuleList([
            CSWinBlock(embed_dim, image_size//32, heads[3], 7, last_stage=True)
        for _ in range(depths[3])])

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


    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)            
        
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
    model = CSWin('B', 'checkpoints/cswin/cswin_b.pth')
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)