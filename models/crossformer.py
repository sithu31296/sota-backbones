import torch
import math
from torch import nn, Tensor
from .layers import DropPath, MLP


class DynamicPositionEmbed(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(pos_dim),
            nn.ReLU(True),
            nn.Linear(pos_dim, pos_dim)
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(pos_dim),
            nn.ReLU(True),
            nn.Linear(pos_dim, pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(pos_dim),
            nn.ReLU(True),
            nn.Linear(pos_dim, num_heads)
        )

    def forward(self, biases: Tensor) -> Tensor:
        return self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))


class Attention(nn.Module):
    def __init__(self, dim: int, head: int, group_size):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.g = group_size
        self.pos = DynamicPositionEmbed(dim // 4, head)
        # generate mother-set
        position_bias_h, position_bias_w = torch.arange(1 - self.g, self.g), torch.arange(1 - self.g, self.g)
        biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w])).flatten(1).transpose(0, 1).float()
        self.register_buffer("biases", biases)

        # get pair-wise relative position index for each token inside the window
        coords_h, coords_w = torch.arange(self.g), torch.arange(self.g)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])).flatten(1)
        relative_coords = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.g - 1
        relative_coords[:, :, 1] += self.g - 1
        relative_coords[:, :, 0] *= 2 * self.g - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.head, C//self.head).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        pos = self.pos(self.biases)
        # select position bias
        relative_position_bias = pos[self.relative_position_index.view(-1)].view(self.g**2, self.g**2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn += relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, head, group_size=7, lasda_flag=0, dpr=0.):
        super().__init__()
        self.g = group_size
        self.lasda_flag = lasda_flag
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, self.g)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: Tensor) -> Tensor:
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        if H <= self.g: self.lasda_flag = 0

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # group embeddings
        if self.lasda_flag: # 0 for SDA
            x = x.reshape(B, H//self.g, self.g, W//self.g, self.g, C).permute(0, 1, 3, 2, 4, 5)
        else:   # 1 for LDA
            x = x.reshape(B, self.g, H//self.g, self.g, W//self.g, C).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B*H*W // self.g**2, self.g**2, C)

        x = self.attn(x)

        # ungroup embeddings
        x = x.reshape(B, H//self.g, W//self.g, self.g, self.g, C)
        if self.lasda_flag == 0:
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        else:
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, H, W, C)
        x = x.view(B, L, C)

        x = shortcut + self.drop_path(x)
        x += self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, patch_sizes=[2]):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.reductions = nn.ModuleList([
            nn.Conv2d(dim, 2*dim // 2**i if i == len(patch_sizes)-1 else 2*dim // 2**(i+1), ps, 2, (ps - 2)//2)
        for i, ps in enumerate(patch_sizes)])

    def forward(self, x: Tensor) -> Tensor:
        B, L, C = x.shape
        H = W =  int(math.sqrt(L))
        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        x_scales = []
        for red in self.reductions:
            x_scales.append(red(x).flatten(2).transpose(1, 2))
        x = torch.cat(x_scales, dim=2)
        return x


class Stage(nn.Module):
    def __init__(self, dim, depth, head, group_size, dpr=0., downsample=True, patch_size_end=[4]):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, head, group_size, 0 if i % 2 == 0 else 1, dpr[i] if isinstance(dpr, list) else dpr)
        for i in range(depth)])

        if downsample:
            self.downsample = PatchMerging(dim, patch_size_end)
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x 


class PatchEmbed(nn.Module):
    def __init__(self, patch_sizes=[4, 8, 16, 32], dim=768):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(3, dim // 2**i if i == len(patch_sizes)-1 else dim // 2**(i+1), ps, patch_sizes[0], (ps - patch_sizes[0])//2)
        for i, ps in enumerate(patch_sizes)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x_scales = []
        for proj in self.projs:
            x_scales.append(proj(x).flatten(2).transpose(1, 2))
        x = torch.cat(x_scales, dim=2)
        x = self.norm(x)
        return x


cf_settings = {
    'T': [64, [1, 1, 8, 6], [2, 4, 8, 16], 0.1],    # [embed_dim, depths, num_heads, drop_path_rate]
    'S': [96, [2, 2, 6, 2], [3, 6, 12, 24], 0.2],
    'B': [96, [2, 2, 18, 2], [3, 6, 12, 24], 0.3],
    'L': [128, [2, 2, 18, 2], [4, 8, 16, 32], 0.5]
}


class CrossFormer(nn.Module):
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in cf_settings.keys(), f"CrossFormer model name should be in {list(cf_settings.keys())}"
        embed_dim, depths, heads, drop_path_rate = cf_settings[model_name]

        self.patch_embed = PatchEmbed([4, 8, 16, 32], embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]    # stochastic depth decay rule
        self.layers = nn.ModuleList([
            Stage(embed_dim*2**i, depths[i], heads[i], 7, dpr[sum(depths[:i]):sum(depths[:i+1])], True if i < len(depths)-1 else False, [2, 4])
        for i in range(len(depths))])

        self.norm = nn.LayerNorm(embed_dim*2**(len(depths)-1))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim*2**(len(depths)-1), num_classes)
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
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).flatten(1)

        x = self.head(x)
        return x


if __name__ == '__main__':
    model = CrossFormer('S', 'checkpoints/crossformer/crossformer-s.pth')
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)