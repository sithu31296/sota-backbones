import torch
from torch import nn, Tensor
from layers import DropPath
from einops import rearrange


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = nn.ReLU6(True)
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, head: int, window_size: int, shuffle: bool = False):
        super().__init__()
        self.head = head
        self.shuffle = shuffle
        self.window_size = window_size
        self.scale = (dim // head) ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2*window_size-1) * (2*window_size-1), head))

        # get pair-wise relative position index for each token inside the window
        coords = torch.stack(torch.meshgrid([torch.arange(window_size), torch.arange(window_size)])).flatten(1)
        relative_coords = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.to_qkv = nn.Conv2d(dim, dim*3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape    # [1, 96, 56, 56]
        qkv = self.to_qkv(x)    # [1, 288, 56, 56]

        if self.shuffle:
            q, k, v = rearrange(qkv, 'b (qkv h d) (ws1 hh) (ws2 ww) -> qkv (b hh ww) h (ws1 ws2) d', h=self.head, qkv=3, ws1=self.window_size, ws2=self.window_size)
        else:
            q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.head, qkv=3, ws1=self.window_size, ws2=self.window_size)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # select position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size**2, self.window_size**2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn += relative_position_bias.unsqueeze(0)     # [1, 3, 49, 49]
        attn = attn.softmax(dim=-1)

        x = attn @ v

        if self.shuffle:
            x = rearrange(x, '(b hh ww) h (ws1 ws2) d -> b (h d) (ws1 hh) (ws2 ww)', h=self.head, b=B, hh=H//self.window_size, ws1=self.window_size, ws2=self.window_size)
        else:
            x = rearrange(x, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.head, b=B, hh=H//self.window_size, ws1=self.window_size, ws2=self.window_size)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, out_dim, head, window_size=7, shuffle=False, dpr=0.):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, head, window_size, shuffle)
        self.local = nn.Conv2d(dim, dim, window_size, 1, window_size//2, groups=dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim, int(dim*4), out_dim)
        self.norm3 = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        x += self.drop_path(self.attn(self.norm1(x)))
        x += self.local(self.norm2(x))
        x += self.drop_path(self.mlp(self.norm3(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, 0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, depth, head, window_size=1, dpr=0.):
        super().__init__()
        if dim != out_dim:
            self.patch_partition = PatchMerging(dim, out_dim)
        else:
            self.patch_partition = None

        self.layers = nn.ModuleList([
            nn.ModuleList([
                Block(out_dim, out_dim, head, window_size, shuffle=False, dpr=dpr),
                Block(out_dim, out_dim, head, window_size, shuffle=True, dpr=dpr)
            ])
        for _ in range(depth // 2)])

    def forward(self, x: Tensor) -> Tensor:
        if self.patch_partition:
            x = self.patch_partition(x)

        for blk, shifted_blk in self.layers:
            x = blk(x)
            x = shifted_blk(x)
        return x 


class PatchEmbed(nn.Module):
    def __init__(self, c1=32, c2=48):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c1, 3, 2, 1),
            nn.BatchNorm2d(c1),
            nn.ReLU6(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU6(True)
        )
        self.conv3 = nn.Conv2d(c2, c2, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv3(self.conv2(self.conv1(x)))


shuffle_settings = {
    'T': [96, [2, 2, 6, 2], [3, 6, 12, 24], 0.1],    # [embed_dim, depths, num_heads, drop_path_rate]
    'S': [96, [2, 2, 18, 2], [3, 6, 12, 24], 0.3],
    'B': [128, [2, 2, 18, 2], [4, 8, 16, 32], 0.5]
}


class Shuffle(nn.Module):
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in shuffle_settings.keys(), f"Shuffle Transformer model name should be in {list(shuffle_settings.keys())}"
        embed_dim, depths, heads, drop_path_rate = shuffle_settings[model_name]
        dims = [i*32 for i in heads]

        self.to_token = PatchEmbed(32, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]    # stochastic depth decay rule
        self.stage1 = BasicLayer(embed_dim, dims[0], depths[0], heads[0], 7, dpr[0])
        self.stage2 = BasicLayer(dims[0], dims[1], depths[1], heads[1], 7, dpr[1])
        self.stage3 = BasicLayer(dims[1], dims[2], depths[2], heads[2], 7, dpr[2])
        self.stage4 = BasicLayer(dims[2], dims[3], depths[3], heads[3], 7, dpr[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dims[-1], num_classes)
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
        x = self.to_token(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x).flatten(1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = Shuffle('S', 'checkpoints/shuffle/shuffle-s.pth')
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)