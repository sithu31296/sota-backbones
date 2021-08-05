import torch
from torch import nn, Tensor
from .layers import DropPath, MLP


class WindowAttention(nn.Module):
    def __init__(self, dim: int, head: int, window_size: int):
        super().__init__()
        self.head = head
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

        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.head, C//self.head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        # select position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size**2, self.window_size**2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn += relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B//nW, nW, self.head, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.head, N, N)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, resolution, dim, head, window_size=7, shift_size=0, dpr=0.):
        super().__init__()
        self.resolution = resolution
        self.shift_size = shift_size
        self.window_size = window_size

        if min(self.resolution) <= self.window_size:
            # if window size is larger than input resolution, don't parition windows
            self.shift_size = 0
            self.window_size = min(self.resolution)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, head, self.window_size)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = img_mask.view(1, H//self.window_size, self.window_size, W//self.window_size, self.window_size, 1)
            mask_windows = img_mask.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, 1).view(-1, self.window_size**2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: Tensor) -> Tensor:
        B, _, C = x.shape
        H, W = self.resolution

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = shifted_x.view(B, H//self.window_size, self.window_size, W//self.window_size, self.window_size, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = attn_windows.view(B, H//self.window_size, W//self.window_size, self.window_size, self.window_size, -1)
        shifted_x = shifted_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H*W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x += self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, resolution, dim):
        super().__init__()
        self.resolution = resolution
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = nn.LayerNorm(4*dim)

    def forward(self, x: Tensor) -> Tensor:
        B, _, C = x.shape
        H, W = self.resolution
        x = x.view(B, H, W, C)

        x0 = x[:, ::2, ::2, :]
        x1 = x[:, 1::2, ::2, :]
        x2 = x[:, ::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4*C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, resolution, dim, depth, head, window_size, dpr=0., downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(resolution, dim, head, window_size, 0 if i % 2 == 0 else window_size // 2, dpr[i] if isinstance(dpr, list) else dpr)
        for i in range(depth)])

        if downsample:
            self.downsample = PatchMerging(resolution, dim)
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x 


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, dim=768):
        super().__init__()
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.proj = nn.Conv2d(3, dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


swin_settings = {
    'T': [96, [2, 2, 6, 2], [3, 6, 12, 24], 0.2],    # [embed_dim, depths, num_heads, drop_path_rate]
    'S': [96, [2, 2, 18, 2], [3, 6, 12, 24], 0.3],
    'B': [128, [2, 2, 18, 2], [4, 8, 16, 32], 0.5],
    'L': [192, [2, 2, 18, 2], [6, 12, 24, 48], 0.5]
}


class Swin(nn.Module):
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in swin_settings.keys(), f"Swin Transformer model name should be in {list(swin_settings.keys())}"
        embed_dim, depths, heads, drop_path_rate = swin_settings[model_name]

        self.patch_embed = PatchEmbed(image_size, 4, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]    # stochastic depth decay rule
        self.layers = nn.ModuleList([
            BasicLayer((self.patch_embed.resolution[0]//2**i, self.patch_embed.resolution[1]//2**i), embed_dim*2**i, depths[i], heads[i], 7, dpr[sum(depths[:i]):sum(depths[:i+1])], True if i < len(depths)-1 else False)
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
    model = Swin('S', 'checkpoints/swin/swin_small_patch4_window7_224.pth', image_size=224)
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)