import torch
from torch import nn, Tensor
from .layers import DropPath


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LearnedAggreationLayer(nn.Module):
    def __init__(self, dim, head=1):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.head, C//self.head).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.head, C//self.head).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.head, C//self.head).permute(0, 2, 1, 3)

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        
        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        
        return x_cls


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, rd_ratio=0.25) -> None:
        super().__init__()
        rd_channels = round(in_chs * rd_ratio)
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1)
        self.act1 = nn.ReLU(True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.act1(self.conv_reduce(x_se))
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class SEBlock(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.qkv_pos = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.GELU(),
            SqueezeExcite(dim),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(-1, -2)
        x = x.reshape(B, C, H, W)
        x = self.qkv_pos(x)
        x = x.reshape(B, C, N)
        x = x.transpose(-1, -2)
        return x


class Block(nn.Module):
    def __init__(self, dim, dpr=0., init_values=1e-6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SEBlock(dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        return x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))


class BlockToken(nn.Module):
    def __init__(self, dim, head, dpr=0., init_values=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LearnedAggreationLayer(dim, head)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*3))
        
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x: Tensor, x_cls: Tensor) -> Tensor:
        u = torch.cat([x_cls, x], dim=1)
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls


class Stem(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Sequential(nn.Conv2d(3, dim//8, 3, 2, 1, bias=False)),
            nn.GELU(),
            nn.Sequential(nn.Conv2d(dim//8, dim//4, 3, 2, 1, bias=False)),
            nn.GELU(),
            nn.Sequential(nn.Conv2d(dim//4, dim//2, 3, 2, 1, bias=False)),
            nn.GELU(),
            nn.Sequential(nn.Conv2d(dim//2, dim, 3, 2, 1, bias=False))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


patchconvnet_settings = {
    'S60': [384, 60, 0.],    # [embed_dim, depth, drop_path_rate]
    'S120': [384, 120, 0.],
    'B60': [768, 60, 0.]
}


class PatchConvnet(nn.Module):
    def __init__(self, model_name: str = 'S60', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in patchconvnet_settings.keys(), f"PatchConvnet model name should be in {list(patchconvnet_settings.keys())}"
        embed_dim, depth, drop_path_rate = patchconvnet_settings[model_name]

        self.patch_embed = Stem(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [drop_path_rate for _ in range(depth)]    # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, dpr[i])
        for i in range(depth)])

        self.blocks_token_only = nn.ModuleList([
            BlockToken(embed_dim, 1, 0)
        for i in range(1)])

        self.norm = nn.LayerNorm(embed_dim)
        self.total_len = 1 + depth

        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            try:
                self.load_state_dict(torch.load(pretrained, map_location='cpu'))
            except RuntimeError:
                pretrained_dict = torch.load(pretrained, map_location='cpu')
                pretrained_dict.popitem()   # remove bias
                pretrained_dict.popitem()   # remove weight
                self.load_state_dict(pretrained_dict, strict=False)
            finally:
                print(f"Loaded imagenet pretrained from {pretrained}")
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
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        for blk in self.blocks:
            x = blk(x)

        for blk in self.blocks_token_only:
            cls_tokens = blk(x, cls_tokens)

        x = torch.cat([cls_tokens, x], dim=1)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x


if __name__ == '__main__':
    model = PatchConvnet('B60', 'C:\\Users\\sithu\\Documents\\weights\\backbones\\patchconvnet\\b60_224_1k.pth')
    x = torch.zeros(1, 3, 224, 224)
    y = model(x)
    print(y.shape)