import torch
from torch import nn, Tensor
from .layers import MLP, DropPath


class CMLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class CBlock(nn.Module):
    def __init__(self, dim, dpr=0.):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = CMLP(dim, int(dim*4))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x))))) 
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, dpr=0.) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, N, H, W)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_ch=3, embed_dim=768) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


uniformer_settings = {
    'S': [3, 4, 8, 3],       # [depth]
    'B': [5, 8, 20, 7]
}


class UniFormer(nn.Module):     
    def __init__(self, model_name: str = 'S', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in uniformer_settings.keys(), f"UniFormer model name should be in {list(uniformer_settings.keys())}"
        depth = uniformer_settings[model_name]

        head_dim = 64
        drop_path_rate = 0.
        embed_dims = [64, 128, 320, 512]
    
        for i in range(4):
            self.add_module(f"patch_embed{i+1}", PatchEmbed(4 if i == 0 else 2, 3 if i == 0 else embed_dims[i-1], embed_dims[i]))

        self.pos_drop = nn.Dropout(0.)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        num_heads = [dim // head_dim for dim in embed_dims]

        self.blocks1 = nn.ModuleList([
            CBlock(embed_dims[0], dpr[i])
        for i in range(depth[0])])

        self.blocks2 = nn.ModuleList([
            CBlock(embed_dims[1], dpr[i+depth[0]])
        for i in range(depth[1])])

        self.blocks3 = nn.ModuleList([
            SABlock(embed_dims[2], num_heads[2], dpr[i+depth[0]+depth[1]])
        for i in range(depth[2])])

        self.blocks4 = nn.ModuleList([
            SABlock(embed_dims[3], num_heads[3], dpr[i+depth[0]+depth[1]+depth[2]])
        for i in range(depth[3])])

        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

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
                
        
    def forward(self, x: torch.Tensor):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)

        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x.flatten(2).mean(2))
        return x

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    model = UniFormer('S', 'C:\\Users\\sithu\\Documents\\weights\\backbones\\uniformer\\uniformer_small_in1k.pth')
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    # flops = FlopCountAnalysis(model, x)
    # print(flop_count_table(flops))