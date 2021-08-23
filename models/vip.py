import torch
from torch import nn, Tensor
from .layers import MLP, DropPath


class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, segment_dim=8):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(dim, dim, bias=False)
        self.mlp_h = nn.Linear(dim, dim, bias=False)
        self.mlp_w = nn.Linear(dim, dim, bias=False)

        self.reweight = MLP(dim, dim // 4, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        S = C // self.segment_dim

        h = x.reshape(B, H, W, self.segment_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.segment_dim, W, H * S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        w = x.reshape(B, H, W, self.segment_dim, S).permute(0, 1, 3, 2, 4).reshape(B, H, self.segment_dim, W * S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, W, S).permute(0, 1, 3, 2, 4).reshape(B, H, W, C)

        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        return self.proj(x)


class PermutatorBlock(nn.Module):
    def __init__(self, dim, segment_dim, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WeightedPermuteMLP(dim, segment_dim)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 3))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding
    """
    def __init__(self, patch_size=16, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                   # b x hidden_dim x 14 x 14
        x = x.permute(0, 2, 3, 1)           
        return x


class Downsample(nn.Module):
    def __init__(self, c1, c2, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, patch_size)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


vip_settings = {
    'S': [7, [4, 3, 8, 3], [32, 16, 16, 16], [192, 384, 384, 384]],    #[patch_size, layers, segment dim, embed dims]
    'M': [7, [4, 3, 14, 3], [32, 32, 16, 16], [256, 256, 512, 512]],     
    'L': [7, [8, 8, 16, 4], [32, 16, 16, 16], [256, 512, 512, 512]]
}


class ViP(nn.Module):
    def __init__(self, model_name: str = 'S', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in vip_settings.keys(), f"Vision Permutator model name should be in {list(vip_settings.keys())}"
        patch_size, layers, segment_dims, embed_dims = vip_settings[model_name]
    
        self.patch_embed = PatchEmbedding(patch_size, embed_dims[0])

        network = []

        for i in range(len(layers)):
            stage = nn.Sequential(*[
                PermutatorBlock(embed_dims[i], segment_dims[i])
            for _ in range(layers[i])])
            
            network.append(stage)

            if (i != len(layers) - 1) and (embed_dims[i] != embed_dims[i+1]):
                network.append(Downsample(embed_dims[i], embed_dims[i+1], 2))

        self.network = nn.ModuleList(network)
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
                
        
    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)          
        
        for blk in self.network:
            x = blk(x)

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
        

if __name__ == '__main__':
    model = ViP('S', pretrained='checkpoints/vip/vip_s7.pth')
    x = torch.zeros(1, 3, 224, 224)    
    y = model(x)
    print(y.shape)