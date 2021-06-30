import torch
from torch import nn

from .layers import MLP, PatchEmbedding

# no norm layer
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class ResMLPLayer(nn.Module):
    def __init__(self, num_patches, dim, layerscale_init=1e-4) -> None:
        super().__init__()
        self.norm1 = Affine(dim)
        self.attn = nn.Linear(num_patches, num_patches)
        self.norm2 = Affine(dim)
        self.mlp = MLP(dim, int(dim*4))

        self.gamma_1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.gamma_2 = nn.Parameter(layerscale_init * torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x += self.gamma_1 * self.attn(self.norm1(x).swapaxes(1, 2)).swapaxes(1, 2)
        x += self.gamma_2 * self.mlp(self.norm2(x))
        return x


resmlp_settings = {
    'S12': [16, 12, 384, 0.1],    # [patch_size, layers, embed dim, init scale]
    'S24': [16, 24, 384, 1e-5],     
    'S36': [16, 36, 384, 1e-6]
}


class ResMLP(nn.Module):
    def __init__(self, model_name: str = 'S12', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in resmlp_settings.keys(), f"ResMLP model name should be in {list(resmlp_settings.keys())}"
        patch_size, layers, embed_dim, init_scale = resmlp_settings[model_name]
    
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        
        self.blocks = nn.ModuleList([
            ResMLPLayer(self.patch_embed.num_patches, embed_dim, layerscale_init=init_scale)
        for _ in range(layers)])

        self.norm = Affine(embed_dim)
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
                
        
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x)    
        for blk in self.blocks:  
            x = blk(x)    
        x = self.norm(x)
        x = x.mean(dim=1)       
        x = self.head(x)
        return x
        

if __name__ == '__main__':
    model = ResMLP('S36', pretrained='checkpoints/resmlp/resmlp_36_dist.pth')
    x = torch.zeros(1, 3, 224, 224)    
    y = model(x)
    print(y.shape)