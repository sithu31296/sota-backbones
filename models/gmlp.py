import torch
from torch import nn

from .layers import PatchEmbedding


class SpatialGatingUnit(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim // 2)
        self.proj = nn.Linear(num_patches, num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v).transpose(-1, -2)
        v = self.proj(v)
        return u * v.transpose(-1, -2)


class GatedMLP(nn.Module):
    def __init__(self, num_patches, dim):
        super().__init__()
        mlp_dim = int(dim * 6)
        self.fc1 = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
        )
        self.gate = SpatialGatingUnit(num_patches, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim // 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.gate(self.fc1(x)))


class SpatialGatingLayer(nn.Module):
    def __init__(self, num_patches, dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = GatedMLP(num_patches, dim)

    def forward(self, x: torch.Tensor):
        x += self.mlp(self.norm(x))
        return x


gmlp_settings = {
    'Ti': [16, 30, 128],    #[patch_size, number_of_layers, embed_dim]
    'S': [16, 30, 256],     
    'B': [16, 30, 512]
}


class gMLP(nn.Module):
    def __init__(self, model_name: str = 'S', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in gmlp_settings.keys(), f"gMLP model name should be in {list(gmlp_settings.keys())}"
        patch_size, num_layers, dim = gmlp_settings[model_name]
    
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim)
        
        self.layers = nn.Sequential(*[
            SpatialGatingLayer(self.patch_embedding.num_patches, dim)
        for _ in range(num_layers)])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

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
        x = self.patch_embedding(x)          
        x = self.layers(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
        

if __name__ == '__main__':
    model = gMLP('S')
    x = torch.zeros(1, 3, 224, 224)    
    y = model(x)
    print(y.shape)