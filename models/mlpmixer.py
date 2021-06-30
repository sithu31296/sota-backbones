import torch
from torch import nn

from .layers import MLP, PatchEmbedding


class MixerLayer(nn.Module):
    def __init__(self, num_patches, embed_dim, tokens_dim) -> None:
        super().__init__()
        # Token Mixing MLP
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp1 = MLP(num_patches, tokens_dim)

        # Channels Mixing MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp2 = MLP(embed_dim, int(embed_dim * 4))

    def forward(self, x: torch.Tensor):
        x += self.mlp1(self.norm1(x).swapaxes(1, 2)).swapaxes(1, 2)
        x += self.mlp2(self.norm2(x))
        return x


mixer_settings = {
    'S': [16, 8, 512, 256],    #[patch_size, number_of_layers, embed_dim, tokens_dim]
    'B': [16, 12, 768, 384],     
    'L': [16, 24, 1024, 512]
}


class MLPMixer(nn.Module):
    def __init__(self, model_name: str = 'B', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in mixer_settings.keys(), f"Mixer model name should be in {list(mixer_settings.keys())}"
        patch_size, num_layers, embed_dim, tokens_dim = mixer_settings[model_name]
    
        self.patch_embedding = PatchEmbedding(image_size, patch_size, embed_dim)
        
        self.mixer_layers = nn.Sequential(*[
            MixerLayer(self.patch_embedding.num_patches, embed_dim, tokens_dim)
        for _ in range(num_layers)])

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
                
        
    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)          
        x = self.mixer_layers(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
        

if __name__ == '__main__':
    model = MLPMixer('B')
    x = torch.zeros(1, 3, 224, 224)    
    y = model(x)
    print(y.shape)