import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim, mlp_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout()
        )

    def forward(self, x):
        return self.net(x)


class MixerLayer(nn.Module):
    def __init__(self, hidden_dim, num_patches, channels_dim, tokens_dim) -> None:
        super().__init__()
        # Token Mixing MLP
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp1 = MLP(num_patches, tokens_dim)

        # Channels Mixing MLP
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp2 = MLP(hidden_dim, channels_dim)

    def forward(self, x: torch.Tensor):
        y = self.norm1(x)
        y = y.swapaxes(1, 2)
        y = self.mlp1(y)
        y = y.swapaxes(1, 2)
        x += y
        y = self.norm2(x)
        y = self.mlp2(y)
        x += y
        return x


mixer_settings = {
    'S/16': [16, 8, 512, 2048, 256],    #[patch_size, number_of_layers, hidden_dim, channels_dim, tokens_dim]
    'B/16': [16, 12, 768, 3072, 384],     
    'L/16': [16, 24, 1024, 4096, 512]
}


class MLPMixer(nn.Module):
    def __init__(self, model_name: str = 'B/16', pretrained: str = None, num_classes: int = 1000, image_size: int = 224) -> None:
        super().__init__()
        assert model_name in ['S/16', 'B/16', 'L/16'], "Mixer model name should be 'S/16' or 'B/16' or 'L/16'"
        
        patch_size, num_layers, hidden_dim, channels_dim, tokens_dim = mixer_settings[model_name]
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'

        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(3, hidden_dim, patch_size, patch_size)
        self.mixer_layers = nn.ModuleList([])

        for _ in range(num_layers):
            self.mixer_layers.append(MixerLayer(hidden_dim, num_patches, channels_dim, tokens_dim))

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self._init_weights(pretrained)


    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu'))
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        
    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)             # b x hidden_dim x 14 x 14
        x = x.view(x.shape[0], -1, x.shape[1])  # b x (14*14) x hidden_dim

        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x
        

if __name__ == '__main__':
    x = torch.zeros(1, 3, 224, 224)

    model = MLPMixer(
        'B/16'
    )
    y = model(x)

    print(y.shape)