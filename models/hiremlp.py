import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .layers import DropPath


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class HireAttn(nn.Module):
    def __init__(self, dim, pixel=2, step=1):
        super().__init__()
        self.pixel = pixel
        self.step = step
        
        self.mlp_h1 = nn.Conv2d(dim*pixel, dim//2, 1, bias=False)
        self.mlp_h1_norm = nn.BatchNorm2d(dim//2)
        self.mlp_h2 = nn.Conv2d(dim//2, dim*pixel, 1)
        self.mlp_w1 = nn.Conv2d(dim*pixel, dim//2, 1, bias=False)
        self.mlp_w1_norm = nn.BatchNorm2d(dim//2)
        self.mlp_w2 = nn.Conv2d(dim//2, dim*pixel, 1)
        self.mlp_c = nn.Conv2d(dim, dim, 1)

        self.act = nn.ReLU()
        self.reweight = MLP(dim, dim//4, dim*3)

        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        Setting of F.pad: (left, right, top, bottom)
        """
        B, C, H, W = x.shape

        pad_h, pad_w = (self.pixel - H % self.pixel) % self.pixel, (self.pixel - W % self.pixel) % self.pixel
        h, w = x.clone(), x.clone()

        if self.step:
            h = torch.roll(h, self.step, -2)
            w = torch.roll(w, self.step, -1)

        h = F.pad(h, (0, 0, 0, pad_h), mode='circular')
        w = F.pad(w, (0, pad_w, 0, 0), mode='circular')

        h = h.reshape(B, C, (H + pad_h) // self.pixel, self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C*self.pixel, (H + pad_h) // self.pixel, W)
        w = w.reshape(B, C, H, (W + pad_w) // self.pixel, self.pixel).permute(0, 1, 4, 2, 3).reshape(B, C*self.pixel, H, (W + pad_w) // self.pixel)

        h = self.mlp_h2(self.act(self.mlp_h1_norm(self.mlp_h1(h))))
        w = self.mlp_w2(self.act(self.mlp_w1_norm(self.mlp_w1(w))))

        h = h.reshape(B, C, self.pixel, (H + pad_h) // self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C, H + pad_h, W)
        w = w.reshape(B, C, self.pixel, H, (W + pad_w) // self.pixel).permute(0, 1, 3, 2, 4).reshape(B, C, H, W + pad_w)

        h = torch.narrow(h, 2, 0, H)
        w = torch.narrow(w, 3, 0, W)

        if self.step:
            h = torch.roll(h, -self.step, -2)
            w = torch.roll(w, -self.step, -1)

        c = self.mlp_c(x)
        a = (h + w + c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        return x

    
class Block(nn.Module):
    def __init__(self, dim, pixel=2, step=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = HireAttn(dim, pixel, step)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlap(nn.Module):
    """Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, stride, padding)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> Tensor:
        return self.act(self.norm(self.proj(x)))


class Downsample(nn.Module):
    """Downsample transition stage"""
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 3, 2, 1)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.proj(x)))


hiremlp_settings = {
    'T': [2, 2, 4, 2],       # [layers]
    'S': [3, 4, 10, 3],
    'B': [4, 6, 24, 3]
}


class HireMLP(nn.Module):     
    def __init__(self, model_name: str = 'B', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in hiremlp_settings.keys(), f"HireMLP model name should be in {list(hiremlp_settings.keys())}"
        layers = hiremlp_settings[model_name]

        pixel = [4, 3, 3, 2]
        step_stride = [2, 2, 3, 2]
        step_dilation = [2, 2, 1, 1]
        embed_dims = [64, 128, 320, 512]
    
        self.patch_embed = PatchEmbedOverlap(7, 4, 2, embed_dims[0])

        network = []

        for i in range(len(layers)):
            stage = nn.Sequential(*[
                Block(embed_dims[i], pixel[i], (j % step_stride[i]) * step_dilation[i])
            for j in range(layers[i])])
            
            network.append(stage)
            if i >= len(layers) - 1: break
            network.append(Downsample(embed_dims[i], embed_dims[i+1]))

        self.network = nn.ModuleList(network)
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
        x = self.patch_embed(x)          

        for blk in self.network:
            x = blk(x)

        x = self.norm(x)
        x = self.head(x.flatten(2).mean(2))
        return x

if __name__ == '__main__':
    model = HireMLP('S', 'C:\\Users\\sithu\\Documents\\weights\\backbones\\hiremlp\\hire_mlp_small.pth')
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)