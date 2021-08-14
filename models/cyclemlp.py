import torch
import math
from torch import nn, Tensor
from torchvision.ops.deform_conv import deform_conv2d
from torch.nn.modules.utils import _pair
from .layers import MLP


class CycleFC(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, bias=True):
        super().__init__()
        self.c1 = c1
        self.k = k
        self.s = _pair(s)
        self.p = _pair(p)
        self.d = _pair(d)
        self.weight = nn.Parameter(torch.empty(c2, c1//g, 1, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(c2))
        else:
            self.register_parameter('bias', None)

        self.register_buffer('offset', self.gen_offset())
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """Offsets to be applied for each position in the convolution kernel
        offset Tensor of shape [B, 2*offset_groups*kernel_height*kernel_width, H, W]
        """
        offset = torch.empty(1, self.c1*2, 1, 1)
        start_idx = (self.k[0] * self.k[1]) // 2
        assert self.k[0] == 1 or self.k[1] == 1, self.k

        for i in range(self.c1):
            if self.k[0] == 1:
                offset[0, 2*i+0, 0, 0] = 0
                offset[0, 2*i+1, 0, 0] = (i+start_idx) % self.k[1] - (self.k[1] // 2)
            else:
                offset[0, 2*i+0, 0, 0] = (i+start_idx) % self.k[0] - (self.k[0] // 2)
                offset[0, 2*i+1, 0, 0] = 0
        return offset

    def forward(self, x: Tensor) -> Tensor:
        """x > shape [B, c1, H, W]"""
        B, _, H, W = x.shape
        return deform_conv2d(x, self.offset.expand(B, -1, H, W), self.weight, self.bias, self.s, self.p, self.d)



class CycleAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=False)
        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)
        self.reweight = MLP(dim, dim//4, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, _, _, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h+w+c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        return x

    
class CycleBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, skip_lam=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CycleAttn(dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio))
        self.skip_lam = skip_lam

    def forward(self, x: Tensor) -> Tensor:
        x += self.attn(self.norm1(x)) / self.skip_lam
        x += self.mlp(self.norm2(x)) / self.skip_lam
        return x


class PatchEmbedOverlap(nn.Module):
    """Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, stride, padding)

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.proj(x)                   # b x hidden_dim x 14 x 14
        return x


class Downsample(nn.Module):
    """Downsample transition stage"""
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 3, 1, 2)   # [B, C, H, W]
        x = self.proj(x)    
        x = x.permute(0, 2, 3, 1)
        return x


cycle_settings = {
    'B1': [[2, 2, 4, 2], [64, 128, 320, 512], [4, 4, 4, 4]],       # [layers, embed_dims, mlp_ratios]
    'B2': [[2, 3, 10, 3], [64, 128, 320, 512], [4, 4, 4, 4]],
    'B3': [[3, 4, 18, 3], [64, 128, 320, 512], [8, 8, 4, 4]],
    'B4': [[3, 8, 27, 3], [64, 128, 320, 512], [8, 8, 4, 4]],
    'B5': [[3, 4, 24, 3], [96, 192, 384, 768], [4, 4, 4, 4]]
}


class CycleMLP(nn.Module):      # this model works with any image size, even non-square image size
    def __init__(self, model_name: str = 'B1', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in cycle_settings.keys(), f"CycleMLP model name should be in {list(cycle_settings.keys())}"
        layers, embed_dims, mlp_ratios = cycle_settings[model_name]
    
        self.patch_embed = PatchEmbedOverlap(7, 4, 2, embed_dims[0])

        network = []

        for i in range(len(layers)):
            stage = nn.Sequential(*[
                CycleBlock(embed_dims[i], mlp_ratios[i])
            for _ in range(layers[i])])
            
            network.append(stage)
            if i >= len(layers) - 1: break
            network.append(Downsample(embed_dims[i], embed_dims[i+1]))

        self.network = nn.ModuleList(network)
        self.norm = nn.LayerNorm(embed_dims[-1])
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
        x = x.permute(0, 2, 3, 1)   # B, H, W, C

        for blk in self.network:
            x = blk(x)

        B, _, _, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        x = self.head(x.mean(dim=1))
        return x

if __name__ == '__main__':
    model = CycleMLP('B2', 'checkpoints/cyclemlp/CycleMLP_B2.pth')
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)