import torch
from torch import nn, Tensor


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible by patch size'

        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(3, embed_dim, patch_size, patch_size)

    def forward(self, x: torch.Tensor) -> Tensor:
        x = self.proj(x)                   # b x hidden_dim x 14 x 14
        x = x.flatten(2).swapaxes(1, 2)     # b x (14*14) x hidden_dim

        return x


