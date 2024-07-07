import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Get_curvature(nn.Module):
    def __init__(self):
        super(Get_curvature, self).__init__()
        kernel_v1 = [[0, -1, 0],
                     [0, 0, 0],
                     [0, 1, 0]]
        kernel_h1 = [[0, 0, 0],
                     [-1, 0, 1],
                     [0, 0, 0]]
        kernel_h2 = [[0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [1, 0, -2, 0, 1],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]
        kernel_v2 = [[0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, -2, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]]
        kernel_w2 = [[1, 0, -1],
                     [0, 0, 0],
                     [-1, 0, 1]]
        kernel_h1 = torch.FloatTensor(kernel_h1).unsqueeze(0).unsqueeze(0)
        kernel_v1 = torch.FloatTensor(kernel_v1).unsqueeze(0).unsqueeze(0)
        kernel_v2 = torch.FloatTensor(kernel_v2).unsqueeze(0).unsqueeze(0)
        kernel_h2 = torch.FloatTensor(kernel_h2).unsqueeze(0).unsqueeze(0)
        kernel_w2 = torch.FloatTensor(kernel_w2).unsqueeze(0).unsqueeze(0)
        self.weight_h1 = nn.Parameter(data=kernel_h1, requires_grad=False)
        self.weight_v1 = nn.Parameter(data=kernel_v1, requires_grad=False)
        self.weight_v2 = nn.Parameter(data=kernel_v2, requires_grad=False)
        self.weight_h2 = nn.Parameter(data=kernel_h2, requires_grad=False)
        self.weight_w2 = nn.Parameter(data=kernel_w2, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v1, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h1, padding=1)
            x_i_v2 = F.conv2d(x_i.unsqueeze(1), self.weight_v2, padding=2)
            x_i_h2 = F.conv2d(x_i.unsqueeze(1), self.weight_h2, padding=2)
            x_i_w2 = F.conv2d(x_i.unsqueeze(1), self.weight_w2, padding=1)
            sum = torch.pow((torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2)), 3 / 2)
            fg = torch.mul(torch.pow(x_i_v, 2), x_i_v2) + 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_h2)
            fh = torch.mul(torch.pow(x_i_v, 2), x_i_h2) - 2 * torch.mul(torch.mul(x_i_v, x_i_h), x_i_w2) + torch.mul(
                torch.pow(x_i_h, 2), x_i_v2)
            x_i = torch.div(torch.abs(fg - fh), sum + 1e-10)
            x_i = torch.div(torch.abs(fh), sum + 1e-10)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=1)
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
            self,
            kernel_size: Tuple[int, int] = (16, 16),
            stride: Tuple[int, int] = (16, 16),
            padding: Tuple[int, int] = (0, 0),
            in_chans: int = 3,
            embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


class get_PMD_embeddings(nn.Module):
    def __init__(self):
        super(get_PMD_embeddings, self).__init__()

        self.PMD_Head = Get_curvature()

        self.patch_embed = PatchEmbed(
            kernel_size=(16, 16),
            stride=(16, 16),
            in_chans=3,
            embed_dim=1280,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        # Initialize absolute positional embedding with pretrain image size.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1024 // 16, 1024 // 16, 1280)
        )

        self.neck = nn.Sequential(
            nn.Conv2d(
                1280,
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def forward(self, images):
        images_PMD = self.PMD_Head(images)

        PMD_patch = self.patch_embed(images_PMD)
        PMD_pos_embeddings = PMD_patch + self.pos_embed

        PMD_embeddings = self.neck(PMD_pos_embeddings.permute(0, 3, 1, 2))

        return(PMD_embeddings)

