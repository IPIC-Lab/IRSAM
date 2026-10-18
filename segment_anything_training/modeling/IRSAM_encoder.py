# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath, \
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Tuple


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f'(drop_prob={self.drop_prob})'
        return msg


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size: Tuple[int, int] = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * \
                           self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x += shortcut
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if (out_dim == 320 or out_dim == 448 or out_dim == 576):
            stride_c = 1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(
            range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N),
                             persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.register_buffer('ab',
                                 self.attention_biases[:, self.attention_bias_idxs],
                                 persistent=False)

    def forward(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -
        1).split([self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class TinyViTBlock(nn.Module):
    r""" TinyViT Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 local_conv_size=3,
                 activation=nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        x = x + self.drop_path(self.mlp(x))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"


class BasicLayer(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        activation: the activation function. Default: nn.GELU
        out_dim: the output dimension of the layer. Default: dim
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            TinyViTBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class SelfAttn(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class UNetEncoderWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoderWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.attention = SelfAttn(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.attention(x)
        x_pooled = self.pool(x)
        return x, x_pooled


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


# ---------------------------------------------------------------------------
# 梯度补偿分支：可学习小波变换梯度特征提取
# ---------------------------------------------------------------------------

# 固定的 Sobel 梯度算子（仅用于产生梯度先验，本身不参与训练）
_SOBEL_X = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
_SOBEL_Y = torch.tensor([[-1., -2., -1.],
                         [0., 0., 0.],
                         [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0


def compute_gradient_prior(image, value_range=255.0, eps=1e-6):
    """提取图像的梯度先验 [g_x, g_y, |g|]，通道数为输入通道数的 3 倍。

    Args:
        image: (C, H, W) 或 (B, C, H, W) 的图像张量。
        value_range: 输入图像的灰度取值范围，用于把梯度归一化到统一量级；
            常规 0~255 图像保持默认值 255.0，已完成归一化的输入传 1.0。
        eps: 计算梯度幅值时避免开方为零的小量。

    Returns:
        与输入 batch 维度一致的梯度先验张量。
    """
    squeeze = image.dim() == 3
    if squeeze:
        image = image.unsqueeze(0)
    image = image.float() / float(value_range)

    b, c, h, w = image.shape
    kx = _SOBEL_X.to(device=image.device, dtype=image.dtype).repeat(c, 1, 1, 1)
    ky = _SOBEL_Y.to(device=image.device, dtype=image.dtype).repeat(c, 1, 1, 1)
    gx = F.conv2d(image, kx, padding=1, groups=c)
    gy = F.conv2d(image, ky, padding=1, groups=c)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    grad = torch.cat([gx, gy, mag], dim=1)
    if squeeze:
        grad = grad.squeeze(0)
    return grad


def _haar_filters(channels):
    """Haar 小波滤波器，形状 (4C, 1, 2, 2)，按 [LL, LH, HL, HH] 通道主序堆叠。

    Haar 基是正交归一基，因此分析与合成共用同一组滤波器即可精确重构；
    这里将其作为可学习参数的初始化值。
    """
    ll = torch.tensor([[1., 1.], [1., 1.]]) / 2.0
    lh = torch.tensor([[1., -1.], [1., -1.]]) / 2.0
    hl = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0
    hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0
    kernel = torch.stack([ll, lh, hl, hh], dim=0).view(4, 1, 2, 2)
    return kernel.repeat(channels, 1, 1, 1)


class LearnableWaveletTransform(nn.Module):
    """可学习二维小波变换（多级 DWT + IDWT）。

    分析滤波器与合成滤波器均以 Haar 小波初始化并可端到端学习；每个子带
    （LL/LH/HL/HH）配有可学习的通道级门控，用于自适应筛选梯度与边缘信息；
    各级重建结果按可学习权重融合，同时保留低频轮廓与高频细节。
    """

    def __init__(self, channels, levels=2, subband_gate_init=1.0):
        super().__init__()
        assert levels >= 1, 'wavelet levels must be >= 1'
        self.channels = channels
        self.levels = levels

        analysis_filters, synthesis_filters, subband_gates = [], [], []
        for _ in range(levels):
            haar = _haar_filters(channels)
            analysis_filters.append(nn.Parameter(haar.clone()))
            synthesis_filters.append(nn.Parameter(haar.clone()))
            subband_gates.append(nn.Parameter(
                torch.full((1, channels, 4, 1, 1), float(subband_gate_init))))
        self.analysis_filters = nn.ParameterList(analysis_filters)
        self.synthesis_filters = nn.ParameterList(synthesis_filters)
        self.subband_gates = nn.ParameterList(subband_gates)
        # 多尺度重建的融合权重（softmax 后使用）
        self.level_logits = nn.Parameter(torch.zeros(levels))

    def _dwt(self, x, level):
        """一级二维离散小波分解，返回 (B, C, 4, H/2, W/2) 的子带系数。"""
        coeff = F.conv2d(x, self.analysis_filters[level], stride=2,
                         groups=self.channels)
        b, _, h, w = coeff.shape
        coeff = coeff.view(b, self.channels, 4, h, w)
        return coeff * self.subband_gates[level]

    def forward(self, x):
        """x: (B, C, H, W) -> 小波重构特征 (B, C, H, W)。"""
        h, w = x.shape[-2:]
        factor = 2 ** self.levels
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 多级分解：LL 子带继续分解，同时保存各级细节子带
        coeffs, approx = [], x
        for level in range(self.levels):
            coeff = self._dwt(approx, level)
            coeffs.append(coeff)
            approx = coeff[:, :, 0]

        # 多级重构：由最粗尺度逐级 IDWT 回原始分辨率，并收集各尺度响应
        responses, rec = [], approx
        for level in reversed(range(self.levels)):
            coeff = coeffs[level]
            detail = torch.cat([rec.unsqueeze(2), coeff[:, :, 1:]], dim=2)
            b_, c_, _, h_, w_ = detail.shape
            detail = detail.reshape(b_, c_ * 4, h_, w_)
            rec = F.conv_transpose2d(detail, self.synthesis_filters[level],
                                     stride=2, groups=c_)
            responses.append(rec)

        # 多尺度融合（以最精细的重建分辨率为基准）
        weights = F.softmax(self.level_logits, dim=0)
        target = responses[-1].shape[-2:]
        out = None
        for i, resp in enumerate(responses):
            if resp.shape[-2:] != target:
                resp = F.interpolate(resp, size=target, mode='bilinear',
                                     align_corners=False)
            term = resp * weights[i]
            out = term if out is None else out + term

        if pad_h > 0 or pad_w > 0:
            out = out[..., :h, :w]
        return out


class WaveletGradientCompensation(nn.Module):
    """梯度补偿分支。

    流程：图像梯度先验 -> 可学习下采样主干 -> 可学习多级小波变换
    -> 多尺度融合 -> 逐层 1x1 投影与可学习门控 -> 与各 layer 的输出相加，
    以补偿深层网络在降采样过程中丢失的梯度/边缘信息。

    Args:
        in_chans (int): 输入图像通道数（梯度先验通道数为 3 * in_chans）。
        out_dims (tuple[int]): 骨干网络各 layer 输出的通道数。
        base_dim (int): 分支内部特征通道数。
        wavelet_levels (int): 小波分解级数。
        stem_stride (int): 梯度先验的下采样步长。
        gamma_init (float): 逐层门控初值，默认 0 以保证训练初期不改变原有网络行为。
    """

    def __init__(self, in_chans=3, out_dims=(128, 160, 320, 320), base_dim=32,
                 wavelet_levels=2, stem_stride=2, gamma_init=0.0):
        super().__init__()
        self.in_chans = in_chans
        self.out_dims = tuple(out_dims)

        self.stem = nn.Sequential(
            Conv2d_BN(in_chans * 3, base_dim, ks=3, stride=stem_stride, pad=1),
            nn.GELU(),
            Conv2d_BN(base_dim, base_dim, ks=3, stride=1, pad=1),
            nn.GELU(),
        )
        self.wavelet = LearnableWaveletTransform(base_dim, levels=wavelet_levels)
        self.fuse = nn.Sequential(
            nn.Conv2d(base_dim, base_dim, kernel_size=1, bias=False),
            LayerNorm2d(base_dim),
            nn.GELU(),
        )
        self.proj = nn.ModuleList([
            nn.Conv2d(base_dim, dim, kernel_size=1) for dim in self.out_dims])
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.full((1,), float(gamma_init)))
            for _ in self.out_dims])

    def forward(self, x, grad_prior=None):
        """x: (B, C, H, W) 编码器输入；grad_prior: 可选的外部梯度先验。"""
        if grad_prior is None:
            # 未提供外部梯度先验时，直接由编码器输入（已完成归一化）计算
            grad_prior = compute_gradient_prior(x, value_range=1.0)
        elif grad_prior.dim() == 3:
            grad_prior = grad_prior.unsqueeze(0)
        assert grad_prior.shape[1] == self.in_chans * 3, \
            f'grad_prior 的通道数 {grad_prior.shape[1]} 与期望的 {self.in_chans * 3} 不一致'
        if grad_prior.dtype != x.dtype or grad_prior.device != x.device:
            grad_prior = grad_prior.to(device=x.device, dtype=x.dtype)
        if grad_prior.shape[-2:] != x.shape[-2:]:
            grad_prior = F.interpolate(grad_prior, size=x.shape[-2:],
                                       mode='bilinear', align_corners=False)

        feat = self.stem(grad_prior)
        feat = self.wavelet(feat)
        feat = self.fuse(feat)
        return feat

    def project_to_layer(self, feat, layer_idx, num_tokens):
        """把梯度补偿特征投影成第 layer_idx 层输出的形状 (B, L, C)。"""
        side = int(round(num_tokens ** 0.5))
        size = (side, side) if side * side == num_tokens else (1, num_tokens)
        f = feat if feat.shape[-2:] == size else F.interpolate(
            feat, size=size, mode='bilinear', align_corners=False)
        f = self.proj[layer_idx](f)
        f = f.flatten(2).transpose(1, 2)
        return f * self.gamma[layer_idx]


class TinyViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 use_gradient_compensation=True,
                 grad_branch_dim=32,
                 grad_wavelet_levels=2,
                 grad_branch_stride=2,
                 grad_branch_gamma_init=0.0,
                 ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                          input_resolution=(patches_resolution[0] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                                            patches_resolution[1] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer))),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                                  i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        self.use_gradient_compensation = bool(use_gradient_compensation)
        if self.use_gradient_compensation:
            layer_out_dims = [
                embed_dims[i_layer + 1] if i_layer < self.num_layers - 1
                else embed_dims[i_layer]
                for i_layer in range(self.num_layers)
            ]
            self.gradient_compensation = WaveletGradientCompensation(
                in_chans=in_chans,
                out_dims=layer_out_dims,
                base_dim=grad_branch_dim,
                wavelet_levels=grad_wavelet_levels,
                stem_stride=grad_branch_stride,
                gamma_init=grad_branch_gamma_init,
            )
        else:
            self.gradient_compensation = None

        # Classifier head
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

        # init weights
        self.apply(self._init_weights)
        self.set_layer_lr_decay(layer_lr_decay)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
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

    def set_layer_lr_decay(self, layer_lr_decay):
        decay_rate = layer_lr_decay

        # layers -> blocks (depth)
        depth = sum(self.depths)
        lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

        # print("LR SCALES:", lr_scales)

        def _set_lr_scale(m, scale):
            for p in m.parameters():
                p.lr_scale = scale

        self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
        i = 0
        for layer in self.layers:
            for block in layer.blocks:
                block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                i += 1
            if layer.downsample is not None:
                layer.downsample.apply(
                    lambda x: _set_lr_scale(x, lr_scales[i - 1]))
        assert i == depth
        for m in [self.norm_head, self.head]:
            m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

        # 梯度补偿分支同样纳入层级学习率衰减：共享部分用浅层尺度，
        # 逐层投影与门控则跟随其目标 layer 的尺度
        grad_branch = getattr(self, 'gradient_compensation', None)
        if grad_branch is not None:
            grad_branch.apply(lambda x: _set_lr_scale(x, lr_scales[0]))
            for i_layer, proj in enumerate(grad_branch.proj):
                scale = lr_scales[min(sum(self.depths[:i_layer + 1]) - 1, depth - 1)]
                proj.apply(lambda x, s=scale: _set_lr_scale(x, s))
                grad_branch.gamma[i_layer].lr_scale = scale

        for k, p in self.named_parameters():
            p.param_name = k

        def _check_lr_scale(m):
            for p in m.parameters():
                assert hasattr(p, 'lr_scale'), p.param_name

        self.apply(_check_lr_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases', 'gamma', 'subband_gates', 'level_logits'}

    def forward_features(self, x, grad_prior=None):
        # x: (N, C, H, W)
        grad_feat = None
        if self.gradient_compensation is not None:
            grad_feat = self.gradient_compensation(x, grad_prior=grad_prior)

        x = self.patch_embed(x)

        x = self.layers[0](x)
        if grad_feat is not None:
            x = x + self.gradient_compensation.project_to_layer(
                grad_feat, 0, x.shape[1])
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
            if grad_feat is not None:
                x = x + self.gradient_compensation.project_to_layer(
                    grad_feat, i, x.shape[1])
            if i == 1:
                interm_embedding = x.view(x.shape[0], 64, 64, -1)
        B, _, C = x.size()
        x = x.view(B, 64, 64, C)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x, interm_embedding

    def forward(self, x, grad_prior=None):
        x = self.forward_features(x, grad_prior=grad_prior)
        # x = self.norm_head(x)
        # x = self.head(x)
        return x


_checkpoint_url_format = \
    'https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth'
_provided_checkpoints = {
    'tiny_vit_5m_224': 'tiny_vit_5m_22kto1k_distill',
    'tiny_vit_11m_224': 'tiny_vit_11m_22kto1k_distill',
    'tiny_vit_21m_224': 'tiny_vit_21m_22kto1k_distill',
    'tiny_vit_21m_384': 'tiny_vit_21m_22kto1k_384_distill',
    'tiny_vit_21m_512': 'tiny_vit_21m_22kto1k_512_distill',
}


def register_tiny_vit_model(fn):
    '''Register a TinyViT model
    It is a wrapper of `register_model` with loading the pretrained checkpoint.
    '''

    def fn_wrapper(pretrained=False, **kwargs):
        model = fn()
        if pretrained:
            model_name = fn.__name__
            assert model_name in _provided_checkpoints, \
                f'Sorry that the checkpoint `{model_name}` is not provided yet.'
            url = _checkpoint_url_format.format(
                _provided_checkpoints[model_name])
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url,
                map_location='cpu', check_hash=False,
            )
            model.load_state_dict(checkpoint['model'])

        return model

    # rename the name of fn_wrapper
    fn_wrapper.__name__ = fn.__name__
    return register_model(fn_wrapper)


@register_tiny_vit_model
def tiny_vit_5m_224(pretrained=False, num_classes=1000, drop_path_rate=0.0):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_11m_224(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_21m_224(pretrained=False, num_classes=1000, drop_path_rate=0.2):
    return TinyViT(
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_21m_384(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        img_size=384,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=drop_path_rate,
    )


@register_tiny_vit_model
def tiny_vit_21m_512(pretrained=False, num_classes=1000, drop_path_rate=0.1):
    return TinyViT(
        img_size=512,
        num_classes=num_classes,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=drop_path_rate,
    )
