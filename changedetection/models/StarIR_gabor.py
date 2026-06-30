"""
StarIR StarModule / DFFN with Gabor replacing patch-wise FFT.
Based on StarIR_orignal.py — same forward logic, GaborFreqTransform instead of rfft2.
Used by ChangeDecoder_Star_GaborFPU (HAM-CD Star-style decoder).
"""

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from MambaCD.changedetection.models.Spatial_Mamba import SpatialMambaBlock
from MambaCD.classification.models.vmamba import Permute
from MambaCD.changedetection.models.sfs_conv import GaborFreqTransform


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super().__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, channel_first, norm_layer, **kwargs):
        super().__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )
        self.gabor_freq = GaborFreqTransform(dim)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = self.gabor_freq(x)
        return x


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class StarModule(nn.Module):
    def __init__(self, dim, bias, channel_first, norm_layer, **kwargs):
        super().__init__()

        self.dim = dim
        self.SMB = nn.Sequential(
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(
                hidden_dim=128,
                drop_path=0.1,
                norm_layer=norm_layer,
                attn_drop_rate=0.0,
                d_state=kwargs["ssm_d_state"],
                dt_init="random",
                use_checkpoint=kwargs["use_checkpoint"],
            ),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.to_hidden = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim, LayerNorm_type="WithBias")
        self.gabor_freq = GaborFreqTransform(dim)
        self.spatial = SpatialOperation(dim)
        self.plain_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        hidden = self.to_hidden(x)
        q, v = self.to_hidden_dw(hidden).split([self.dim, self.dim], dim=1)

        out = self.gabor_freq(q)
        out = self.norm(out)
        out = self.spatial(v) * out
        out = self.plain_channel(out) * out
        out = self.project_out(out)
        return out
