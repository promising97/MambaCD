"""
SFS-Conv: Space-Frequency Selection Convolution (CVPR 2024)
https://github.com/like413/SFS-Conv

Shunt-perceive-select strategy:
  - SPU: multi-scale spatial perception with hierarchical residual
  - FPU: fractional Gabor transform for frequency/texture features
  - CSU: parameter-free channel selection via softmax on GAP statistics
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FractionalGaborFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super().__init__()
        filters = []
        for angle in angles:
            for scale in scales:
                filters.append(
                    self._generate_fractional_gabor(
                        out_channels, kernel_size, order, angle, scale
                    )
                )
        self.register_buffer("filters", torch.stack(filters, dim=0))

    @staticmethod
    def _generate_fractional_gabor(out_channels, size, order, angle, scale):
        angle_rad = math.radians(angle)
        x, y = np.meshgrid(
            np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1])
        )
        x_theta = x * math.cos(angle_rad) + y * math.sin(angle_rad)
        y_theta = -x * math.sin(angle_rad) + y * math.cos(angle_rad)
        real_part = np.exp(
            -((x_theta ** 2 + (y_theta / scale) ** 2) ** order)
        ) * np.cos(2 * np.pi * x_theta / scale)
        weight = torch.tensor(real_part, dtype=torch.float32).view(1, 1, size[0], size[1])
        return weight.repeat(out_channels, 1, 1, 1)

    def forward(self, x):
        return sum(f * x for f in self.filters.unbind(0))


class GaborSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, order, angles, scales):
        super().__init__()
        self.gabor = FractionalGaborFilter(
            in_channels, out_channels, kernel_size, order, angles, scales
        )
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        kernel = self.gabor(self.t)
        pad = (kernel.shape[-2] - 1) // 2
        out = F.conv2d(x, kernel, stride=1, padding=pad)
        out = self.relu(out)
        out = F.dropout(out, 0.3, training=self.training)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class GaborFreqTransform(nn.Module):
    """Size-preserving fractional Gabor transform (replaces patch-wise FFT in StarBlock)."""

    def __init__(
        self,
        channels,
        kernel_size=3,
        order=0.25,
        angles=(0, 45, 90, 135),
        scales=(1, 2, 3, 4),
    ):
        super().__init__()
        k = (kernel_size, kernel_size)
        self.gabor = FractionalGaborFilter(channels, channels, k, order, angles, scales)
        self.t = nn.Parameter(torch.randn(channels, channels, kernel_size, kernel_size))
        nn.init.normal_(self.t)

    def forward(self, x):
        kernel = self.gabor(self.t)
        pad = (kernel.shape[-2] - 1) // 2
        return F.conv2d(x, kernel, stride=1, padding=pad)


class GaborFPU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        order=0.25,
        angles=(0, 45, 90, 135),
        scales=(1, 2, 3, 4),
    ):
        super().__init__()
        self.gabor = GaborSingle(
            in_channels // 4, out_channels // 4, (3, 3), order, angles, scales
        )
        self.fc = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, dim=1)
        x_out = torch.cat(
            [self.gabor(x1), self.gabor(x2), self.gabor(x3), self.gabor(x4)], dim=1
        )
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class FrFTFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super().__init__()
        self.register_buffer(
            "weight",
            self._generate_frft_filter(in_channels, out_channels, kernel_size, f, order),
        )

    @staticmethod
    def _generate_frft_filter(in_channels, out_channels, kernel, f, p):
        d_x, d_y = kernel[0], kernel[1]
        x = np.linspace(1, d_x, d_x)
        y = np.linspace(1, d_y, d_y)
        x_grid, y_grid = np.meshgrid(x, y)

        real_filter = np.zeros([d_x, d_y, out_channels])
        for i in range(out_channels):
            real_x = np.cos(
                -f * x_grid / math.sin(p)
                + (f * f + x_grid * x_grid) / (2 * math.tan(p))
            )
            real_y = np.cos(
                -f * y_grid / math.sin(p)
                + (f * f + y_grid * y_grid) / (2 * math.tan(p))
            )
            real_filter[:, :, i] = real_y * real_x

        g_f = np.zeros((kernel[0], kernel[1], in_channels, out_channels))
        for i in range(out_channels):
            g_f[:, :, :, i] = np.repeat(real_filter[:, :, i : i + 1], in_channels, axis=2)
        g_f_real = g_f.reshape((out_channels, in_channels, kernel[0], kernel[1]))
        return torch.tensor(g_f_real, dtype=torch.float32)

    def forward(self, x):
        return x * self.weight


class FrFTSingle(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, f, order):
        super().__init__()
        self.fft = FrFTFilter(in_channels, out_channels, kernel_size, f, order)
        self.t = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        nn.init.normal_(self.t)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        kernel = self.fft(self.t)
        pad = (kernel.shape[-2] - 1) // 2
        out = F.conv2d(x, kernel, stride=1, padding=pad)
        out = self.relu(out)
        out = F.dropout(out, 0.3, training=self.training)
        out = F.pad(out, (1, 0, 1, 0), mode="constant", value=0)
        out = F.max_pool2d(out, 2, stride=1, padding=0)
        return out


class FourierFPU(nn.Module):
    def __init__(self, in_channels, out_channels, order=0.25):
        super().__init__()
        self.fft1 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.25, order)
        self.fft2 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.50, order)
        self.fft3 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 0.75, order)
        self.fft4 = FrFTSingle(in_channels // 4, out_channels // 4, (3, 3), 1.00, order)
        self.fc = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        channels_per_group = x.shape[1] // 4
        x1, x2, x3, x4 = torch.split(x, channels_per_group, dim=1)
        x_out = torch.cat(
            [self.fft1(x1), self.fft2(x2), self.fft3(x3), self.fft4(x4)], dim=1
        )
        x_out = self.fc(x_out)
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class SPU(nn.Module):
    """Spatial Perception Unit: multi-scale depthwise conv with hierarchical residual."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        half = in_channels // 2
        self.c1 = Conv(half, half, 3, g=half)
        self.c2 = Conv(half, half, 5, g=half)
        self.c3 = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        x1 = self.c1(x1)
        x2 = self.c2(x2 + x1)
        x_out = self.c3(torch.cat([x1, x2], dim=1))
        if x.shape[1] == x_out.shape[1]:
            x_out = x_out + x
        return x_out


class SFS_Conv(nn.Module):
    """
    Space-Frequency Selection Convolution.
    Shunts input into spatial (SPU) and frequency (FPU) branches,
    then fuses via parameter-free channel selection (CSU).
    """

    def __init__(self, in_channels, out_channels, order=0.25, filter_type="FrGT"):
        super().__init__()
        self.pwc0 = Conv(in_channels, in_channels // 2, 1)
        self.pwc1 = Conv(in_channels, in_channels // 2, 1)
        self.spu = SPU(in_channels // 2, out_channels)

        if filter_type not in ("FrFT", "FrGT"):
            raise ValueError("filter_type must be 'FrFT' or 'FrGT'")
        if filter_type == "FrFT":
            self.fpu = FourierFPU(in_channels // 2, out_channels, order)
        else:
            self.fpu = GaborFPU(in_channels // 2, out_channels, order)

        self.pwc_o = Conv(out_channels, out_channels, 1)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_spa = self.spu(self.pwc0(x))
        x_fre = self.fpu(self.pwc1(x))
        out = torch.cat([x_spa, x_fre], dim=1)
        # CSU: parameter-free channel selection
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return self.pwc_o(out1 + out2)


class SFSConvBlock(nn.Module):
    """Residual wrapper for drop-in replacement of ICSFBlock / ResBlock."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.sfs = SFS_Conv(out_channels, out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.input_proj(x)
        out = self.sfs(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        elif identity.shape != out.shape:
            identity = F.interpolate(identity, size=out.shape[-2:], mode="bilinear", align_corners=False)
            if identity.shape[1] != out.shape[1]:
                identity = identity[:, : out.shape[1]]
        return out + identity
