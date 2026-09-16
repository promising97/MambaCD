"""
StarIR StarModule / DFFN used by the isolated COPE-S experiment stack.

The DFGF and DFFN implementations are intentionally unchanged here. The
renamed copy keeps the COPE-S experiment independent from the original files
and makes checkpoint/model-stack mismatches easier to detect.

The StarModule can switch between:
  * the original GaborFreqTransform (use_dfgf=False), and
  * DirectionalFrequencyGatedGabor (use_dfgf=True, default).

DFFN is intentionally left unchanged so that the new mechanism can be
evaluated as a clean StarModule-only ablation.
"""

import math
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


class DirectionalFrequencyGatedGabor(nn.Module):
    """Direction-frequency decoupled Gabor filtering.

    The channels are split into orientation heads. Each head is filtered by a
    fixed, physically interpretable Gabor kernel and then receives an adaptive
    learnable mask in the Fourier domain. A lightweight direction selector
    reweights the four heads before a residual connection is applied.

    The grouped design avoids materializing ``num_orientations`` full copies
    of the input and keeps the output shape identical to the input shape.
    """

    def __init__(
        self,
        dim,
        num_orientations=4,
        kernel_size=5,
        sigma=2.0,
        wavelength=3.0,
        gamma=0.5,
        mask_size=16,
        reduction=8,
        residual_scale=0.1,
        use_frequency_gate=True,
        use_direction_selector=True,
    ):
        super().__init__()

        if dim % num_orientations != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_orientations "
                f"({num_orientations})."
            )
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        if mask_size < 2:
            raise ValueError("mask_size must be at least 2.")

        self.dim = dim
        self.num_orientations = num_orientations
        self.channels_per_orientation = dim // num_orientations
        self.kernel_size = kernel_size
        self.use_frequency_gate = use_frequency_gate
        self.use_direction_selector = use_direction_selector

        angles = [
            i * math.pi / num_orientations for i in range(num_orientations)
        ]
        kernels = [
            self._make_gabor_kernel(
                kernel_size, sigma, angle, wavelength, gamma
            )
            for angle in angles
        ]
        self.register_buffer(
            "gabor_kernels",
            torch.stack(kernels, dim=0).unsqueeze(1),
            persistent=True,
        )

        if use_frequency_gate:
            # A resolution-independent learnable frequency prior. It is resized
            # to the current rFFT grid. Initial sigmoid(4) ~= 0.982, i.e. almost
            # pass-through.
            rfft_mask_width = mask_size // 2 + 1
            self.frequency_prior = nn.Parameter(
                torch.full(
                    (num_orientations, mask_size, rfft_mask_width),
                    4.0,
                )
            )

            gate_hidden = max(4, dim // (num_orientations * reduction))
            self.frequency_refine = nn.Sequential(
                nn.Conv2d(
                    1, gate_hidden, kernel_size=3, padding=1, bias=True
                ),
                nn.GELU(),
                nn.Conv2d(gate_hidden, 1, kernel_size=1, bias=True),
            )
            # Start from the learnable prior only; content adaptation is
            # introduced smoothly during optimization.
            nn.init.zeros_(self.frequency_refine[-1].weight)
            nn.init.zeros_(self.frequency_refine[-1].bias)
        else:
            self.register_parameter("frequency_prior", None)
            self.frequency_refine = None

        if use_direction_selector:
            direction_hidden = max(8, dim // reduction)
            self.direction_selector = nn.Sequential(
                nn.Linear(dim, direction_hidden, bias=True),
                nn.GELU(),
                nn.Linear(
                    direction_hidden, num_orientations, bias=True
                ),
            )
            nn.init.zeros_(self.direction_selector[-1].weight)
            nn.init.zeros_(self.direction_selector[-1].bias)
        else:
            self.direction_selector = None

        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale))
        )

    @staticmethod
    def _make_gabor_kernel(
        kernel_size, sigma, theta, wavelength, gamma
    ):
        radius = kernel_size // 2
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")

        x_theta = xx * math.cos(theta) + yy * math.sin(theta)
        y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
        gaussian = torch.exp(
            -(x_theta.square() + gamma**2 * y_theta.square())
            / (2.0 * sigma**2)
        )
        carrier = torch.cos(2.0 * math.pi * x_theta / wavelength)
        kernel = gaussian * carrier

        # Remove the DC component and normalize the response scale.
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
        return kernel

    def _frequency_gate(self, subband, orientation_index):
        """Apply a content-adaptive real mask while preserving FFT phase."""
        original_dtype = subband.dtype
        fft_input = (
            subband.float()
            if subband.dtype in (torch.float16, torch.bfloat16)
            else subband
        )
        spectrum = torch.fft.rfft2(fft_input, norm="ortho")

        # Channel-averaged log magnitude makes the mask robust to activation
        # scale while keeping a separate mask for each direction head.
        magnitude = torch.log1p(spectrum.abs()).mean(dim=1, keepdim=True)
        mean = magnitude.mean(dim=(-2, -1), keepdim=True)
        std = magnitude.std(dim=(-2, -1), keepdim=True, unbiased=False)
        magnitude = (magnitude - mean) / (std + 1e-6)

        gate_dtype = self.frequency_refine[0].weight.dtype
        content_logits = self.frequency_refine(magnitude.to(gate_dtype))
        prior = F.interpolate(
            self.frequency_prior.unsqueeze(0),
            size=spectrum.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, orientation_index : orientation_index + 1]
        gate = torch.sigmoid(prior.to(content_logits.dtype) + content_logits)
        gated_spectrum = spectrum * gate.to(spectrum.real.dtype)

        gated = torch.fft.irfft2(
            gated_spectrum,
            s=subband.shape[-2:],
            norm="ortho",
        )
        return gated.to(original_dtype)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4-D BCHW tensor, got shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected {self.dim} channels, got {x.shape[1]}."
            )

        if self.use_direction_selector:
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            selector_dtype = self.direction_selector[0].weight.dtype
            direction_weights = self.direction_selector(
                pooled.to(selector_dtype)
            )
            # Multiplication by O keeps every head at unit gain at initialization.
            direction_weights = (
                direction_weights.softmax(dim=-1)
                * self.num_orientations
            ).to(x.dtype)
        else:
            direction_weights = x.new_ones(
                x.shape[0], self.num_orientations
            )

        chunks = x.chunk(self.num_orientations, dim=1)
        outputs = []
        padding = self.kernel_size // 2
        for orientation_index, chunk in enumerate(chunks):
            kernel = self.gabor_kernels[
                orientation_index : orientation_index + 1
            ]
            kernel = kernel.expand(
                self.channels_per_orientation, 1, -1, -1
            ).to(dtype=chunk.dtype)
            subband = F.conv2d(
                chunk,
                kernel.contiguous(),
                padding=padding,
                groups=self.channels_per_orientation,
            )
            if self.use_frequency_gate:
                subband = self._frequency_gate(
                    subband, orientation_index
                )
            weight = direction_weights[
                :, orientation_index
            ].view(-1, 1, 1, 1)
            outputs.append(subband * weight)

        directional_feature = torch.cat(outputs, dim=1)
        return x + self.residual_scale.to(x.dtype) * directional_feature


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
        self.use_dfgf = kwargs.get("use_dfgf", True)
        if self.use_dfgf:
            self.gabor_freq = DirectionalFrequencyGatedGabor(
                dim=dim,
                num_orientations=kwargs.get("dfgf_num_orientations", 4),
                kernel_size=kwargs.get("dfgf_kernel_size", 5),
                sigma=kwargs.get("dfgf_sigma", 2.0),
                wavelength=kwargs.get("dfgf_wavelength", 3.0),
                gamma=kwargs.get("dfgf_gamma", 0.5),
                mask_size=kwargs.get("dfgf_mask_size", 16),
                reduction=kwargs.get("dfgf_reduction", 8),
                residual_scale=kwargs.get("dfgf_residual_scale", 0.1),
                use_frequency_gate=kwargs.get(
                    "dfgf_use_frequency_gate", True
                ),
                use_direction_selector=kwargs.get(
                    "dfgf_use_direction_selector", True
                ),
            )
        else:
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
