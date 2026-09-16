"""Change-Oriented Pyramid Enhancement for near-homogeneous bi-temporal data.

COPE-S is the same-/near-domain variant used for pseudo-SAR/SAR or SAR/SAR
change detection. It combines four direction-specific Gabor experts with a
bi-temporal change guide and sample-adaptive orientation reweighting.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels for an NCHW tensor."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _make_gabor_kernel(kernel_size, sigma, wavelength, aspect_ratio, theta):
    if kernel_size % 2 == 0:
        raise ValueError("COPE-S requires an odd Gabor kernel size")

    radius = kernel_size // 2
    axis = torch.arange(-radius, radius + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x_theta = xx * cos_theta + yy * sin_theta
    y_theta = -xx * sin_theta + yy * cos_theta

    gaussian = torch.exp(
        -(x_theta.square() + (aspect_ratio * y_theta).square())
        / (2.0 * sigma**2)
    )
    carrier = torch.cos(2.0 * math.pi * x_theta / wavelength)
    kernel = gaussian * carrier
    kernel = kernel - kernel.mean()
    kernel = kernel / kernel.abs().sum().clamp_min(1e-6)
    return kernel


class OrientedGaborExpert(nn.Module):
    """One fixed orientation prior followed by an independent light refiner."""

    def __init__(
        self,
        channels,
        theta,
        kernel_size=5,
        sigma=2.0,
        wavelength=3.0,
        aspect_ratio=0.5,
    ):
        super().__init__()
        kernel = _make_gabor_kernel(
            kernel_size,
            sigma,
            wavelength,
            aspect_ratio,
            theta,
        )
        self.register_buffer(
            "gabor_kernel",
            kernel.view(1, 1, kernel_size, kernel_size),
            persistent=True,
        )
        self.channels = channels
        self.padding = kernel_size // 2
        self.refine = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        kernel = self.gabor_kernel.to(dtype=x.dtype).expand(
            self.channels,
            1,
            -1,
            -1,
        )
        oriented = F.conv2d(
            x,
            kernel,
            padding=self.padding,
            groups=self.channels,
        )
        return oriented + self.refine(oriented)


class COPES(nn.Module):
    """COPE-S: change-guided directional detail enhancement.

    Args:
        channels: Channels of the projected decoder feature ``z_k``.
        feature_channels: Channels of each raw encoder feature at this scale.
        enabled: False gives the branch-removal ablation and returns zeros.
        use_change_guide: Enable the CGG spatial change mask.
        use_direction_reweight: Enable the DRS sample-wise orientation weights.
        use_residual: Enable RCF residual output; otherwise return the projection.
    """

    def __init__(
        self,
        channels=128,
        feature_channels=None,
        num_orientations=4,
        kernel_size=5,
        sigma=2.0,
        wavelength=3.0,
        aspect_ratio=0.5,
        reduction=8,
        residual_scale=0.1,
        enabled=True,
        use_change_guide=True,
        use_direction_reweight=True,
        use_residual=True,
    ):
        super().__init__()
        feature_channels = feature_channels or channels
        if num_orientations != 4:
            raise ValueError("The current COPE-S implementation uses 4 orientations")
        if channels % num_orientations != 0:
            raise ValueError("channels must be divisible by num_orientations")

        self.channels = channels
        self.num_orientations = num_orientations
        self.enabled = enabled
        self.use_change_guide = use_change_guide
        self.use_direction_reweight = use_direction_reweight
        self.use_residual = use_residual

        self.pre_projection = nn.Sequential(
            nn.Conv2d(feature_channels, channels, kernel_size=1, bias=False),
            ChannelLayerNorm2d(channels),
        )
        self.post_projection = nn.Sequential(
            nn.Conv2d(feature_channels, channels, kernel_size=1, bias=False),
            ChannelLayerNorm2d(channels),
        )
        self.change_mask = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        expert_channels = channels // num_orientations
        orientations = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
        self.experts = nn.ModuleList(
            [
                OrientedGaborExpert(
                    expert_channels,
                    theta,
                    kernel_size=kernel_size,
                    sigma=sigma,
                    wavelength=wavelength,
                    aspect_ratio=aspect_ratio,
                )
                for theta in orientations
            ]
        )

        hidden_channels = max(channels // reduction, num_orientations)
        self.direction_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                num_orientations,
                kernel_size=1,
                bias=True,
            ),
        )

        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def _channel_shuffle(self, x):
        batch, channels, height, width = x.shape
        groups = self.num_orientations
        x = x.reshape(batch, groups, channels // groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch, channels, height, width)

    def forward(self, z_k, feat_pre, feat_post, return_aux=False):
        if not self.enabled:
            output = torch.zeros_like(z_k)
            if return_aux:
                return output, {}
            return output
        if feat_pre.shape[-2:] != z_k.shape[-2:]:
            feat_pre = F.interpolate(
                feat_pre,
                size=z_k.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if feat_post.shape[-2:] != z_k.shape[-2:]:
            feat_post = F.interpolate(
                feat_post,
                size=z_k.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        pre_aligned = self.pre_projection(feat_pre)
        post_aligned = self.post_projection(feat_post)
        change_feature = torch.abs(pre_aligned - post_aligned)

        if self.use_change_guide:
            spatial_weight = torch.sigmoid(self.change_mask(change_feature))
        else:
            spatial_weight = torch.ones_like(change_feature[:, :1])

        if self.use_direction_reweight:
            direction_weight = torch.softmax(
                self.direction_selector(change_feature),
                dim=1,
            )
        else:
            direction_weight = change_feature.new_full(
                (change_feature.shape[0], self.num_orientations, 1, 1),
                1.0 / self.num_orientations,
            )

        chunks = torch.chunk(z_k, self.num_orientations, dim=1)
        responses = []
        for index, (chunk, expert) in enumerate(zip(chunks, self.experts)):
            response = expert(chunk)
            response = response * spatial_weight
            response = response * direction_weight[:, index : index + 1]
            responses.append(response)

        enhanced = torch.cat(responses, dim=1)
        enhanced = self._channel_shuffle(enhanced)
        enhanced = self.output_projection(enhanced)
        output = z_k + self.residual_scale * enhanced if self.use_residual else enhanced

        if return_aux:
            return output, {
                "change_mask": spatial_weight,
                "direction_weights": direction_weight,
            }
        return output


ChangeOrientedPyramidEnhancementS = COPES
