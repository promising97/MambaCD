"""SpatialMamba BCD model with the direction-frequency gated Star decoder."""

import glob
import os

import torch.nn as nn
import torch.nn.functional as F

from MambaCD.changedetection.models.ChangeDecoder_Star_GaborFPU_DFGF import (
    ChangeDecoder,
)
from MambaCD.changedetection.models.Spatial_Mamba_backbone import (
    Backbone_SpatialMamba,
)
# This model uses Backbone_SpatialMamba, so the SpatialMamba ImageNet weight
# must take priority over the VSSM weight when both files are present.
DEFAULT_SPATIALMAMBA_WEIGHT = "spatialmamba_small_224_1k.pth"


def _resolve_pretrained_path(pretrained=None):
    """Resolve the SpatialMamba encoder checkpoint.

    Args:
        pretrained:
            * False: disable encoder pretraining.
            * Existing path: use that exact checkpoint.
            * Relative path/name: search it under the repository and weight
              directories.
            * None/True: automatically select a compatible SpatialMamba
              checkpoint.
    """
    if pretrained is False:
        return None

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    search_dirs = [
        repo_root,
        os.path.join(repo_root, "pretrained_weight"),
        os.path.join(repo_root, "pretrained_weights"),
        os.path.join(repo_root, "changedetection", "pretrained_weight"),
        os.path.join(repo_root, "changedetection", "pretrained_weights"),
        os.getcwd(),
        os.path.join(os.getcwd(), "pretrained_weight"),
        os.path.join(os.getcwd(), "pretrained_weights"),
    ]

    if isinstance(pretrained, (str, os.PathLike)):
        pretrained = os.fspath(pretrained)
        explicit_candidates = [
            pretrained,
            os.path.join(repo_root, pretrained),
        ]
        explicit_candidates.extend(
            os.path.join(root, os.path.basename(pretrained))
            for root in search_dirs
        )
        for path in explicit_candidates:
            if os.path.isfile(path):
                return os.path.abspath(path)

        print(
            "Warning: the requested pretrained checkpoint was not found: "
            f"{pretrained}. Continuing without pretrained weights."
        )
        return None

    # Compatible SpatialMamba weights are intentionally searched first.
    common_names = [
        DEFAULT_SPATIALMAMBA_WEIGHT,
        "spatialmamba_small.pth",
        "MambaBCD_spatial.pth",
    ]

    for root in search_dirs:
        for name in common_names:
            path = os.path.join(root, name)
            if os.path.isfile(path):
                return os.path.abspath(path)

    # Do not silently fall back to a VSSM checkpoint: this model instantiates
    # Backbone_SpatialMamba and the two state dictionaries are not guaranteed
    # to be compatible.
    spatial_patterns = (
        "*spatialmamba*.pth",
        "*spatial_mamba*.pth",
        "*spatialmamba*.pt",
        "*spatial_mamba*.pt",
        "*spatialmamba*.ckpt",
        "*spatial_mamba*.ckpt",
    )
    for root in search_dirs:
        for pattern in spatial_patterns:
            matches = sorted(glob.glob(os.path.join(root, pattern)))
            if matches:
                return os.path.abspath(matches[0])

    return None


class STMambaBCD(nn.Module):
    def __init__(
        self,
        pretrained=None,
        use_dfgf=True,
        dfgf_use_frequency_gate=True,
        dfgf_use_direction_selector=True,
        dfgf_num_orientations=4,
        dfgf_kernel_size=5,
        dfgf_sigma=2.0,
        dfgf_wavelength=3.0,
        dfgf_gamma=0.5,
        dfgf_mask_size=16,
        dfgf_reduction=8,
        dfgf_residual_scale=0.1,
        **kwargs,
    ):
        super().__init__()

        resolved_pretrained = _resolve_pretrained_path(pretrained)
        if resolved_pretrained is not None:
            print(f"Using SpatialMamba pretrained weight: {resolved_pretrained}")
        else:
            print("SpatialMamba encoder will be initialized without pretrained weights.")

        # All DFGF arguments are explicit constructor arguments, so none of
        # them leak into Backbone_SpatialMamba through **kwargs.
        self.encoder = Backbone_SpatialMamba(
            out_indices=(0, 1, 2, 3),
            pretrained=resolved_pretrained,
            **kwargs,
        )

        norm_layer = nn.LayerNorm
        ssm_act_layer = nn.SiLU
        mlp_act_layer = nn.GELU

        # These three arguments are supplied explicitly below and must not
        # appear a second time inside **clean_kwargs.
        clean_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "norm_layer",
                "ssm_act_layer",
                "mlp_act_layer",
            }
        }

        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            use_dfgf=use_dfgf,
            dfgf_use_frequency_gate=dfgf_use_frequency_gate,
            dfgf_use_direction_selector=dfgf_use_direction_selector,
            dfgf_num_orientations=dfgf_num_orientations,
            dfgf_kernel_size=dfgf_kernel_size,
            dfgf_sigma=dfgf_sigma,
            dfgf_wavelength=dfgf_wavelength,
            dfgf_gamma=dfgf_gamma,
            dfgf_mask_size=dfgf_mask_size,
            dfgf_reduction=dfgf_reduction,
            dfgf_residual_scale=dfgf_residual_scale,
            **clean_kwargs,
        )

        self.main_clf = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=1,
        )

    @staticmethod
    def _upsample_add(x, y):
        """Retained for compatibility; the current forward does not call it."""
        _, _, height, width = y.size()
        return (
            F.interpolate(
                x,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            + y
        )

    def forward(self, pre_data, post_data):
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        output = self.decoder(pre_features, post_features)
        output = self.main_clf(output)
        output = F.interpolate(
            output,
            size=pre_data.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return output
