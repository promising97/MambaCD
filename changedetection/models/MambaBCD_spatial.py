import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
from MambaCD.changedetection.models.Spatial_Mamba_backbone import Backbone_SpatialMamba
from MambaCD.changedetection.models.Mamba_backbone import Backbone_VSSM
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import glob
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
#from MambaCD.changedetection.models.ChangeDecoder import ChangeDecoder
#from MambaCD.changedetection.models.ChangeDecoder_spatialMamba import ChangeDecoder
#from MambaCD.changedetection.models.ChangeDecoder_spatialMamba_GFM import ChangeDecoder
#from MambaCD.changedetection.models.ChangeDecoder_spatialMamba_small import ChangeDecoder
#from MambaCD.changedetection.models.ChangeDecoder_spatialMamba_small_ICFKFusion import ChangeDecoder

#from MambaCD.changedetection.models.ChangeDecoder_spatialMamba_small_ICFKFusion_DualTransformer_MDTA import ChangeDecoder
# from MambaCD.changedetection.models.ChangeDecoder_spatialMamba_small_ICFKFusion_DualTransformer_MDTA_four_blocks_with_one_embeding_layer import ChangeDecoder
# from MambaCD.changedetection.models.ChangeDecoder_hny import ChangeDecoder
from MambaCD.changedetection.models.ChangeDecoder_Star_GaborFPU import ChangeDecoder
#from MambaCD.changedetection.models.ChangeDecoder_SFSConv import ChangeDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count


def _resolve_pretrained_path(pretrained):
    if pretrained:
        if os.path.exists(pretrained):
            return pretrained
        return None

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    search_dirs = [
        repo_root,
        os.path.join(repo_root, 'pretrained_weight'),
        os.path.join(repo_root, 'pretrained_weights'),
        os.path.join(repo_root, 'changedetection', 'pretrained_weight'),
        os.path.join(repo_root, 'changedetection', 'pretrained_weights'),
        os.getcwd(),
        os.path.join(os.getcwd(), 'pretrained_weight'),
        os.path.join(os.getcwd(), 'pretrained_weights'),
    ]

    common_names = [
        'vssm_tiny_0229_ckpt_epoch_222.pth',
        'vssm_small_0229_ckpt_epoch_222.pth',
        'vssm_base_0229_ckpt_epoch_237.pth',
        'vmamba_tiny_0229_ckpt_epoch_222.pth',
        'vmamba_small_0229_ckpt_epoch_222.pth',
        'vmamba_base_0229_ckpt_epoch_237.pth',
        'spatialmamba_small.pth',
        'spatialmamba_small_224_1k.pth',
        'MambaBCD_spatial.pth',
    ]

    for root in search_dirs:
        for name in common_names:
            path = os.path.join(root, name)
            if os.path.exists(path):
                return path

        for pattern in ('*.pth', '*.pt', '*.ckpt'):
            matches = sorted(glob.glob(os.path.join(root, pattern)))
            if matches:
                return matches[0]

    return None


class STMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(STMambaBCD, self).__init__()
        resolved_pretrained = _resolve_pretrained_path(pretrained)
        if resolved_pretrained is not None:
            print(f"Using pretrained weight: {resolved_pretrained}")
        self.encoder = Backbone_SpatialMamba(out_indices=(0, 1, 2, 3), pretrained=resolved_pretrained, **kwargs)
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
 

        #norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        #ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        #mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)
        norm_layer: nn.Module = nn.LayerNorm
        ssm_act_layer: nn.Module = nn.SiLU
        mlp_act_layer: nn.Module = nn.GELU
        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims,
            channel_first=self.encoder.channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # Decoder processing - passing encoder outputs to the decoder
        output = self.decoder(pre_features, post_features)

        output = self.main_clf(output)
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output

