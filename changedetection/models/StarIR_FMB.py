import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from  mamba_ssm import Mamba

from MambaCD.changedetection.models.Spatial_Mamba import SpatialMambaBlock
from MambaCD.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Feed-forward network
class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, channel_first, norm_layer, **kwargs):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim

        self.spatial = nn.Sequential(
            # nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=hidden_features * 2, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate=0., d_state=kwargs['ssm_d_state'], dt_init="random",
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):

        # x = self.to_hidden(x)

        # x = self.project_in(x)
        # x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = self.project_in(x)
        x = self.dwconv(x)

        x1, x2 = self.spatial(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
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

##########################################################################
## Star module
class StarModule(nn.Module):
    def __init__(self, dim, bias, channel_first, norm_layer, **kwargs):
        super(StarModule, self).__init__()

        self.patch_size = 8
        self.dim = dim

        self.SMB = nn.Sequential(
            # nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3] * 2, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            SpatialMambaBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer,
                              attn_drop_rate=0., d_state=kwargs['ssm_d_state'], dt_init="random",
                              use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.to_hidden = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)


        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

        self.fft_FSAS = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

        self.spatial = SpatialOperation(dim)

        self.plain_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )


    def forward(self, x):
        B, C, H, W = x.shape
        # hidden = self.to_hidden(x)
        # x_seq = rearrange(x, 'b c h w -> b (h w) c')
        #
        # # forward
        # out_seq = self.to_hidden(x_seq)
        #
        # # reshape back 3D -> 4D
        # hidden = rearrange(out_seq, 'b (h w) c -> b c h w', h=H, w=W)

        hidden = self.to_hidden(x)
        q, v = self.to_hidden_dw(hidden).split([self.dim, self.dim], dim=1)

        v = self.SMB(v)


        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)

        q_fft = torch.fft.rfft2(q_patch.float())

        out = q_fft * self.fft_FSAS
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        # star
        out = self.spatial(v) * out
        # CMU
        out = self.plain_channel(out) * out
        out = self.project_out(out)

        return out

##########################################################################
class StarBlock(nn.Module):
    def __init__(self, dim, channel_first, norm_layer, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',  **kwargs):
        super(StarBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = StarModule(dim, bias, channel_first=channel_first, norm_layer=norm_layer, **kwargs)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias, channel_first=channel_first, norm_layer=norm_layer, **kwargs)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Fuse(nn.Module):
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.conv = nn.Conv2d(n_feat * 2, n_feat, 1, 1, 0)

    def forward(self, enc, dnc):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- StarIR -----------------------
class StarIR(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 48,
        num_blocks = [3,3,7],
        num_refinement_blocks = 4,
        ffn_expansion_factor = 3,
        bias = False
    ):
        super(StarIR, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            StarBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            StarBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = nn.Sequential(*[
            StarBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            StarBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            StarBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            StarBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            StarBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias)
            for i in range(num_refinement_blocks)])

        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1