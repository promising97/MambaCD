import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
from einops import rearrange
import numbers

from mamba_ssm import Mamba


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

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)

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

class StarModule_Mod(nn.Module):
    def __init__(self, dim, bias):
        super(StarModule_Mod, self).__init__()

        self.patch_size = 8
        self.dim = dim
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
        hidden = self.to_hidden(x)

        q, v = self.to_hidden_dw(hidden).split([self.dim, self.dim], dim=1)

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

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)

        x1, x2 = self.dwconv(x).chunk(2, dim=1)

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

def main():
    # 假设输入 batch=1, channel=64, 高宽=64
    B, C, H, W = 8, 128, 8, 8
    x = torch.randn(B, C, H, W)

    # mamba = Mamba(d_model=C, d_state=64, d_conv=4, expand=2)
    #
    # # # 初始化 StarModule_Mod
    # # star_module = StarModule_Mod(dim=C, bias=True)
    # #
    # # # 前向计算
    # # out = star_module(x)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x = x.to(device)
    # mamba = mamba.to(device)
    #
    #
    # x_seq = rearrange(x, 'b c h w -> b (h w) c')
    #
    # # forward
    # out_seq = mamba(x_seq)
    #
    # # reshape back 3D -> 4D
    # out = rearrange(out_seq, 'b (h w) c -> b c h w', h=H, w=W)
    #
    # # 打印 shape
    # print(f"Input shape : {x.shape}")
    # print(f"Output shape: {out.shape}")

    dffn = DFFN(128, ffn_expansion_factor=2.66, bias=False)
    y = dffn(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")
    # to_hidden = nn.Conv2d(C, C * 2, kernel_size=1)
    # to_hidden_dw = nn.Conv2d(C * 2, C * 2, kernel_size=3, stride=1, padding=1, groups=C * 2)
    # hidden = to_hidden(x)
    # s = to_hidden_dw(hidden)
    # q, v = to_hidden_dw(hidden).split([C, C], dim=1)
    #
    # print(f"Input shape : {x.shape}")
    # print(f"hidden shape : {hidden.shape}")
    # print(f"s shape : {s.shape}")
    # print(f"Output shape: {q.shape}")
    # print(f"Output shape: {v.shape}")

if __name__ == "__main__":
    main()
