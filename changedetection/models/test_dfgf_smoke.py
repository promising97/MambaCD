"""Minimal forward/backward smoke test for DirectionalFrequencyGatedGabor."""

import torch

from MambaCD.changedetection.models.StarIR_gabor_DFGF import (
    DirectionalFrequencyGatedGabor,
)


def run_case(height, width, device):
    block = DirectionalFrequencyGatedGabor(dim=128).to(device)
    x = torch.randn(
        2, 128, height, width, device=device, requires_grad=True
    )
    y = block(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()

    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None
        for parameter in block.parameters()
        if parameter.requires_grad
    )
    print(
        f"PASS: device={device}, shape={tuple(x.shape)}, "
        f"params={sum(p.numel() for p in block.parameters())}"
    )


if __name__ == "__main__":
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    run_case(32, 32, target_device)
    run_case(31, 27, target_device)
