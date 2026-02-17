"""8-bit min-max quantization for autoencoder latent tensors.

Scales float32 latent values to uint8 [0, 255] using per-tensor min/max,
preserving enough fidelity for the decoder while shrinking storage by 4×.
"""

import torch

_EPS = 1e-8  # Guard against division-by-zero on flat latent regions


def quantize_latent(
    latent: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    """Quantize a float32 latent tensor to uint8.

    Args:
        latent: Float32 tensor of any shape (typically 1×256×16×16).

    Returns:
        (quantized_uint8, min_val, max_val) — the uint8 tensor and
        the scaling constants needed to dequantize later.
    """
    l_min = latent.min().item()
    l_max = latent.max().item()

    scale = (l_max - l_min) + _EPS
    quantized = ((latent - l_min) / scale * 255).to(torch.uint8)

    return quantized, l_min, l_max


def dequantize_latent(
    quantized: torch.Tensor,
    l_min: float,
    l_max: float,
) -> torch.Tensor:
    """Dequantize a uint8 latent tensor back to float32.

    Args:
        quantized: uint8 tensor produced by quantize_latent().
        l_min: Original minimum value.
        l_max: Original maximum value.

    Returns:
        Float32 tensor approximating the original latent values.
    """
    return (quantized.to(torch.float32) / 255.0) * (l_max - l_min) + l_min
