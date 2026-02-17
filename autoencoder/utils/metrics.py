"""Image quality metrics for evaluating autoencoder reconstruction."""

import numpy as np


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        original:      (H, W, C) uint8 numpy array.
        reconstructed: (H, W, C) uint8 numpy array, same shape.

    Returns:
        PSNR in dB. Returns float('inf') if images are identical.
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0**2 / mse)


def calculate_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    C1: float = 6.5025,
    C2: float = 58.5225,
) -> float:
    """Calculate (simplified) Structural Similarity Index between two images.

    Uses the mean-luminance / contrast / structure formula without
    windowing, which gives a reasonable global SSIM estimate.

    Args:
        original:      (H, W, C) uint8 numpy array.
        reconstructed: (H, W, C) uint8 numpy array, same shape.
        C1: Stabilisation constant for luminance (default: (0.01*255)^2).
        C2: Stabilisation constant for contrast  (default: (0.03*255)^2).

    Returns:
        SSIM value in [0, 1]. Higher is better.
    """
    img1 = original.astype(np.float64)
    img2 = reconstructed.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim)
