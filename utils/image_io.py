"""Image loading, saving, and comparison utilities."""

import os

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image file and return it as a uint8 numpy array (H, W, C)."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(
    image: np.ndarray,
    path: str,
    quality: int = 70,
    optimize: bool = True,
) -> None:
    """Save a numpy array as a JPEG image.

    Args:
        image: (H, W, C) uint8 numpy array.
        path: Output file path.
        quality: JPEG quality (1-95). Lower = smaller file.
    """
    img = Image.fromarray(image)
    img.save(path, "JPEG", quality=quality, optimize=True)


def print_stats(original_path: str, compressed_path: str) -> None:
    """Print a before / after size comparison."""
    orig = os.path.getsize(original_path) / (1024 * 1024)
    comp = os.path.getsize(compressed_path) / (1024 * 1024)
    saved = ((orig - comp) / orig) * 100 if orig else 0

    print(f"Original Size:   {orig:.2f} MB")
    print(f"Compressed Size: {comp:.2f} MB")
    print(f"Space Saved:     {saved:.1f}%")
