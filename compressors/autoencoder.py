"""Autoencoder compressor — plugs the CNN autoencoder into the compressor registry.

Handles compression to .cae (Convolutional AutoEncoder) bitstream files
and decompression back to standard image formats.
"""

import os

import numpy as np
import torch

from compressors.base import BaseCompressor
from autoencoder.engine.codec import load_model, encode, decode

# ── Custom file extension for the compressed bitstream ───────────────────────
CAE_EXTENSION = ".cae"


class AutoencoderCompressor(BaseCompressor):
    """Image compression using a pre-trained convolutional autoencoder
    with 8-bit latent quantization.

    Compress: image → encoder → 8-bit quantize → save .cae payload
    Decompress: load .cae → dequantize → decoder → save image
    """

    name = "autoencoder"

    def __init__(self):
        self._model = None
        self._model_path = None
        self._model_type = None

    def _ensure_model(self, model_path: str, model_type: str) -> None:
        """Lazy-load model weights on first use (or if path/type changes)."""
        if (self._model is None
                or self._model_path != model_path
                or self._model_type != model_type):
            if not os.path.isfile(model_path):
                raise FileNotFoundError(
                    f"Model weights not found at '{model_path}'. "
                )
            self._model = load_model(model_path, model_type)
            self._model_path = model_path
            self._model_type = model_type

    def compress(self, image: np.ndarray, **params) -> np.ndarray:
        """Compress an image: encode → quantize → save .cae file.

        Also returns the reconstructed image (via decode) so the CLI
        can display a preview / compute stats.

        Args:
            image:  (H, W, 3) uint8 numpy array.
            **params:
                model_path:  Path to .pth weights file.
                output_path: Where to save the .cae bitstream.

        Returns:
            (H, W, 3) uint8 numpy array — the reconstructed preview.
        """
        defaults = self.default_params()
        model_path = params.get("model_path", defaults["model_path"])
        model_type = params.get("model_type", defaults["model_type"])
        output_path = params.get("output_path", "compressed.cae")
        self._ensure_model(model_path, model_type)

        # Encode → quantize → payload dict
        payload = encode(image, self._model)

        # Save the quantized payload as a .cae bitstream
        cae_path = _ensure_cae_extension(output_path)
        torch.save(payload, cae_path)
        print(f"Saved compressed bitstream -> {cae_path}")

        # Decode for preview / stats
        reconstructed = decode(payload, self._model)
        return reconstructed

    def decompress(self, input_path: str, **params) -> np.ndarray:
        """Decompress a .cae file back to an image.

        Args:
            input_path: Path to the .cae bitstream.
            **params:
                model_path: Path to .pth weights file.

        Returns:
            (H, W, 3) uint8 numpy array — the reconstructed image.
        """
        defaults = self.default_params()
        model_path = params.get("model_path", defaults["model_path"])
        model_type = params.get("model_type", defaults["model_type"])
        self._ensure_model(model_path, model_type)

        payload = torch.load(input_path, map_location="cpu", weights_only=False)
        return decode(payload, self._model)

    @staticmethod
    def default_params() -> dict:
        return {
            "model_type": "residual",
            "model_path": "weights/celeb_res_perceptual.pth",
        }


def _ensure_cae_extension(path: str) -> str:
    """Swap the file extension to .cae if it isn't already."""
    base, ext = os.path.splitext(path)
    if ext.lower() != CAE_EXTENSION:
        return base + CAE_EXTENSION
    return path
