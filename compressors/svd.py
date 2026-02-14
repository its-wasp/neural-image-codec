import numpy as np
from scipy.linalg import svd

from compressors.base import BaseCompressor


class SVDCompressor(BaseCompressor):
    """Image compression using Singular Value Decomposition (SVD).

    Decomposes each color channel into U, S, Vh matrices and reconstructs
    using only the top-k singular values. Lower k → more compression / more blur.
    """

    name = "svd"

    def default_params(self) -> dict:
        return {"k": 100}

    def compress(self, image: np.ndarray, **params) -> np.ndarray:
        k = int(params.get("k", self.default_params()["k"]))

        channels = [image[:, :, c] for c in range(image.shape[2])]
        compressed_channels = [self._compress_channel(ch, k) for ch in channels]

        result = np.stack(compressed_channels, axis=2)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _compress_channel(channel: np.ndarray, k: int) -> np.ndarray:
        """Compress a single color channel using truncated SVD."""
        U, s, Vh = svd(channel, full_matrices=False)
        return U[:, :k] @ np.diag(s[:k]) @ Vh[:k, :]
