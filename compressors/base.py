from abc import ABC, abstractmethod
import numpy as np


class BaseCompressor(ABC):
    """Abstract base class for all image compression techniques.

    To create a new compressor:
        1. Create a new file in the compressors/ directory
        2. Subclass BaseCompressor
        3. Set the `name` class attribute
        4. Implement `compress()` and `default_params()`
    It will be auto-discovered and available via the CLI.
    """

    name: str = ""  # Human-readable name, e.g. "svd", "autoencoder"

    @abstractmethod
    def default_params(self) -> dict:
        """Return a dict of default parameters for this compression technique."""
        ...

    @abstractmethod
    def compress(self, image: np.ndarray, **params) -> np.ndarray:
        """Compress an image (H x W x C uint8 numpy array) and return the result.

        Args:
            image: Input image as a numpy array with shape (H, W, C) and dtype uint8.
            **params: Technique-specific parameters (see default_params()).

        Returns:
            Compressed image as a numpy array with shape (H, W, C) and dtype uint8.
        """
        ...

    def decompress(self, input_path: str, **params) -> np.ndarray:
        """Decompress from a compressed file back to an image.

        Optional — only compressors that produce custom file formats
        (e.g. .cae) need to implement this. Raises NotImplementedError
        by default.

        Args:
            input_path: Path to the compressed file.
            **params: Technique-specific parameters.

        Returns:
            Reconstructed image as a numpy array (H, W, C) uint8.
        """
        raise NotImplementedError(
            f"'{self.name}' compressor does not support decompression."
        )
