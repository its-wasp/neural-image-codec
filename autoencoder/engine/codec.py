"""Encode / decode pipeline for the CNN autoencoder.

Handles image preprocessing, 8-bit quantization, model inference,
and postprocessing. Automatically selects GPU (CUDA) if available,
otherwise falls back to CPU.
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from autoencoder.core.model import ConvAutoencoder
from autoencoder.core.quantization import quantize_latent, dequantize_latent

# ── Device selection ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Fixed input size expected by the model ───────────────────────────────────
MODEL_INPUT_SIZE = (256, 256)

# ── Preprocessing transform ─────────────────────────────────────────────────
_preprocess = transforms.Compose([
    transforms.Resize(MODEL_INPUT_SIZE),
    transforms.ToTensor(),  # HWC uint8 [0,255] → CHW float [0,1]
])


def load_model(weights_path: str) -> ConvAutoencoder:
    """Load a ConvAutoencoder with pre-trained weights.

    Args:
        weights_path: Path to the .pth state-dict file.

    Returns:
        The model in eval mode on the best available device.
    """
    model = ConvAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    return model


def encode(image: np.ndarray, model: ConvAutoencoder) -> dict:
    """Compress an image into a quantized payload.

    Pipeline: preprocess → encoder → 8-bit quantize.

    Args:
        image: (H, W, 3) uint8 numpy array.
        model: A loaded ConvAutoencoder.

    Returns:
        Payload dict with keys:
            data:          uint8 tensor (1, 256, 16, 16)
            min:           float — latent minimum (for dequantization)
            max:           float — latent maximum (for dequantization)
            original_size: (W, H) tuple — original image dimensions
    """
    original_h, original_w = image.shape[:2]

    pil_img = Image.fromarray(image)
    tensor = _preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        latent = model.encoder(tensor)

    quantized, l_min, l_max = quantize_latent(latent)

    return {
        "data": quantized.cpu(),
        "min": l_min,
        "max": l_max,
        "original_size": (original_w, original_h),
    }


def decode(payload: dict, model: ConvAutoencoder) -> np.ndarray:
    """Reconstruct an image from a quantized payload.

    Pipeline: dequantize → decoder → postprocess → resize to original.

    Args:
        payload: Dict from encode() or loaded from a .cae file.
        model:   A loaded ConvAutoencoder.

    Returns:
        (H, W, 3) uint8 numpy array at the original resolution.
    """
    latent = dequantize_latent(
        payload["data"].to(DEVICE), payload["min"], payload["max"]
    )

    with torch.no_grad():
        output = model.decoder(latent)

    # CHW float [0,1] → HWC uint8 [0,255]
    img_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # Resize back to original dimensions
    original_size = payload["original_size"]
    pil_out = Image.fromarray(img_np)
    pil_out = pil_out.resize(original_size, Image.LANCZOS)

    return np.array(pil_out)
