# 🗜️ ImageCompression

A **modular, extensible** image compression toolkit supporting multiple compression techniques through a plugin-based architecture. Ships with **SVD** (traditional linear algebra) and **CNN Autoencoder** (deep learning with 8-bit quantization) compressors.

---

## ✨ Features

- **Pluggable Architecture** — Add new compression techniques by dropping a Python file into `compressors/`. Auto-discovered via the registry.
- **SVD Compression** — Classic truncated Singular Value Decomposition with tunable `k` parameter.
- **CNN Autoencoder Compression** — Two model variants:
  - **Baseline** — Standard convolutional autoencoder trained with MSE loss
  - **Residual** *(default)* — Residual blocks + hybrid loss (MSE + SSIM + VGG perceptual) for sharper reconstructions
- **8-bit Quantization** — Latent representations are min-max quantized to `uint8`, producing compact ~65 KB `.cae` bitstream files.
- **GPU Acceleration** — Automatically uses CUDA GPU if available, falls back to CPU.
- **Two-Way Pipeline** — Compress to `.cae` bitstreams and decompress back to standard image formats.
- **Quality Metrics** — Built-in PSNR and SSIM calculations.

---

## 📁 Project Structure

```
ImageCompression/
│
├── main.py                          # CLI entry point
├── download_weights.py              # Auto-download model weights
├── requirements.txt                 # Python dependencies
│
├── compressors/                     # Compression plugin system
│   ├── __init__.py                  # Auto-discovery registry
│   ├── base.py                      # BaseCompressor abstract class
│   ├── svd.py                       # SVD compressor
│   └── autoencoder.py               # Autoencoder compressor (thin wrapper)
│
├── autoencoder/                     # Autoencoder package
│   ├── core/
│   │   ├── model.py                 # ConvAutoencoder + ResAutoencoder models
│   │   └── quantization.py          # 8-bit min-max quantization
│   ├── engine/
│   │   └── codec.py                 # Encode/decode pipeline
│   └── utils/
│       └── metrics.py               # PSNR, SSIM calculations
│
├── utils/                           # General utilities
│   └── image_io.py                  # Image load/save/stats helpers
│
├── weights/                         # Model weights (git-ignored)
│   ├── celeb_ae_engine.pth          # Baseline model (~3 MB)
│   └── celeb_res_perceptual.pth     # Residual + perceptual model (~6 MB)
│
└── notebooks/                       # Kaggle training notebooks
    ├── train-image-compression-cnn-autoencoder.ipynb
    ├── train-image-compression-cnn-autoencoder-with-vgg-perceptual.ipynb
    └── inference-imagecompression-cnn-autoencoder.ipynb
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/its-wasp/image-compression-library.git
cd ImageCompression

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download model weights
python download_weights.py
```

> **Note:** PyTorch will automatically use your GPU (CUDA) if available. For a CPU-only install (~200MB lighter), use:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### Model Weights

The pre-trained model weights are hosted on [GitHub Releases](https://github.com/its-wasp/image-compression-library/releases) and are **not** included in the git repo.

| Model | File | Size | Trained With |
|-------|------|------|-------------|
| Baseline | `celeb_ae_engine.pth` | ~3 MB | MSE loss |
| Residual | `celeb_res_perceptual.pth` | ~6 MB | MSE + SSIM + VGG perceptual loss |

**Option A: Auto-download (recommended)**

```bash
python download_weights.py
```

**Option B: Manual download**

1. Go to the [latest release](https://github.com/its-wasp/image-compression-library/releases/latest)
2. Download the `.pth` files and place them in the `weights/` directory

> **For maintainers:** To publish weights, create a GitHub Release, attach the `.pth` files, and update the `MODELS` dict in `download_weights.py`.

---

## 📖 Usage

### List Available Methods

```bash
python main.py --list
```

```
Available compression methods:
  • autoencoder       params: model_type=residual, model_path=weights/celeb_res_perceptual.pth
  • svd               params: k=100
```

### SVD Compression

```bash
# Default (k=100 singular values)
python main.py -i input.png -o output.jpg -m svd

# More compression (lower k = more blur)
python main.py -i input.png -o output.jpg -m svd --param k=50

# Higher quality
python main.py -i input.png -o output.jpg -m svd --param k=200
```

### Autoencoder Compression

```bash
# Compress with residual model (default — best quality)
python main.py -i input.png -o compressed.cae -m autoencoder

# Compress with baseline model
python main.py -i input.png -o compressed.cae -m autoencoder \
  --param model_type=baseline \
  --param model_path=weights/celeb_ae_engine.pth

# Decompress .cae back to a viewable image
python main.py -i compressed.cae -o reconstructed.png -m autoencoder --decompress
```

### CLI Reference

| Flag | Description |
|------|-------------|
| `-i, --input` | Input image path (or `.cae` file for decompression) |
| `-o, --output` | Output path |
| `-m, --method` | Compression method: `svd`, `autoencoder` (default: `svd`) |
| `-q, --quality` | JPEG save quality 1–95 (default: 85) |
| `--param key=value` | Method-specific parameters (repeatable) |
| `--list` | List all available compression methods |
| `--decompress` | Decompress a `.cae` file back to an image |

#### Autoencoder Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `model_type` | `residual`, `baseline` | `residual` | Model architecture to use |
| `model_path` | path to `.pth` | `weights/celeb_res_perceptual.pth` | Path to model weights |

---

## 🧠 How It Works

### SVD Compression

Each RGB channel is decomposed using **Singular Value Decomposition**:

```
Channel = U × Σ × Vᵀ
```

By keeping only the top-`k` singular values, the image is reconstructed with fewer components. Lower `k` = more compression but more blur.

### Autoencoder Compression

Two model architectures are available, both trained on CelebA-HQ 256×256 face images:

**Baseline (`ConvAutoencoder`):**
- 4 Conv2d encoder layers + 4 ConvTranspose2d decoder layers
- Trained with MSE loss only

**Residual (`ResAutoencoder`):**
- Same structure + ResidualBlocks (skip connections) after each layer
- Trained with hybrid loss: `0.7×MSE + 1.4×SSIM + 2.0×VGG_perceptual`
- Produces sharper, more perceptually pleasing reconstructions

**Compression pipeline:**

```
Input Image
  → Resize to 256×256
  → Encoder (Conv2d layers, ±ResidualBlocks)
  → Latent: 16×16×256 float32 tensor
  → 8-bit Quantization: float32 → uint8 (min-max scaling)
  → Saved as .cae file (~65 KB)

.cae file
  → Dequantize: uint8 → float32
  → Decoder (ConvTranspose2d layers, ±ResidualBlocks)
  → Resize back to original dimensions
  → Reconstructed Image
```

**Key properties:**
- **Lossy** — the reconstruction is approximate, not pixel-perfect
- **Fixed internal resolution** — images are resized to 256×256 for the model, then resized back
- **Trained on faces** — works best on face images but generalises to other content
- **8-bit quantized** — latent space is stored as uint8 with < 0.2 dB PSNR drop vs float32

---

## 🔌 Adding a New Compressor

1. Create a new file in `compressors/` (e.g., `compressors/wavelet.py`)
2. Subclass `BaseCompressor`
3. Set the `name` class attribute
4. Implement `compress()` and `default_params()`
5. It will be auto-discovered — no registration code needed!

```python
import numpy as np
from compressors.base import BaseCompressor


class WaveletCompressor(BaseCompressor):
    name = "wavelet"

    def default_params(self) -> dict:
        return {"level": 3}

    def compress(self, image: np.ndarray, **params) -> np.ndarray:
        level = params.get("level", 3)
        # ... your compression logic here ...
        return compressed_image
```

```bash
python main.py -i input.png -o output.jpg -m wavelet --param level=5
```

---

## 📊 Compression Comparison

| Method | Type | Compressed Size | Quality Control |
|--------|------|----------------|----------------|
| **SVD** | Linear algebra | Varies with `k` | `k` parameter |
| **Autoencoder (baseline)** | Deep Learning | ~65 KB (`.cae`) | MSE-trained |
| **Autoencoder (residual)** | Deep Learning | ~65 KB (`.cae`) | MSE + SSIM + VGG perceptual |

---

## 📝 License

This project is for educational and research purposes.
