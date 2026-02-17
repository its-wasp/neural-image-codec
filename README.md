# 🗜️ ImageCompression

A **modular, extensible** image compression toolkit supporting multiple compression techniques through a plugin-based architecture. Currently ships with **SVD** (traditional linear algebra) and **CNN Autoencoder** (deep learning with 8-bit quantization) compressors.

---

## ✨ Features

- **Pluggable Architecture** — Add new compression techniques by simply dropping a Python file into `compressors/`. Auto-discovered via the registry.
- **SVD Compression** — Classic truncated Singular Value Decomposition. Lossless-to-lossy control via the `k` parameter (number of singular values retained).
- **CNN Autoencoder Compression** — Deep learning-based compression using a convolutional autoencoder trained on CelebA-HQ 256×256 images.
  - **8-bit Quantization** — Latent representations are min-max quantized to `uint8`, reducing the bitstream to ~65 KB per image.
  - **Custom `.cae` Format** — Compressed bitstreams are saved as `.cae` (Convolutional AutoEncoder) files containing the quantized latent, scaling constants, and original dimensions.
  - **GPU Acceleration** — Automatically uses CUDA GPU if available, falls back to CPU.
- **Two-Way Pipeline** — Compress images to `.cae` bitstreams and decompress them back to standard image formats (PNG/JPG).
- **Quality Metrics** — Built-in PSNR and SSIM calculations for evaluating reconstruction quality.
- **CLI Interface** — Simple command-line tool for all operations.

---

## 📁 Project Structure

```
ImageCompression/
│
├── main.py                          # CLI entry point
│
├── compressors/                     # Compression plugin system
│   ├── __init__.py                  # Auto-discovery registry
│   ├── base.py                      # BaseCompressor abstract class
│   ├── svd.py                       # SVD compressor
│   └── autoencoder.py               # Autoencoder compressor (thin wrapper)
│
├── autoencoder/                     # Autoencoder package
│   ├── core/
│   │   ├── model.py                 # ConvAutoencoder (nn.Module)
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
│   └── celeb_ae_engine.pth          # Pre-trained autoencoder weights
│
└── notebooks/                       # Training & inference notebooks
    ├── train-image-compression-cnn-autoencoder.ipynb
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
git clone <repo-url>
cd ImageCompression

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install numpy pillow scipy torch torchvision
```

> **Note:** PyTorch will automatically use your GPU (CUDA) if available. For a CPU-only install (~200MB lighter), use:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### Model Weights

The pre-trained model weights (~3 MB) are hosted on [GitHub Releases](https://github.com/its-wasp/image-compression-library/releases) and are **not** included in the git repo.

**Option A: Auto-download (recommended)**

```bash
python download_weights.py
```

This downloads all required weights into the `weights/` directory automatically.

**Option B: Manual download**

1. Go to the [latest release](https://github.com/its-wasp/image-compression-library/releases/latest)
2. Download `celeb_ae_engine.pth`
3. Place it in the `weights/` directory

> **For maintainers:** To publish weights for a new release, create a GitHub Release (e.g., `v1.0`), and attach the `.pth` file as a release asset. Then update the `MODELS` dict in `download_weights.py` with the new filename and tag.

---

## 📖 Usage

### List Available Methods

```bash
python main.py --list
```

```
Available compression methods:
  • autoencoder       params: model_path=weights/celeb_ae_engine.pth
  • svd               params: k=100
```

### SVD Compression

```bash
# Default (k=100 singular values)
python main.py -i input.png -o output.jpg -m svd

# Custom k value (lower = more compression, more blur)
python main.py -i input.png -o output.jpg -m svd --param k=50

# Higher quality (more singular values retained)
python main.py -i input.png -o output.jpg -m svd --param k=200
```

### Autoencoder Compression

```bash
# Compress to .cae bitstream (also saves a preview JPEG)
python main.py -i input.png -o compressed.cae -m autoencoder
```

```
Saved compressed bitstream -> compressed.cae
Original Size:   236.2 KB
Bitstream Size:  65.6 KB  (compressed.cae)
Compression:     27.8%
Preview saved:   compressed_preview.jpg
```

```bash
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

---

## 🧠 How It Works

### SVD Compression

Each RGB channel is decomposed using **Singular Value Decomposition**:

```
Channel = U × Σ × Vᵀ
```

By keeping only the top-`k` singular values, the image is reconstructed with fewer components. Lower `k` = more compression but more blur.

### Autoencoder Compression

A **convolutional autoencoder** trained on CelebA-HQ 256×256 face images:

```
Input (256×256×3)
  → Encoder: 4× Conv2d layers (stride=2, ReLU)
  → Latent: 16×16×256 tensor
  → 8-bit Quantization: float32 → uint8 (min-max scaling)
  → Saved as .cae file (~65 KB)

.cae file
  → Dequantize: uint8 → float32
  → Decoder: 4× ConvTranspose2d layers (stride=2, Sigmoid)
  → Reconstructed Image (256×256×3)
  → Resize back to original dimensions
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

Then use it immediately:

```bash
python main.py -i input.png -o output.jpg -m wavelet --param level=5
```

---

## 📊 Compression Comparison

| Method | Type | Compressed Size | Control |
|--------|------|----------------|---------|
| **SVD** | Traditional (linear algebra) | Varies with `k` | `k` parameter (1–∞) |
| **Autoencoder** | Deep Learning (CNN) | ~65 KB (`.cae` file) | Fixed architecture |

---

## 📝 License

This project is for educational and research purposes.
