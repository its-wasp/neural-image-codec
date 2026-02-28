"""CNN Autoencoder model definitions.

Two architectures are available:

  ConvAutoencoder (baseline):
    Encoder: 256x256x3 -> 16x16x256  (4x stride-2 Conv2d)
    Decoder: 16x16x256 -> 256x256x3  (4x stride-2 ConvTranspose2d)

  ResAutoencoder (residual):
    Same spatial structure as baseline, but with ResidualBlocks
    after each conv layer for better gradient flow and sharper
    reconstructions. Trained with hybrid loss (MSE + SSIM + VGG).
"""

import torch.nn as nn


# ── Shared Building Blocks ───────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """Two-conv residual block with skip connection.

    Input passes through two Conv2d layers and is added back
    to the output (skip connection), followed by ReLU.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


# ── Model Architectures ─────────────────────────────────────────────────────


class ConvAutoencoder(nn.Module):
    """Baseline convolutional autoencoder (no residual connections).

    Trained on CelebA-HQ 256x256 with MSE loss.
    Weights: celeb_ae_engine.pth (~3 MB)
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ResAutoencoder(nn.Module):
    """Residual autoencoder with skip connections.

    Same spatial structure as ConvAutoencoder but with ResidualBlocks
    after each conv/deconv layer. Trained on CelebA-HQ 256x256 with
    hybrid loss: 0.7*MSE + 1.4*SSIM + 2.0*VGG_perceptual.

    Weights: celeb_res_perceptual.pth (~6 MB)
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),      # 256 -> 128
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # 128 -> 64
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),    # 64  -> 32
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),   # 32  -> 16
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            nn.ReLU(inplace=True),
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),   # 32 -> 64
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),    # 64 -> 128
            nn.ReLU(inplace=True),
            ResidualBlock(32),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),     # 128 -> 256
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Factory ──────────────────────────────────────────────────────────────────

_MODELS = {
    "baseline": ConvAutoencoder,
    "residual": ResAutoencoder,
}


def get_model(model_type: str = "residual") -> nn.Module:
    """Instantiate an autoencoder by name.

    Args:
        model_type: 'baseline' (ConvAutoencoder) or 'residual' (ResAutoencoder).

    Returns:
        An uninitialized model instance.
    """
    if model_type not in _MODELS:
        available = ", ".join(_MODELS.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Available: {available}")
    return _MODELS[model_type]()
