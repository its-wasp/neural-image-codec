"""CNN Autoencoder model definition.

Architecture matches the Kaggle-trained model (celeb_ae_engine.pth):
  Encoder: 256×256×3 → 16×16×256  (4× stride-2 Conv2d)
  Decoder: 16×16×256 → 256×256×3  (4× stride-2 ConvTranspose2d)
"""

import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for image compression.

    Trained on CelebA-HQ 256×256 images. The encoder compresses a
    256×256×3 RGB image into a 16×16×256 latent tensor, and the decoder
    reconstructs it back to 256×256×3.
    """

    def __init__(self):
        super().__init__()

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16

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

        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
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
