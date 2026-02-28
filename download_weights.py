"""Download pre-trained model weights from GitHub Releases.

Usage:
    python download_weights.py

Downloads the celeb_ae_engine.pth file from the latest GitHub Release
and places it in the weights/ directory.
"""

import os
import sys
import urllib.request
import urllib.error

# ── Configuration ────────────────────────────────────────────────────────────
REPO = "its-wasp/neural-image-codec"
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# Map of filename → GitHub Release tag
MODELS = {
    "celeb_ae_engine.pth": "v1.0",
}


def download_file(url: str, dest: str) -> None:
    """Download a file with a progress indicator."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "=" * int(pct // 2) + ">" + " " * (50 - int(pct // 2))
            sys.stdout.write(f"\r  [{bar}] {pct:.0f}%")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress bar
    except urllib.error.HTTPError as e:
        print(f"\nError: HTTP {e.code} — {e.reason}")
        print(f"Make sure the release '{url}' exists on GitHub.")
        sys.exit(1)


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    for filename, tag in MODELS.items():
        dest = os.path.join(WEIGHTS_DIR, filename)

        if os.path.isfile(dest):
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"Already exists: {dest} ({size_mb:.1f} MB) — skipping.")
            continue

        url = f"https://github.com/{REPO}/releases/download/{tag}/{filename}"
        download_file(url, dest)

        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"Done: {dest} ({size_mb:.1f} MB)")

    print("\nAll weights downloaded successfully!")


if __name__ == "__main__":
    main()
