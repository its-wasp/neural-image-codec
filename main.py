"""Image Compression CLI — supports multiple compression techniques."""

import argparse
import os

from compressors import get_compressor, list_compressors
from utils import load_image, save_image, print_stats


def parse_params(raw: list[str] | None) -> dict:
    """Parse 'key=value' strings into a dict, casting numeric values."""
    if not raw:
        return {}
    params: dict = {}
    for item in raw:
        if "=" not in item:
            print(f"Warning: ignoring malformed param '{item}' (expected key=value)")
            continue
        key, val = item.split("=", 1)
        # Try to cast to int or float
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        params[key] = val
    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress images using pluggable compression techniques.",
    )
    parser.add_argument("-i", "--input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output path (default: compressed_<input>)")
    parser.add_argument("-m", "--method", default="svd", help="Compression method (default: svd)")
    parser.add_argument("-q", "--quality", type=int, default=85, help="JPEG save quality 1-95 (default: 85)")
    parser.add_argument("--param", action="append", metavar="key=value", help="Method-specific params, e.g. --param k=200 --param model_type=baseline")
    parser.add_argument("--list", action="store_true", help="List all available compression methods")
    parser.add_argument("--decompress", action="store_true", help="Decompress a .cae file back to an image")

    args = parser.parse_args()

    # ── List mode ───────────────────────────────────────────────────────────
    if args.list:
        print("Available compression methods:")
        for name in list_compressors():
            compressor = get_compressor(name)
            defaults = compressor.default_params()
            param_str = ", ".join(f"{k}={v}" for k, v in defaults.items())
            print(f"  • {name:16s}  params: {param_str}")
        return

    # ── Input required for compress / decompress ─────────────────────────────
    if not args.input:
        parser.error("--input / -i is required (or use --list)")

    compressor = get_compressor(args.method)

    # Merge default params with user overrides
    params = compressor.default_params()
    params.update(parse_params(args.param))

    # ── Decompress mode ──────────────────────────────────────────────────────
    if args.decompress:
        output_path = args.output or f"reconstructed_{os.path.splitext(os.path.basename(args.input))[0]}.png"

        print(f"Method:     {compressor.name}")
        print(f"Mode:       decompress")
        print(f"Input:      {args.input}")
        print(f"Output:     {output_path}")
        print()

        reconstructed = compressor.decompress(args.input, **params)
        save_image(reconstructed, output_path, quality=args.quality)

        cae_size = os.path.getsize(args.input) / (1024 * 1024)
        out_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Bitstream Size:     {cae_size:.4f} MB")
        print(f"Reconstructed Size: {out_size:.2f} MB")
        return

    # ── Compress mode ────────────────────────────────────────────────────────
    output_path = args.output or f"compressed_{args.input}"

    # Pass output_path so compressors can save custom file formats
    params["output_path"] = output_path

    print(f"Method:  {compressor.name}")
    print(f"Params:  {params}")
    print(f"Input:   {args.input}")
    print(f"Output:  {output_path}")
    print()

    image = load_image(args.input)
    compressed = compressor.compress(image, **params)

    # If the compressor saved a custom bitstream (.cae), save the
    # reconstructed preview alongside it instead of overwriting.
    cae_path = os.path.splitext(output_path)[0] + ".cae"
    if os.path.isfile(cae_path):
        preview_path = os.path.splitext(cae_path)[0] + "_preview.jpg"
        save_image(compressed, preview_path, quality=args.quality)

        cae_kb = os.path.getsize(cae_path) / 1024
        orig_kb = os.path.getsize(args.input) / 1024
        print(f"Original Size:   {orig_kb:.1f} KB")
        print(f"Bitstream Size:  {cae_kb:.1f} KB  ({cae_path})")
        print(f"Compression:     {cae_kb / orig_kb * 100:.1f}%")
        print(f"Preview saved:   {preview_path}")
    else:
        save_image(compressed, output_path, quality=args.quality)
        print_stats(args.input, output_path)


if __name__ == "__main__":
    main()
