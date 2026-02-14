"""Image Compression CLI — supports multiple compression techniques."""

import argparse
import sys

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
    parser.add_argument("-o", "--output", help="Output image path (default: compressed_<input>)")
    parser.add_argument("-m", "--method", default="svd", help="Compression method (default: svd)")
    parser.add_argument("-q", "--quality", type=int, default=85, help="JPEG save quality 1-95 (default: 85)")
    parser.add_argument("--param", nargs="*", metavar="key=value", help="Method-specific params, e.g. --param k=200")
    parser.add_argument("--list", action="store_true", help="List all available compression methods")

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

    # ── Compress mode ───────────────────────────────────────────────────────
    if not args.input:
        parser.error("--input / -i is required (or use --list)")

    compressor = get_compressor(args.method)

    # Merge default params with user overrides
    params = compressor.default_params()
    params.update(parse_params(args.param))

    output_path = args.output or f"compressed_{args.input}"

    print(f"Method:  {compressor.name}")
    print(f"Params:  {params}")
    print(f"Input:   {args.input}")
    print(f"Output:  {output_path}")
    print()

    image = load_image(args.input)
    compressed = compressor.compress(image, **params)
    save_image(compressed, output_path, quality=args.quality)
    print_stats(args.input, output_path)


if __name__ == "__main__":
    main()
