import argparse
import os

from processor import (
    apply_contrast_enhancement,
    apply_edge_detection,
    apply_grayscale,
    apply_ndvi_like,
    load_image,
)
from utils import ensure_dir, save_image, stem_with_suffix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple satellite image processing MVP")
    p.add_argument("--input", required=True, help="Path to input image (.jpg, .png, .tif)")
    p.add_argument(
        "--operation",
        required=True,
        choices=["grayscale", "enhance", "edges", "ndvi", "all"],
        help="Operation to apply",
    )
    p.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "output"),
        help="Output directory (default: satellite_processor/output)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(args.output)

    img = load_image(args.input)
    ops = [args.operation] if args.operation != "all" else ["grayscale", "enhance", "edges", "ndvi"]

    for op in ops:
        if op == "grayscale":
            out = apply_grayscale(img)
        elif op == "enhance":
            out = apply_contrast_enhancement(img)
        elif op == "edges":
            out = apply_edge_detection(img)
        elif op == "ndvi":
            out = apply_ndvi_like(img)
            if out is None:
                print(f"input={args.input} operation=ndvi output=SKIPPED (needs 3+ bands)")
                continue
        else:
            raise ValueError(f"Unknown operation: {op}")

        out_path = os.path.join(args.output, stem_with_suffix(args.input, op) + ".png")
        save_image(out_path, out)
        print(f"input={args.input} operation={op} output={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
