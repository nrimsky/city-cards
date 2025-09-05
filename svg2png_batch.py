#!/usr/bin/env python3
"""
python svg2png_batch.py ./country_cards ./country_cards_png --width 750

 --height 1125
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import cairosvg
except ImportError:
    sys.stderr.write(
        "Error: CairoSVG is not installed.\n" "Install with:  pip install cairosvg\n"
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch convert SVGs to PNGs at a desired size (aspect ratio preserved)."
    )
    p.add_argument("input_dir", type=Path, help="Folder containing SVG files.")
    p.add_argument("output_dir", type=Path, help="Folder to write PNG files.")
    size = p.add_mutually_exclusive_group(required=False)
    size.add_argument("--width", type=int, help="Desired PNG width in pixels.")
    size.add_argument("--height", type=int, help="Desired PNG height in pixels.")
    # Allow providing both width and height if you like:
    p.add_argument(
        "--both",
        action="store_true",
        help="If provided with --width and --height, force both (must match aspect ratio).",
    )
    p.add_argument(
        "-r", "--recursive", action="store_true", help="Search for SVGs recursively."
    )
    p.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing PNGs."
    )
    p.add_argument(
        "--bg",
        default=None,
        help="Background color (e.g., '#FFFFFF' or 'white'). Default: transparent.",
    )
    return p.parse_args()


def find_svgs(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.svg" if recursive else "*.svg"
    # Case-insensitive: also pick up .SVG
    paths = list(root.glob(pattern)) + list(root.glob(pattern.upper()))
    # Deduplicate while preserving order
    seen, unique = set(), []
    for p in paths:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique


def convert_one(
    svg_path: Path,
    png_path: Path,
    width: int | None,
    height: int | None,
    force: bool,
    bg: str | None,
    use_both: bool,
) -> None:
    if png_path.exists() and not force:
        print(f"skip (exists): {png_path}")
        return

    png_path.parent.mkdir(parents=True, exist_ok=True)

    # If only width OR height is provided, CairoSVG will compute the other to keep aspect ratio.
    kwargs = {}
    if width is not None:
        kwargs["output_width"] = int(width)
    if height is not None:
        kwargs["output_height"] = int(height)
    if not use_both:
        # If both were provided but --both not set, prefer width-only or height-only to preserve ratio strictly.
        if width is not None and height is not None:
            # Default to width-only; drop height so aspect ratio is guaranteed by CairoSVG.
            kwargs.pop("output_height", None)

    if bg:
        kwargs["background_color"] = bg

    # Use url= for correct relative path handling for linked assets
    cairosvg.svg2png(
        url=str(svg_path.resolve()), write_to=str(png_path.resolve()), **kwargs
    )
    print(f"ok: {svg_path.name} -> {png_path.name}")


def main():
    args = parse_args()

    if not args.input_dir.is_dir():
        sys.stderr.write(f"Input directory not found: {args.input_dir}\n")
        sys.exit(2)

    if args.width is None and args.height is None:
        sys.stderr.write(
            "You must provide --width or --height (or both with --both).\n"
        )
        sys.exit(2)

    svgs = find_svgs(args.input_dir, args.recursive)
    if not svgs:
        sys.stderr.write("No SVG files found.\n")
        sys.exit(3)

    # Mirror input structure under output_dir if recursive; otherwise just flat.
    for svg in svgs:
        rel = svg.relative_to(args.input_dir) if args.recursive else svg.name
        rel = Path(rel)
        out_rel = rel.with_suffix(".png")
        out_path = (
            (args.output_dir / out_rel)
            if args.recursive
            else (args.output_dir / out_rel.name)
        )

        try:
            convert_one(
                svg, out_path, args.width, args.height, args.force, args.bg, args.both
            )
        except Exception as e:
            sys.stderr.write(f"fail: {svg} -> {out_path} ({e})\n")


if __name__ == "__main__":
    main()
