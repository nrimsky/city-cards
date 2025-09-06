from typing import Tuple, Optional, Dict
import xml.etree.ElementTree as ET
import os
import base64
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from PIL import Image

CARD_TEMPLATE = "border_card.svg"

FONT_MAPPING: Dict[Tuple[str, str], str] = {
    ("Sans", "normal"): "./fonts/Cabin-Regular.ttf",
    ("Sans", "bold"): "./fonts/Cabin-Bold.ttf",
    ("Mono", "bold"): "./fonts/SourceCodePro-Bold.ttf",
}

CARD_WIDTH = 180
CARD_HEIGHT = 270

_FONT_CACHE: Dict[Tuple[str, str], TTFont] = {}


def get_font(family: str, weight: str) -> Optional[TTFont]:
    """Loads a font from the FONT_MAPPING, using a cache for performance."""
    key = (family, weight)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    font_path = FONT_MAPPING.get(key)
    if not font_path or not os.path.exists(font_path):
        # Fallback to normal weight if specific weight is not found
        fallback_key = (family, "normal")
        font_path = FONT_MAPPING.get(fallback_key)
        if not font_path or not os.path.exists(font_path):
            print(f"Warning: Font for '{family}' ({weight}) not found.")
            print(
                f"  Attempted paths: {FONT_MAPPING.get(key)} and {FONT_MAPPING.get(fallback_key)}"
            )
            return None

    try:
        font = TTFont(font_path)
        _FONT_CACHE[key] = font
        return font
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        return None


def text_to_svg_group(
    text: str,
    x: float,
    y: float,
    font_family: str,
    font_size: float,
    font_weight: str = "normal",
    text_anchor: str = "start",
    fill: str = "#000",
    **kwargs,
) -> Optional[ET.Element]:
    """
    Converts a text string to an SVG group of paths using fontTools.

    Returns:
        An ET.Element <g> containing the text rendered as <path> elements, or None.
    """
    font = get_font(font_family, font_weight)
    if font is None:
        # Return a fallback text element if font is missing
        error_text = ET.Element(
            "text",
            {
                "x": str(x),
                "y": str(y),
                "font-size": str(font_size / 2),
                "fill": "red",
                "text-anchor": text_anchor,
            },
        )
        error_text.text = f"[Font '{font_family}' not found]"
        return error_text

    glyph_set = font.getGlyphSet()
    cmap = font.get("cmap").getBestCmap()
    hmtx = font.get("hmtx")
    units_per_em = font["head"].unitsPerEm
    scale = font_size / units_per_em

    # Calculate total width for text-anchor
    total_width = 0
    for char in text:
        if ord(char) in cmap:
            glyph_name = cmap[ord(char)]
            total_width += hmtx[glyph_name][0]
    total_width *= scale

    # Adjust starting x position based on text-anchor
    start_x = x
    if text_anchor == "middle":
        start_x = x - (total_width / 2)
    elif text_anchor == "end":
        start_x = x - total_width

    # Adjust baseline y position for vertical alignment
    y_baseline = y

    group = ET.Element("g", {"fill": fill, **kwargs})
    current_x = start_x

    # Generate a path for each character
    for char in text:
        if ord(char) not in cmap:
            continue
        glyph_name = cmap[ord(char)]

        pen = SVGPathPen(glyph_set)
        glyph = glyph_set[glyph_name]
        glyph.draw(pen)

        path_d = pen.getCommands()
        if (
            path_d
        ):  # Only create a path if it has drawing commands (e.g., not for space)
            # Y-axis is flipped in font coordinates vs. SVG
            transform = f"translate({current_x} {y_baseline}) scale({scale} {-scale})"
            ET.SubElement(group, "path", {"d": path_d, "transform": transform})

        # Advance to the next character position
        advance_width = hmtx[glyph_name][0] * scale
        current_x += advance_width

    return group


def get_text_width(
    text: str, font_family: str, font_size: float, font_weight: str = "normal"
) -> float:
    """Calculates the rendered width of a text string for a given font."""
    font = get_font(font_family, font_weight)
    if font is None:
        return 0  # Cannot calculate width if font is missing

    cmap = font.get("cmap").getBestCmap()
    hmtx = font.get("hmtx")
    units_per_em = font["head"].unitsPerEm
    scale = font_size / units_per_em

    total_width_in_font_units = 0
    for char in text:
        if ord(char) in cmap:
            glyph_name = cmap[ord(char)]
            total_width_in_font_units += hmtx[glyph_name][0]

    return total_width_in_font_units * scale


def embed_image_as_base64(image_path: str) -> str:
    """
    Convert image to base64 data URI for embedding in SVG.
    Returns the data URI string or None if image cannot be loaded.
    """
    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            # Determine MIME type from extension
            ext = os.path.splitext(image_path)[1].lower()
            mime_types = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
            }
            mime_type = mime_types.get(ext, "image/jpeg")
            base64_string = base64.b64encode(img_data).decode("utf-8")
            return f"data:{mime_type};base64,{base64_string}"
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None


def load_card_template(filepath: str = CARD_TEMPLATE) -> str:
    """
    Load the card template SVG as a string.
    Args:
        filepath: Path to the card.svg template
    Returns:
        SVG string content
    """
    with open(filepath, "r") as f:
        content = f.read()
        # Remove any namespace prefixes if present
        content = content.replace("ns0:", "").replace(":ns0", "")
        return content


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """Gets the width and height of an image file using Pillow."""
    try:
        with Image.open(image_path) as img:
            return img.size  # returns (width, height)
    except Exception as e:
        print(f"Warning: Could not read dimensions from image {image_path}: {e}")
        return None
