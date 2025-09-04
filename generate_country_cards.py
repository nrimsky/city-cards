#!/usr/bin/env python3
"""
Generate playing cards with country information and mini continent maps.

This version uses fonttools to convert all text elements into paths,
removing dependencies on external fonts in the output SVG files.
"""
import json
from typing import Tuple, Optional, List, Dict
import xml.etree.ElementTree as ET
import os
import base64
import csv
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from generate_country_images import make_image
from PIL import Image
import math


FONT_MAPPING: Dict[Tuple[str, str], str] = {
    ("Arial", "normal"): "/System/Library/Fonts/Supplemental/Arial.ttf",
    ("Arial", "bold"): "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ("Courier New", "bold"): "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
}

# Fixed continent windows
# Format: (min_lon, min_lat, max_lon, max_lat)
CONTINENT_WINDOWS: Dict[str, Dict] = {
    "europe": {"bbox": (-11.5, 34.0, 45.5, 71.5)},  # excludes Azores (-31°)
    "africa": {"bbox": (-20.0, -36.0, 52.0, 38.0)},
    "asia": {"bbox": (25.0, -10.0, 170.0, 80.0)},
    "north america": {"bbox": (-170.0, 5.0, -52.0, 83.0)},
    "south america": {"bbox": (-82.0, -56.0, -34.0, 13.0)},
    # Oceania spans the dateline; use wrap_center=160 so 200°E is "just east"
    "oceania": {"bbox": (110.0, -50.0, 210.0, 10.0), "wrap_center": 160.0},
    "antarctica": {"bbox": (-180.0, -90.0, 180.0, -60.0)},
}

# Per-country overrides if you ever need them
COUNTRY_WINDOWS: Dict[str, Dict] = {
    # "portugal": {"bbox": (-12.0, 34.0, 43.0, 72.0)},
    # "france": {"bbox": (-11.5, 34.0, 45.5, 71.5)},
}

# A simple cache to avoid reloading font files from disk repeatedly.
_FONT_CACHE: Dict[Tuple[str, str], TTFont] = {}
_GEO_DATA_CACHE: Optional[Dict] = None

TEMPLATE_PATH = "border_card.svg"


def _rewrap_lon(lon: float, center: float) -> float:
    """Rewrap longitude into [center-180, center+180)."""
    span_min = center - 180.0
    span_max = center + 180.0
    while lon < span_min:
        lon += 360.0
    while lon >= span_max:
        lon -= 360.0
    return lon


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


# --- END FONTTOOLS INTEGRATION ---

# Card dimensions
CARD_WIDTH = 180
CARD_HEIGHT = 270


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


def get_geodata() -> Optional[Dict]:
    """
    Fetch world countries GeoJSON data, prioritizing a source with continent info.
    Caches the result to avoid repeated downloads.
    """
    global _GEO_DATA_CACHE
    if _GEO_DATA_CACHE:
        return _GEO_DATA_CACHE

    with open("world.geojson", "r") as f:
        _GEO_DATA_CACHE = json.load(f)
    return _GEO_DATA_CACHE


def calculate_bounding_box(
    features: List[Dict],
) -> Optional[Tuple[float, float, float, float]]:
    """Calculates the bounding box [min_lon, min_lat, max_lon, max_lat] for a list of GeoJSON features."""
    if not features:
        return None

    min_lon, min_lat = 180.0, 90.0
    max_lon, max_lat = -180.0, -90.0

    for feature in features:
        geometry = feature.get("geometry", {})
        coords = geometry.get("coordinates")
        geom_type = geometry.get("type")

        if not coords:
            continue

        polygons = []
        if geom_type == "Polygon":
            polygons = [coords]
        elif geom_type == "MultiPolygon":
            polygons = coords

        for polygon in polygons:
            for ring in polygon:
                for lon, lat in ring:
                    min_lon = min(min_lon, lon)
                    max_lon = max(max_lon, lon)
                    min_lat = min(min_lat, lat)
                    max_lat = max(max_lat, lat)

    return min_lon, min_lat, max_lon, max_lat


def project_coordinates(
    lon: float,
    lat: float,
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    wrap_center: Optional[float] = None,  # <— new
) -> Tuple[float, float]:
    """Projects geo-coordinates to SVG coordinates based on a bounding box."""
    min_lon, min_lat, max_lon, max_lat = bbox

    # Keep bbox and coordinate longitudes in the same wrap space when requested
    if wrap_center is not None:
        lon = _rewrap_lon(lon, wrap_center)
        min_lon = _rewrap_lon(min_lon, wrap_center)
        max_lon = _rewrap_lon(max_lon, wrap_center)

    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    if lon_range == 0 or lat_range == 0:
        return 0.0, 0.0

    x = (lon - min_lon) * (width / lon_range)
    y = (max_lat - lat) * (height / lat_range)  # SVG y-down
    return x, y


def geojson_to_svg_path(
    coordinates: List,
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int,
    wrap_center: Optional[float] = None,  # <— new
) -> str:
    if not coordinates:
        return ""
    path_parts = []
    for polygon in coordinates:
        if not polygon:
            continue
        rings = [polygon] if isinstance(polygon[0][0], float) else polygon
        for ring in rings:
            if not ring:
                continue
            first_point = ring[0]
            x, y = project_coordinates(
                first_point[0], first_point[1], bbox, width, height, wrap_center
            )
            path_parts.append(f"M {x:.2f},{y:.2f}")
            for point in ring[1:]:
                x, y = project_coordinates(
                    point[0], point[1], bbox, width, height, wrap_center
                )
                path_parts.append(f"L {x:.2f},{y:.2f}")
            path_parts.append("Z")
    return " ".join(path_parts)


def lookup_continent(country: str):
    file = "continents.csv"
    with open(file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"].lower().strip() == country.lower().strip():
                return row["Continent"]
    print(f"Warning: No continent found for country '{country}'.")
    return "unknown"


def create_mini_map_group(country: str, continent: str) -> ET.Element:
    """
    Create a group element containing the continent map with the country highlighted.
    """
    world_data = get_geodata()
    # Filter features for the given continent
    features = world_data.get("features", [])
    named_features = [
        (f.get("properties", {}).get("name", "").lower(), f) for f in features
    ]
    continent_features = [
        f
        for name, f in named_features
        if lookup_continent(name).lower() == continent.lower()
    ]

    if not continent_features:
        raise ValueError(f"Warning: No countries found for continent '{continent}'.")

    # Choose a fixed window: country override, else continent default, else fallback to computed
    lower_country = country.lower().strip()
    lower_continent = continent.lower().strip()

    window_cfg = COUNTRY_WINDOWS.get(lower_country) or CONTINENT_WINDOWS.get(
        lower_continent
    )
    wrap_center = None
    if window_cfg and "bbox" in window_cfg:
        min_lon, min_lat, max_lon, max_lat = window_cfg["bbox"]
        wrap_center = window_cfg.get("wrap_center")
    else:
        # Fallback: compute from data (may be "too big" for France/US etc.)
        min_lon, min_lat, max_lon, max_lat = calculate_bounding_box(continent_features)

    # Add a tiny padding and match aspect ratio to avoid distortion
    padding_factor = 1.02  # 2% pad
    lon_range = (max_lon - min_lon) * padding_factor
    lat_range = (max_lat - min_lat) * padding_factor
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    # Padded window
    min_lon, max_lon = center_lon - lon_range / 2.0, center_lon + lon_range / 2.0
    min_lat, max_lat = center_lat - lat_range / 2.0, center_lat + lat_range / 2.0

    # Recompute spans after padding/centering
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat

    # If we're wrapping (e.g., Oceania), keep the span in the same wrap space
    if wrap_center is not None:
        min_w = _rewrap_lon(min_lon, wrap_center)
        max_w = _rewrap_lon(max_lon, wrap_center)
        lon_span = max_w - min_w

    # Mean cosine across the latitude band (better than using only center_lat)
    phi_min = math.radians(min_lat)
    phi_max = math.radians(max_lat)
    if abs(phi_max - phi_min) < 1e-12:
        cos_eff = math.cos(math.radians(center_lat))
    else:
        # mean(cos φ) = (sin φ2 − sin φ1) / (φ2 − φ1)
        cos_eff = (math.sin(phi_max) - math.sin(phi_min)) / (phi_max - phi_min)

    # Avoid blow-ups near the poles
    cos_eff = max(cos_eff, 1e-3)

    # Height/width ratio that equalizes meters-per-pixel near this latitude band
    target_aspect = lat_span / (lon_span * cos_eff)

    map_width = 80.0
    map_height = map_width * target_aspect

    map_max_height = 60.0
    if map_height > map_max_height:
        shrink_ratio = map_max_height / map_height
        map_height = map_max_height
        map_width *= shrink_ratio

    map_x, map_y = 20, CARD_HEIGHT - map_height - 20

    bbox = (min_lon, min_lat, max_lon, max_lat)

    # ---- drawing ----
    map_group = ET.Element(
        "g", {"id": "mini-map", "transform": f"translate({map_x}, {map_y})"}
    )

    # Clip anything outside the mini-map rectangle (so outliers never show)
    clip_id = f"mini-map-clip-{abs(hash((country, continent))) % 65535}"
    defs = ET.SubElement(map_group, "defs")
    clip = ET.SubElement(defs, "clipPath", {"id": clip_id})
    ET.SubElement(
        clip,
        "rect",
        {"x": "0", "y": "0", "width": str(map_width), "height": str(map_height)},
    )

    countries_subgroup = ET.SubElement(
        map_group, "g", {"id": "countries", "clip-path": f"url(#{clip_id})"}
    )

    special_countries = {
        "united kingdom": ["england", "wales", "scotland", "northern ireland"]
    }

    for feature in continent_features:
        properties = feature.get("properties", {})
        country_name = properties.get("name", "") or properties.get("NAME", "")
        geometry = feature.get("geometry", {}) or {}
        path_d = geojson_to_svg_path(
            geometry.get("coordinates", []),
            bbox,
            map_width,
            map_height,
            wrap_center=wrap_center,  # <— keep Oceania tidy
        )
        if path_d:
            is_target = (
                country_name.lower() == lower_country
                or country_name.lower() in special_countries.get(lower_country, [])
            )
            ET.SubElement(
                countries_subgroup,
                "path",
                {
                    "d": path_d,
                    "fill": "#EC1E28" if is_target else "#f0f0f0",
                    "stroke": "#EC1E28" if is_target else "#888",
                    "stroke-width": "0.8" if is_target else "0.3",
                },
            )

    return map_group


def load_card_template(filepath: str = TEMPLATE_PATH) -> str:
    """Load the card template SVG as a string."""
    with open(filepath, "r") as f:
        content = f.read()
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


def create_country_card(
    country: str,
    continent: str,
    code: str,
    image_path: str,
    template_path: str = TEMPLATE_PATH,
) -> str:
    """
    Create a playing card for a country with text converted to paths.
    """
    template_content = load_card_template(template_path)
    closing_tag_pos = template_content.rfind("</svg>")
    if closing_tag_pos == -1:
        print("Warning: Could not find closing SVG tag in template")
        closing_tag_pos = len(template_content)

    additional_content = []

    # Add image with preserved aspect ratio
    if os.path.exists(image_path):
        original_dims = get_image_dimensions(image_path)
        if original_dims:
            original_width, original_height = original_dims
            BOUNDING_BOX_WIDTH = 130
            BOUNDING_BOX_HEIGHT = 130
            ratio_w = BOUNDING_BOX_WIDTH / original_width
            ratio_h = BOUNDING_BOX_HEIGHT / original_height
            scale_ratio = min(ratio_w, ratio_h)
            img_width = original_width * scale_ratio
            img_height = original_height * scale_ratio
            img_x = (CARD_WIDTH / 2) - (img_width / 2)
            img_y = 125 - (img_height / 2)
            image_data = embed_image_as_base64(image_path)
            if image_data:
                additional_content.append(
                    f'<g id="country-image"><image x="{img_x:.2f}" y="{img_y:.2f}" width="{img_width:.2f}" height="{img_height:.2f}" href="{image_data}"/></g>'
                )

    # --- Text to Path Conversion ---
    country_info_group = ET.Element("g", {"id": "country-info"})
    MAX_COUNTRY_WIDTH = 140
    country_font_size = 18.0
    initial_width = get_text_width(
        text=country,
        font_family="Arial",
        font_size=country_font_size,
        font_weight="bold",
    )
    if initial_width > MAX_COUNTRY_WIDTH:
        country_font_size *= MAX_COUNTRY_WIDTH / initial_width
        print(
            f"  - Resizing country '{country}' font to {country_font_size:.2f}pt to fit."
        )

    country_paths = text_to_svg_group(
        text=country,
        x=CARD_WIDTH // 2,
        y=50,
        font_family="Arial",
        font_size=country_font_size,
        font_weight="bold",
        text_anchor="middle",
        fill="#2B2B2B",
    )
    if country_paths is not None:
        country_info_group.append(country_paths)

    additional_content.append(
        f'\n{ET.tostring(country_info_group, encoding="unicode")}'
    )

    # Add mini continent map
    map_group = create_mini_map_group(country, continent)
    additional_content.append(f'\n{ET.tostring(map_group, encoding="unicode")}')

    # Add code in bottom right
    code_group = ET.Element("g", {"id": "country-code"})
    ET.SubElement(
        code_group,
        "circle",
        {
            "cx": str(CARD_WIDTH - 40),
            "cy": str(CARD_HEIGHT - 40),
            "r": "16",
            "fill": "#2BA6DE",
            "opacity": "0.1",
        },
    )
    code_paths = text_to_svg_group(
        text=code,
        x=CARD_WIDTH - 40,
        y=CARD_HEIGHT - 36,
        font_family="Courier New",
        font_size=15,
        font_weight="bold",
        text_anchor="middle",
        fill="#2BA6DE",
    )
    if code_paths is not None:
        code_group.append(code_paths)

    additional_content.append(f'\n{ET.tostring(code_group, encoding="unicode")}')

    final_svg = (
        template_content[:closing_tag_pos]
        + "\n".join(additional_content)
        + "\n"
        + template_content[closing_tag_pos:]
    )
    return final_svg


def generate_card_from_country(
    country: str,
    continent: str,
    code: str,
    image_path: str,
    output_dir: str = "country_cards",
    template_path: str = TEMPLATE_PATH,
    skip_if_exists: bool = True,
):
    """Generate and save a card for a country."""
    filename = f"{country.lower().replace(' ', '_')}.svg"
    filepath = os.path.join(output_dir, filename)
    if skip_if_exists and os.path.exists(filepath):
        print(f"Card for {country} already exists. Skipping...")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating card for {country}...")

    svg_content = create_country_card(
        country,
        continent,
        code=code,
        image_path=image_path,
        template_path=template_path,
    )

    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"  ✓ Saved to {filepath}")


def main():
    """Main function to generate country cards from CSV file."""
    csv_path = "countries.csv"
    try:
        with open(csv_path, "r") as f:
            data = list(csv.DictReader(f))
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file {csv_path} not found.")

    codes_seen = set()
    for i, elem in enumerate(data):
        if i > 1:
            break
        try:
            name = elem["Name"]
            continent = lookup_continent(name)
            code = elem["Code"]
            if code in codes_seen:
                raise ValueError(f"Duplicate code {code} for {name}")
            codes_seen.add(code)
        except Exception as e:
            print(f"Error reading row {elem}: {e}")
            continue

        img_path = make_image(country=name)
        generate_card_from_country(
            country=name, continent=continent, code=code, image_path=img_path
        )


if __name__ == "__main__":
    main()
