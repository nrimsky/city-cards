#!/usr/bin/env python3
"""
Generate playing cards with world city/country information and mini maps.

This version uses fonttools to convert all text elements into paths,
removing dependencies on external fonts in the output SVG files.
"""
import json
import urllib.request
import urllib.parse
from typing import Tuple, Optional, List, Dict
import xml.etree.ElementTree as ET
import random
import os
import base64
import csv
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from generate_world_city_images import make_image
from PIL import Image
import math

CARD_TEMPLATE = "border_card.svg"

FONT_MAPPING: Dict[Tuple[str, str], str] = {
    ("Arial", "normal"): "/System/Library/Fonts/Supplemental/Arial.ttf",
    ("Arial", "bold"): "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ("Courier New", "bold"): "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
}

# A simple cache to avoid reloading font files from disk repeatedly.
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


def get_city_coordinates(city: str, country: str) -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude for a city using Nominatim geocoding service.
    Args:
        city: City name
        country: Country name
    Returns:
        Tuple of (longitude, latitude) or None if not found
    """
    query = f"{city}, {country}"
    encoded_query = urllib.parse.quote(query)
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"
    headers = {"User-Agent": "WorldMapGenerator/1.0"}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data:
                lon = float(data[0]["lon"])
                lat = float(data[0]["lat"])
                return (lon, lat)
    except Exception as e:
        print(f"Error geocoding {city}, {country}: {e}")
    return None


def get_country_geojson(country_name: str) -> dict:
    """
    Fetch country boundary GeoJSON data.
    Args:
        country_name: Name of the country
    Returns:
        GeoJSON dictionary with country boundaries
    """
    with open("world.geojson", "r") as f:
        data = json.load(f)
    # Filter to only include the specified country
    name_fields = ["name", "name_long", "admin", "sovereignt", "name_en"]
    filtered_features = []
    for feature in data["features"]:
        properties = feature.get("properties", {})
        # Check various name fields for match
        for field in name_fields:
            if properties.get(field, "").lower() == country_name.lower():
                filtered_features.append(feature)
                break

    if filtered_features:
        return {"type": "FeatureCollection", "features": filtered_features}
    else:
        # If country not found by exact match, try partial match
        for feature in data["features"]:
            properties = feature.get("properties", {})
            for field in name_fields:
                if country_name.lower() in properties.get(field, "").lower():
                    filtered_features.append(feature)
                    break
            if filtered_features:
                break

        return {"type": "FeatureCollection", "features": filtered_features}


def get_bounds_for_country(country_data: dict, country_name: str = None):
    min_lon, max_lon = float("inf"), float("-inf")
    min_lat, max_lat = float("inf"), float("-inf")

    MAX_WIDTH = 80
    MAX_HEIGHT = 80

    # Check if we have override bounds for specific countries
    if country_name:
        country_lower = country_name.lower()

        # Overrides for countries with overseas territories or special cases
        country_bounds_overrides = {
            "united kingdom": (-8.5, 2.0, 49.5, 61.0),  # British Isles only
            "uk": (-8.5, 2.0, 49.5, 61.0),  # British Isles only
            "great britain": (-8.5, 2.0, 49.5, 61.0),  # British Isles only
            "france": (-5.0, 10.0, 41.0, 51.5),  # Metropolitan France only
            "netherlands": (3.0, 7.5, 50.5, 53.5),  # European Netherlands only
            "portugal": (-10.0, -6.0, 36.5, 42.5),  # Continental Portugal only
            "spain": (-10.0, 5.0, 35.5, 44.0),  # Iberian Peninsula only
            "denmark": (8.0, 13.0, 54.5, 58.0),  # Denmark proper only
            "norway": (4.0, 31.5, 57.5, 71.5),  # Mainland Norway
            "united states": (-125.0, -66.0, 24.0, 50.0),  # Continental US only
            "usa": (-125.0, -66.0, 24.0, 50.0),  # Continental US only
            "russia": (27.0, 180.0, 41.0, 82.0),  # Main landmass
            "china": (73.0, 135.0, 18.0, 54.0),  # Mainland China
            "australia": (112.0, 154.0, -44.0, -10.0),  # Mainland Australia
            "new zealand": (166.0, 179.0, -47.5, -34.0),  # Main islands
        }

        if country_lower in country_bounds_overrides:
            min_lon, max_lon, min_lat, max_lat = country_bounds_overrides[country_lower]
        else:
            # Calculate from GeoJSON if no override
            for feature in country_data.get("features", []):
                geometry = feature.get("geometry", {})
                coords_list = []

                if geometry.get("type") == "Polygon":
                    coords_list = geometry.get("coordinates", [])
                elif geometry.get("type") == "MultiPolygon":
                    for polygon in geometry.get("coordinates", []):
                        coords_list.extend(polygon)

                for ring in coords_list:
                    for coord in ring:
                        if isinstance(coord[0], (int, float)):
                            lon, lat = coord[0], coord[1]
                            min_lon = min(min_lon, lon)
                            max_lon = max(max_lon, lon)
                            min_lat = min(min_lat, lat)
                            max_lat = max(max_lat, lat)
    else:
        # Default calculation from GeoJSON
        for feature in country_data.get("features", []):
            geometry = feature.get("geometry", {})
            coords_list = []

            if geometry.get("type") == "Polygon":
                coords_list = geometry.get("coordinates", [])
            elif geometry.get("type") == "MultiPolygon":
                for polygon in geometry.get("coordinates", []):
                    coords_list.extend(polygon)

            for ring in coords_list:
                for coord in ring:
                    if isinstance(coord[0], (int, float)):
                        lon, lat = coord[0], coord[1]
                        min_lon = min(min_lon, lon)
                        max_lon = max(max_lon, lon)
                        min_lat = min(min_lat, lat)
                        max_lat = max(max_lat, lat)

    # Add 10% padding to the bounds
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    padding_lon = 0.1 * lon_range
    padding_lat = 0.1 * lat_range

    min_lon -= padding_lon
    max_lon += padding_lon
    min_lat -= padding_lat
    max_lat += padding_lat

    # Recalculate ranges after padding
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    # Account for latitude distortion (simple mercator-like adjustment)
    # At higher latitudes, longitude degrees cover less distance
    avg_lat = (min_lat + max_lat) / 2
    lat_correction = abs(math.cos(math.radians(avg_lat)))

    # Apply correction to longitude range for aspect ratio calculation
    corrected_lon_range = lon_range * lat_correction

    # Calculate the natural aspect ratio of the region (width/height)
    aspect_ratio = corrected_lon_range / lat_range if lat_range > 0 else 1

    # Calculate actual dimensions to fit in the box while preserving aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        actual_width = MAX_WIDTH
        actual_height = MAX_WIDTH / aspect_ratio
        if actual_height > MAX_HEIGHT:
            actual_height = MAX_HEIGHT
            actual_width = MAX_HEIGHT * aspect_ratio
    else:  # Taller than wide
        actual_height = MAX_HEIGHT
        actual_width = MAX_HEIGHT * aspect_ratio
        if actual_width > MAX_WIDTH:
            actual_width = MAX_WIDTH
            actual_height = MAX_WIDTH / aspect_ratio

    # Now adjust the bounds to maintain the aspect ratio
    # We need to ensure that the lon/lat ranges match the aspect ratio we're drawing
    target_aspect = actual_width / actual_height
    current_aspect = corrected_lon_range / lat_range

    if (
        abs(target_aspect - current_aspect) > 0.01
    ):  # Only adjust if significantly different
        if target_aspect > current_aspect:
            # Need to expand longitude range
            new_corrected_lon_range = lat_range * target_aspect
            expansion = (new_corrected_lon_range - corrected_lon_range) / lat_correction
            min_lon -= expansion / 2
            max_lon += expansion / 2
        else:
            # Need to expand latitude range
            new_lat_range = corrected_lon_range / target_aspect
            expansion = new_lat_range - lat_range
            min_lat -= expansion / 2
            max_lat += expansion / 2

    return (
        min_lon,
        max_lon,
        min_lat,
        max_lat,
        actual_width,
        actual_height,
    )


def project_coordinates_for_country(
    lon: float,
    lat: float,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    """
    Project longitude/latitude to SVG coordinates for a specific country.
    Uses a simple equirectangular projection with aspect ratio preservation.

    Args:
        lon: Longitude
        lat: Latitude
        min_lon, max_lon, min_lat, max_lat: Country bounds
        width: SVG width
        height: SVG height
    Returns:
        Tuple of (x, y) SVG coordinates
    """
    # Simple linear mapping - the aspect ratio correction is already in the bounds
    x_ratio = (lon - min_lon) / (max_lon - min_lon) if max_lon != min_lon else 0.5
    y_ratio = (lat - min_lat) / (max_lat - min_lat) if max_lat != min_lat else 0.5

    # Map to SVG coordinates (no additional padding since it's in the bounds)
    x = x_ratio * width
    y = height - (y_ratio * height)  # Invert Y axis for SVG

    return (x, y)


def geojson_to_svg_path(
    coordinates: List,
    width: int,
    height: int,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
) -> str:
    """
    Convert GeoJSON coordinates to SVG path string.
    Args:
        coordinates: GeoJSON coordinates array
        width: SVG width
        height: SVG height
        min_lon, max_lon, min_lat, max_lat: Bounds for projection
    """
    if not coordinates:
        return ""
    path_parts = []
    for polygon in coordinates:
        if isinstance(polygon[0][0], list):
            rings = polygon
        else:
            rings = [polygon]
        for ring in rings:
            if not ring:
                continue
            first_point = ring[0]
            x, y = project_coordinates_for_country(
                first_point[0],
                first_point[1],
                min_lon,
                max_lon,
                min_lat,
                max_lat,
                width,
                height,
            )
            path_parts.append(f"M {x:.2f},{y:.2f}")
            for point in ring[1:]:
                x, y = project_coordinates_for_country(
                    point[0],
                    point[1],
                    min_lon,
                    max_lon,
                    min_lat,
                    max_lat,
                    width,
                    height,
                )
                path_parts.append(f"L {x:.2f},{y:.2f}")
            path_parts.append("Z")
    return " ".join(path_parts)


def create_mini_map_group(city: str, country: str) -> ET.Element:
    """
    Create a group element containing the mini country map with city marker.
    Args:
        city: City name
        country: Country name
        x_offset: X position for the map group
        y_offset: Y position for the map group
    Returns:
        ET.Element group containing the map
    """
    # Get country boundaries
    country_data = get_country_geojson(country)

    # Get city coordinates first (needed for filtering)
    city_coords = get_city_coordinates(city, country)

    # Get country bounds with filtering
    min_lon, max_lon, min_lat, max_lat, actual_width, actual_height = (
        get_bounds_for_country(country_data, country)
    )

    map_x, map_y = 20, CARD_HEIGHT - actual_height - 20

    # Create group for the mini map
    map_group = ET.Element(
        "g", {"id": "mini-map", "transform": f"translate({map_x}, {map_y})"}
    )

    # Draw country
    country_subgroup = ET.SubElement(map_group, "g", {"id": "country"})
    for feature in country_data.get("features", []):
        geometry = feature.get("geometry", {})

        # Filter out disconnected territories
        if geometry.get("type") == "Polygon":
            coordinates = geometry.get("coordinates", [])
            path_d = geojson_to_svg_path(
                coordinates,
                actual_width,
                actual_height,
                min_lon,
                max_lon,
                min_lat,
                max_lat,
            )
        elif geometry.get("type") == "MultiPolygon":
            path_d = ""
            for polygon in geometry.get("coordinates", []):
                path_d += (
                    geojson_to_svg_path(
                        [polygon],
                        actual_width,
                        actual_height,
                        min_lon,
                        max_lon,
                        min_lat,
                        max_lat,
                    )
                    + " "
                )
        else:
            continue
        if path_d:
            ET.SubElement(
                country_subgroup,
                "path",
                {
                    "d": path_d,
                    "fill": "#e8e8e8",
                    "stroke": "#888",
                    "stroke-width": "0.3",
                },
            )

    # Add city marker if coordinates found
    if city_coords:
        city_x, city_y = project_coordinates_for_country(
            city_coords[0],
            city_coords[1],
            min_lon,
            max_lon,
            min_lat,
            max_lat,
            actual_width,
            actual_height,
        )

        # Add a pulsing effect for the marker
        marker_group = ET.SubElement(map_group, "g", {"id": "city-marker"})
        # Outer ring
        ET.SubElement(
            marker_group,
            "circle",
            {
                "cx": str(city_x),
                "cy": str(city_y),
                "r": "3",
                "fill": "none",
                "stroke": "#EC1E28",
                "stroke-width": "0.5",
                "opacity": "0.5",
            },
        )
        # Main marker dot
        ET.SubElement(
            marker_group,
            "circle",
            {"cx": str(city_x), "cy": str(city_y), "r": "1.5", "fill": "#EC1E28"},
        )
    return map_group


def load_card_template(filepath: str = CARD_TEMPLATE) -> str:
    """
    Load the card template SVG as a string.
    Args:
        filepath: Path to the card.svg template
    Returns:
        SVG string content
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
        # Remove any namespace prefixes if present
        content = content.replace("ns0:", "").replace(":ns0", "")
        return content
    except Exception as e:
        print(f"Error loading card template: {e}")
        # Return a basic card template if file not found
        return """<svg width="180" height="270" viewBox="0 0 180 270" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="180" height="270" fill="white"/>
<path d="M9 251.919C9 256.89 13.029 260.919 18 260.919H162.012C166.983 260.919 171.012 256.89 171.012 251.919V17.919C171.012 12.948 166.983 8.91901 162.012 8.91901H18C13.029 8.91901 9 12.948 9 17.919V251.919Z" stroke="#EC1E28"/>
<path d="M18 21.825C18 19.669 19.525 17.925 21.408 17.925H158.593C160.476 17.925 162.001 19.669 162.001 21.825V248.013C162.001 250.167 160.476 251.913 158.593 251.913H21.408C19.526 251.913 18 250.167 18 248.013V21.825Z" stroke="#2BA6DE" stroke-dasharray="1.36 1.36"/>
</svg>"""


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """Gets the width and height of an image file using Pillow."""
    try:
        with Image.open(image_path) as img:
            return img.size  # returns (width, height)
    except Exception as e:
        print(f"Warning: Could not read dimensions from image {image_path}: {e}")
        return None


def create_city_card(
    city: str,
    country: str,
    code: str,
    image_path: str = None,
    template_path: str = CARD_TEMPLATE,
) -> str:
    """
    Create a playing card for a city/country combination with text converted to paths.
    Args:
        city: City name
        country: Country name
        code: Optional 3-letter code
        image_path: Path to city image
        template_path: Path to card.svg template
    Returns:
        SVG string for the card
    """
    template_content = load_card_template(template_path)
    closing_tag_pos = template_content.rfind("</svg>")
    if closing_tag_pos == -1:
        print("Warning: Could not find closing SVG tag in template")
        closing_tag_pos = len(template_content)

    additional_content = []

    # Add image with preserved aspect ratio
    if image_path:
        # First, get the image's original dimensions
        original_dims = get_image_dimensions(image_path)

        if original_dims:
            original_width, original_height = original_dims

            # Define the bounding box for the image on the card
            BOUNDING_BOX_WIDTH = 130
            BOUNDING_BOX_HEIGHT = 130

            # Calculate scaling ratios for width and height
            ratio_w = BOUNDING_BOX_WIDTH / original_width
            ratio_h = BOUNDING_BOX_HEIGHT / original_height
            # Use the smaller ratio to ensure the image fits entirely
            scale_ratio = min(ratio_w, ratio_h)

            # Calculate the new dimensions that preserve the aspect ratio
            img_width = original_width * scale_ratio
            img_height = original_height * scale_ratio

            # Center the resized image in the same area as before
            # The desired center point is (90, 125)
            img_x = (CARD_WIDTH / 2) - (img_width / 2)
            img_y = 130 - (img_height / 2)

            # Now, embed the image data as base64
            image_data = embed_image_as_base64(image_path)
            if image_data:
                # The width and height now perfectly match the image's aspect ratio
                additional_content.append(
                    f'<g id="city-image"><image x="{img_x:.2f}" y="{img_y:.2f}" width="{img_width:.2f}" height="{img_height:.2f}" href="{image_data}"/></g>'
                )

    # Text to Path Conversion
    city_info_group = ET.Element("g", {"id": "city-info"})

    MAX_CITY_WIDTH = 140
    city_font_size = 18.0

    initial_width = get_text_width(
        text=city, font_family="Arial", font_size=city_font_size, font_weight="bold"
    )

    if initial_width > MAX_CITY_WIDTH:
        city_font_size = (MAX_CITY_WIDTH / initial_width) * city_font_size
        print(
            f"  - Resizing city '{city}' font to {city_font_size:.2f}pt to fit 140px width."
        )

    city_paths = text_to_svg_group(
        text=city,
        x=CARD_WIDTH // 2,
        y=40,
        font_family="Arial",
        font_size=city_font_size,
        font_weight="bold",
        text_anchor="middle",
        fill="#2B2B2B",
    )
    if city_paths is not None:
        city_info_group.append(city_paths)

    country_paths = text_to_svg_group(
        text=country,
        x=CARD_WIDTH // 2,
        y=58,
        font_family="Arial",
        font_size=12,
        font_weight="normal",
        text_anchor="middle",
        fill="#666",
    )
    if country_paths is not None:
        city_info_group.append(country_paths)

    additional_content.append(f'\n{ET.tostring(city_info_group, encoding="unicode")}')

    map_group = create_mini_map_group(city, country)
    additional_content.append(f'\n{ET.tostring(map_group, encoding="unicode")}')

    code_group = ET.Element("g", {"id": "city-code"})
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


def generate_card_from_city(
    city: str,
    country: str,
    code: str,
    image_path: str = None,
    output_dir: str = "world_city_cards",
    template_path: str = CARD_TEMPLATE,
    skip_if_exists: bool = True,
):
    filename = (
        f"{city.lower().replace(' ', '_')}_{country.lower().replace(' ', '_')}.svg"
    )
    filepath = os.path.join(output_dir, filename)
    if skip_if_exists and os.path.exists(filepath):
        print(f"Card for {city}, {country} already exists. Skipping...")
        return
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating card for {city}, {country}...")
    # Generate card
    svg_content = create_city_card(
        city, country, code=code, image_path=image_path, template_path=template_path
    )
    # Save to file
    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"  ✓ Saved to {filepath}")


def main():
    data = csv.DictReader(open("world_cities.csv"))
    codes_seen = set()
    for i, elem in enumerate(data):
        if i > 2:
            break
        try:
            name = elem["Name"]
            country = elem["Country"]
            code = random.randint(0, 999)
            while code in codes_seen:
                code = random.randint(0, 999)
            codes_seen.add(code)
            # convert to DDD string
            code = f"{code:03d}"
        except Exception as e:
            print(f"Error reading row {elem}: {e}")
            continue
        img_path = make_image(city=name, country=country)
        generate_card_from_city(
            city=name, country=country, code=code, image_path=img_path
        )


if __name__ == "__main__":
    main()
