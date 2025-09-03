#!/usr/bin/env python3
"""
Generate playing cards with US city/state information and mini maps.

This version uses fonttools to convert all text elements into paths,
removing dependencies on external fonts in the output SVG files.
"""
import json
import urllib.request
import urllib.parse
from typing import Tuple, Optional, List, Dict
import xml.etree.ElementTree as ET
import hashlib
import os
import base64
import csv
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from generate_city_images import make_image
from PIL import Image

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

# Constants for mini map dimensions
MAP_WIDTH = 80
MAP_HEIGHT = 50
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


def get_city_coordinates(city: str, state: str) -> Optional[Tuple[float, float]]:
    """
    Get latitude and longitude for a city using Nominatim geocoding service.
    Args:
        city: City name
        state: State name or abbreviation
    Returns:
        Tuple of (longitude, latitude) or None if not found
    """
    query = f"{city}, {state}, USA"
    encoded_query = urllib.parse.quote(query)
    url = f"https://nominatim.openstreetmap.org/search?q={encoded_query}&format=json&limit=1"
    headers = {"User-Agent": "USMapGenerator/1.0"}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data:
                lon = float(data[0]["lon"])
                lat = float(data[0]["lat"])
                return (lon, lat)
    except Exception as e:
        print(f"Error geocoding {city}, {state}: {e}")
    return None


def get_us_states_geojson(state_filter: str = None) -> dict:
    """
    Fetch US states GeoJSON data from a public source.
    Args:
        state_filter: If provided, only return this specific state
    Returns:
        GeoJSON dictionary with state boundaries
    """
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            if state_filter:
                # Filter to only include the specified state
                filtered_features = []
                for feature in data["features"]:
                    state_name = feature.get("properties", {}).get("name", "")
                    if state_name.lower() == state_filter.lower():
                        filtered_features.append(feature)
                return {"type": "FeatureCollection", "features": filtered_features}
            else:
                # Return continental US only (exclude Alaska, Hawaii, Puerto Rico)
                filtered_features = []
                for feature in data["features"]:
                    state_name = feature.get("properties", {}).get("name", "")
                    if state_name not in ["Alaska", "Hawaii", "Puerto Rico"]:
                        filtered_features.append(feature)
                return {"type": "FeatureCollection", "features": filtered_features}
    except Exception as e:
        print(f"Error fetching state boundaries: {e}")
        return None


def project_coordinates_for_state(
    lon: float, lat: float, state: str, width: int = MAP_WIDTH, height: int = MAP_HEIGHT
) -> Tuple[float, float]:
    """
    Project longitude/latitude to SVG coordinates for a specific state.
    Args:
        lon: Longitude
        lat: Latitude
        state: State name
        width: SVG width
        height: SVG height
    Returns:
        Tuple of (x, y) SVG coordinates
    """
    state_lower = state.lower()
    # Define bounds for each disconnected state
    if state_lower in ["alaska", "ak"]:
        min_lon, max_lon = -180, -130
        min_lat, max_lat = 52, 72
    elif state_lower in ["hawaii", "hi"]:
        min_lon, max_lon = -161, -154
        min_lat, max_lat = 18, 23
    elif state_lower in ["puerto rico", "pr"]:
        min_lon, max_lon = -68, -65
        min_lat, max_lat = 17.5, 18.6
    else:
        # Continental US
        min_lon, max_lon = -125, -66
        min_lat, max_lat = 24, 50
    # Add padding
    padding = 0.1
    x_ratio = (lon - min_lon) / (max_lon - min_lon)
    y_ratio = (lat - min_lat) / (max_lat - min_lat)
    x = x_ratio * width * (1 - 2 * padding) + width * padding
    y = height - (y_ratio * height * (1 - 2 * padding) + height * padding)
    return (x, y)


def geojson_to_svg_path(
    coordinates: List, width: int, height: int, state: str = None
) -> str:
    """
    Convert GeoJSON coordinates to SVG path string.
    Args:
        coordinates: GeoJSON coordinates array
        width: SVG width
        height: SVG height
        state: State name for proper projection (if disconnected state)
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
            if state:
                x, y = project_coordinates_for_state(
                    first_point[0], first_point[1], state, width, height
                )
            else:
                # Use simple projection for continental US
                min_lon, max_lon = -125, -66
                min_lat, max_lat = 24, 50
                x_ratio = (first_point[0] - min_lon) / (max_lon - min_lon)
                y_ratio = (first_point[1] - min_lat) / (max_lat - min_lat)
                x = x_ratio * width * 0.9 + width * 0.05
                y = height - (y_ratio * height * 0.85 + height * 0.1)
            path_parts.append(f"M {x:.2f},{y:.2f}")
            for point in ring[1:]:
                if state:
                    x, y = project_coordinates_for_state(
                        point[0], point[1], state, width, height
                    )
                else:
                    x_ratio = (point[0] - min_lon) / (max_lon - min_lon)
                    y_ratio = (point[1] - min_lat) / (max_lat - min_lat)
                    x = x_ratio * width * 0.9 + width * 0.05
                    y = height - (y_ratio * height * 0.85 + height * 0.1)
                path_parts.append(f"L {x:.2f},{y:.2f}")
            path_parts.append("Z")
    return " ".join(path_parts)


def create_mini_map_group(
    city: str, state: str, x_offset: int, y_offset: int
) -> ET.Element:
    """
    Create a group element containing the mini US map with city marker.
    Args:
        city: City name
        state: State name or abbreviation
        x_offset: X position for the map group
        y_offset: Y position for the map group
    Returns:
        ET.Element group containing the map
    """
    # Check if state is disconnected
    disconnected_states = {
        "alaska": "Alaska",
        "ak": "Alaska",
        "hawaii": "Hawaii",
        "hi": "Hawaii",
        "puerto rico": "Puerto Rico",
        "pr": "Puerto Rico",
    }
    state_lower = state.lower()
    is_disconnected = state_lower in disconnected_states
    state_filter = disconnected_states.get(state_lower) if is_disconnected else None
    # Get state boundaries - either single state or continental US
    states_data = get_us_states_geojson(state_filter)
    # Create group for the mini map
    map_group = ET.Element(
        "g", {"id": "mini-map", "transform": f"translate({x_offset}, {y_offset})"}
    )
    if not states_data:
        # Add error text if map data couldn't be loaded
        ET.SubElement(
            map_group, "text", {"x": "5", "y": "25", "font-size": "8", "fill": "#666"}
        ).text = "Map unavailable"
        return map_group

    # Get city coordinates
    city_coords = get_city_coordinates(city, state)
    # Draw states
    states_subgroup = ET.SubElement(map_group, "g", {"id": "states"})
    for feature in states_data.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "Polygon":
            coordinates = geometry.get("coordinates", [])
            path_d = geojson_to_svg_path(
                coordinates, MAP_WIDTH, MAP_HEIGHT, state_filter
            )
        elif geometry.get("type") == "MultiPolygon":
            path_d = ""
            for polygon in geometry.get("coordinates", []):
                path_d += (
                    geojson_to_svg_path([polygon], MAP_WIDTH, MAP_HEIGHT, state_filter)
                    + " "
                )
        else:
            continue
        if path_d:
            # Use different fill color for single state view
            fill_color = "#e8e8e8" if is_disconnected else "#f0f0f0"
            ET.SubElement(
                states_subgroup,
                "path",
                {
                    "d": path_d,
                    "fill": fill_color,
                    "stroke": "#888",
                    "stroke-width": "0.3",
                },
            )
    # Add city marker if coordinates found
    if city_coords:
        if is_disconnected:
            # Use special projection for disconnected states
            city_x, city_y = project_coordinates_for_state(
                city_coords[0], city_coords[1], state, MAP_WIDTH, MAP_HEIGHT
            )
        else:
            # Use regular projection for continental US
            min_lon, max_lon = -125, -66
            min_lat, max_lat = 24, 50
            x_ratio = (city_coords[0] - min_lon) / (max_lon - min_lon)
            y_ratio = (city_coords[1] - min_lat) / (max_lat - min_lat)
            city_x = x_ratio * MAP_WIDTH * 0.9 + MAP_WIDTH * 0.05
            city_y = MAP_HEIGHT - (y_ratio * MAP_HEIGHT * 0.85 + MAP_HEIGHT * 0.1)
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


def generate_city_code(city: str, state: str) -> str:
    """
    Generate a unique 3-letter code for a city/state combination.
    Args:
        city: City name
        state: State name or abbreviation
    Returns:
        3-letter code
    """
    # Create a hash of the city and state
    combined = f"{city.upper()}{state.upper()}"
    hash_obj = hashlib.md5(combined.encode())
    hash_hex = hash_obj.hexdigest()
    # Convert hash to uppercase letters
    code = ""
    for i in range(0, 6, 2):
        # Convert pairs of hex digits to letters A-Z
        num = int(hash_hex[i : i + 2], 16) % 26
        code += chr(65 + num)
    return code[:3]


def load_card_template(filepath: str = "card.svg") -> str:
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
    state: str,
    code: str = None,
    image_path: str = None,
    template_path: str = "card.svg",
) -> str:
    """
    Create a playing card for a city/state combination with text converted to paths.
    Args:
        city: City name
        state: State name or abbreviation
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

    # --- UPDATED: Add image with preserved aspect ratio ---
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
            img_y = 125 - (img_height / 2)

            # Now, embed the image data as base64
            image_data = embed_image_as_base64(image_path)
            if image_data:
                # The width and height now perfectly match the image's aspect ratio
                additional_content.append(
                    f'<g id="city-image"><image x="{img_x:.2f}" y="{img_y:.2f}" width="{img_width:.2f}" height="{img_height:.2f}" href="{image_data}"/></g>'
                )

    # --- Text to Path Conversion ---
    # (The rest of the function remains the same as before)
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

    state_paths = text_to_svg_group(
        text=state.upper(),
        x=CARD_WIDTH // 2,
        y=58,
        font_family="Arial",
        font_size=14,
        font_weight="normal",
        text_anchor="middle",
        fill="#666",
    )
    if state_paths is not None:
        city_info_group.append(state_paths)

    additional_content.append(f'\n{ET.tostring(city_info_group, encoding="unicode")}')

    map_x, map_y = 20, CARD_HEIGHT - MAP_HEIGHT - 20
    map_group = create_mini_map_group(city, state, map_x, map_y)
    additional_content.append(f'\n{ET.tostring(map_group, encoding="unicode")}')

    if code is None:
        code = generate_city_code(city, state)

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
    state: str,
    code: str = None,
    image_path: str = None,
    output_dir: str = "city_cards",
    template_path: str = "card.svg",
    skip_if_exists: bool = True,
):
    filename = f"{city.lower().replace(' ', '_')}_{state.lower().replace(' ', '_')}.svg"
    filepath = os.path.join(output_dir, filename)
    if skip_if_exists and os.path.exists(filepath):
        print(f"Card for {city}, {state} already exists. Skipping...")
        return
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating card for {city}, {state}...")
    # Generate card
    svg_content = create_city_card(
        city, state, code=code, image_path=image_path, template_path=template_path
    )
    # Save to file
    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"  ✓ Saved to {filepath}")


def main():
    data = csv.DictReader(open("cities.csv"))
    codes_seen = set()
    for i, elem in enumerate(data):
        # if i >= 1:
        #     break
        try:
            name = elem["Name"]
            # if len(name) < 16:
            #     continue
            state = elem["State"]
            code = elem.get("Code", None)
            if code is not None:
                if code in codes_seen:
                    raise ValueError(f"Duplicate code {code} for {name}, {state}")
                codes_seen.add(code)
        except Exception as e:
            print(f"Error reading row {elem}: {e}")
            continue
        img_path = make_image(city=name, state=state)
        generate_card_from_city(city=name, state=state, code=code, image_path=img_path)


if __name__ == "__main__":
    main()
