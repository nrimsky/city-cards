#!/usr/bin/env python3
"""
Generate playing cards with US state information and mini maps.

This version uses fonttools to convert all text elements into paths,
removing dependencies on external fonts in the output SVG files.
"""
import json
import urllib.request
from typing import Tuple, Optional, List, Dict
import xml.etree.ElementTree as ET
import hashlib
import os
import base64
import csv
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from generate_state_images import make_image
from PIL import Image

TEMPLATE_PATH = "border_card.svg"

FONT_MAPPING: Dict[Tuple[str, str], str] = {
    ("Arial", "normal"): "/System/Library/Fonts/Supplemental/Arial.ttf",
    ("Arial", "bold"): "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ("Courier New", "bold"): "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
}

# A simple cache to avoid reloading font files from disk repeatedly.
_FONT_CACHE: Dict[Tuple[str, str], TTFont] = {}

# Constants for mini map dimensions
MAP_WIDTH = 80
MAP_HEIGHT = 50
# Card dimensions
CARD_WIDTH = 180
CARD_HEIGHT = 270


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
            # Handle space character
            if char == " ":
                # Advance by space width (typically 0.25em)
                current_x += font_size * 0.25
            continue
        glyph_name = cmap[ord(char)]

        pen = SVGPathPen(glyph_set)
        glyph = glyph_set[glyph_name]
        glyph.draw(pen)

        path_d = pen.getCommands()
        if path_d:  # Only create a path if it has drawing commands
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
        elif char == " ":
            # Approximate space width
            total_width_in_font_units += units_per_em * 0.25

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


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """Gets the width and height of an image file using Pillow."""
    try:
        with Image.open(image_path) as img:
            return img.size  # returns (width, height)
    except Exception as e:
        print(f"Warning: Could not read dimensions from image {image_path}: {e}")
        return None


def generate_state_code(state_name: str, state_abbr: str) -> str:
    """
    Generate a unique 3-letter code for a state.
    Args:
        state_name: State name
        state_abbr: State abbreviation
    Returns:
        3-letter code
    """
    # Create a hash of the state name and abbreviation
    combined = f"{state_name.upper()}{state_abbr.upper()}"
    hash_obj = hashlib.md5(combined.encode())
    hash_hex = hash_obj.hexdigest()
    # Convert hash to uppercase letters
    code = ""
    for i in range(0, 6, 2):
        # Convert pairs of hex digits to letters A-Z
        num = int(hash_hex[i : i + 2], 16) % 26
        code += chr(65 + num)
    return code[:3]


def get_us_states_geojson(
    state_filter: str = None, highlight_state: str = None
) -> dict:
    """
    Fetch US states GeoJSON data from a public source.
    Args:
        state_filter: If provided, only return this specific state
        highlight_state: State to be highlighted in the map
    Returns:
        GeoJSON dictionary with state boundaries
    """
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))

            # Mark which state should be highlighted
            if highlight_state:
                for feature in data["features"]:
                    state_name = feature.get("properties", {}).get("name", "")
                    feature["properties"]["highlighted"] = (
                        state_name.lower() == highlight_state.lower()
                    )

            if state_filter:
                # Filter to only include the specified state
                filtered_features = []
                for feature in data["features"]:
                    state_name = feature.get("properties", {}).get("name", "")
                    if state_name.lower() == state_filter.lower():
                        filtered_features.append(feature)
                return {"type": "FeatureCollection", "features": filtered_features}
            else:
                # Return continental US only (exclude Alaska, Hawaii, Puerto Rico) for minimap
                # unless we're highlighting one of those states
                filtered_features = []
                excluded_states = ["Alaska", "Hawaii", "Puerto Rico"]
                show_disconnected = highlight_state and any(
                    highlight_state.lower() == s.lower() for s in excluded_states
                )

                if show_disconnected:
                    # If highlighting a disconnected state, only show that state
                    for feature in data["features"]:
                        state_name = feature.get("properties", {}).get("name", "")
                        if state_name.lower() == highlight_state.lower():
                            filtered_features.append(feature)
                else:
                    # Show continental US
                    for feature in data["features"]:
                        state_name = feature.get("properties", {}).get("name", "")
                        if state_name not in excluded_states:
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


def create_mini_map_group(state_name: str, x_offset: int, y_offset: int) -> ET.Element:
    """
    Create a group element containing the mini US map with state highlighted.
    Args:
        state_name: State name to highlight
        x_offset: X position for the map group
        y_offset: Y position for the map group
    Returns:
        ET.Element group containing the map
    """
    # Check if state is disconnected
    disconnected_states = {
        "alaska": "Alaska",
        "hawaii": "Hawaii",
        "puerto rico": "Puerto Rico",
        "district of columbia": "District of Columbia",
    }
    state_lower = state_name.lower()
    is_disconnected = state_lower in disconnected_states
    state_filter = disconnected_states.get(state_lower) if is_disconnected else None

    # Get state boundaries with highlight info
    states_data = get_us_states_geojson(state_filter, highlight_state=state_name)

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

    # Draw states
    states_subgroup = ET.SubElement(map_group, "g", {"id": "states"})
    for feature in states_data.get("features", []):
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})
        is_highlighted = properties.get("highlighted", False)

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
            # Use different colors for highlighted state
            if is_highlighted:
                fill_color = "#2BA6DE"
                stroke_color = "#1976D2"
                stroke_width = "0.8"
                opacity = "0.7"
            else:
                fill_color = "#f0f0f0"
                stroke_color = "#888"
                stroke_width = "0.3"
                opacity = "1.0"

            ET.SubElement(
                states_subgroup,
                "path",
                {
                    "d": path_d,
                    "fill": fill_color,
                    "stroke": stroke_color,
                    "stroke-width": stroke_width,
                    "opacity": opacity,
                },
            )

    return map_group


def load_card_template(filepath: str = TEMPLATE_PATH) -> str:
    """
    Load the card template SVG as a string.
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


def create_state_card(
    state_name: str,
    state_abbr: str,
    code: str = None,
    image_path: str = None,
    template_path: str = TEMPLATE_PATH,
) -> str:
    """
    Create a playing card for a US state with text converted to paths.
    Args:
        state_name: Full state name
        state_abbr: State abbreviation
        code: Optional 3-letter code for the state
        image_path: Path to state image
        template_path: Path to card template
    Returns:
        SVG string for the card
    """
    template_content = load_card_template(template_path)
    closing_tag_pos = template_content.rfind("</svg>")
    if closing_tag_pos == -1:
        print("Warning: Could not find closing SVG tag in template")
        closing_tag_pos = len(template_content)

    additional_content = []

    # --- Add image with preserved aspect ratio ---
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
                    f'<g id="state-image"><image x="{img_x:.2f}" y="{img_y:.2f}" width="{img_width:.2f}" height="{img_height:.2f}" href="{image_data}"/></g>'
                )

    # --- State Info Group ---
    state_info_group = ET.Element("g", {"id": "state-info"})

    # State name (title)
    MAX_STATE_WIDTH = 140
    state_font_size = 18.0

    initial_width = get_text_width(
        text=state_name,
        font_family="Arial",
        font_size=state_font_size,
        font_weight="bold",
    )

    if initial_width > MAX_STATE_WIDTH:
        state_font_size = (MAX_STATE_WIDTH / initial_width) * state_font_size
        print(
            f"  - Resizing state '{state_name}' font to {state_font_size:.2f}pt to fit 140px width."
        )

    state_paths = text_to_svg_group(
        text=state_name,
        x=CARD_WIDTH // 2,
        y=40,
        font_family="Arial",
        font_size=state_font_size,
        font_weight="bold",
        text_anchor="middle",
        fill="#2B2B2B",
    )
    if state_paths is not None:
        state_info_group.append(state_paths)

    # State abbreviation (subtitle)
    abbr_paths = text_to_svg_group(
        text=state_abbr.upper(),
        x=CARD_WIDTH // 2,
        y=58,
        font_family="Arial",
        font_size=14,
        font_weight="normal",
        text_anchor="middle",
        fill="#666",
    )
    if abbr_paths is not None:
        state_info_group.append(abbr_paths)

    additional_content.append(f'\n{ET.tostring(state_info_group, encoding="unicode")}')

    # --- Mini Map with State Highlighted ---
    map_x, map_y = 20, CARD_HEIGHT - MAP_HEIGHT - 20
    map_group = create_mini_map_group(state_name, map_x, map_y)
    additional_content.append(f'\n{ET.tostring(map_group, encoding="unicode")}')

    # --- State Code Circle ---
    if code is None:
        code = generate_state_code(state_name, state_abbr)

    code_group = ET.Element("g", {"id": "state-code"})
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

    # Combine all elements
    final_svg = (
        template_content[:closing_tag_pos]
        + "\n".join(additional_content)
        + "\n"
        + template_content[closing_tag_pos:]
    )
    return final_svg


def generate_state_card_file(
    state_name: str,
    state_abbr: str,
    code: str = None,
    image_path: str = None,
    output_dir: str = "state_cards",
    template_path: str = TEMPLATE_PATH,
    skip_if_exists: bool = True,
):
    """
    Generate and save a state card to file.
    """
    filename = f"{state_name.lower().replace(' ', '_')}.svg"
    filepath = os.path.join(output_dir, filename)

    if skip_if_exists and os.path.exists(filepath):
        print(f"Card for {state_name} already exists. Skipping...")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating card for {state_name} ({state_abbr})...")

    # Generate card
    svg_content = create_state_card(
        state_name,
        state_abbr,
        code=code,
        image_path=image_path,
        template_path=template_path,
    )

    # Save to file
    with open(filepath, "w") as f:
        f.write(svg_content)
    print(f"  ✓ Saved to {filepath}")


def main():
    """
    Main function to generate state cards from CSV file.
    """
    print("Starting US State Cards Generation")
    print("=" * 50)

    # Read states from CSV file
    # Expected columns: Name, Abbreviation, Code (optional)
    csv_file = "states.csv"

    try:
        with open(csv_file, "r") as f:
            data = csv.DictReader(f)
            codes_seen = set()

            for i, elem in enumerate(data):
                if i > 0:
                    break
                try:
                    # Read state information from CSV
                    state_name = elem.get("Name", "").strip()
                    state_abbr = elem.get("Abbreviation", "").strip()
                    code = elem.get("Code", None)

                    if not state_name or not state_abbr:
                        print(
                            f"Warning: Skipping row {i+1} - missing Name or Abbreviation"
                        )
                        continue

                    # Check for duplicate codes
                    if code is not None:
                        code = code.strip() if code else None
                        if code and code in codes_seen:
                            raise ValueError(
                                f"Duplicate code {code} for {state_name}, {state_abbr}"
                            )
                        if code:
                            codes_seen.add(code)

                    # Generate image using make_image from generate_state_images
                    img_path = make_image(state=state_name)

                    # Generate the card
                    generate_state_card_file(
                        state_name=state_name,
                        state_abbr=state_abbr,
                        code=code,
                        image_path=img_path,
                        output_dir="state_cards",
                        template_path=TEMPLATE_PATH,
                        skip_if_exists=True,
                    )

                except Exception as e:
                    print(f"Error processing row {i+1} ({elem}): {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        print(
            "Please create a CSV file with columns: Name, Abbreviation, Code (optional)"
        )
        print("\nExample CSV content:")
        print("Name,Abbreviation,Code")
        print("California,CA,CAL")
        print("Texas,TX,TEX")
        print("New York,NY,")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print("=" * 50)
    print("State cards generation complete!")
    print(f"Cards saved to: state_cards/")


if __name__ == "__main__":
    main()
