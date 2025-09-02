#!/usr/bin/env python3
"""
Generate playing cards with US city/state information and mini maps.
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
from generate_images import make_image

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

    # Add white background for the map area
    # ET.SubElement(map_group, 'rect', {
    #     'width': str(MAP_WIDTH),
    #     'height': str(MAP_HEIGHT),
    #     'fill': 'white'
    # })

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


def create_city_card(
    city: str,
    state: str,
    code: str = None,
    image_path: str = None,
    template_path: str = "card.svg",
) -> str:
    """
    Create a playing card for a city/state combination.

    Args:
        city: City name
        state: State name or abbreviation
        template_path: Path to card.svg template

    Returns:
        SVG string for the card
    """
    # Load template as string
    template_content = load_card_template(template_path)

    # Find the closing </svg> tag to insert our content before it
    closing_tag_pos = template_content.rfind("</svg>")
    if closing_tag_pos == -1:
        print("Warning: Could not find closing SVG tag in template")
        closing_tag_pos = len(template_content)

    # Build the additional SVG content
    additional_content = []

    # Add image if provided
    if image_path:
        image_data = embed_image_as_base64(image_path)
        if image_data:
            # Center the image in the middle of the card
            img_width = 130
            img_height = 130
            img_x = (CARD_WIDTH - img_width) // 2
            img_y = (CARD_HEIGHT - img_height) // 2 - 10  # Slightly higher than center

            additional_content.append(
                f"""
        <!-- City Image -->
        <g id="city-image">
            <image x="{img_x}" y="{img_y}" width="{img_width}" height="{img_height}" 
                href="{image_data}" 
                preserveAspectRatio="xMidYMid meet"/>
        </g>"""
            )

    # Add city/state text at the top
    additional_content.append(
        f"""
    <!-- City and State Information -->
    <g id="city-info">
        <text x="{CARD_WIDTH // 2}" y="40" font-family="Arial, sans-serif" font-size="18" font-weight="bold" text-anchor="middle" fill="#2B2B2B">{city}</text>
        <text x="{CARD_WIDTH // 2}" y="58" font-family="Arial, sans-serif" font-size="14" text-anchor="middle" fill="#666">{state.upper()}</text>
    </g>"""
    )

    # Create mini map
    map_x = 20
    map_y = CARD_HEIGHT - MAP_HEIGHT - 20

    # Get the mini map as an element first
    map_group = create_mini_map_group(city, state, map_x, map_y)
    map_string = ET.tostring(map_group, encoding="unicode")
    additional_content.append(f"\n    <!-- Mini Map -->\n    {map_string}")

    # Add 3-letter code in bottom right corner
    if code is None:
        code = generate_city_code(city, state)
    additional_content.append(
        f"""
    <!-- City Code -->
    <g id="city-code">
        <circle cx="{CARD_WIDTH - 40}" cy="{CARD_HEIGHT - 40}" r="16" fill="#2BA6DE" opacity="0.1"/>
        <text x="{CARD_WIDTH - 40}" y="{CARD_HEIGHT - 40}" font-family="Consolas, monospace" font-size="15" font-weight="bold" text-anchor="middle" fill="#2BA6DE" dominant-baseline="central">{code}</text>
    </g>"""
    )

    # Insert the additional content before the closing </svg> tag
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
    output_dir: str = "cards",
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
        if i > 10:
            break
        try:
            name = elem["Name"]
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
