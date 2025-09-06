import json
import urllib.request
from typing import Tuple, Optional, List
import xml.etree.ElementTree as ET


def get_city_coordinates(city: str, state: str) -> Optional[Tuple[float, float]]:
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


def get_data():
    with open("us.geojson", "r") as f:
        data = json.load(f)
        return data


def get_us_states_geojson(
    state_filter: str = None, highlight_state: str = None
) -> dict:
    data = get_data()

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


def project_coordinates_for_state(
    lon: float, lat: float, state: str, width: int, height: int
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


def create_mini_map_group(
    state_name: str,
    x_offset: int,
    y_offset: int,
    map_width: int,
    map_height: int,
    city_name: Optional[str] = None,
) -> ET.Element:
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
    states_data = get_us_states_geojson(
        state_filter, highlight_state=state_name if city_name is None else None
    )

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
                coordinates, map_width, map_height, state_filter
            )
        elif geometry.get("type") == "MultiPolygon":
            path_d = ""
            for polygon in geometry.get("coordinates", []):
                path_d += (
                    geojson_to_svg_path([polygon], map_width, map_height, state_filter)
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

    if city_name is not None:
        city_coords = get_city_coordinates(city_name, state_name)
        city_x, city_y = project_coordinates_for_state(
            city_coords[0], city_coords[1], state_name, map_width, map_height
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
