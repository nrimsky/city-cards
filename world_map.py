import json
from typing import Tuple, Optional, List, Dict
import xml.etree.ElementTree as ET
import math
import urllib.request
import urllib.parse

_GEO_DATA_CACHE: Optional[Dict] = None


def _unwrap_mean_lon(lons: List[float]) -> float:
    """
    Compute a mean longitude by unwrapping to avoid dateline issues.
    """
    if not lons:
        return 0.0
    base = lons[0]
    total = 0.0
    prev = base
    for lon in lons:
        # unwrap relative to previous so jumps are < 180°
        while lon - prev > 180.0:
            lon -= 360.0
        while lon - prev < -180.0:
            lon += 360.0
        total += lon
        prev = lon
    mean = total / len(lons)
    # wrap back to [-180, 180]
    while mean <= -180.0:
        mean += 360.0
    while mean > 180.0:
        mean -= 360.0
    return mean


def _country_centroid(features: List[Dict]) -> Tuple[float, float]:
    """
    Very simple geographic centroid for a country's features.
    Uses an unwrapped mean of longitudes and arithmetic mean of latitudes.
    """
    lons: List[float] = []
    lats: List[float] = []
    for f in features:
        geom = f.get("geometry", {})
        coords = geom.get("coordinates")
        gtype = geom.get("type")
        if not coords:
            continue
        polygons = (
            [coords]
            if gtype == "Polygon"
            else coords if gtype == "MultiPolygon" else []
        )
        for poly in polygons:
            # poly may be [ring] or [[ring, ...]]
            rings = [poly] if (poly and isinstance(poly[0][0], float)) else poly
            for ring in rings:
                for lon, lat in ring:
                    lons.append(lon)
                    lats.append(lat)
    if not lons or not lats:
        return (0.0, 0.0)
    return (_unwrap_mean_lon(lons), sum(lats) / len(lats))


def _ortho_project(
    lon_deg: float,
    lat_deg: float,
    lon0: float,
    lat0: float,
    radius: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, bool]:
    # Normalize Δλ into [-180, 180]
    dlon = lon_deg - lon0
    while dlon > 180.0:
        dlon -= 360.0
    while dlon < -180.0:
        dlon += 360.0

    lam = math.radians(dlon)
    phi = math.radians(lat_deg)
    phi0 = math.radians(lat0)

    # Front-hemisphere visibility
    cosc = math.sin(phi0) * math.sin(phi) + math.cos(phi0) * math.cos(phi) * math.cos(
        lam
    )
    visible = cosc > 0.0

    # Orthographic projection
    x = radius * (math.cos(phi) * math.sin(lam))
    y = radius * (
        math.cos(phi0) * math.sin(phi) - math.sin(phi0) * math.cos(phi) * math.cos(lam)
    )
    return (cx + x, cy - y, visible)


def _geojson_to_ortho_path(
    coordinates: List,
    lon0: float,
    lat0: float,
    radius: float,
    cx: float,
    cy: float,
) -> str:
    """
    Convert (Multi)Polygon coordinates to an SVG path using an orthographic projection
    centered at (lon0, lat0). Points on the far hemisphere are skipped.
    """
    if not coordinates:
        return ""

    path_parts: List[str] = []
    polygons = (
        [coordinates]
        if (coordinates and isinstance(coordinates[0][0][0], float))
        else coordinates
    )
    for poly in polygons:
        rings = [poly] if (poly and isinstance(poly[0][0], float)) else poly
        for ring in rings:
            started = False
            last_vis = False
            if len(ring) > 0 and not isinstance(ring[0], list):
                ring = [ring]
            for i, (lon, lat) in enumerate(ring):
                x, y, vis = _ortho_project(lon, lat, lon0, lat0, radius, cx, cy)
                if vis:
                    if not started:
                        path_parts.append(f"M {x:.2f},{y:.2f}")
                        started = True
                    else:
                        # if we were previously invisible, start a new subpath to avoid drawing across the horizon
                        if not last_vis:
                            path_parts.append(f"M {x:.2f},{y:.2f}")
                        else:
                            path_parts.append(f"L {x:.2f},{y:.2f}")
                last_vis = vis
            # Don't close with Z — closing can draw straight lines across the hidden hemisphere
    return " ".join(path_parts)


def _get_geodata() -> Optional[Dict]:
    """
    Fetch world countries GeoJSON data
    """
    global _GEO_DATA_CACHE
    if _GEO_DATA_CACHE:
        return _GEO_DATA_CACHE

    with open("world.geojson", "r") as f:
        _GEO_DATA_CACHE = json.load(f)
    return _GEO_DATA_CACHE


def _get_city_coordinates(
    city: str, country: str | None
) -> Optional[Tuple[float, float]]:
    query = f"{city}, {country}" if country else city
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
        raise ValueError(f"Error geocoding {city}, {country}: {e}")


def create_mini_map_group(
    country: str, card_height: float, city: Optional[str] = None
) -> ET.Element:
    """
    Create a group element containing a globe (orthographic projection) centered
    on the target country, with that country highlighted in red.
    """
    world_data = _get_geodata()
    features = world_data.get("features", [])

    lower_country = country.lower().strip()

    # Some countries are represented by sub-features in some datasets (e.g., UK)
    special_countries = {
        "united kingdom": ["england", "wales", "scotland", "northern ireland"],
    }

    target_features = [
        f
        for f in features
        if f.get("properties", {}).get("name", "").lower() == lower_country
        or f.get("properties", {}).get("name_long", "").lower() == lower_country
        or f.get("properties", {}).get("name_en", "").lower() == lower_country
        or f.get("properties", {}).get("name", "").lower()
        in special_countries.get(lower_country, [])
        or f.get("properties", {}).get("name_long", "").lower()
        in special_countries.get(lower_country, [])
        or f.get("properties", {}).get("name_en", "").lower()
        in special_countries.get(lower_country, [])
    ]

    if not target_features:
        raise ValueError(
            f"Warning: Could not find geometry for '{country}' in world data."
        )

    # Center the globe on the (approximate) centroid of the target country
    if city is not None:
        try:
            center_lon, center_lat = _get_city_coordinates(city, country)
        except Exception:
            center_lon, center_lat = _get_city_coordinates(city, None)
    else:
        center_lon, center_lat = _country_centroid(target_features)

    # Globe dimensions/placement (keep the same bottom-left anchor as before)
    radius = 35
    size = radius * 2.0
    map_x, map_y = 20.0, card_height - size - 20.0
    cx, cy = radius, radius  # local center inside the group

    # Build the group and a circular clip (so only the front hemisphere shows)
    map_group = ET.Element(
        "g", {"id": "mini-map", "transform": f"translate({map_x}, {map_y})"}
    )

    clip_id = f"mini-globe-clip-{abs(hash((country, 'globe'))) % 65535}"
    defs = ET.SubElement(map_group, "defs")
    clip = ET.SubElement(defs, "clipPath", {"id": clip_id})
    ET.SubElement(clip, "circle", {"cx": str(cx), "cy": str(cy), "r": str(radius)})

    # Ocean/sphere background
    ET.SubElement(
        map_group,
        "circle",
        {
            "cx": str(cx),
            "cy": str(cy),
            "r": str(radius),
            "fill": "#d8eef8",
            "stroke": "#888",
            "stroke-width": "0.5",
        },
    )

    # Container for land with clipping to the globe
    land_group = ET.SubElement(map_group, "g", {"clip-path": f"url(#{clip_id})"})

    # Draw all countries (background land)
    for feat in features:
        geom = feat.get("geometry", {})
        d = _geojson_to_ortho_path(
            geom.get("coordinates", []),
            lon0=center_lon,
            lat0=center_lat,
            radius=radius,
            cx=cx,
            cy=cy,
        )
        if d:
            ET.SubElement(
                land_group,
                "path",
                {"d": d, "fill": "#f0f0f0", "stroke": "#888", "stroke-width": "0.3"},
            )

    if city is None:
        # Draw the target country on top in highlight color
        for feat in target_features:
            geom = feat.get("geometry", {})
            d = _geojson_to_ortho_path(
                geom.get("coordinates", []),
                lon0=center_lon,
                lat0=center_lat,
                radius=radius,
                cx=cx,
                cy=cy,
            )
            if d:
                ET.SubElement(
                    land_group,
                    "path",
                    {
                        "d": d,
                        "fill": "#968283",
                        "stroke": "#EC1E28",
                        "stroke-width": "0.8",
                    },
                )
    else:
        x, y, vis = _ortho_project(
            center_lon, center_lat, center_lon, center_lat, radius, cx, cy
        )
        if vis:
            marker_group = ET.SubElement(map_group, "g", {"id": "city-marker"})
            # Outer ring
            ET.SubElement(
                marker_group,
                "circle",
                {
                    "cx": f"{x:.2f}",
                    "cy": f"{y:.2f}",
                    "r": "2",
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
                {"cx": f"{x:.2f}", "cy": f"{y:.2f}", "r": "0.75", "fill": "#EC1E28"},
            )

    return map_group
