"""
Coordinate Correction Pipeline
==============================
Cleans and standardizes raw coordinate strings into valid closed GeoJSON polygons.

Pipeline Steps:
1. Strip all internal and external whitespace.
2. Verify valid comma delimiter.
3. Auto-inject missing decimal points (e.g. '164418' -> '16.4418').
4. Cast components to float.
5. Maharashtra regional bounding box validation (Lat: 15.0-22.5, Lon: 72.0-81.5).
6. Auto-detect and correct inverted Lat/Lon pairs.
7. Apply Convex Hull algorithm via Shapely to close 4-point rings into valid polygons.
8. Reproject to UTM Zone 43N (EPSG:32643) for true ground area measurement in hectares.
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
from shapely.geometry import MultiPoint

logger = logging.getLogger(__name__)

# Regional bounding box for Maharashtra and neighbouring districts
LAT_MIN, LAT_MAX = 15.0, 22.5
LON_MIN, LON_MAX = 72.0, 81.5


def _inject_decimal(num_str: str) -> str:
    """Inserts a decimal point after the first 2 digits if none is present (e.g., '16789' -> '16.789')."""
    if "." not in num_str and len(num_str) > 2:
        return num_str[:2] + "." + num_str[2:]
    return num_str


def clean_coord_string(raw: str) -> Optional[Tuple[float, float]]:
    """
    Parses and sanitizes a single 'lat,lon' string from field notes or Excel.

    Returns:
        tuple: (latitude, longitude) as floats, or None if unrecoverable.
    """
    if not raw or str(raw).strip().lower() in ("nan", "none", "", "null", "-"):
        return None

    raw_str = str(raw).strip()
    telemetry = []

    # Step 1: Strip all spaces (handles '16.4418, 74.1393' and internal typos like '17. 504503')
    cleaned = raw_str.replace(" ", "")
    if cleaned != raw_str:
        telemetry.append(f"whitespace stripped: '{raw_str}' -> '{cleaned}'")

    # Step 2: Must contain exactly one comma separating latitude and longitude
    if cleaned.count(",") != 1:
        logger.warning(f"Invalid coordinate format (missing or multiple commas): {raw_str}")
        return None

    part_a, part_b = cleaned.split(",")

    # Step 3: Auto-inject missing decimals
    fixed_a = _inject_decimal(part_a)
    fixed_b = _inject_decimal(part_b)
    if fixed_a != part_a:
        telemetry.append(f"decimal injected: '{part_a}' -> '{fixed_a}'")
    if fixed_b != part_b:
        telemetry.append(f"decimal injected: '{part_b}' -> '{fixed_b}'")

    # Step 4: Cast to floating-point numbers
    try:
        val1 = float(fixed_a)
        val2 = float(fixed_b)
    except ValueError as err:
        logger.warning(f"Could not convert coordinate to float '{raw_str}': {err}")
        return None

    # Step 5 & 6: Validate Maharashtra bounds and correct swapped Lat/Lon
    is_lat1 = LAT_MIN <= val1 <= LAT_MAX
    is_lon1 = LON_MIN <= val1 <= LON_MAX
    is_lat2 = LAT_MIN <= val2 <= LAT_MAX
    is_lon2 = LON_MIN <= val2 <= LON_MAX

    if is_lat1 and is_lon2:
        lat, lon = val1, val2  # Standard order: (Lat, Lon)
    elif is_lon1 and is_lat2:
        lat, lon = val2, val1  # Inverted order: (Lon, Lat) -> Swap to (Lat, Lon)
        telemetry.append(f"lat/lon inverted: ({val1}, {val2}) -> lat={lat}, lon={lon}")
    elif is_lat1 and not is_lon2:
        lat, lon = val1, val2
        logger.warning(f"Longitude slightly out of Maharashtra bounds: {raw_str} -> {val2}")
    else:
        logger.warning(f"Both coordinates outside regional bounds, skipped: {raw_str}")
        return None

    if telemetry:
        logger.debug(f"Coordinate corrections for '{raw_str}': {' | '.join(telemetry)}")

    return lat, lon


def build_closed_ring(raw_coords: List[str]) -> Optional[Dict[str, Any]]:
    """
    Takes 4 raw coordinate strings (A, B, C, D) and builds a closed GeoJSON polygon.
    Applies Convex Hull to resolve irregular order and closes the coordinate ring.

    Returns:
        dict: Standardized payload with GeoJSON Polygon, area in hectares, and correction logs.
    """
    corrections = []
    points_latlon = []

    for index, raw in enumerate(raw_coords):
        point_label = chr(65 + index)  # Point A, B, C, D
        parsed = clean_coord_string(raw)
        if parsed is None:
            corrections.append(f"Point {point_label} ('{raw}') could not be parsed and was skipped.")
            continue
        points_latlon.append(parsed)

    # A polygon requires at least 3 distinct vertices
    if len(points_latlon) < 3:
        logger.error("Insufficient valid points (< 3) to form a field boundary polygon.")
        return None

    # Step 7: Shapely geometry uses (x = longitude, y = latitude)
    shapely_points = [(lon, lat) for lat, lon in points_latlon]
    hull = MultiPoint(shapely_points).convex_hull

    # Handle degenerate geometries (e.g. points in a straight line)
    if hull.geom_type in ("Point", "LineString"):
        logger.error("ConvexHull failed: points are collinear or identical.")
        return None

    # Exterior coordinates of ConvexHull are automatically closed (first == last)
    ring_lonlat = list(hull.exterior.coords)
    geojson_coords = [list(ring_lonlat)]

    # Compute ground area in hectares
    area_ha = _calculate_area_ha(hull)

    corrections.append(
        f"ConvexHull formed a valid closed polygon with {len(ring_lonlat)} vertices from {len(points_latlon)} points."
    )

    return {
        "geojson": {
            "type": "Polygon",
            "coordinates": geojson_coords,
        },
        "area_ha": round(area_ha, 3),
        "corrections": corrections,
    }


def parse_kml_polygon(kml_text: str) -> Optional[Dict[str, Any]]:
    """
    Parses a Google Earth KML string to extract polygon boundary coordinates.
    """
    from lxml import etree

    try:
        root = etree.fromstring(kml_text.encode("utf-8"))
    except etree.XMLSyntaxError as err:
        logger.error(f"KML XML syntax parse error: {err}")
        return None

    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    coord_element = root.find(".//kml:coordinates", namespace)
    if coord_element is None:
        # Fallback search without XML namespace
        coord_element = root.find(".//coordinates")

    if coord_element is None or not coord_element.text:
        logger.error("No <coordinates> tag found in KML file.")
        return None

    raw_coords = coord_element.text.strip()
    ring_lonlat = []

    # KML coordinates format: "lon,lat,alt lon,lat,alt ..."
    for token in raw_coords.split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                ring_lonlat.append((lon, lat))
            except ValueError:
                continue

    if len(ring_lonlat) < 3:
        return None

    # Deduplicate repeated closing vertex if already closed
    if ring_lonlat[0] == ring_lonlat[-1]:
        ring_lonlat = ring_lonlat[:-1]

    # Re-close with ConvexHull to guarantee valid winding and geometry
    hull = MultiPoint(ring_lonlat).convex_hull
    closed_ring = list(hull.exterior.coords)
    area_ha = _calculate_area_ha(hull)

    return {
        "geojson": {
            "type": "Polygon",
            "coordinates": [closed_ring],
        },
        "area_ha": round(area_ha, 3),
        "corrections": [
            f"Parsed KML: extracted {len(ring_lonlat)} points -> closed ring with {len(closed_ring)} vertices."
        ],
        "source": "kml",
    }


def _calculate_area_ha(hull_geom) -> float:
    """
    Projects the polygon from WGS84 (EPSG:4326) to UTM Zone 43N (EPSG:32643)
    to calculate true surface area in hectares (1 ha = 10,000 m²).
    """
    try:
        from pyproj import Transformer
        from shapely.ops import transform

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
        projected_poly = transform(transformer.transform, hull_geom)
        return projected_poly.area / 10_000.0
    except Exception as err:
        logger.warning(f"Area projection calculation failed ({err}); returning 0.0 ha.")
        return 0.0