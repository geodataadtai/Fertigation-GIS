"""
Geometry Agent
==============
Responsible for extracting and standardizing field boundary geometry for a farmer.

Workflow:
1. Primary Source: MinIO KML storage (high-precision closed polygon file).
2. Fallback Source: 4 GPS corner coordinates (A, B, C, D) stored in the farmer's profile.

Returns a standardized GeoJSON polygon structure and calculated acreage/hectares.
"""

import logging
from typing import Dict, Any

from app.core.database import minio_client
from app.services.coord_correction import build_closed_ring, parse_kml_polygon

logger = logging.getLogger(__name__)

# Global in-memory geometry cache to prevent repeated MinIO network trips
_GEOMETRY_CACHE: Dict[str, Dict[str, Any]] = {}


async def get_geometry(farmer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves the farmer's field boundary polygon and converts it to GeoJSON.

    Strategy:
    - First checks in-memory cache and farmer doc.
    - Next attempts to load and parse the official KML file from MinIO.
    - If the KML is missing or cannot be parsed, falls back to the 4 corner
      coordinates (coord_a, coord_b, coord_c, coord_d) from the farmer's record.

    Args:
        farmer (dict): Farmer document containing phone, kml_filename, and/or corner coordinates.

    Returns:
        dict: Standardized geometry payload containing:
            - geojson: GeoJSON Polygon dictionary
            - area_ha: Plot area in hectares
            - source: 'kml' or 'excel'
            - corrections: List of coordinate telemetry/correction notes
    """
    phone = str(farmer.get("phone", "")).strip()
    if phone and phone in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[phone]

    if isinstance(farmer.get("geometry"), dict) and "geojson" in farmer["geometry"]:
        if phone:
            _GEOMETRY_CACHE[phone] = farmer["geometry"]
        return farmer["geometry"]

    kml_filename = farmer.get("kml_filename")
    sr_no = farmer.get("sr_no")

    # -------------------------------------------------------------------------
    # Option 1: Load high-resolution polygon from MinIO KML file (or local disk fallback)
    # -------------------------------------------------------------------------
    candidates = []
    if kml_filename:
        candidates.append(kml_filename)
    if phone:
        candidates.append(f"{phone}.kml")
        if sr_no:
            candidates.append(f"{phone}_{sr_no}.kml")

    import os
    kml_dir = os.path.join(os.path.dirname(__file__), "..", "data", "kml")
    if os.path.exists(kml_dir):
        for f in os.listdir(kml_dir):
            if f.startswith(phone) and f.endswith(".kml") and f not in candidates:
                candidates.append(f)

    for fname in candidates:
        kml_text = None
        try:
            response = minio_client.get_object("kml-storage", fname)
            kml_text = response.data.decode("utf-8")
            response.close()
            response.release_conn()
        except Exception:
            pass

        if not kml_text and os.path.exists(kml_dir):
            disk_path = os.path.join(kml_dir, fname)
            if os.path.exists(disk_path):
                try:
                    with open(disk_path, "r", encoding="utf-8") as f:
                        kml_text = f.read()
                except Exception:
                    pass

        if kml_text:
            result = parse_kml_polygon(kml_text)
            if result:
                result["source"] = "kml"
                logger.info(f"Geometry agent: successfully parsed KML ({fname}) for phone {phone}")
                if phone:
                    _GEOMETRY_CACHE[phone] = result
                return result

    # -------------------------------------------------------------------------
    # Option 2: Fallback to 4-corner GPS coordinates (A, B, C, D)
    # -------------------------------------------------------------------------
    logger.info(f"Geometry agent: using Excel coordinates fallback for phone {phone}")
    raw_coords = [
        str(farmer.get("coord_a", "") or ""),
        str(farmer.get("coord_b", "") or ""),
        str(farmer.get("coord_c", "") or ""),
        str(farmer.get("coord_d", "") or ""),
    ]

    result = build_closed_ring(raw_coords)
    if result is not None:
        result["source"] = "excel"
        if phone:
            _GEOMETRY_CACHE[phone] = result
        return result

    # -------------------------------------------------------------------------
    # Option 3: Fallback to center point (latitude, longitude) + area
    # -------------------------------------------------------------------------
    lat = farmer.get("latitude")
    lon = farmer.get("longitude")
    if lat is not None and lon is not None:
        try:
            lat = float(lat)
            lon = float(lon)
            area_acre = float(farmer.get("area_acre") or 1.0)
            area_m2 = area_acre * 4046.86
            half_side_m = (area_m2 ** 0.5) / 2.0
            import math
            d_lat = half_side_m / 111000.0
            d_lon = half_side_m / (111000.0 * math.cos(math.radians(lat)))
            closed_ring = [
                (round(lon - d_lon, 6), round(lat - d_lat, 6)),
                (round(lon + d_lon, 6), round(lat - d_lat, 6)),
                (round(lon + d_lon, 6), round(lat + d_lat, 6)),
                (round(lon - d_lon, 6), round(lat + d_lat, 6)),
                (round(lon - d_lon, 6), round(lat - d_lat, 6)),
            ]
            fallback_res = {
                "geojson": {
                    "type": "Polygon",
                    "coordinates": [closed_ring],
                },
                "area_ha": round(area_acre * 0.404686, 3),
                "corrections": [
                    f"Generated rectangular boundary from centroid ({lat}, {lon}) and area ({area_acre} acres)."
                ],
                "source": "centroid",
            }
            if phone:
                _GEOMETRY_CACHE[phone] = fallback_res
            return fallback_res
        except Exception as err:
            logger.warning(f"Centroid boundary generation failed: {err}")

    raise ValueError(f"Could not build a valid boundary polygon for farmer with phone: {phone}")