"""
Agent Orchestrator
==================
The central coordinator for farmer dashboard requests.

Workflow:
1. Fetch farmer profile from MongoDB.
2. Resolve field geometry (KML from MinIO or GPS coordinates).
3. Check 24-hour cache for satellite indices.
4. Retrieve fresh Sentinel-2 / Sentinel-1 data if cache expired.
5. Correlate remote-sensing indices (NDVI) with soil test nitrogen for agronomic advisory.
6. Verify active fertigation window and schedule approval status.
7. Compute 3-Zone Variable Rate Application (VRA) prescription.
8. Log request metadata for administration monitoring.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from ninja.errors import HttpError

from app.core.database import database
from app.agents.geometry_agent import get_geometry
from app.agents import indices_agent
from app.services import fertilizer_service, vra_service

logger = logging.getLogger(__name__)

# Ultra-fast in-memory cache for farmer dashboard payloads (1 hour TTL)
_DASHBOARD_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TIMESTAMP: Dict[str, float] = {}
CACHE_TTL_SECONDS = 3600


def invalidate_farmer_cache(phone: str):
    """Purges the in-memory cache for a farmer so fresh data is computed on next request."""
    clean_phone = str(phone).strip()
    _DASHBOARD_MEMORY_CACHE.pop(clean_phone, None)
    _CACHE_TIMESTAMP.pop(clean_phone, None)
    # Also invalidate VRA memory cache keys for this farmer
    keys_to_del = [k for k in vra_service._VRA_MEMORY_CACHE if k.startswith(clean_phone)]
    for k in keys_to_del:
        vra_service._VRA_MEMORY_CACHE.pop(k, None)


def _generate_integrated_advisory(
    indices: Dict[str, Any],
    farmer_doc: Dict[str, Any],
    schedule: Dict[str, Any] = None,
) -> Dict[str, str]:
    """
    Multi-Factorial AI Agronomic Reasoning Engine.
    Dynamically correlates real-time satellite spectral & SAR telemetry with
    laboratory soil chemistry and crop phenology:
      - Optical Vigor & Greenness (NDVI)
      - Deep-Layer Chlorophyll & Canopy Nitrogen (NDRE)
      - Foliar & Soil Moisture Hydration (NDMI)
      - Sentinel-1 SAR Radar Stalk Volume & Tillering Roughness (RVI)
      - Soil Chemical Reaction (pH, N, P, K)
      - Crop Stage: Days After Planting (DAP)

    Args:
        indices: Mean vegetation and biophysical indices dictionary.
        farmer_doc: MongoDB farmer record containing soil_parameters.
        schedule: Fertigation schedule object including current_dap and active period.

    Returns:
        dict: Dynamic diagnostic summary with 'type', 'status', and recommended 'action'.
    """
    schedule = schedule or {}
    soil_params = farmer_doc.get("soil_parameters", {}) or {}

    ndvi = float(indices.get("NDVI") or 0.58)
    ndre = float(indices.get("NDRE") or 0.35)
    ndmi = float(indices.get("NDMI") or 0.30)
    rvi = float(indices.get("RVI") or 0.60)

    ph = float(soil_params.get("pH") or soil_params.get("ph") or 7.8)
    n_val = float(soil_params.get("N") or soil_params.get("nitrogen") or 165.0)
    p_val = float(soil_params.get("P") or soil_params.get("phosphorus") or 18.0)
    k_val = float(soil_params.get("K") or soil_params.get("potassium") or 240.0)

    cur_dap = int(schedule.get("current_dap") or 60)
    cur_period = schedule.get("current_period") or {}
    period_name = cur_period.get("period") or f"{cur_dap} DAP"

    # 1. Critical Water Deficit / Drought Stress
    if ndmi < 0.20:
        return {
            "type": "critical",
            "status": f"Acute Moisture Deficit (NDMI: {ndmi:.2f})",
            "action": f"Soil moisture is severely depressed. Pre-irrigate via drip for 45 mins before injecting {period_name} fertilizers to avoid osmotic root burn.",
        }

    # 2. Alkaline Soil Reaction & Phosphorus Lockup
    if ph >= 8.1 and p_val < 16.0:
        return {
            "type": "warning",
            "status": f"Alkaline Phosphorus Lockup (pH: {ph:.2f}, P: {p_val:.1f} kg/ha)",
            "action": f"Calcareous soil reaction binds phosphate. Deliver upcoming {period_name} fertilizer via acidic drip fertigation directly to root tips.",
        }

    # 3. Sentinel-1 Radar Tillering & Stalk Density Lag
    if rvi < 0.50 and cur_dap >= 45:
        return {
            "type": "warning",
            "status": f"Stalk Tillering & Density Lag (RVI: {rvi:.2f})",
            "action": f"Sentinel-1 radar detects delayed tillering at {cur_dap} DAP. Apply scheduled booster dose with uniform emitter pressure to stimulate basal cane buds.",
        }

    # 4. Deep-Canopy Nitrogen Exhaustion (RedEdge chlorophyll depression)
    if ndre < 0.35 and n_val < 200.0:
        return {
            "type": "critical",
            "status": f"Deep-Canopy Nitrogen Exhaustion (NDRE: {ndre:.2f})",
            "action": f"RedEdge chlorophyll reflects nitrogen depletion in lower leaves. Prioritize scheduled urea/nitrogen fertigation split for {period_name}.",
        }

    # 5. Topographic Moisture Saturation & Leaching Hazard
    if ndmi >= 0.48 and ndvi >= 0.70:
        return {
            "type": "info",
            "status": f"Moisture Pooling & Lodging Hazard (NDMI: {ndmi:.2f})",
            "action": f"High soil moisture saturation and heavy canopy cover. Reduce irrigation run-time and curtail nitrogen by -25% in dense patches to prevent lodging.",
        }

    # 6. Calcareous Soil & Marginal Potash
    if ph >= 8.2 and k_val < 220.0:
        return {
            "type": "warning",
            "status": f"Potassium Antagonism (pH: {ph:.2f}, K: {k_val:.1f} kg/ha)",
            "action": f"Alkaline soil restricts potash uptake. Apply water-soluble sulphate of potash (0:0:50) to strengthen cane rinds and lodging resistance.",
        }

    # 7. Vigorous Target Growth (Optimal optical and radar alignment)
    if ndvi >= 0.70 and rvi >= 0.70:
        return {
            "type": "success",
            "status": f"Vigorous Target Canopy (NDVI: {ndvi:.2f}, RVI: {rvi:.2f})",
            "action": f"Excellent canopy greenness and radar stalk development at {cur_dap} DAP. Continue scheduled {period_name} fertigation with precision VRA zone modulation.",
        }

    # 8. Standard Stable Growth (Default)
    return {
        "type": "success",
        "status": f"Balanced Crop Growth (NDVI: {ndvi:.2f})",
        "action": f"Canopy vigor and soil parameters are well balanced. Proceed with scheduled {period_name} fertigation dose.",
    }


async def run_orchestrator(phone: str) -> Dict[str, Any]:
    """
    Main orchestration pipeline that synthesizes farmer data, satellite imagery,
    fertigation scheduling, and precision VRA recommendations into one unified response.

    Args:
        phone: Registered 10-digit mobile number of the farmer.

    Returns:
        dict: Complete dashboard data payload (farmer, geometry, indices, advisory, schedule, vra).
    """
    start_time = time.time()
    clean_phone = str(phone).strip()

    # Fast-path: return from high-speed in-memory cache if fresh
    now_ts = time.time()
    if clean_phone in _DASHBOARD_MEMORY_CACHE and (now_ts - _CACHE_TIMESTAMP.get(clean_phone, 0)) < CACHE_TTL_SECONDS:
        logger.info(f"Fast In-Memory Cache Hit for phone {clean_phone} (0ms fetch)")
        return _DASHBOARD_MEMORY_CACHE[clean_phone]

    logger.info(f"Orchestrator pipeline started for phone: {clean_phone}")

    # -------------------------------------------------------------------------
    # Step 1: Retrieve Farmer Profile
    # -------------------------------------------------------------------------
    farmer_doc = await database["farmers"].find_one({"phone": phone}, {"_id": 0})
    if not farmer_doc:
        raise HttpError(404, f"No farmer registered with phone: {phone}")

    # -------------------------------------------------------------------------
    # Step 2: Retrieve Field Boundary Geometry
    # -------------------------------------------------------------------------
    geom_data = await get_geometry(farmer_doc)
    logger.info(f"  Field geometry loaded: area = {geom_data.get('area_ha', 0.0)} ha")

    # -------------------------------------------------------------------------
    # Step 3: Evaluate 24-Hour Cache for Satellite Indices
    # -------------------------------------------------------------------------
    now = datetime.utcnow()
    cache_data = farmer_doc.get("cache", {})
    cached_stats = None
    is_cache_hit = False

    has_recent_cache = (
        cache_data.get("last_updated")
        and cache_data.get("stats")
        and "image_urls" in cache_data.get("stats", {})
    )

    raw_last = cache_data.get("last_updated")
    last_dt = None
    if isinstance(raw_last, datetime):
        last_dt = raw_last
    elif isinstance(raw_last, str):
        try:
            clean_str = raw_last.replace("Z", "+00:00")
            last_dt = datetime.fromisoformat(clean_str)
            if last_dt.tzinfo is not None:
                last_dt = last_dt.replace(tzinfo=None)
        except Exception:
            last_dt = None

    if has_recent_cache and last_dt:
        # Check if cache is strictly less than 24 hours old
        if (now - last_dt) < timedelta(hours=24):
            cached_stats = cache_data["stats"]
            is_cache_hit = True
            logger.info("  Cache Hit: Using cached Sentinel statistics and TIFF URLs (0 PU).")
        else:
            logger.info("  Cache Expired: Existing cache is over 24 hours old.")

    # -------------------------------------------------------------------------
    # Step 4: Fetch Satellite Indices (Cache vs Live Copernicus CDSE / Mock)
    # -------------------------------------------------------------------------
    if is_cache_hit:
        indices_result = cached_stats
        indices_result["source"] = "local_cache"
        indices_result["pu_used"] = 0.0

        # Ensure advanced biophysical indicators (Chlorophyll & Biomass proxy) exist in cache
        if "chlorophyll" not in indices_result or "biomass" not in indices_result:
            enrichment = indices_agent.compute_chlorophyll_biomass_indicators(
                ndvi=indices_result.get("NDVI", 0.58),
                ndre=indices_result.get("NDRE", 0.35),
                evi=indices_result.get("EVI", 0.42),
                ndmi=indices_result.get("NDMI", 0.22),
            )
            indices_result.update(enrichment)
            await database["farmers"].update_one(
                {"phone": phone},
                {"$set": {"cache.stats": indices_result}},
            )
    else:
        # Run blocking network call to Copernicus in a background worker thread
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        indices_result = await loop.run_in_executor(
            None,
            indices_agent._cdse_indices_or_mock,
            phone,
            geom_data["geojson"],
            cached_stats,
        )

        # Persist fresh satellite statistics to MongoDB for future 24h requests
        if "NDVI" in indices_result:
            new_stats = {"capture_date": indices_result.get("capture_date")}
            for key, val in indices_result.items():
                if key not in ["capture_date", "source", "pu_used"]:
                    new_stats[key] = val

            await database["farmers"].update_one(
                {"phone": phone},
                {"$set": {"cache": {"last_updated": now, "stats": new_stats}}},
            )
            logger.info("  Cache Updated: Stored fresh 24h satellite data in database.")

    cdse_pu_used = indices_result.get("pu_used", 0.0)

    # -------------------------------------------------------------------------
    # Step 5: Check Fertigation Window & Advisory Approvals
    # -------------------------------------------------------------------------
    status_info = fertilizer_service.check_farmer_window_and_approval(farmer_doc)
    farmer_doc["window_active_today"] = status_info["window_active_today"]
    farmer_doc["is_pending"] = status_info["is_pending"]
    farmer_doc["cycle_status"] = status_info["cycle_status"]

    # Attach schedule and crop season metadata
    fert_schedule = status_info.get("schedule", {})
    farmer_doc["schedule"] = fert_schedule
    farmer_doc["crop_season"] = (
        farmer_doc.get("crop_season")
        or farmer_doc.get("season")
        or fert_schedule.get("season")
    )

    # Dynamic multi-factorial real-time agronomic advisory synthesis
    integrated_adv = _generate_integrated_advisory(indices_result, farmer_doc, fert_schedule)
    db_advisory = farmer_doc.get("approved_advisory") or {}
    integrated_adv["isApproved"] = True
    integrated_adv["approved_at"] = db_advisory.get("approved_at") or now.isoformat()
    if "admin_notes" in db_advisory:
        integrated_adv["admin_notes"] = db_advisory["admin_notes"]

    farmer_doc["approved_advisory"] = integrated_adv

    # -------------------------------------------------------------------------
    # Step 6: Log Telemetry for Admin Dashboard Monitoring
    # -------------------------------------------------------------------------
    processing_time_ms = int((time.time() - start_time) * 1000)
    await database["api_logs"].insert_one({
        "timestamp": now,
        "phone": phone,
        "name": farmer_doc.get("name"),
        "processing_time_ms": processing_time_ms,
        "is_cache_hit": is_cache_hit,
        "cdse_pu_used": cdse_pu_used,
    })

    # -------------------------------------------------------------------------
    # Step 7: Calculate 3-Zone Variable Rate Application (VRA) Prescription
    # -------------------------------------------------------------------------
    vra_data = vra_service.compute_farmer_vra(
        farmer=farmer_doc,
        schedule=fert_schedule,
        indices=indices_result,
        geometry=geom_data,
    )

    # -------------------------------------------------------------------------
    # Step 8: Return Complete Unified Assembled Package
    # -------------------------------------------------------------------------
    stcr_data = fertilizer_service.compute_farmer_stcr_full(farmer_doc)
    soil_insights = fertilizer_service.get_farmer_soil_insights(farmer_doc)

    cursor = database["schedule_compliance"].find({"phone": clean_phone}, {"_id": 0})
    compliance_logs = await cursor.to_list(length=None)
    compliance_by_period = {log["period"]: log for log in compliance_logs}

    response_payload = {
        "farmer": farmer_doc,
        "farmer_profile": {
            "sr_no": farmer_doc.get("sr_no"),
            "name": farmer_doc.get("name"),
            "phone": clean_phone,
            "address": farmer_doc.get("address") or (
                f"{farmer_doc.get('village', 'Kagal')}, Tal. {farmer_doc.get('taluka', 'Kagal')}, "
                f"Dist. {farmer_doc.get('district', 'Kolhapur')} - {farmer_doc.get('pincode', 416216)}"
            ),
            "prn": farmer_doc.get("prn"),
            "variety": farmer_doc.get("variety", "86032"),
            "transplanted": farmer_doc.get("transplanted"),
            "crop_season": fert_schedule.get("season"),
            "season": fert_schedule.get("season"),
            "area_acre": farmer_doc.get("area_acre"),
            "latitude": farmer_doc.get("latitude"),
            "longitude": farmer_doc.get("longitude"),
            "approved_advisory": integrated_adv,
            "window_active_today": status_info.get("window_active_today", False),
            "is_pending": False,
            "initial_login_snapshot": farmer_doc.get("initial_login_snapshot"),
            "first_login_at": (farmer_doc.get("initial_login_snapshot") or {}).get("first_login_at"),
            "login_dap": (farmer_doc.get("initial_login_snapshot") or {}).get("login_dap"),
            "live_dap": fert_schedule.get("current_dap"),
        },
        "soil_report": farmer_doc.get("soil_parameters", {}),
        "stcr": stcr_data,
        "rdf": stcr_data,
        "soil_insights": soil_insights,
        "compliance": compliance_by_period,
        "geometry": geom_data,
        "indices": indices_result,
        "advisory": integrated_adv,
        "schedule": fert_schedule,
        "vra": vra_data,
    }

    _DASHBOARD_MEMORY_CACHE[clean_phone] = response_payload
    _CACHE_TIMESTAMP[clean_phone] = time.time()
    return response_payload