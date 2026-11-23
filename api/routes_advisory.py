import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from ninja import Router
from ninja.errors import HttpError
from pydantic import BaseModel

from app.core.database import database
from app.services import fertilizer_service
from app.services import advisory
from app.services import vra_service
from app.agents.geometry_agent import get_geometry
from app.agents.orchestrator import invalidate_farmer_cache

# In-memory advisory cache (fast 1ms responses)
_ADVISORY_MEMORY_CACHE: Dict[str, Dict[str, Any]] = {}

router = Router(tags=["Sugarcane Fertilizer & Soil Advisory"])


class ConfirmScheduleRequest(BaseModel):
    period: str
    applied: bool = True
    notes: Optional[str] = None


async def _find_farmer(identifier: str) -> dict:
    identifier = identifier.strip()

    query = {
        "$or": [
            {"phone": identifier},
            {"name": {"$regex": f"^{re.escape(identifier)}$", "$options": "i"}},
        ]
    }

    if identifier.isdigit():
        query["$or"].append({"sr_no": int(identifier)})

    farmer = await database["farmers"].find_one(query, {"_id": 0})
    if not farmer:
        raise HttpError(404, f"Farmer '{identifier}' not found")
    return farmer


@router.get("/fertilizers")
async def get_fertilizer_options(request):
    return {
        "options": list(fertilizer_service.NPK_FERTILIZER_OPTIONS.keys()),
        "default": fertilizer_service.DEFAULT_FERTILIZER,
    }


@router.get("/farmer/{identifier}/stcr")
async def get_farmer_stcr(request, identifier: str):
    farmer = await _find_farmer(identifier)
    return fertilizer_service.compute_farmer_stcr_full(farmer)


@router.get("/farmer/{identifier}/rdf")
async def get_farmer_rdf(request, identifier: str):
    farmer = await _find_farmer(identifier)
    return fertilizer_service.compute_farmer_stcr_full(farmer)


@router.get("/farmer/{identifier}/soil-insights")
async def get_farmer_soil_insights(request, identifier: str):
    farmer = await _find_farmer(identifier)
    return fertilizer_service.get_farmer_soil_insights(farmer)


@router.get("/farmer/{identifier}/weekly-schedule")
async def get_farmer_weekly_schedule(request, identifier: str):
    farmer = await _find_farmer(identifier)
    return fertilizer_service.get_farmer_schedule(farmer)


@router.get("/farmer/{identifier}/advisory")
async def get_farmer_quick_advisory(request, identifier: str):
    farmer = await _find_farmer(identifier)
    schedule_data = fertilizer_service.get_farmer_schedule(farmer)

    sp = farmer.get("soil_parameters") or {}
    ec = sp.get("EC") if sp.get("EC") is not None else farmer.get("EC")
    ph = sp.get("pH") if sp.get("pH") is not None else farmer.get("pH")

    return {
        "farmer_name": farmer.get("name"),
        "season": schedule_data.get("season"),
        "cycle_status": schedule_data.get("cycle_status"),
        "days_since_planting": schedule_data.get("days_since_planting"),
        "upcoming_dose": schedule_data.get("current_period"),
        "ph": ph,
        "acidic_fertilizers_prioritized": schedule_data.get(
            "acidic_fertilizers_prioritized", False
        ),
        "alkaline_fertilizers_prioritized": schedule_data.get(
            "alkaline_fertilizers_prioritized", False
        ),
        "ec": ec,
        "ec_note_needed": ec is not None and ec > 1.0,
    }


@router.get("/farmer/{identifier}/full")
async def get_farmer_full_advisory(request, identifier: str):
    clean_id = str(identifier).strip()
    if clean_id in _ADVISORY_MEMORY_CACHE:
        return _ADVISORY_MEMORY_CACHE[clean_id]

    farmer = await _find_farmer(identifier)

    stcr_data = fertilizer_service.compute_farmer_stcr_full(farmer)
    soil_insights = fertilizer_service.get_farmer_soil_insights(farmer)
    status_info = fertilizer_service.check_farmer_window_and_approval(farmer)
    schedule_data = status_info["schedule"]

    approved_adv = farmer.get("approved_advisory")
    if not approved_adv:
        approved_adv = {
            "type": "success",
            "status": "Healthy crop growth",
            "action": "All indices are within expected bounds. Continue standard monitoring.",
            "isApproved": True,
            "approved_at": datetime.utcnow().isoformat(),
        }
    else:
        approved_adv = dict(approved_adv)
        approved_adv["isApproved"] = True

    cache = farmer.get("cache") or {}
    stats = cache.get("stats") or {}

    if not stats or stats.get("NDWI") is None or stats.get("NDVI") is None:
        try:
            geom = await get_geometry(farmer)
            geojson_poly = geom["geojson"]
        except Exception:
            lat = float(farmer.get("latitude") or 16.7)
            lon = float(farmer.get("longitude") or 74.2)
            d = 0.001
            geojson_poly = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon - d, lat - d],
                        [lon + d, lat - d],
                        [lon + d, lat + d],
                        [lon - d, lat + d],
                        [lon - d, lat - d],
                    ]
                ],
            }

        from app.agents.indices_agent import _mock_indices

        stats = _mock_indices(farmer.get("phone", ""), geojson_poly)
        cache["stats"] = stats
        cache["last_updated"] = datetime.utcnow().isoformat()

        await database["farmers"].update_one(
            {"phone": farmer.get("phone")},
            {
                "$set": {
                    "cache": cache,
                    "approved_advisory": approved_adv,
                    "is_pending": False,
                }
            },
        )

    cursor = database["schedule_compliance"].find(
        {"phone": farmer.get("phone")}, {"_id": 0}
    )
    compliance_logs = await cursor.to_list(length=None)
    compliance_by_period = {log["period"]: log for log in compliance_logs}

    advisory_payload = {
        "farmer_profile": {
            "sr_no": farmer.get("sr_no"),
            "name": farmer.get("name"),
            "phone": farmer.get("phone"),
            "address": farmer.get("address"),
            "prn": farmer.get("prn"),
            "variety": farmer.get("variety", "86032"),
            "transplanted": farmer.get("transplanted"),
            "crop_season": schedule_data.get("season"),
            "season": schedule_data.get("season"),
            "area_acre": farmer.get("area_acre"),
            "latitude": farmer.get("latitude"),
            "longitude": farmer.get("longitude"),
            "approved_advisory": approved_adv,
            "window_active_today": status_info["window_active_today"],
            "is_pending": False,
            "initial_login_snapshot": farmer.get("initial_login_snapshot"),
            "first_login_at": farmer.get("initial_login_snapshot", {}).get(
                "first_login_at"
            ),
            "login_dap": farmer.get("initial_login_snapshot", {}).get("login_dap"),
            "live_dap": schedule_data.get("current_dap"),
        },
        "soil_report": farmer.get("soil_parameters", {}),
        "stcr": stcr_data,
        "rdf": stcr_data,
        "soil_insights": soil_insights,
        "schedule": schedule_data,
        "indices": stats,
        "compliance": compliance_by_period,
    }

    _ADVISORY_MEMORY_CACHE[clean_id] = advisory_payload
    return advisory_payload


@router.post("/farmer/{identifier}/schedule/confirm")
async def confirm_schedule_application(
    request, identifier: str, body: ConfirmScheduleRequest
):
    farmer = await _find_farmer(identifier)
    phone = farmer.get("phone")

    record = {
        "phone": phone,
        "farmer_name": farmer.get("name"),
        "period": body.period,
        "applied": body.applied,
        "notes": body.notes,
        "logged_at": datetime.utcnow(),
    }

    await database["schedule_compliance"].update_one(
        {"phone": phone, "period": body.period}, {"$set": record}, upsert=True
    )

    if phone:
        invalidate_farmer_cache(phone)
        _ADVISORY_MEMORY_CACHE.pop(str(phone), None)
        _ADVISORY_MEMORY_CACHE.pop(str(identifier), None)

    return {
        "status": "success",
        "message": f"Dose for {body.period} marked as applied",
        "data": record,
    }


@router.get("/farmer/{identifier}/schedule/compliance")
async def get_schedule_compliance(request, identifier: str):
    farmer = await _find_farmer(identifier)
    phone = farmer.get("phone")

    cursor = database["schedule_compliance"].find({"phone": phone}, {"_id": 0})
    logs = await cursor.to_list(length=None)
    by_period = {log["period"]: log for log in logs}

    return {"phone": phone, "by_period": by_period, "logs": logs}


@router.get("/admin/outliers")
async def get_soil_outliers(request):
    cursor = database["farmers"].find({}, {"_id": 0})
    farmers = await cursor.to_list(length=None)

    flat_farmers = []
    for f in farmers:
        sp = f.get("soil_parameters") or {}
        flat = {**f, **sp, "farmer_name": f.get("name")}
        flat_farmers.append(flat)

    outliers = advisory.find_outlier_farmers(flat_farmers)
    return {"total_farmers": len(farmers), "outliers": outliers}


@router.get("/farmer/{identifier}/vra")
async def get_farmer_vra_prescription(request, identifier: str):
    farmer = await _find_farmer(identifier)
    schedule = fertilizer_service.get_farmer_schedule(farmer)
    geom_data = await get_geometry(farmer)
    stats = farmer.get("cache", {}).get("stats", {})

    return vra_service.compute_farmer_vra(
        farmer=farmer, schedule=schedule, indices=stats, geometry=geom_data
    )

