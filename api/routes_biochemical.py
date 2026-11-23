import datetime as dt
import re
from typing import Any, Optional, Dict, List
from ninja import Router
from ninja.errors import HttpError
from pydantic import BaseModel, Field

from app.core.database import database
from app.agrogeo_model import runner
from app.agrogeo_model.crops import CROPS

router = Router(tags=["Biochemical Soil & Water Analysis"])


class BiochemicalInputs(BaseModel):
    climate: str = "deccan"
    soc_pct: float = Field(0.6, ge=0.1, le=10, description="Soil organic carbon percentage")
    ph: float = Field(7.2, ge=3.5, le=10, description="Soil pH")
    bulk_density: float = Field(1.4, ge=0.9, le=1.8, description="Soil bulk density in g/cm3")
    ece: Optional[float] = Field(None, ge=0, le=40, description="Soil salinity ECe in dS/m")
    esp: Optional[float] = Field(None, ge=0, le=60, description="Exchangeable Sodium Percentage")
    irrigation: str = "irrigated"
    tillage: str = "conventional"
    residue_pct: float = Field(30, ge=0, le=100, description="Crop residue retention percentage")
    fym_t_ha: float = Field(0, ge=0, le=100, description="Farmyard manure applied in tons/ha")
    water: Optional[Dict[str, float]] = None
    fert_plan: Optional[List[Dict[str, Any]]] = None
    sowing_date: Optional[str] = None


async def _find_farmer(identifier: str) -> Dict[str, Any]:
    cleaned_id = identifier.strip()
    query: Dict[str, Any] = {
        "$or": [
            {"phone": cleaned_id},
            {"name": {"$regex": f"^{re.escape(cleaned_id)}$", "$options": "i"}},
        ]
    }
    if cleaned_id.isdigit():
        query["$or"].append({"sr_no": int(cleaned_id)})

    farmer = await database["farmers"].find_one(query, {"_id": 0})
    if not farmer:
        raise HttpError(404, f"Farmer '{cleaned_id}' not found")
    return farmer


def _num(source: Dict[str, Any], keys: tuple, default: Any) -> Any:
    for key in keys:
        val = source.get(key)
        if val is not None and val != "":
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return default


def _sowing_date(farmer: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        try:
            return dt.date.fromisoformat(override).isoformat()
        except ValueError:
            raise HttpError(422, "sowing_date must use YYYY-MM-DD format.")

    raw = (
        farmer.get("transplanted")
        or farmer.get("planting_date")
        or farmer.get("sowing_date")
    )
    if raw:
        for fmt in ("%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return dt.datetime.strptime(str(raw).strip(), fmt).date().isoformat()
            except ValueError:
                continue
    return dt.date.today().isoformat()


@router.get("/farmer/{identifier}/defaults")
async def biochemical_defaults(request, identifier: str):
    farmer = await _find_farmer(identifier)
    sp = farmer.get("soil_parameters") or {}

    from app.services.fertilizer_service import calculate_organic_manure

    oc_val = _num(sp, ("soc_pct", "organic_carbon", "OC", "oc"), None)
    default_fym = calculate_organic_manure(oc_val) if oc_val is not None else 0.0

    return {
        "farmer": {
            "name": farmer.get("name"),
            "phone": farmer.get("phone"),
            "area_acre": farmer.get("area_acre"),
            "address": farmer.get("address"),
            "latitude": farmer.get("latitude"),
            "longitude": farmer.get("longitude"),
            "variety": farmer.get("variety", "86032"),
            "transplanted": farmer.get("transplanted"),
        },
        "inputs": {
            "climate": "deccan",
            "soc_pct": _num(sp, ("soc_pct", "organic_carbon", "OC", "oc"), 0.6),
            "ph": _num(sp, ("pH", "ph", "PH"), 7.2),
            "bulk_density": _num(sp, ("bulk_density", "BD", "bd"), 1.4),
            "ece": _num(sp, ("ECe", "ece", "EC", "ec"), None),
            "esp": _num(sp, ("ESP", "esp"), None),
            "irrigation": "irrigated",
            "tillage": "conventional",
            "residue_pct": 30,
            "fym_t_ha": _num(sp, ("fym_t_ha", "organic_manure", "manure"), default_fym),
            "sowing_date": _sowing_date(farmer, None),
            "water": None,
            "fert_plan": None,
        },
        "soil_parameters_available": sorted(sp.keys()),
        "notes": [
            "Values found in the farmer's soil_parameters are prefilled.",
            "Missing soil measurements use clearly labelled indicative defaults; edit them before relying on the model.",
        ],
    }


@router.post("/farmer/{identifier}/analyze")
async def analyze_farmer_biochemical(request, identifier: str, body: BiochemicalInputs):
    farmer = await _find_farmer(identifier)
    sp = farmer.get("soil_parameters") or {}

    inputs = body.model_dump()
    if inputs["climate"] not in (
        "deccan",
        "nwindia",
        "gangetic",
        "westcoast",
        "temperate",
    ):
        raise HttpError(422, "Unsupported climate preset.")
    if inputs["irrigation"] not in ("irrigated", "rainfed"):
        raise HttpError(422, "irrigation must be 'irrigated' or 'rainfed'.")
    if inputs["tillage"] not in ("conventional", "reduced", "notill"):
        raise HttpError(422, "Unsupported tillage value.")

    water = inputs.get("water")
    if water:
        required = ("Ca", "Mg", "Na", "K", "Cl", "SO4", "HCO3", "pH", "temp")
        if any(k not in water for k in required):
            raise HttpError(
                422,
                "Water test requires: Ca, Mg, Na, K, Cl, SO4, HCO3, pH, and temp.",
            )

    fert_plan = inputs.get("fert_plan")
    if fert_plan is None:
        from app.services import fertilizer_service

        try:
            stcr = fertilizer_service.compute_farmer_stcr_full(farmer)
            n_per_acre = float(
                stcr.get(
                    "final_stcr_n_kg_per_acre",
                    stcr.get(
                        "stcr_n_kg_per_acre",
                        stcr.get("final_rdf_n_kg_per_acre", 48.6),
                    ),
                )
                or 48.6
            )
            n_kg_ha = n_per_acre * 2.47105
        except Exception:
            n_kg_ha = 120.0

        n_kg_ha = max(0, min(n_kg_ha, 600))
        fert_plan = [
            {"day": 0, "type": "dap", "n_kg": n_kg_ha * 0.25},
            {"day": 30, "type": "urea", "n_kg": n_kg_ha * 0.375},
            {"day": 60, "type": "urea", "n_kg": n_kg_ha * 0.375},
        ]

    field_data = {
        "name": farmer.get("name") or identifier,
        "area_acres": _num(farmer, ("area_acre", "area_acres"), 1),
        "climate": inputs["climate"],
        "latitude": _num(farmer, ("latitude",), 18.5),
        "soc_pct": inputs["soc_pct"],
        "ph": inputs["ph"],
        "bulk_density": inputs["bulk_density"],
        "ece": inputs["ece"],
        "esp": inputs["esp"],
        "crop": "sugarcane",
        "sowing_date": _sowing_date(farmer, inputs.get("sowing_date")),
        "irrigation": inputs["irrigation"],
        "tillage": inputs["tillage"],
        "residue_pct": inputs["residue_pct"],
        "fym_t_ha": inputs["fym_t_ha"],
        "water": water,
        "fert_plan": fert_plan,
    }

    try:
        result = runner.run_field(field_data)
    except Exception as exc:
        raise HttpError(
            422,
            f"Biochemical simulation could not run with these inputs: {exc}",
        ) from exc

    sim = result["sim"]
    totals = sim["totals"]
    total_n_losses = totals["leach"] + sim["n2o_total"] + totals["n2"] + totals["nh3"]

    return {
        "status": "success",
        "farmer": {
            "name": farmer.get("name"),
            "phone": farmer.get("phone"),
            "area_acre": farmer.get("area_acre"),
            "variety": farmer.get("variety", "86032"),
        },
        "inputs_used": {
            **inputs,
            "sowing_date": field_data["sowing_date"],
            "crop": "sugarcane",
            "area_acres": field_data["area_acres"],
            "fert_plan": fert_plan,
        },
        "results": {
            "soil_health": result["scores"],
            "expected_yield_t_ha": sim["yield_t_ha"],
            "expected_yield_q_acre": sim["yield_t_ha"] * 4.047,
            "nitrogen_applied_kg_ha": result["fert_n_total"],
            "nitrogen_losses_kg_ha": total_n_losses,
            "nitrate_leached_kg_ha": totals["leach"],
            "n2o_kg_ha": sim["n2o_total"],
            "irrigation_mm": sim["irrigation_mm"],
            "water_chemistry": result["water"],
            "gypsum": result["gypsum"],
            "carbon": result["carbon"],
            "nitrogen_rates": result["n_rates"],
            "nitrogen_recommendation_kg_ha": result["n_rec"],
            "advice": result["advice"],
            "model_architecture": result.get("model_architecture"),
            "scientific_references": result.get("scientific_references"),
        },
        "model_notice": (
            "Biochemical outputs are indicative model estimates, not a substitute for a laboratory "
            "soil/water test or local agricultural-university advice."
        ),
    }

