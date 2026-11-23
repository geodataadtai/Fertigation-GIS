"""
Sugarcane Fertilizer & Soil Advisory Service
============================================
Core engine for Soil Test Crop Response (STCR) targeted yield calculations,
organic manure budgeting, and season-specific fertigation requirements.

Key Scientific Concepts:
1. STCR Targeted Yield Equation:
   Calculates exact fertilizer requirement (kg/ha and kg/acre) based on crop target yield,
   soil test available nutrients (N, P, K), and organic manure application.
2. Crop Season Calibration:
   - Adsali: 250 ton/ha target yield (standard 400:170:170 kg NPK/ha)
   - Pre-Seasonal: 220 ton/ha target yield (standard 340:170:170 kg NPK/ha)
   - Suru: 200 ton/ha target yield (standard 250:115:115 kg NPK/ha)
3. Bounded STCR Rule:
   Applies an 80% STCR targeted dose clamped within ±20% of the crop season standard,
   preventing over- or under-fertilization.
4. 'To Be Applied' Balance:
   Bounded STCR Target minus nutrients already applied till date by the farmer.
"""

import math
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List

from app.services import advisory

# =============================================================================
# Constants & Agronomic Benchmarks
# =============================================================================

EXPECTED_YIELD = 250          # ton/ha standard baseline yield for Adsali sugarcane
HA_TO_ACRE = 2.47105          # Conversion: 1 hectare = 2.47105 acres

# Complex fertilizer catalog for sugarcane
NPK_FERTILIZER_OPTIONS: Dict[str, Dict[str, float]] = {
    "24:24:00": {"N": 24.0, "P": 24.0, "K": 0.0},
    "14:35:14": {"N": 14.0, "P": 35.0, "K": 14.0},
    "10:26:26": {"N": 10.0, "P": 26.0, "K": 26.0},
    "12:32:16": {"N": 12.0, "P": 32.0, "K": 16.0},
}
DEFAULT_FERTILIZER = "24:24:00"

# Standard seasonal nutrient recommendations and yield goals
SEASON_STANDARDS: Dict[str, Dict[str, Any]] = {
    "adsali": {
        "expected_yield_ton_per_ha": 250,
        "standard_n_kg_per_ha": 400.0,
        "standard_p_kg_per_ha": 170.0,
        "standard_k_kg_per_ha": 170.0,
    },
    "pre-seasonal": {
        "expected_yield_ton_per_ha": 220,
        "standard_n_kg_per_ha": 340.0,
        "standard_p_kg_per_ha": 170.0,
        "standard_k_kg_per_ha": 170.0,
    },
    "suru": {
        "expected_yield_ton_per_ha": 200,
        "standard_n_kg_per_ha": 250.0,
        "standard_p_kg_per_ha": 115.0,
        "standard_k_kg_per_ha": 115.0,
    },
}

# In-memory cache for schedule generation
SCHEDULE_CACHE: Dict[str, Any] = {}


# =============================================================================
# Utility Functions
# =============================================================================

def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts string, numeric, or None values to float with fallback."""
    if val is None:
        return default
    try:
        if isinstance(val, str):
            val = val.strip()
            if not val or val.lower() in ("null", "none", "nan", "n/a", "undefined", "-"):
                return default
        f = float(val)
        return default if math.isnan(f) else f
    except (ValueError, TypeError):
        return default


def assign_fertilizer(farmer_name: str) -> str:
    """Deterministically assigns one complex fertilizer per farmer based on SHA-256 hash of their name."""
    options = sorted(NPK_FERTILIZER_OPTIONS.keys())
    name_str = str(farmer_name or "Farmer")
    digest = hashlib.sha256(f"fertilizer-assignment::{name_str}".encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(options)
    return options[index]


def get_season_standard(season_str: Any) -> Dict[str, Any]:
    """Resolves standard season parameters from Marathi or English text."""
    s = str(season_str or "").strip().lower()
    if "adsali" in s or "आडसाली" in s:
        return SEASON_STANDARDS["adsali"]
    elif "suru" in s or "सुरू" in s or "seasonal" in s:
        return SEASON_STANDARDS["suru"]
    return SEASON_STANDARDS["pre-seasonal"]


# =============================================================================
# Organic Manure & STCR Targeted Yield Calculations
# =============================================================================

def calculate_organic_manure(oc_value: Any) -> float:
    """
    Organic Manure recommendation (ton/ha) based on Organic Carbon (OC %):
    Starts at 28 ton/ha for low OC [0.1, 0.2%) and decreases by 1 ton/ha per 0.1% increase in OC.
    """
    oc_val = safe_float(oc_value, 0.5)
    bracket_index = math.floor((oc_val - 0.1) / 0.1 + 1e-9)
    om_value = 28 - bracket_index
    return round(float(om_value), 2)


def calculate_stcr(oc: Any, n: Any, p: Any, k: Any, season: Any = "Adsali") -> Dict[str, Any]:
    """
    Calculates Soil Test Crop Response (STCR) targeted yield equations for Sugarcane:

    Formulas (Mahatma Phule Krishi Vidyapeeth, MPKV Rahuri calibration):
      - N (kg/ha) = 4.03 * TargetYield - 1.43 * Soil_N - 3.81 * OrganicManure
      - P (kg/ha) = 1.23 * TargetYield - 2.44 * Soil_P - 1.83 * OrganicManure
      - K (kg/ha) = 2.26 * TargetYield - 0.55 * Soil_K - 1.40 * OrganicManure
    """
    oc_clean = safe_float(oc, 0.5)
    n_clean = safe_float(n, 100.0)
    p_clean = safe_float(p, 10.0)
    k_clean = safe_float(k, 300.0)

    season_std = get_season_standard(season)
    expected_yield = season_std["expected_yield_ton_per_ha"]

    organic_manure = calculate_organic_manure(oc_clean)
    organic_manure_ton_per_acre = round(organic_manure / 2.5, 2)

    stcr_n_ha = (4.03 * expected_yield) - (1.43 * n_clean) - (3.81 * organic_manure)
    stcr_p_ha = (1.23 * expected_yield) - (2.44 * p_clean) - (1.83 * organic_manure)
    stcr_k_ha = (2.26 * expected_yield) - (0.55 * k_clean) - (1.40 * organic_manure)

    stcr_n_acre = stcr_n_ha / HA_TO_ACRE
    stcr_p_acre = stcr_p_ha / HA_TO_ACRE
    stcr_k_acre = stcr_k_ha / HA_TO_ACRE

    return {
        "organic_manure_ton_per_ha": organic_manure,
        "organic_manure_ton_per_acre": organic_manure_ton_per_acre,
        "expected_yield_ton_per_ha": expected_yield,
        "stcr_n_kg_per_ha": round(stcr_n_ha, 2),
        "stcr_p_kg_per_ha": round(stcr_p_ha, 2),
        "stcr_k_kg_per_ha": round(stcr_k_ha, 2),
        "stcr_n_kg_per_acre": round(stcr_n_acre, 2),
        "stcr_p_kg_per_acre": round(stcr_p_acre, 2),
        "stcr_k_kg_per_acre": round(stcr_k_acre, 2),
        # Backward-compatibility aliases
        "rdf_n_kg_per_ha": round(stcr_n_ha, 2),
        "rdf_p_kg_per_ha": round(stcr_p_ha, 2),
        "rdf_k_kg_per_ha": round(stcr_k_ha, 2),
        "rdf_n_kg_per_acre": round(stcr_n_acre, 2),
        "rdf_p_kg_per_acre": round(stcr_p_acre, 2),
        "rdf_k_kg_per_acre": round(stcr_k_acre, 2),
    }


calculate_rdf = calculate_stcr


def calculate_basal_dose(*args, **kwargs) -> Dict[str, Any]:
    """Stub retained for backwards compatibility."""
    return {}


# =============================================================================
# Full Farmer STCR Targeted Dossier
# =============================================================================

def compute_farmer_stcr_full(farmer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes full STCR targeted yield requirements, bounded target limits,
    and remaining 'To Be Applied' NPK for the farmer's plot.
    """
    sp = farmer.get("soil_parameters") or {}
    oc = safe_float(sp.get("OC") if sp.get("OC") is not None else farmer.get("OC"), 0.5)
    n = safe_float(sp.get("N") if sp.get("N") is not None else farmer.get("N"), 100.0)
    p = safe_float(sp.get("P") if sp.get("P") is not None else farmer.get("P"), 10.0)
    k = safe_float(sp.get("K") if sp.get("K") is not None else farmer.get("K"), 300.0)
    season = farmer.get("crop_season") or farmer.get("season", "Adsali")

    fertilizer_key = assign_fertilizer(farmer.get("name") or farmer.get("farmer_name", "Farmer"))
    calc = calculate_stcr(oc, n, p, k, season)

    final_n = calc["stcr_n_kg_per_acre"]
    final_p = calc["stcr_p_kg_per_acre"]
    final_k = calc["stcr_k_kg_per_acre"]

    # Nutrients already applied till date by the farmer (kg/acre)
    applied_data = sp.get("applied_till_date") or farmer.get("applied_till_date") or {}
    applied_n = safe_float(applied_data.get("N"), 0.0)
    applied_p = safe_float(applied_data.get("P"), 0.0)
    applied_k = safe_float(applied_data.get("K"), 0.0)

    # Standard per-acre season target
    season_key = str(season or "").strip().lower()
    if "adsali" in season_key or "आडसाली" in season_key:
        std_target_n, std_target_p, std_target_k = 160.0, 68.0, 68.0
    elif "suru" in season_key or "सुरू" in season_key:
        std_target_n, std_target_p, std_target_k = 100.0, 46.0, 46.0
    else:
        std_target_n, std_target_p, std_target_k = 136.0, 68.0, 68.0

    # Bounded STCR Rule: 80% STCR clamped within ±20% of Crop Season Standard
    raw_80_n = 0.80 * final_n
    raw_80_p = 0.80 * final_p
    raw_80_k = 0.80 * final_k

    bounded_stcr_target_n = round(max(0.80 * std_target_n, min(1.20 * std_target_n, raw_80_n)), 2)
    bounded_stcr_target_p = round(max(0.80 * std_target_p, min(1.20 * std_target_p, raw_80_p)), 2)
    bounded_stcr_target_k = round(max(0.80 * std_target_k, min(1.20 * std_target_k, raw_80_k)), 2)

    # To Be Applied = Bounded STCR Target - Nutrients Applied Till Date
    to_be_applied_n = round(max(0.0, bounded_stcr_target_n - applied_n), 2)
    to_be_applied_p = round(max(0.0, bounded_stcr_target_p - applied_p), 2)
    to_be_applied_k = round(max(0.0, bounded_stcr_target_k - applied_k), 2)
    to_be_applied = {
        "N": to_be_applied_n,
        "P": to_be_applied_p,
        "K": to_be_applied_k,
    }

    farm_area = safe_float(farmer.get("area_acre") or farmer.get("area_acres") or farmer.get("area"), 1.0)
    if farm_area <= 0:
        farm_area = 1.0

    season_std = get_season_standard(season)

    return {
        "farmer_name": farmer.get("name") or farmer.get("farmer_name"),
        "phone": farmer.get("phone") or farmer.get("mobile"),
        "transplanted": farmer.get("transplanted") or farmer.get("plantation_date"),
        "crop_season": season,
        "area_acre": safe_float(farmer.get("area_acre"), 1.0) if farmer.get("area_acre") is not None else None,
        "farm_area_acre": farm_area,
        "soil_parameters": {
            k: v for k, v in {
                "OC": sp.get("OC") if sp.get("OC") is not None else farmer.get("OC"),
                "N": n, "P": p, "K": k
            }.items() if v is not None
        },
        **calc,
        "recommendation_text": (
            f"Recommended N:P:K = {round(calc['stcr_n_kg_per_acre'])} : "
            f"{round(calc['stcr_p_kg_per_acre'])} : {round(calc['stcr_k_kg_per_acre'])} kg/acre"
        ),
        "final_stcr_n_kg_per_acre": final_n,
        "final_stcr_p_kg_per_acre": final_p,
        "final_stcr_k_kg_per_acre": final_k,
        "final_recommendation_text": (
            f"STCR N:P:K = {round(final_n)} : {round(final_p)} : {round(final_k)} kg/acre"
        ),
        "applied_till_date": {
            "N": applied_n,
            "P": applied_p,
            "K": applied_k,
        },
        "to_be_applied": to_be_applied,
        "farm_to_be_applied": {
            "N": round(to_be_applied_n * farm_area, 2),
            "P": round(to_be_applied_p * farm_area, 2),
            "K": round(to_be_applied_k * farm_area, 2),
        },
        "farm_applied_till_date": {
            "N": round(applied_n * farm_area, 2),
            "P": round(applied_p * farm_area, 2),
            "K": round(applied_k * farm_area, 2),
        },
        "farm_stcr_target_npk": {
            "N": round(bounded_stcr_target_n * farm_area, 2),
            "P": round(bounded_stcr_target_p * farm_area, 2),
            "K": round(bounded_stcr_target_k * farm_area, 2),
        },
        "stcr_target_npk": {
            "N": bounded_stcr_target_n,
            "P": bounded_stcr_target_p,
            "K": bounded_stcr_target_k,
        },
        "raw_stcr_80_npk": {
            "N": round(raw_80_n, 2),
            "P": round(raw_80_p, 2),
            "K": round(raw_80_k, 2),
        },
        "season_standard_npk": {
            "N": std_target_n,
            "P": std_target_p,
            "K": std_target_k,
        },
        "standard_dose": {
            "N_kg_per_ha": season_std["standard_n_kg_per_ha"],
            "P_kg_per_ha": season_std["standard_p_kg_per_ha"],
            "K_kg_per_ha": season_std["standard_k_kg_per_ha"],
            "N_kg_per_acre": round(season_std["standard_n_kg_per_ha"] / HA_TO_ACRE, 2),
            "P_kg_per_acre": round(season_std["standard_p_kg_per_ha"] / HA_TO_ACRE, 2),
            "K_kg_per_acre": round(season_std["standard_k_kg_per_ha"] / HA_TO_ACRE, 2),
        },
        # Backward-compatibility aliases
        "final_rdf_n_kg_per_acre": final_n,
        "final_rdf_p_kg_per_acre": final_p,
        "final_rdf_k_kg_per_acre": final_k,
    }


compute_farmer_rdf_full = compute_farmer_stcr_full


# =============================================================================
# Fertigation Schedule & Window Status
# =============================================================================

def get_farmer_schedule(farmer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates the DAP-based fertigation schedule for a farmer using the KVK Excel advisory.
    Adapts based on soil pH (acidic/alkaline) and crop season cutoff days.
    """
    from app.services import fertigation_schedule

    farmer_name = farmer.get("name") or farmer.get("farmer_name", "Farmer")
    phone = farmer.get("phone") or farmer.get("mobile", "")
    snapshot = farmer.get("initial_login_snapshot") or {}
    login_dap = snapshot.get("login_dap")
    area_acre = safe_float(farmer.get("area_acre") or farmer.get("area_acres") or farmer.get("area"), 1.0)
    if area_acre <= 0:
        area_acre = 1.0

    stcr_data = compute_farmer_stcr_full(farmer)
    to_be_applied_npk = stcr_data.get("to_be_applied")
    applied_data = stcr_data.get("applied_till_date") or {}
    app_sig = f"{applied_data.get('N', 0)}_{applied_data.get('P', 0)}_{applied_data.get('K', 0)}"

    cache_key = f"fertigation_{phone}_{farmer_name}_dap{login_dap}_area{area_acre}_app_{app_sig}"
    if cache_key in SCHEDULE_CACHE:
        return SCHEDULE_CACHE[cache_key]

    sp = farmer.get("soil_parameters") or {}
    ph_val = sp.get("pH") if sp.get("pH") is not None else farmer.get("pH")
    season_val = farmer.get("crop_season") or farmer.get("season")
    plantation_date = (
        farmer.get("transplanted")
        or farmer.get("transplanted_date")
        or farmer.get("plantation_date")
    )

    result = fertigation_schedule.get_fertigation_schedule(
        ph=ph_val,
        season=season_val,
        plantation_date=plantation_date,
        to_be_applied_npk=to_be_applied_npk,
        login_dap=login_dap,
        applied_till_date=applied_data,
        area_acre=area_acre,
    )

    # Attach summary metadata
    result["farmer_name"] = farmer_name
    result["area_acre"] = area_acre
    result["to_be_applied"] = to_be_applied_npk
    result["applied_till_date"] = applied_data
    result["login_dap"] = login_dap
    result["stcr_target_npk"] = stcr_data.get("stcr_target_npk")
    result["raw_stcr_80_npk"] = stcr_data.get("raw_stcr_80_npk")
    result["season_standard_npk"] = stcr_data.get("season_standard_npk")
    result["farm_to_be_applied"] = stcr_data.get("farm_to_be_applied")
    result["farm_applied_till_date"] = stcr_data.get("farm_applied_till_date")
    result["farm_stcr_target_npk"] = stcr_data.get("farm_stcr_target_npk")

    SCHEDULE_CACHE[cache_key] = result
    return result


def get_farmer_soil_insights(farmer: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates salinity/sodicity risk and composite soil health index for a farmer."""
    sp = farmer.get("soil_parameters") or {}
    flat_farmer = {**farmer, **sp}

    return {
        "farmer_name": farmer.get("name") or farmer.get("farmer_name"),
        "salinity_sodicity": advisory.assess_salinity_sodicity(flat_farmer),
        "soil_health_index": advisory.calculate_soil_health_index(flat_farmer),
    }


def check_farmer_window_and_approval(farmer: Dict[str, Any], today: datetime = None) -> Dict[str, Any]:
    """
    Evaluates whether the farmer has an active fertigation window today
    and checks cycle completion status.
    """
    today = today or datetime.now()
    schedule = get_farmer_schedule(farmer)
    cycle_status = schedule.get("cycle_status")
    cur_period = schedule.get("current_period")
    current_dap = schedule.get("current_dap")
    app_groups = schedule.get("application_groups", [])

    has_upcoming_doses = False
    if cycle_status == "in_progress":
        if current_dap is not None:
            has_upcoming_doses = any(g.get("dap", 0) >= current_dap for g in app_groups)
        else:
            has_upcoming_doses = len(app_groups) > 0

    window_active_today = False
    if cur_period:
        sd = cur_period.get("start_date")
        ed = cur_period.get("end_date")
        if sd and ed:
            try:
                s_dt = datetime.strptime(sd, "%d-%b-%Y").date()
                e_dt = datetime.strptime(ed, "%d-%b-%Y").date()
                window_active_today = (s_dt <= today.date() <= e_dt)
            except Exception:
                window_active_today = True
        else:
            window_active_today = True

    return {
        "window_active_today": window_active_today,
        "has_upcoming_doses": has_upcoming_doses,
        "is_pending": False,
        "is_approved": True,
        "admin_approved": True,
        "current_period": cur_period,
        "cycle_status": cycle_status,
        "schedule": schedule,
    }
