"""
FertiNex Unified Bio-Geochemical Model Runner
=============================================
Coordinates the single unified FertiNex Bio-Geochemical Simulation Engine
for field-level agro-ecological analysis:

1. Hydrochemistry & Soil Salinity:
   Evaluates irrigation water quality, Sodium Adsorption Ratio (SAR), Residual Sodium
   Carbonate (RSC), aqueous chemical speciation, and calculates gypsum requirement.
2. Agro-Ecosystem Nitrogen & Water Balance:
   Simulates daily soil-water-crop dynamics, crop yield, nitrate leaching, and greenhouse
   gas emissions (N2O, NH3, N2).
3. Soil Organic Carbon Dynamics:
   Projects long-term soil organic matter (SOM) fractions, active/slow carbon pools,
   and 30-year carbon sequestration potential based on residue retention and tillage practices.
"""

from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache
from typing import Dict, Any, List

from . import advice
from .fertinex_biogeo import fertinex_biogeo_engine, FertinexBioGeoEngine
from . import weather as wx
from .crops import CROPS, FERT_TYPES

# Standard nitrogen test rates (kg N/ha) for generating the yield response curve
N_RATES = [0, 30, 60, 90, 120, 150, 180, 210, 240]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_MODEL_KEYS = (
    "climate", "latitude", "soc_pct", "ph", "bulk_density", "ece", "esp", "crop",
    "sowing_date", "irrigation", "tillage", "residue_pct", "fym_t_ha", "water", "fert_plan"
)


# =============================================================================
# Public Entry Point
# =============================================================================

def run_field(field: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public entry point for running field-level biochemical simulations.
    Caches results using an LRU cache based on model-relevant input parameters.
    """
    cache_key = json.dumps({k: field.get(k) for k in _MODEL_KEYS}, sort_keys=True, default=str)
    return _run_cached(cache_key)


# =============================================================================
# Helper Functions
# =============================================================================

def _sim(field_data: Dict[str, Any], soil: Dict[str, Any], plan: List[Dict[str, Any]], year_type: str = "normal"):
    """Invokes the FertiNex soil-nitrogen-water simulation engine."""
    return fertinex_biogeo_engine.simulate_soil_nitrogen_water(
        soil=soil,
        crop_key=field_data["crop"],
        sow_date=dt.date.fromisoformat(field_data["sowing_date"]),
        irrigated=(field_data["irrigation"] == "irrigated"),
        fert_plan=plan,
        fym_t_ha=field_data["fym_t_ha"],
        residue_retention=field_data["residue_pct"] / 100.0,
        ece=field_data["ece_used"],
        preset=field_data["climate"],
        lat=field_data.get("latitude"),
        year_type=year_type,
    )


def _summary(sim_result: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregates crop yield, nitrogen losses, and water metrics into a compact summary."""
    totals = sim_result["totals"]
    total_n_losses = totals["leach"] + sim_result["n2o_total"] + totals["n2"] + totals["nh3"]

    return {
        "yield": sim_result["yield_t_ha"],
        "leach": totals["leach"],
        "leach_season": sim_result["leach_season"],
        "n2o": sim_result["n2o_total"],
        "n2": totals["n2"],
        "nh3": totals["nh3"],
        "losses": total_n_losses,
        "uptake": totals["uptake"],
        "irr": totals["irr"],
        "fert": totals["fert"] + totals["manure"],
    }


# =============================================================================
# Cached Core Simulation
# =============================================================================

@lru_cache(maxsize=256)
def _run_cached(key: str) -> Dict[str, Any]:
    """Internal cached runner for the combined biochemical simulation pipeline."""
    field_data = json.loads(key)
    crop_info = CROPS[field_data["crop"]]
    soil_params = {
        "soc_pct": field_data.get("soc_pct", 0.6),
        "ph": field_data.get("ph", 7.2),
        "bulk_density": field_data.get("bulk_density", 1.4),
    }

    out: Dict[str, Any] = {"crop_label": crop_info["label"]}

    # -------------------------------------------------------------------------
    # 1. Irrigation Water Chemistry (FertiNex Hydrochemistry)
    # -------------------------------------------------------------------------
    water_analysis = None
    water_test = field_data.get("water")

    if water_test:
        milligrams = {k: water_test[k] for k in ("Ca", "Mg", "Na", "K", "Cl", "SO4", "HCO3")}
        water_analysis = fertinex_biogeo_engine.analyze_water(milligrams, water_test["pH"], water_test["temp"])
        water_analysis["concentration"] = fertinex_biogeo_engine.concentration_path(milligrams, water_test["temp"])
        water_analysis["gypsum"] = fertinex_biogeo_engine.gypsum_response(milligrams, water_test["temp"])

    out["water"] = water_analysis

    # Infer soil salinity (ECe) from water EC if soil measurement is absent
    field_data["ece_assumed"] = field_data.get("ece") is None
    if field_data["ece_assumed"]:
        field_data["ece_used"] = round(1.5 * water_analysis["EC"], 2) if water_analysis else 1.0
    else:
        field_data["ece_used"] = field_data["ece"]

    out["ece_used"] = field_data["ece_used"]
    out["ece_assumed"] = field_data["ece_assumed"]

    # -------------------------------------------------------------------------
    # 2. Nitrogen Dynamics & Crop Yield (FertiNex Soil-Crop Engine)
    # -------------------------------------------------------------------------
    fert_plan = [item for item in field_data["fert_plan"] if item["n_kg"] > 0]
    base_sim = _sim(field_data, soil_params, fert_plan)

    out["sim"] = base_sim
    out["plan_total_n"] = sum(item["n_kg"] for item in fert_plan)
    out["fert_n_total"] = base_sim["totals"]["fert"] + base_sim["totals"]["manure"]

    # Generate N response curve across multiple rates
    default_shape = fert_plan if fert_plan else fertinex_biogeo_engine.default_fert_plan(120)
    shape_total = sum(item["n_kg"] for item in default_shape)
    rate_runs = []

    for rate in N_RATES:
        scaled_plan = fertinex_biogeo_engine.scale_plan(default_shape, rate / shape_total)
        run_sim = _sim(field_data, soil_params, scaled_plan)
        run_summary = _summary(run_sim)
        run_summary["rate"] = rate
        rate_runs.append(run_summary)

    zero_n_run = rate_runs[0]
    for summary in rate_runs:
        if summary["rate"]:
            summary["recovery"] = (summary["uptake"] - zero_n_run["uptake"]) / summary["rate"]
            summary["agron_eff"] = (summary["yield"] - zero_n_run["yield"]) * 1000.0 / summary["rate"]
        else:
            summary["recovery"] = None
            summary["agron_eff"] = None

    # Economic/Agronomic recommended N: rate achieving >= 98% of peak grid yield
    max_yield_in_grid = max(s["yield"] for s in rate_runs)
    n_recommendation = next(s["rate"] for s in rate_runs if s["yield"] >= 0.98 * max_yield_in_grid)
    out["n_rates"] = rate_runs
    out["n_rec"] = n_recommendation

    # Multi-year weather variability scenarios (dry, normal, wet)
    year_scenarios = {}
    for year_type in ("dry", "normal", "wet"):
        year_scenarios[year_type] = (
            _summary(base_sim) if year_type == "normal"
            else _summary(_sim(field_data, soil_params, fert_plan, year_type))
        )
    out["year_types"] = year_scenarios

    # Fertilizer application timing comparison (single basal vs 3-split vs current plan)
    timing_comparisons = {"plan": _summary(base_sim)}
    if out["plan_total_n"] > 0:
        timing_comparisons["split3"] = _summary(_sim(field_data, soil_params, fertinex_biogeo_engine.split_plan(fert_plan, "split3")))
        timing_comparisons["basal"] = _summary(_sim(field_data, soil_params, fertinex_biogeo_engine.split_plan(fert_plan, "basal")))
    out["timing"] = timing_comparisons

    # Monthly time-series aggregations for charts
    start_date = dt.date.fromisoformat(base_sim["start"])
    month_indices = []
    month_labels = []

    series_data = base_sim["series"]
    for day_idx in range(len(series_data["leach"])):
        current_date = start_date + dt.timedelta(days=day_idx)
        month_tag = (current_date.year, current_date.month)

        if not month_indices or month_indices[-1][0] != month_tag:
            month_indices.append([month_tag, []])
            month_labels.append(MONTHS[current_date.month - 1])
        month_indices[-1][1].append(day_idx)

    out["monthly"] = {
        "labels": month_labels,
        "leach": [sum(series_data["leach"][idx] for idx in days) for _, days in month_indices],
        "n2o":   [sum(series_data["n2o"][idx] for idx in days) for _, days in month_indices],
        "n2":    [sum(series_data["n2"][idx] for idx in days) for _, days in month_indices],
        "nh3":   [sum(series_data["nh3"][idx] for idx in days) for _, days in month_indices],
        "rain":  [sum(series_data["rain"][idx] for idx in days) for _, days in month_indices],
        "irr":   [sum(series_data["irr"][idx] for idx in days) for _, days in month_indices],
    }

    irrigation_calendar = [0.0] * 12
    for day_idx, vol in enumerate(series_data["irr"]):
        m_idx = (start_date + dt.timedelta(days=day_idx)).month - 1
        irrigation_calendar[m_idx] += vol

    month_ticks = []
    for day_idx in range(len(series_data["rain"])):
        d = start_date + dt.timedelta(days=day_idx)
        if d.day == 1:
            month_ticks.append((day_idx, MONTHS[d.month - 1]))
    out["month_ticks"] = month_ticks

    # -------------------------------------------------------------------------
    # 3. Soil Organic Carbon Modeling (FertiNex Multi-Pool Carbon Engine)
    # -------------------------------------------------------------------------
    climate_normals = wx.monthly_normals(field_data["climate"], field_data.get("latitude"))
    carbon_projection = fertinex_biogeo_engine.project_carbon_dynamics(
        soil=soil_params,
        clim=climate_normals,
        crop_key=field_data["crop"],
        sow_date=dt.date.fromisoformat(field_data["sowing_date"]),
        yield_t=base_sim["yield_t_ha"],
        current={
            "retention": field_data["residue_pct"] / 100.0,
            "fym_t_ha": field_data["fym_t_ha"],
            "tillage": field_data["tillage"],
        },
        irr_month=irrigation_calendar,
    )
    out["carbon"] = carbon_projection

    # -------------------------------------------------------------------------
    # 4. Gypsum Soil Reclamation Requirements
    # -------------------------------------------------------------------------
    # Estimate Cation Exchange Capacity (CEC) from soil texture and organic matter
    estimated_cec = 16.0 + 3.0 * field_data["soc_pct"]
    current_esp = field_data.get("esp")
    if current_esp is None and water_analysis:
        current_esp = water_analysis["ESP"]

    gyp_info = {
        "cec": estimated_cec,
        "esp": current_esp,
        "esp_source": "soil test" if field_data.get("esp") is not None else "estimated from water SAR",
    }

    if current_esp is not None:
        gyp_info["soil_kg_ha"] = fertinex_biogeo_engine.soil_gypsum_requirement(
            esp_initial=current_esp,
            esp_target=8.0,
            cec=estimated_cec,
            depth_m=0.15,
            bulk_density=field_data["bulk_density"],
        )

    if water_analysis:
        # Find minimum gypsum dose to lower SAR <= 6 and RSC <= 1.25 meq/L
        rec_water_gypsum = next(
            (g for g in water_analysis["gypsum"] if g["SAR"] <= 6 and g["RSC"] <= 1.25),
            water_analysis["gypsum"][-1],
        )
        gyp_info["water_dose"] = rec_water_gypsum["dose"]
        gyp_info["water_sar_after"] = rec_water_gypsum["SAR"]
        total_irr_mm = base_sim["irrigation_mm"]
        gyp_info["water_kg_ha"] = (
            fertinex_biogeo_engine.gypsum_kg_per_ha(rec_water_gypsum["dose"], total_irr_mm)
            if total_irr_mm > 0 else None
        )

    out["gypsum"] = gyp_info

    # -------------------------------------------------------------------------
    # 5. Composite Agronomic Scores & Expert Advice
    # -------------------------------------------------------------------------
    out["scores"] = advice.scores(field_data, out)
    out["advice"] = advice.build(field_data, out)
    out["field_echo"] = {
        "fert_products": [
            {
                "day": item["day"],
                "label": FERT_TYPES[item["type"]]["label"],
                "n_kg": item["n_kg"],
                "product_kg_ha": item["n_kg"] / (FERT_TYPES[item["type"]]["n_pct"] / 100.0),
            }
            for item in fert_plan
        ]
    }

    # -------------------------------------------------------------------------
    # 6. Scientific & Peer-Reviewed Grounding Metadata
    # -------------------------------------------------------------------------
    out["model_architecture"] = "FertiNex-BioGeo Unified Soil Bio-Geochemical Dynamics Engine"
    out["scientific_references"] = fertinex_biogeo_engine.get_scientific_references()

    return out
