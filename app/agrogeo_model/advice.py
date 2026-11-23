"""
AgroGeo Advice and Soil Health Scoring Module
=============================================
This module interprets the output of coupled biophysical models (water chemistry,
daily nitrogen/water dynamics, and multi-decadal carbon projections) and turns
them into:
  1. A composite Soil Health Score (0 to 100) across 5 weighted categories.
  2. Prioritized, plain-language advisory recommendations for the farmer.
"""
from __future__ import annotations
from typing import Any, Dict, List
from .crops import CROPS

# Conversion factor: 1 hectare = 2.47105 acres, or 1 acre = 0.4047 hectares
ACRE_PER_HA = 0.4047


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a numerical value between lower and upper bounds."""
    return max(lower, min(upper, value))


# =============================================================================
# 1. COMPOSITE SOIL HEALTH SCORING (0 to 100)
# =============================================================================

def scores(farmer_params: Dict[str, Any], sim_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a composite 100-point Soil Health Score based on 5 agronomic pillars:
      - Soil Organic Carbon Level  (25 pts max): Normalized against 1.5% target
      - Soil Acidity / pH Range   (20 pts max): Optimal between 6.0 and 7.6
      - Salinity & Sodicity Risk  (25 pts max): Governed by Maas-Hoffman & ESP
      - Nitrogen Management Loss  (15 pts max): Ratio of leaching/volatilization to inputs
      - 30-Year Carbon Trend      (15 pts max): Projected change in carbon stock
    """
    crop = CROPS[farmer_params["crop"]]
    sim = sim_results["sim"]

    # 1. Soil Organic Carbon Score (25 points)
    # Reaches full score at 1.5% SOC
    soc_score = _clamp(farmer_params["soc_pct"] / 1.5)

    # 2. Soil pH Balance Score (20 points)
    # Optimum pH is between 6.0 and 7.6; penalizes both acidic and overly alkaline ranges
    ph = farmer_params["ph"]
    if 6.0 <= ph <= 7.6:
        ph_score = 1.0
    elif ph < 6.0:
        ph_score = _clamp(1.0 - (6.0 - ph) / 1.5)
    else:
        ph_score = _clamp(1.0 - (ph - 7.6) / 1.4)

    # 3. Salinity and Sodicity Score (25 points)
    # Minimum of the crop salinity tolerance factor and soil sodicity (ESP) factor
    salinity_factor = sim["sal_factor"]
    esp = sim_results["gypsum"]["esp"]
    sodicity_factor = 1.0 if esp is None else _clamp(1.0 - (esp - 6.0) / 14.0)
    salsod_score = min(salinity_factor, sodicity_factor)

    # 4. Nitrogen Efficiency Score (15 points)
    # Penalizes lost nitrogen (leaching + N2O + N2 + NH3 volatilization) relative to total inputs
    totals = sim["totals"]
    total_inputs = totals["fert"] + totals["manure"] + totals["min_soil"] + totals["min_res"]
    total_losses = totals["leach"] + sim["n2o_total"] + totals["n2"] + totals["nh3"]
    n_score = _clamp(1.0 - (total_losses / total_inputs) / 0.5) if total_inputs > 0 else 0.5

    # 5. Carbon Trend Score (15 points)
    # Evaluates projected 30-year organic carbon trajectory under current practice
    current_scenario = sim_results["carbon"]["scenarios"]["current"]
    start_stock = sim_results["carbon"]["start_stock"]
    c_score = _clamp(0.5 + current_scenario["change"] / (0.3 * start_stock)) if start_stock > 0 else 0.5

    # Weighted aggregate total score (0 to 100)
    total_score = (
        25.0 * soc_score +
        20.0 * ph_score +
        25.0 * salsod_score +
        15.0 * n_score +
        15.0 * c_score
    )

    # Qualitative label & badge tone
    if total_score >= 75.0:
        label, tone = "Good", "ok"
    elif total_score >= 50.0:
        label, tone = "Fair", "warn"
    else:
        label, tone = "Needs attention", "bad"

    return {
        "total": round(total_score),
        "overall_index": round(total_score),
        "label": label,
        "tone": tone,
        "parts": [
            ("Soil carbon", soc_score, 25),
            ("Acidity (pH)", ph_score, 20),
            ("Salt & sodium", salsod_score, 25),
            ("Nitrogen losses", n_score, 15),
            ("Carbon trend", c_score, 15),
        ],
    }


# =============================================================================
# 2. PLAIN-LANGUAGE ADVISORY BUILDER
# =============================================================================

def build(farmer_params: Dict[str, Any], sim_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate actionable, ranked agricultural recommendations by evaluating:
      - Water chemistry risks (SAR, RSC, Salinity, Charge Balance)
      - Soil physical & chemical constraints (EC, pH, ESP gypsum requirement, SOC)
      - Nitrogen fertilizer utilization (leaching, volatilization, over/under fertilization)
      - Long-term carbon sequestration trajectory
    """
    items: List[Dict[str, Any]] = []

    def add_recommendation(level: str, area: str, title: str, text: str) -> None:
        items.append({"level": level, "area": area, "title": title, "text": text})

    crop = CROPS[farmer_params["crop"]]
    sim = sim_results["sim"]
    totals = sim["totals"]
    water = sim_results["water"]
    gypsum = sim_results["gypsum"]

    # -------------------------------------------------------------------------
    # A. Water Chemistry Hazards (SAR, RSC, EC)
    # -------------------------------------------------------------------------
    if water:
        sodicity_class, _ = water["sodicity_class"]

        # High sodium hazard (SAR >= 10 or soil ESP >= 10)
        if water["SAR"] >= 10 or (gypsum.get("esp") or 0) >= 10:
            kg_ha = gypsum.get("water_kg_ha")
            dose = gypsum.get("water_dose")
            extra_msg = (
                f" That is about {kg_ha:,.0f} kg gypsum per hectare "
                f"({kg_ha * ACRE_PER_HA:,.0f} kg per acre) over the {sim['irrigation_mm']:.0f} mm "
                f"of irrigation this crop gets, which brings the water SAR down to about {gypsum['water_sar_after']:.1f}."
            ) if kg_ha else ""

            add_recommendation(
                level="high",
                area="Water chemistry",
                title="Irrigation water will build up sodium in your soil",
                text=(
                    f"Water SAR is {water['SAR']:.1f} (sodium hazard {sodicity_class}); long-term this points to about "
                    f"{water['ESP']:.0f}% exchangeable sodium, which makes soil crusty and slow to drain. "
                    f"Adding gypsum so the water carries about {dose:g} mmol/L extra calcium fixes this.{extra_msg}"
                ),
            )
        elif water["SAR"] >= 6:
            add_recommendation(
                level="medium",
                area="Water chemistry",
                title="Sodium in water is moderate",
                text=(
                    f"Water SAR is {water['SAR']:.1f}. Safe for now on most soils, but watch infiltration and consider "
                    f"occasional gypsum on heavy soils."
                ),
            )

        # Residual Sodium Carbonate (RSC) hazard
        if water["RSC"] > 1.25:
            severity = "medium" if water["RSC"] <= 2.5 else "high"
            add_recommendation(
                level=severity,
                area="Water chemistry",
                title="Water is rich in bicarbonate",
                text=(
                    f"Residual sodium carbonate is {water['RSC']:.1f} meq/L ({water['rsc_class'][0]}). As the soil dries, calcium "
                    f"drops out as lime and the sodium share rises. The model shows soil-solution SAR rising from "
                    f"{water['concentration'][0]['SAR']:.1f} to {water['concentration'][3]['SAR']:.1f} when water is concentrated 4×. "
                    f"Gypsum or organic matter helps."
                ),
            )

        # Water Electrical Conductivity (EC) Salinity Hazard
        salinity_class, salinity_tone = water["salinity_class"]
        if salinity_tone != "ok":
            ec_threshold = crop["ec_thr"]
            denominator = 5 * ec_threshold - water["EC"]
            leaching_req = (
                f"To keep the root zone below your crop's tolerance, add roughly {100 * water['EC'] / denominator:.0f}% extra water as a leaching fraction."
                if denominator > 0
                else "This water is too salty for this crop to be grown safely with it."
            )
            add_recommendation(
                level="medium" if salinity_tone == "warn" else "high",
                area="Water chemistry",
                title="Irrigation water is salty",
                text=f"Water EC is about {water['EC']:.2f} dS/m (salinity hazard {salinity_class}). {leaching_req}",
            )

        # Cation-Anion Charge Balance Error Check
        if abs(water["cbe"]) > 10:
            add_recommendation(
                level="info",
                area="Water chemistry",
                title="Check the lab report",
                text=(
                    f"The cations and anions differ by {water['cbe']:.0f}%, which usually means a typing slip or a missing ion. "
                    f"Re-check the numbers before relying on the chemistry results."
                ),
            )

    # -------------------------------------------------------------------------
    # B. Soil Health Constraints (Salinity, pH, ESP, Carbon)
    # -------------------------------------------------------------------------
    ece_used = sim_results["ece_used"]
    if sim["sal_factor"] < 0.97:
        severity = "high" if sim["sal_factor"] < 0.85 else "medium"
        assumed_suffix = " (assumed from your water)" if sim_results["ece_assumed"] else ""
        add_recommendation(
            level=severity,
            area="Soil",
            title="Soil salinity is cutting yield",
            text=(
                f"Soil EC of {ece_used:g} dS/m{assumed_suffix} is above what "
                f"{crop['label'].split(' (')[0].lower()} tolerates ({crop['ec_thr']:g} dS/m). Expect about "
                f"{100 * (1 - sim['sal_factor']):.0f}% yield loss. Good drainage plus extra irrigation to flush salts helps."
            ),
        )

    # Soil pH Extremes
    if farmer_params["ph"] < 5.5:
        add_recommendation(
            level="high",
            area="Soil",
            title="Soil is too acidic",
            text=(
                f"pH {farmer_params['ph']:.1f} locks up phosphorus and harms roots. Apply agricultural lime "
                f"based on a lime-requirement test and recheck pH after a season."
            ),
        )
    elif farmer_params["ph"] > 8.5:
        add_recommendation(
            level="high",
            area="Soil",
            title="Soil is strongly alkaline",
            text=(
                f"pH {farmer_params['ph']:.1f} usually means sodium is high. Gypsum, compost and "
                f"good drainage are the usual fixes; micronutrients (zinc, iron) may need foliar sprays."
            ),
        )

    # Soil ESP Reclamation Requirement
    if gypsum.get("esp") is not None and gypsum["esp"] > 10 and gypsum.get("soil_kg_ha"):
        severity = "high" if gypsum["esp"] > 15 else "medium"
        soil_kg = gypsum["soil_kg_ha"]
        add_recommendation(
            level=severity,
            area="Soil",
            title="Soil sodium (ESP) needs gypsum",
            text=(
                f"Exchangeable sodium is {gypsum['esp']:.0f}% ({gypsum['esp_source']}). A reclamation dose to bring it to 8% in the top 15 cm is "
                f"about {soil_kg:,.0f} kg gypsum/ha ({soil_kg * ACRE_PER_HA:,.0f} kg per acre), using an estimated CEC of {gypsum['cec']:.0f} cmolc/kg."
            ),
        )

    # Low Soil Organic Carbon
    if farmer_params["soc_pct"] < 0.5:
        add_recommendation(
            level="medium",
            area="Soil",
            title="Very low soil carbon",
            text=f"{farmer_params['soc_pct']:.2f}% organic carbon is low; water and nutrient holding will be poor.",
        )

    # -------------------------------------------------------------------------
    # C. Nitrogen Management & Losses
    # -------------------------------------------------------------------------
    planned_n = sim_results["plan_total_n"]
    fertilizer_total = totals["fert"] + totals["manure"]
    fallow_leaching = totals["leach"] - sim["leach_season"]

    if fallow_leaching > 20:
        add_recommendation(
            level="medium",
            area="Nitrogen",
            title="Nitrate is lost after harvest",
            text=(
                f"About {fallow_leaching:.0f} kg N/ha washes out of the root zone outside the crop season, mostly in the rains. "
                f"Sow a quick catch crop or green manure, keep residues on the field, and avoid heavy late top-dressing."
            ),
        )

    if fertilizer_total > 0 and sim["leach_season"] > 0.2 * fertilizer_total:
        add_recommendation(
            level="high",
            area="Nitrogen",
            title="Fertiliser is washing out during the season",
            text=(
                f"{sim['leach_season']:.0f} kg N/ha ({100 * sim['leach_season'] / fertilizer_total:.0f}% of applied N) leaches while the crop grows. "
                f"Split the N into smaller doses and avoid applying before heavy rain or irrigation."
            ),
        )

    if fertilizer_total > 0 and totals["nh3"] > 0.12 * fertilizer_total:
        add_recommendation(
            level="medium",
            area="Nitrogen",
            title="Urea is losing ammonia to the air",
            text=(
                f"The model loses about {totals['nh3']:.0f} kg N/ha as ammonia gas (pH {farmer_params['ph']:.1f}, warm weather). Irrigate right after "
                f"top-dressing, mix urea into the soil, or use a coated urea."
            ),
        )

    # Optimal N Rate Comparison
    recommended_n = sim_results["n_rec"]
    if planned_n > recommended_n + 30:
        rates_grid = {step["rate"]: step for step in sim_results["n_rates"]}
        opt_yield = rates_grid.get(recommended_n, {}).get("yield", 0.0)
        add_recommendation(
            level="medium",
            area="Nitrogen",
            title="You may be over-fertilising",
            text=(
                f"Yield levels off at about {recommended_n} kg N/ha ({opt_yield:.1f} t/ha). Your plan applies {planned_n:.0f} kg N/ha. "
                f"The extra N mostly adds losses, not yield. Treat this as an indication from an uncalibrated model and compare it with your local agriculture university's recommendation."
            ),
        )
    elif planned_n < recommended_n - 30:
        add_recommendation(
            level="medium",
            area="Nitrogen",
            title="Nitrogen may be limiting yield",
            text=(
                f"The model reaches near-full yield at about {recommended_n} kg N/ha; your plan gives {planned_n:.0f} kg N/ha. "
                f"Nitrogen supply is the main yield limit ({100 * (1 - sim['n_index']):.0f}% short)."
            ),
        )

    # Split Application Timing Check
    timing_scenarios = sim_results.get("timing", {})
    if (
        "split3" in timing_scenarios
        and timing_scenarios["split3"]["losses"] < timing_scenarios["plan"]["losses"] - 8
        and timing_scenarios["split3"]["yield"] >= 0.98 * timing_scenarios["plan"]["yield"]
    ):
        diff_loss = timing_scenarios["plan"]["losses"] - timing_scenarios["split3"]["losses"]
        add_recommendation(
            level="info",
            area="Nitrogen",
            title="Splitting N into three equal doses would cut losses",
            text=(
                f"Same total N in three equal doses (sowing, +30 d, +60 d) would lose {diff_loss:.0f} kg N/ha less "
                f"with similar yield."
            ),
        )

    # Water Stress Check
    if sim["water_index"] < 0.85:
        severity = "high" if sim["water_index"] < 0.7 else "medium"
        irrigation_advice = (
            "Add irrigation at critical stages (flowering, grain fill) or use mulch."
            if farmer_params["irrigation"] != "irrigated"
            else "Irrigate more often or check how much water is actually applied."
        )
        add_recommendation(
            level=severity,
            area="Water",
            title="Crop is water-stressed",
            text=f"The crop gets only {100 * sim['water_index']:.0f}% of the water it could use. {irrigation_advice}",
        )

    # -------------------------------------------------------------------------
    # D. Long-term Soil Carbon Trajectory
    # -------------------------------------------------------------------------
    carbon_current = sim_results["carbon"]["scenarios"]["current"]
    carbon_improved = sim_results["carbon"]["scenarios"]["improved"]

    if carbon_current["change"] < -1.5:
        severity = "high" if carbon_current["change"] < -4 else "medium"
        add_recommendation(
            level=severity,
            area="Carbon",
            title="Soil carbon is declining",
            text=(
                f"At current practice the top 30 cm loses about {-carbon_current['change']:.1f} t C/ha over 30 years "
                f"(to {carbon_current['pct_end']:.2f}% SOC). Keeping all residues, adding 10 t/ha compost and reducing tillage would instead "
                f"gain {carbon_improved['change']:.1f} t C/ha."
            ),
        )
    elif carbon_improved["change"] - carbon_current["change"] > 2:
        diff_gain = carbon_improved["change"] - carbon_current["change"]
        add_recommendation(
            level="info",
            area="Carbon",
            title="Room to build soil carbon",
            text=f"Residue retention, compost and no-till could add {diff_gain:.1f} t C/ha more than your current practice over 30 years.",
        )

    if farmer_params["residue_pct"] < 30:
        add_recommendation(
            level="medium",
            area="Carbon",
            title="Residues are being removed or burnt",
            text="Returning even half of the straw feeds soil life and cuts nitrate loss because it ties up N as it decomposes.",
        )

    # Default fallback when no issues are discovered
    if not items:
        add_recommendation(
            level="good",
            area="Overview",
            title="No major problems found",
            text="The models found nothing alarming for this field. Keep monitoring soil tests each year.",
        )

    # Sort items by severity priority: high -> medium -> info -> good
    priority_order = {"high": 0, "medium": 1, "info": 2, "good": 3}
    items.sort(key=lambda x: priority_order.get(x["level"], 99))
    return items
