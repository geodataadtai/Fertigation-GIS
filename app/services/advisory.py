"""
Soil Insights & Agronomic Advisory Service
==========================================
Computes soil health classifications, salinity/sodicity risk, and population
statistical outliers based purely on standard soil-test report values:
- pH (Soil acidity / alkalinity)
- EC (Electrical Conductivity, dS/m - salinity)
- OC (Organic Carbon, %)
- Primary Nutrients: Available N, P, K (kg/ha)
- Sodium (Na) and Free Calcium Carbonate (CaCO3, %)
"""

import statistics
from typing import Dict, Any, List, Optional, Tuple

# =============================================================================
# 1. Salinity & Sodicity Risk Evaluation
# =============================================================================

# Electrical Conductivity (EC in dS/m) classification thresholds for sugarcane
EC_NORMAL_MAX = 1.0           # <= 1.0: Optimum for sugarcane root zone
EC_SLIGHTLY_SALINE_MAX = 2.0  # 1.0 - 2.0: Mild salinity (yield begins slight drop)
EC_SALINE_MAX = 4.0           # 2.0 - 4.0: Saline (significant yield limitation)
                              # > 4.0: Strongly saline (severe osmotic stress)

# Coarse threshold for elevated exchangeable Sodium (Na)
NA_ELEVATED_THRESHOLD = 0.50

# Free Calcium Carbonate (CaCO3 %) threshold - locks up phosphorus & micronutrients
CACO3_CALCAREOUS_THRESHOLD = 10.0         # Calcareous soil
CACO3_HIGHLY_CALCAREOUS_THRESHOLD = 15.0  # Highly calcareous soil


def assess_salinity_sodicity(farmer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates soil salinity and sodicity risk from EC, Na, and CaCO3 values.

    Note:
    A definitive sodicity diagnosis requires an Exchangeable Sodium Percentage (ESP)
    and Cation Exchange Capacity (CEC) laboratory test. The Na and EC thresholds here
    act as an early-warning screening tool for extension officers.

    Args:
        farmer: Dictionary containing soil test fields (EC, Na, CaCO3).

    Returns:
        dict: Status flags for EC, Na, CaCO3, and composite sodicity risk.
    """
    ec = farmer.get("EC")
    na = farmer.get("Na")
    caco3 = farmer.get("CaCO3")

    # 1. Electrical Conductivity (Salinity)
    if ec is None:
        ec_status = "unknown"
    elif ec <= EC_NORMAL_MAX:
        ec_status = "normal"
    elif ec <= EC_SLIGHTLY_SALINE_MAX:
        ec_status = "slightly_saline"
    elif ec <= EC_SALINE_MAX:
        ec_status = "saline"
    else:
        ec_status = "strongly_saline"

    # 2. Exchangeable Sodium
    if na is None:
        na_status = "unknown"
    elif na > NA_ELEVATED_THRESHOLD:
        na_status = "elevated"
    else:
        na_status = "normal"

    # 3. Calcium Carbonate (Lime content)
    if caco3 is None:
        caco3_status = "unknown"
    elif caco3 >= CACO3_HIGHLY_CALCAREOUS_THRESHOLD:
        caco3_status = "highly_calcareous"
    elif caco3 >= CACO3_CALCAREOUS_THRESHOLD:
        caco3_status = "calcareous"
    else:
        caco3_status = "normal"

    # Sodicity risk flag: High sodium combined with saline conditions
    sodicity_risk = (na_status == "elevated") and (ec_status in ("saline", "strongly_saline"))

    return {
        "ec": ec,
        "ec_status": ec_status,
        "na": na,
        "na_status": na_status,
        "caco3": caco3,
        "caco3_status": caco3_status,
        "sodicity_risk_flag": sodicity_risk,
        "gypsum_recommended": sodicity_risk,
    }


# =============================================================================
# 2. Composite Soil Health Index (Score: 0 to 100)
# =============================================================================

SOIL_HEALTH_WEIGHTS: Dict[str, float] = {
    "OC": 0.25,  # Organic Carbon gets highest weight (soil biology & structure)
    "pH": 0.15,  # Soil chemical balance
    "EC": 0.15,  # Osmotic & salt balance
    "N":  0.15,  # Available nitrogen
    "P":  0.15,  # Available phosphorus
    "K":  0.15,  # Available potassium
}


def _band_score(
    value: Optional[float],
    low_optimal: float,
    high_optimal: float,
    hard_min: float,
    hard_max: float
) -> Optional[float]:
    """
    Calculates a trapezoidal scoring curve (0 to 100):
    - Values within [low_optimal, high_optimal] score 100.
    - Values below low_optimal fall off linearly to 0 at hard_min.
    - Values above high_optimal fall off linearly to 0 at hard_max.
    - Clamped within [0, 100]. Returns None if value is None.
    """
    if value is None:
        return None

    if low_optimal <= value <= high_optimal:
        return 100.0

    if value < low_optimal:
        if value <= hard_min:
            return 0.0
        return round(100.0 * (value - hard_min) / (low_optimal - hard_min), 1)

    if value >= hard_max:
        return 0.0

    return round(100.0 * (hard_max - value) / (hard_max - high_optimal), 1)


def calculate_soil_health_index(farmer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates an overall Soil Health Index (0-100) by combining component scores.

    Ideal Agronomic Bands for Sugarcane:
    - pH: 6.5 to 7.5 (hard range: 4.0 - 9.5)
    - EC: 0.0 to 1.0 dS/m (hard range: -1.0 - 4.0)
    - OC: 0.75% to 1.50% (hard range: 0.0% - 2.5%)
    - N : 80 to 160 kg/ha (hard range: 0 - 280)
    - P : 15 to 30 kg/ha (hard range: 0 - 60)
    - K : 150 to 300 kg/ha (hard range: 0 - 500)
    """
    component_scores = {
        "pH": _band_score(farmer.get("pH"), 6.5, 7.5, 4.0, 9.5),
        "EC": _band_score(farmer.get("EC"), 0.0, 1.0, -1.0, 4.0),
        "OC": _band_score(farmer.get("OC"), 0.75, 1.50, 0.0, 2.5),
        "N":  _band_score(farmer.get("N"), 80, 160, 0, 280),
        "P":  _band_score(farmer.get("P"), 15, 30, 0, 60),
        "K":  _band_score(farmer.get("K"), 150, 300, 0, 500),
    }

    # Only include parameters that are actually reported
    available = {k: v for k, v in component_scores.items() if v is not None}
    if not available:
        return {"score": None, "rating": "unknown", "component_scores": component_scores}

    total_weight = sum(SOIL_HEALTH_WEIGHTS[k] for k in available)
    weighted_sum = sum(SOIL_HEALTH_WEIGHTS[k] * v for k, v in available.items())
    final_score = round(weighted_sum / total_weight, 1)

    # Qualitative Rating
    if final_score >= 75:
        rating = "good"
    elif final_score >= 50:
        rating = "moderate"
    else:
        rating = "poor"

    return {
        "score": final_score,
        "rating": rating,
        "component_scores": component_scores,
    }


# =============================================================================
# 3. Cross-Farmer Outlier Detection (Admin & Extension Officer Tool)
# =============================================================================

OUTLIER_FIELDS = ["pH", "EC", "OC", "N", "P", "K"]
Z_SCORE_THRESHOLD = 2.0  # Statistical outlier cutoff (|z| >= 2.0 standard deviations)


def find_outlier_farmers(farmers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Identifies farmers with statistical outlier soil parameters compared to the
    regional population. Helps agricultural officers prioritize targeted field visits.

    Args:
        farmers: List of farmer dictionaries containing soil parameters.

    Returns:
        list: Farmers with detected anomalies, sorted by severity.
    """
    # 1. Collect all valid numeric values per field
    field_data: Dict[str, List[float]] = {field: [] for field in OUTLIER_FIELDS}
    for farmer in farmers:
        for field in OUTLIER_FIELDS:
            val = farmer.get(field)
            if val is not None:
                field_data[field].append(float(val))

    # 2. Compute population mean and standard deviation
    population_stats: Dict[str, Tuple[float, float]] = {}
    for field, values in field_data.items():
        if len(values) >= 2:
            mean = statistics.mean(values)
            stdev = statistics.pstdev(values)
            population_stats[field] = (mean, stdev)

    # 3. Calculate z-scores and flag outliers
    flagged_farmers = []
    for farmer in farmers:
        flags = []
        for field in OUTLIER_FIELDS:
            val = farmer.get(field)
            if val is None or field not in population_stats:
                continue

            mean, stdev = population_stats[field]
            if stdev == 0:
                continue

            z_score = (val - mean) / stdev
            if abs(z_score) >= Z_SCORE_THRESHOLD:
                flags.append({
                    "field": field,
                    "value": val,
                    "population_mean": round(mean, 2),
                    "z_score": round(z_score, 2),
                    "direction": "high" if z_score > 0 else "low",
                })

        if flags:
            flagged_farmers.append({
                "farmer_name": farmer.get("farmer_name") or farmer.get("name"),
                "sr_no": farmer.get("sr_no"),
                "flags": sorted(flags, key=lambda f: abs(f["z_score"]), reverse=True),
            })

    # Sort farmers so the ones with the most extreme deviations appear first
    flagged_farmers.sort(
        key=lambda entry: max(abs(f["z_score"]) for f in entry["flags"]),
        reverse=True,
    )
    return flagged_farmers
