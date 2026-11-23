"""Crop and fertiliser parameter tables (indicative, editable)."""
from __future__ import annotations

# n_uptake: total crop N uptake at potential yield (kg N/ha)
# fix: fraction of crop N supplied by biological fixation (legumes)
# gn: fraction of crop N removed in the harvested product
# straw / root: dry-matter residue and root ratio per tonne of harvested product
# ec_thr / ec_slope: Maas-Hoffman salinity tolerance (ECe dS/m, % yield loss per dS/m)
CROPS = {
    "wheat": dict(label="Wheat", sow="11-15", days=120, ymax=4.8, n_uptake=110, fix=0.0, gn=0.75,
                  flooded=False, kc_mid=1.15, straw=1.3, root=0.35, cn=80, ec_thr=6.0, ec_slope=7.1),
    "rice": dict(label="Rice (puddled, flooded)", sow="07-05", days=115, ymax=5.5, n_uptake=95, fix=0.0, gn=0.65,
                 flooded=True, kc_mid=1.20, straw=1.2, root=0.30, cn=70, ec_thr=3.0, ec_slope=12.0),
    "maize": dict(label="Maize", sow="06-25", days=110, ymax=6.5, n_uptake=160, fix=0.0, gn=0.65,
                  flooded=False, kc_mid=1.20, straw=1.2, root=0.40, cn=60, ec_thr=1.7, ec_slope=12.0),
    "soybean": dict(label="Soybean", sow="06-25", days=100, ymax=2.6, n_uptake=190, fix=0.5, gn=0.80,
                    flooded=False, kc_mid=1.10, straw=1.5, root=0.40, cn=30, ec_thr=5.0, ec_slope=20.0),
    "cotton": dict(label="Cotton (seed cotton)", sow="06-10", days=180, ymax=2.4, n_uptake=130, fix=0.0, gn=0.60,
                   flooded=False, kc_mid=1.15, straw=3.0, root=0.50, cn=60, ec_thr=7.7, ec_slope=5.2),
    "sugarcane": dict(label="Sugarcane (cane yield)", sow="02-01", days=330, ymax=85.0, n_uptake=220, fix=0.0, gn=0.55,
                      flooded=False, kc_mid=1.15, straw=0.20, root=0.12, cn=90, ec_thr=1.7, ec_slope=5.9),
}

# n_pct: % N in the product; nh4/no3/urea: how the N is split on application
FERT_TYPES = {
    "urea": dict(label="Urea (46% N)", n_pct=46, nh4=0.0, no3=0.0, urea=1.0),
    "dap": dict(label="DAP 18-46-0 (18% N)", n_pct=18, nh4=1.0, no3=0.0, urea=0.0),
    "can": dict(label="CAN / ammonium nitrate (26% N)", n_pct=26, nh4=0.5, no3=0.5, urea=0.0),
    "ammsulf": dict(label="Ammonium sulphate (21% N)", n_pct=21, nh4=1.0, no3=0.0, urea=0.0),
    "complex": dict(label="Complex NPK 12-32-16 (12% N)", n_pct=12, nh4=1.0, no3=0.0, urea=0.0),
    "nitrate": dict(label="Nitrate fertiliser (15% N)", n_pct=15, nh4=0.0, no3=1.0, urea=0.0),
}

# Farmyard manure / compost (fresh weight basis)
FYM_N_KG_PER_T = 5.0      # kg N per tonne fresh
FYM_NH4_FRACTION = 0.20   # share of manure N that is ammoniacal on application
FYM_C_T_PER_T = 0.10      # t C per tonne fresh
FYM_CN = 20.0


def kc_curve(frac: float, kc_mid: float) -> float:
    """Trapezoidal crop coefficient over crop-season fraction 0..1."""
    if frac < 0.2:
        return 0.4
    if frac < 0.45:
        return 0.4 + (kc_mid - 0.4) * (frac - 0.2) / 0.25
    if frac < 0.75:
        return kc_mid
    return kc_mid + (0.5 - kc_mid) * (frac - 0.75) / 0.25


def n_demand_fraction(day: int, total_days: int) -> float:
    """Cumulative fraction of N demand reached at crop day ``day`` (logistic)."""
    if day <= 0:
        return 0.0
    if day >= total_days:
        return 1.0
    import math

    k = 8.0 / total_days
    mid = 0.45 * total_days

    def lg(t):
        return 1.0 / (1.0 + math.exp(-k * (t - mid)))

    return (lg(day) - lg(0)) / (lg(total_days) - lg(0))
