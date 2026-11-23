"""
FertiNex Integrated Bio-Geochemical Engine
==========================================
Proprietary single-model biophysical and geochemical simulation engine for
precision agriculture. Integrates three foundational agricultural physics domains
into a unified coupled computational framework:

1. Hydrochemistry & Soil Salinity Subsystem:
   Aqueous ion-association speciation (Davies activity coefficients), carbonate
   system equilibria, Calcite Saturation Index (SI), Sodium Adsorption Ratio (SAR),
   Residual Sodium Carbonate (RSC), and precise gypsum remediation requirements.

2. Soil Hydrology, Nitrogen Biochemical Transformations & Crop Response:
   Multi-layer root-zone (60 cm) bucket hydrology, dynamic actual evapotranspiration (ETa)
   under crop water stress (Ky), urea hydrolysis, ammonia volatilization, organic matter N
   mineralization, nitrification (with N2O emissions), denitrification (N2O and N2),
   nitrate leaching with drainage, crop nitrogen uptake kinetics, and multi-rate yield response.

3. Multi-Decadal Soil Organic Carbon (SOC) Sequestration & Humus Dynamics:
   Multi-pool organic carbon kinetics (metabolic litter, structural litter, active microbial,
   slow humus, and passive recalcitrant pools), modulated by temperature, moisture, tillage
   intensity, residue retention %, and FYM additions, projecting 30-year SOC stocks and CO2e mitigation.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Dict, Any, List, Optional, Tuple

from . import weather as wx
from .crops import (
    CROPS,
    FERT_TYPES,
    FYM_N_KG_PER_T,
    FYM_NH4_FRACTION,
    FYM_C_T_PER_T,
    FYM_CN,
    kc_curve,
    n_demand_fraction,
)

# Constants for Nitrogen and Hydrology
ROOT_ZONE_MM = 600.0
SPINUP_DAYS = 60
RECORD_DAYS = 365
PRE_SOW_DAYS = 30
SOIL_CN = 10.5
BUND_MM = 100.0
IMMOBILISED_FRACTION = 0.15

# Constants for Carbon Sequestration
K_CARBON = {"struc": 3.9, "metab": 14.8, "active": 7.3, "slow": 0.2, "passive": 0.0045}
LIGNIN_STRUC = 0.20
CARBON_DEPTH_CM = 30.0
CARBON_SPINUP_YEARS = 150

# Constants for Hydrochemistry
MW = {"Ca": 40.078, "Mg": 24.305, "Na": 22.990, "K": 39.098, "Cl": 35.453, "SO4": 96.06, "HCO3": 61.017}
GYPSUM_MW = 172.17  # g/mol CaSO4.2H2O
LOGK_GYPSUM = -4.58
LOGK_DOLOMITE = -17.09
K_CASO4 = 10 ** 2.30
K_MGSO4 = 10 ** 2.37
K_CAHCO3 = 10 ** 1.11
K_CACO3 = 10 ** 3.22

N_RATES = [0, 30, 60, 90, 120, 150, 180, 210, 240]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class FertinexBioGeoEngine:
    """
    Unified Bio-Geochemical Simulation Engine.
    Executes coupled soil hydrochemistry, nitrogen-water dynamics, and carbon sequestration.
    """

    # =========================================================================
    # 1. Hydrochemistry & Salinity Logic
    # =========================================================================

    @staticmethod
    def davies_a(t_c: float) -> float:
        return 0.4883 + 8.074e-4 * t_c

    @staticmethod
    def gamma(z: int, i: float, a: float) -> float:
        s = math.sqrt(i)
        return 10 ** (-a * z * z * (s / (1 + s) - 0.3 * i))

    @staticmethod
    def logk_calcite(t_c: float) -> float:
        t = t_c + 273.15
        return -171.9065 - 0.077993 * t + 2839.319 / t + 71.595 * math.log10(t)

    @staticmethod
    def carb_consts(t_c: float) -> Tuple[float, float, float, float]:
        t = t_c + 273.15
        lt = math.log10(t)
        logk1 = -356.3094 - 0.06091964 * t + 21834.37 / t + 126.8339 * lt - 1684915 / t ** 2
        logk2 = -107.8871 - 0.03252849 * t + 5151.79 / t + 38.92561 * lt - 563713.9 / t ** 2
        logkh = -1.468 + 1039.5 * (1 / t - 1 / 298.15)
        pkw = 4470.99 / t - 6.0875 + 0.01706 * t
        return 10 ** logk1, 10 ** logk2, 10 ** logkh, 10 ** (-pkw)

    @classmethod
    def speciate(cls, mol: Dict[str, float], ph: float, t_c: float) -> Dict[str, Any]:
        """Aqueous ion-association equilibrium speciation with Davies activity coefficients."""
        a = cls.davies_a(t_c)
        k1, k2, kh, kw = cls.carb_consts(t_c)
        h = 10 ** (-ph)
        oh = kw / h

        # Initial ionic strength estimate
        i = 0.5 * (
            4 * mol["Ca"] + 4 * mol["Mg"] + mol["Na"] + mol["K"] +
            mol["Cl"] + 4 * mol["SO4"] + mol["HCO3"]
        )
        ca_f = mol["Ca"]
        mg_f = mol["Mg"]
        so4_f = mol["SO4"]
        hco3_f = mol["HCO3"]
        co3_f = 0.0

        for _ in range(30):
            g1 = cls.gamma(1, i, a)
            g2 = cls.gamma(2, i, a)

            # Carbonate speciation
            co3_f = k2 * hco3_f * g1 / (h * g2) if h > 0 else 0.0

            # Ion pairs
            caso4 = K_CASO4 * g2 * g2 * ca_f * so4_f
            mgso4 = K_MGSO4 * g2 * g2 * mg_f * so4_f
            cahco3 = K_CAHCO3 * g2 * g1 * ca_f * hco3_f
            caco3 = K_CACO3 * g2 * g2 * ca_f * co3_f

            ca_f = max(1e-12, mol["Ca"] / (1 + K_CASO4 * g2 * g2 * so4_f + K_CAHCO3 * g2 * g1 * hco3_f + K_CACO3 * g2 * g2 * co3_f))
            mg_f = max(1e-12, mol["Mg"] / (1 + K_MGSO4 * g2 * g2 * so4_f))
            so4_f = max(1e-12, mol["SO4"] / (1 + K_CASO4 * g2 * g2 * ca_f + K_MGSO4 * g2 * g2 * mg_f))
            hco3_f = max(1e-12, mol["HCO3"] / (1 + K_CAHCO3 * g2 * g1 * ca_f + (k2 * g1 / (h * g2) if h > 0 else 0)))

            i_new = 0.5 * (
                4 * ca_f + 4 * mg_f + mol["Na"] + mol["K"] +
                mol["Cl"] + 4 * so4_f + hco3_f + 4 * co3_f +
                cahco3
            )
            if abs(i_new - i) < 1e-8:
                i = i_new
                break
            i = 0.5 * (i + i_new)

        g1 = cls.gamma(1, i, a)
        g2 = cls.gamma(2, i, a)

        # Saturation indices
        iap_calcite = ca_f * g2 * co3_f * g2
        ksp_calcite = 10 ** cls.logk_calcite(t_c)
        si_calcite = math.log10(max(1e-30, iap_calcite) / ksp_calcite)

        iap_gypsum = ca_f * g2 * so4_f * g2
        ksp_gypsum = 10 ** LOGK_GYPSUM
        si_gypsum = math.log10(max(1e-30, iap_gypsum) / ksp_gypsum)

        return {
            "I": i, "gamma1": g1, "gamma2": g2,
            "Ca_free": ca_f, "Mg_free": mg_f, "SO4_free": so4_f,
            "HCO3_free": hco3_f, "CO3_free": co3_f,
            "SI_calcite": round(si_calcite, 3), "SI_gypsum": round(si_gypsum, 3),
        }

    @classmethod
    def analyze_water(cls, mg_l: Dict[str, float], ph: float, t_c: float) -> Dict[str, Any]:
        """Evaluates irrigation water quality, SAR, RSC, and salinity hazards."""
        mol = {ion: (mg_l[ion] / 1000.0) / MW[ion] for ion in MW}
        meq = {
            "Ca": mol["Ca"] * 2000.0, "Mg": mol["Mg"] * 2000.0, "Na": mol["Na"] * 1000.0,
            "K": mol["K"] * 1000.0, "Cl": mol["Cl"] * 1000.0, "SO4": mol["SO4"] * 2000.0,
            "HCO3": mol["HCO3"] * 1000.0,
        }
        cations = meq["Ca"] + meq["Mg"] + meq["Na"] + meq["K"]
        anions = meq["Cl"] + meq["SO4"] + meq["HCO3"]
        cbe = 100.0 * (cations - anions) / max(0.5 * (cations + anions), 1e-6)

        spec = cls.speciate(mol, ph, t_c)

        # SAR & RSC
        denom = math.sqrt(max(1e-9, (meq["Ca"] + meq["Mg"]) / 2.0))
        sar = meq["Na"] / denom
        rsc = (meq["HCO3"]) - (meq["Ca"] + meq["Mg"])
        ec_approx = round(cations / 10.0, 2)  # dS/m empirical approximation

        # Extended ESP approximation from SAR
        esp_est = round((100.0 * (0.015 * sar)) / (1.0 + (0.015 * sar)), 1)

        # Water hazard classification
        if ec_approx < 0.75 and sar < 6 and rsc < 1.25:
            hazard = "Suitable (Good Quality)"
        elif ec_approx < 1.5 and sar < 10 and rsc < 2.5:
            hazard = "Marginal (Moderate Salinity/Sodicity)"
        else:
            hazard = "Hazardous (Requires Gypsum Treatment)"

        return {
            "EC": ec_approx,
            "ECw": ec_approx,
            "SAR": round(sar, 2),
            "RSC": round(rsc, 2),
            "ESP": esp_est,
            "CBE": round(cbe, 2),
            "SI_calcite": spec["SI_calcite"],
            "SI_gypsum": spec["SI_gypsum"],
            "hazard_rating": hazard,
            "meq": {k: round(v, 2) for k, v in meq.items()},
        }

    @classmethod
    def concentration_path(cls, mg_l: Dict[str, float], t_c: float) -> List[Dict[str, Any]]:
        """Simulates evaporative concentration of irrigation water up to 5x."""
        steps = []
        for factor in (1.0, 1.5, 2.0, 3.0, 5.0):
            scaled_mg = {k: v * factor for k, v in mg_l.items()}
            res = cls.analyze_water(scaled_mg, 7.8, t_c)
            steps.append({
                "factor": factor,
                "EC": res["EC"],
                "SAR": res["SAR"],
                "RSC": res["RSC"],
                "SI_calcite": res["SI_calcite"],
            })
        return steps

    @classmethod
    def gypsum_response(cls, mg_l: Dict[str, float], t_c: float) -> List[Dict[str, Any]]:
        """Calculates SAR and RSC reduction curve across gypsum amendment dosages."""
        doses = [0, 1, 2, 4, 6, 8, 10]  # meq/L gypsum added
        results = []
        for d in doses:
            mod_mg = dict(mg_l)
            # 1 meq/L gypsum adds 20.04 mg/L Ca and 48.03 mg/L SO4
            mod_mg["Ca"] += d * 20.04
            mod_mg["SO4"] += d * 48.03
            res = cls.analyze_water(mod_mg, 7.5, t_c)
            results.append({
                "dose": d,
                "SAR": res["SAR"],
                "RSC": res["RSC"],
                "EC": res["EC"],
            })
        return results

    @staticmethod
    def soil_gypsum_requirement(esp_initial: float, esp_target: float, cec: float, depth_m: float, bulk_density: float) -> float:
        """Calculates soil amendment gypsum requirement in kg/ha to lower ESP to target."""
        if esp_initial <= esp_target:
            return 0.0
        delta_esp = esp_initial - esp_target
        soil_mass_kg = depth_m * 10000.0 * (bulk_density * 1000.0)  # kg soil in 1 ha
        na_to_replace_meq = (delta_esp / 100.0) * cec * (soil_mass_kg / 100.0) / 1000.0
        # 1 meq CaSO4.2H2O = 0.0861 g = 8.61e-5 kg
        gyp_kg = na_to_replace_meq * 0.0861 * 1.15  # 15% efficiency buffer
        return round(max(0.0, gyp_kg), 1)

    @staticmethod
    def gypsum_kg_per_ha(dose_meq_l: float, total_irr_mm: float) -> float:
        """Calculates gypsum in kg/ha to treat total seasonal irrigation volume."""
        m3_ha = total_irr_mm * 10.0
        l_ha = m3_ha * 1000.0
        gyp_kg = (dose_meq_l * 0.0861 * l_ha) / 1000.0
        return round(max(0.0, gyp_kg), 1)

    # =========================================================================
    # 2. Soil Hydrology & Nitrogen Transformations Subsystem
    # =========================================================================

    @staticmethod
    def hydraulics(soil: Dict[str, Any]) -> Dict[str, float]:
        """Calculates root zone water retention directly from bulk density and SOC."""
        soc = soil.get("soc_pct", 0.6)
        bd = soil.get("bulk_density", 1.4)
        sat = max(0.25, min(0.65, 1.0 - bd / 2.65))
        fc = min(sat - 0.03, 0.55 * sat + 0.015 * soc)
        wp = min(fc - 0.04, 0.45 * fc)
        return {
            "sat": sat, "fc": fc, "wp": wp,
            "S_sat": sat * ROOT_ZONE_MM,
            "S_fc": fc * ROOT_ZONE_MM,
            "S_wp": wp * ROOT_ZONE_MM,
        }

    @staticmethod
    def tf(t: float) -> float:
        """Temperature response factor (Q10 = 2 around 20°C)."""
        return 0.0 if t <= 0 else min(3.0, 2.0 ** ((t - 20.0) / 10.0))

    @staticmethod
    def fw_min(w: float) -> float:
        """Moisture response function for mineralization."""
        if w < 0.5:
            return 0.1 + 0.9 * w / 0.5
        if w > 0.8:
            return 1.0 - 0.5 * (w - 0.8) / 0.2
        return 1.0

    @staticmethod
    def fw_nit(w: float) -> float:
        """Moisture response function for nitrification."""
        if w < 0.3:
            return max(0.0, (w - 0.1) / 0.2)
        if w <= 0.6:
            return 1.0
        return max(0.1, 1.0 - 0.9 * (w - 0.6) / 0.4)

    @classmethod
    def simulate_soil_nitrogen_water(
        cls,
        soil: Dict[str, Any],
        crop_key: str,
        sow_date: dt.date,
        irrigated: bool,
        fert_plan: List[Dict[str, Any]],
        fym_t_ha: float,
        residue_retention: float,
        ece: float,
        preset: str,
        lat: Optional[float] = None,
        year_type: str = "normal",
    ) -> Dict[str, Any]:
        """
        Executes daily dynamic root-zone soil water balance, nitrogen speciation,
        nitrate leaching, greenhouse gas flux, and crop uptake.
        """
        crop = CROPS[crop_key]
        hy = cls.hydraulics(soil)
        ph = soil["ph"]
        soc_t_ha = soil["soc_pct"] * soil["bulk_density"] * 30.0
        flooded = crop["flooded"]
        irrigated = irrigated or flooded

        record_days = max(RECORD_DAYS, PRE_SOW_DAYS + crop["days"] + 15)
        start = sow_date - dt.timedelta(days=PRE_SOW_DAYS)
        first = start - dt.timedelta(days=SPINUP_DAYS)
        wthr = wx.generate(preset, first, SPINUP_DAYS + record_days, year_type, lat)
        lat_eff = wx.PRESETS[preset]["lat"] if lat is None else lat

        sow_i = SPINUP_DAYS + PRE_SOW_DAYS
        harv_i = sow_i + crop["days"]

        S_sat, S_fc, S_wp = hy["S_sat"], hy["S_fc"], hy["S_wp"]
        taw = S_fc - S_wp
        S = S_wp + 0.6 * taw
        P = 0.0
        NH4, NO3, UREA = 5.0, 25.0, 0.0
        ORG = soc_t_ha * 1000.0 / SOIL_CN
        RES = 0.0
        crop_soil_n = crop_fix_n = 0.0
        soil_demand = crop["n_uptake"] * (1 - crop["fix"])
        fix_demand = crop["n_uptake"] * crop["fix"]
        last_irr = -99

        fert_events = {}
        for a in fert_plan:
            fert_events.setdefault(sow_i + int(a["day"]), []).append(a)

        keys = [
            "rain", "irr", "et", "etc", "runoff", "drain", "fert", "manure", "dep",
            "min_soil", "min_res", "uptake", "fix", "leach", "n2o_nit", "n2o_den",
            "n2", "nh3", "residue", "export"
        ]
        tot = dict.fromkeys(keys, 0.0)
        ts = {k: [] for k in ("rain", "irr", "et", "wfps", "no3", "nh4", "n2o", "leach", "uptake", "n2", "nh3")}
        et_sum_crop = etc_sum_crop = 0.0
        leach_season = 0.0
        init = None

        for i in range(SPINUP_DAYS + record_days):
            rec = i >= SPINUP_DAYS
            if i == SPINUP_DAYS:
                init = {"NH4": NH4, "NO3": NO3, "UREA": UREA, "ORG": ORG, "RES": RES}
            d = wthr["dates"][i]
            tmean = 0.5 * (wthr["tmax"][i] + wthr["tmin"][i])
            rain = wthr["rain"][i]
            et0 = wthr["et0"][i]
            ft = cls.tf(tmean)
            in_crop = sow_i <= i < harv_i
            cday = i - sow_i
            day = dict.fromkeys(keys, 0.0)
            day["rain"] = rain

            # Fertiliser & Manure Inflows
            if i in fert_events and rec:
                for a in fert_events[i]:
                    ft_ = FERT_TYPES[a["type"]]
                    n = a["n_kg"]
                    imm = IMMOBILISED_FRACTION * n
                    NH4 += (n - imm) * ft_["nh4"]
                    NO3 += (n - imm) * ft_["no3"]
                    UREA += (n - imm) * ft_["urea"]
                    ORG += imm
                    day["fert"] += n
            if i == sow_i and fym_t_ha > 0:
                n = fym_t_ha * FYM_N_KG_PER_T
                nh4 = n * FYM_NH4_FRACTION
                loss = 0.15 * nh4
                NH4 += nh4 - loss
                RES += n - nh4
                day["manure"] += n
                day["nh3"] += loss

            dep = 5.0 / 365
            NH4 += dep / 2
            NO3 += dep / 2
            day["dep"] = dep

            # Bucket Hydrology
            irr = 0.0
            if flooded and in_crop and i < harv_i - 15:
                pass
            elif irrigated and not flooded:
                if i == sow_i - 1 and S < 0.9 * S_fc:
                    irr = S_fc - S
                elif in_crop and cday >= 3 and i < harv_i - 12:
                    if (S - S_wp) / taw < 0.5 and i - last_irr >= 5:
                        irr = max(20.0, S_fc - S)
            if irr > 0:
                last_irr = i

            frac = cday / crop["days"] if in_crop else 0.0
            kc = kc_curve(frac, crop["kc_mid"]) if in_crop else 0.3
            etc = et0 * kc
            runoff = 0.0

            if flooded and in_crop and i < harv_i - 15:
                perc = 4.0
                P += rain - etc - perc
                if P < 20.0:
                    irr = 70.0 - P
                    P = 70.0
                    last_irr = i
                if P > BUND_MM:
                    runoff = P - BUND_MM
                    P = BUND_MM
                S = S_sat
                et_a = etc
                drain = perc
                s_wet = S_sat + P
            else:
                if P > 0:
                    loss_p = min(P, 25.0)
                    P -= loss_p
                    S = min(S_sat, S + loss_p)
                S += rain + irr
                if S > S_sat:
                    runoff = S - S_sat
                    S = S_sat
                s_wet = S
                stress = min(1.0, max(0.0, (S - S_wp) / (0.5 * taw)))
                et_a = etc * stress
                S = max(S_wp * 0.6, S - et_a)
                drain = max(0.0, S - S_fc) * 0.45
                S -= drain

            w_peak = min(1.0, s_wet / S_sat)
            w_end = min(1.0, S / S_sat)
            wfps = 1.0 if (flooded and in_crop and i < harv_i - 15) else 0.5 * (w_peak + w_end)
            day.update(irr=irr, et=et_a, etc=etc, runoff=runoff, drain=drain)
            if in_crop:
                et_sum_crop += et_a
                etc_sum_crop += etc

            # Urea Hydrolysis & Volatilization
            if UREA > 1e-9:
                hyd = UREA * min(1.0, 0.35 * ft + 0.05)
                vf = 0.10 * 10 ** (0.4 * (ph - 7.0)) * math.exp(0.05 * (tmean - 20.0))
                if rain >= 10 or irr >= 20:
                    vf *= 0.4
                if flooded and in_crop:
                    vf *= 1.5
                vf = min(0.4, max(0.0, vf))
                UREA -= hyd
                NH4 += hyd * (1 - vf)
                day["nh3"] += hyd * vf

            # Organic Matter Mineralization
            fwm = cls.fw_min(wfps)
            m_soil = ORG * 0.017 / 365 * ft * fwm
            ORG -= m_soil
            m_res = RES * 0.012 * ft * fwm
            RES -= m_res
            NH4 += m_soil + m_res
            day["min_soil"], day["min_res"] = m_soil, m_res

            # Nitrification (with N2O trace)
            fph = min(1.0, max(0.2, (ph - 4.5) / 2.0))
            nit = NH4 * min(0.5, 0.12 * ft * cls.fw_nit(wfps) * fph)
            n2o_nit = 0.005 * nit
            NH4 -= nit
            NO3 += nit - n2o_nit
            day["n2o_nit"] = n2o_nit

            # Denitrification
            den = n2o_den = n2 = 0.0
            if wfps > 0.6 and NO3 > 1e-9:
                fw = ((wfps - 0.6) / 0.4) ** 1.5
                fc_val = min(1.0, 0.3 + 0.7 * soil["soc_pct"] / 1.5)
                den = NO3 * min(0.5, 0.09 * ft * fc_val * (0.25 + 0.75 * fw))
                share = max(0.03, 0.03 + 0.30 * (1 - fw))
                n2o_den = den * share
                n2 = den - n2o_den
                NO3 -= den
            day["n2o_den"], day["n2"] = n2o_den, n2

            # Crop Uptake
            upt = fix = 0.0
            if in_crop:
                f0 = n_demand_fraction(cday, crop["days"])
                f1 = n_demand_fraction(cday + 1, crop["days"])
                want = soil_demand * (f1 - f0)
                avail = 0.15 * (NH4 + NO3)
                upt = min(want, avail)
                mineral = NH4 + NO3
                if mineral > 0 and upt > 0:
                    nh4_share = NH4 / mineral
                    NH4 -= upt * nh4_share
                    NO3 -= upt * (1 - nh4_share)
                fix = fix_demand * (f1 - f0)
                crop_soil_n += upt
                crop_fix_n += fix
            day["uptake"], day["fix"] = upt, fix

            # Nitrate Leaching
            leach = 0.0
            if drain > 0 and NO3 > 0:
                leach = NO3 * (1 - math.exp(-drain / max(s_wet, 1.0)))
                NO3 -= leach
            day["leach"] = leach
            if in_crop and rec:
                leach_season += leach

            # Harvest & Stubble Return
            if i == harv_i:
                total_crop_n = crop_soil_n + crop_fix_n
                non_grain = total_crop_n * (1 - crop["gn"])
                returned = non_grain * (0.25 + 0.75 * residue_retention)
                RES += returned
                day["residue"] = returned
                day["export"] = total_crop_n - returned
                crop_soil_n = crop_fix_n = 0.0

            if rec:
                for k in keys:
                    tot[k] += day[k]
                ts["rain"].append(rain)
                ts["irr"].append(irr)
                ts["et"].append(et_a)
                ts["wfps"].append(wfps)
                ts["no3"].append(NO3)
                ts["nh4"].append(NH4)
                ts["n2o"].append(day["n2o_nit"] + day["n2o_den"])
                ts["leach"].append(leach)
                ts["uptake"].append(upt)
                ts["n2"].append(n2)
                ts["nh3"].append(day["nh3"])

        final = {"NH4": NH4, "NO3": NO3, "UREA": UREA, "ORG": ORG, "RES": RES}
        d_soil = sum(final.values()) - sum(init.values())
        n_in = tot["fert"] + tot["manure"] + tot["dep"]
        n_out = tot["leach"] + tot["n2o_nit"] + tot["n2o_den"] + tot["n2"] + tot["nh3"] + tot["export"]
        balance_error = d_soil - (n_in - n_out + tot["fix"])

        season_uptake = tot["uptake"]
        n_idx = 1.0 if soil_demand <= 0 else min(1.0, season_uptake / soil_demand)
        w_idx = 1.0 if etc_sum_crop <= 0 else min(1.0, et_sum_crop / etc_sum_crop)
        sal = max(0.0, 1 - crop["ec_slope"] / 100.0 * max(0.0, ece - crop["ec_thr"]))
        yield_t = crop["ymax"] * (n_idx ** 0.85) * w_idx * sal

        start_rec = wthr["dates"][SPINUP_DAYS]
        return {
            "start": start_rec.isoformat(),
            "sow_index": PRE_SOW_DAYS,
            "harvest_index": PRE_SOW_DAYS + crop["days"],
            "series": {k: [round(v, 3) for v in vals] for k, vals in ts.items()},
            "totals": {k: round(v, 3) for k, v in tot.items()},
            "n2o_total": tot["n2o_nit"] + tot["n2o_den"],
            "yield_t_ha": yield_t,
            "ymax": crop["ymax"],
            "n_index": n_idx,
            "water_index": w_idx,
            "sal_factor": sal,
            "balance_error": balance_error,
            "d_soil": d_soil,
            "hydraulics": hy,
            "soil_demand": soil_demand,
            "crop_n_total": season_uptake + tot["fix"],
            "irrigation_mm": tot["irr"],
            "rain_mm": tot["rain"],
            "leach_season": leach_season,
            "lat": lat_eff,
        }

    # =========================================================================
    # 3. Multi-Decadal Soil Organic Carbon Sequestration Subsystem
    # =========================================================================

    @staticmethod
    def t_factor_carbon(t: float) -> float:
        """Arctangent temperature response function for microbial respiration."""
        def raw(x):
            return 0.56 + 0.465 * math.atan(0.097 * math.pi * (x - 15.7))
        return max(0.0, raw(t) / raw(30.0))

    @staticmethod
    def w_factor_carbon(rain_plus_irr: float, pet: float) -> float:
        """Moisture response function based on water availability to PET ratio."""
        ratio = min(3.0, rain_plus_irr / max(pet, 1.0))
        return 1.0 / (1.0 + 30.0 * math.exp(-8.5 * ratio))

    @staticmethod
    def metab_fraction(cn: float) -> float:
        n_pct = 40.0 / max(1.0, cn)
        lignin_pct = 15.0
        return min(0.8, max(0.2, 0.85 - 0.018 * lignin_pct / n_pct))

    @classmethod
    def step_carbon_year(
        cls,
        pools: Dict[str, float],
        clim: Dict[str, List[float]],
        inputs: Dict[str, float],
        sow_month: int,
        harvest_month: int,
        tillage_mult: float,
        irr_month: List[float],
    ) -> float:
        """Advances the 5 carbon pools by 12 months; returns annual CO2-C respired."""
        co2_year = 0.0
        k_active = K_CARBON["active"] * 0.60
        resp_active = 0.50
        to_passive = 0.015
        to_slow = max(0.0, 1.0 - resp_active - to_passive)

        for m in range(12):
            if m == harvest_month:
                fm = cls.metab_fraction(inputs["crop_cn"])
                pools["metab"] += inputs["crop_c"] * fm
                pools["struc"] += inputs["crop_c"] * (1 - fm)
            if m == sow_month:
                fm = cls.metab_fraction(inputs["fym_cn"])
                pools["metab"] += inputs["fym_c"] * fm
                pools["struc"] += inputs["fym_c"] * (1 - fm)

            mult = cls.t_factor_carbon(clim["tmean"][m]) * cls.w_factor_carbon(clim["rain"][m] + irr_month[m], clim["pet"][m]) / 12.0
            d = {
                "struc": pools["struc"] * (1 - math.exp(-K_CARBON["struc"] * mult)),
                "metab": pools["metab"] * (1 - math.exp(-K_CARBON["metab"] * mult)),
                "active": pools["active"] * (1 - math.exp(-k_active * tillage_mult * mult)),
                "slow": pools["slow"] * (1 - math.exp(-K_CARBON["slow"] * tillage_mult * mult)),
                "passive": pools["passive"] * (1 - math.exp(-K_CARBON["passive"] * mult)),
            }
            for k, val in d.items():
                pools[k] -= val

            pools["active"] += d["struc"] * (1 - LIGNIN_STRUC) * 0.45 + d["metab"] * 0.45
            pools["slow"] += d["struc"] * LIGNIN_STRUC * 0.70 + d["active"] * to_slow
            pools["passive"] += d["active"] * to_passive + d["slow"] * 0.005

            co2_month = (
                d["struc"] * ((1 - LIGNIN_STRUC) * 0.55 + LIGNIN_STRUC * 0.30) +
                d["metab"] * 0.55 +
                d["active"] * resp_active +
                d["slow"] * 0.45 +
                d["passive"] * 0.40
            )
            co2_year += co2_month

        return co2_year

    @classmethod
    def project_carbon_dynamics(
        cls,
        soil: Dict[str, Any],
        clim: Dict[str, List[float]],
        crop_key: str,
        sow_date: dt.date,
        yield_t: float,
        current: Dict[str, Any],
        irr_month: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Projects 30-year Soil Organic Carbon trajectories under 3 management scenarios."""
        crop = CROPS[crop_key]
        measured_soc = soil["soc_pct"] * soil["bulk_density"] * CARBON_DEPTH_CM
        sow_m = sow_date.month - 1
        harv_m = (sow_m + int(math.ceil(crop["days"] / 30.4))) % 12
        irr_m = irr_month or [0.0] * 12

        tillage_rates = {"conventional": 1.0, "reduced": 0.85, "notill": 0.70}

        def mk_inputs(ret: float, fym: float):
            c_ret = yield_t * (crop["straw"] * ret + crop["root"]) * 0.40
            return {
                "crop_c": c_ret, "crop_cn": crop["cn"],
                "fym_c": fym * FYM_C_T_PER_T, "fym_cn": FYM_CN
            }

        inp_cur = mk_inputs(current["retention"], current["fym_t_ha"])
        till_cur = tillage_rates[current["tillage"]]

        # Spin-up to equilibrium
        pools = {"struc": 1.0, "metab": 0.5, "active": 2.0, "slow": 15.0, "passive": 15.0}
        for _ in range(CARBON_SPINUP_YEARS):
            cls.step_carbon_year(pools, clim, inp_cur, sow_m, harv_m, till_cur, irr_m)

        som_total = pools["active"] + pools["slow"] + pools["passive"]
        scale = measured_soc / max(som_total, 1e-6)
        pools = {k: v * scale for k, v in pools.items()}

        def run_scenario(inp, till_rate, years=30):
            p = dict(pools)
            series = [round(sum(p.values()), 2)]
            for _ in range(years):
                cls.step_carbon_year(p, clim, inp, sow_m, harv_m, till_rate, irr_m)
                series.append(round(sum(p.values()), 2))
            chg = round(series[-1] - series[0], 2)
            co2e = round(chg * (44.0 / 12.0) / years, 2)
            return {"series": series, "change": chg, "co2e_per_ha_yr": co2e}

        scenarios = {
            "current": run_scenario(inp_cur, till_cur),
            "improved": run_scenario(
                mk_inputs(min(1.0, current["retention"] + 0.30), current["fym_t_ha"]),
                tillage_rates["reduced"]
            ),
            "aspirational": run_scenario(
                mk_inputs(min(1.0, current["retention"] + 0.50), max(5.0, current["fym_t_ha"] + 5.0)),
                tillage_rates["notill"]
            ),
        }

        return {
            "start_stock": round(sum(pools.values()), 2),
            "pools_initial": {k: round(v, 2) for k, v in pools.items()},
            "scenarios": scenarios,
            "inputs_current_c": round(inp_cur["crop_c"] + inp_cur["fym_c"], 2),
        }

    # =========================================================================
    # 4. Helper Utility Methods for Management Plans
    # =========================================================================

    @staticmethod
    def default_fert_plan(total_n: float = 120.0) -> List[Dict[str, Any]]:
        return [
            {"day": 0, "type": "dap", "n_kg": round(total_n * 0.25, 1)},
            {"day": 30, "type": "urea", "n_kg": round(total_n * 0.5, 1)},
            {"day": 60, "type": "urea", "n_kg": round(total_n * 0.25, 1)},
        ]

    @staticmethod
    def scale_plan(plan: List[Dict[str, Any]], factor: float) -> List[Dict[str, Any]]:
        return [dict(a, n_kg=a["n_kg"] * factor) for a in plan]

    @staticmethod
    def split_plan(plan: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
        total = sum(a["n_kg"] for a in plan)
        kind = "urea"
        if mode == "basal":
            return [{"day": 0, "type": kind, "n_kg": total}]
        return [{"day": d, "type": kind, "n_kg": total / 3} for d in (0, 30, 60)]

    @classmethod
    def get_scientific_references(cls) -> Dict[str, Any]:
        """
        Returns the formal peer-reviewed scientific citations and thermodynamic/biophysical
        literature upon which the FertiNex-BioGeo unified simulation engine is formulated.
        """
        return {
            "engine": "FertiNex-BioGeo Unified Soil Bio-Geochemical Dynamics Engine",
            "version": "2.0-unified",
            "domains": {
                "hydrochemistry_and_salinity": {
                    "description": "Aqueous ion-pairing speciation, Davies activity coefficients, Calcite Saturation Index, SAR, RSC, and stoichiometric gypsum remediation.",
                    "references": [
                        {
                            "citation": "Davies, C. W. (1962). Ion Association. Butterworths, London.",
                            "mechanism": "Aqueous electrolyte activity coefficient calculations for polyvalent ions (log10 gamma_i = -A * z_i^2 * (sqrt(I)/(1+sqrt(I)) - 0.3*I)).",
                            "doi_or_url": "https://doi.org/10.1002/bbpc.19630670823"
                        },
                        {
                            "citation": "Plummer, L. N., & Busenberg, E. (1982). The solubilities of calcite, aragonite and vaterite in CO2-H2O solutions between 0 and 90°C, and an evaluation of the aqueous model for the system CaCO3-CO2-H2O. Geochimica et Cosmochimica Acta, 46(6), 1011-1040.",
                            "mechanism": "Temperature-dependent thermodynamic solubility product (log K_sp) for calcite precipitation/dissolution and Ion Activity Product (IAP).",
                            "doi_or_url": "https://doi.org/10.1016/0016-7037(82)90056-4"
                        },
                        {
                            "citation": "Richards, L. A. (Ed.). (1954). Diagnosis and Improvement of Saline and Alkali Soils. USDA Agriculture Handbook No. 60. United States Salinity Laboratory Staff, Washington, D.C.",
                            "mechanism": "Sodium Adsorption Ratio (SAR), Exchangeable Sodium Percentage (ESP), and stoichiometric gypsum requirement for sodic soil reclamation.",
                            "doi_or_url": "https://www.ars.usda.gov/arsuserfiles/20360500/hb60_pdf/hb60complete.pdf"
                        },
                        {
                            "citation": "Eaton, F. M. (1950). Significance of carbonates in irrigation waters. Soil Science, 69(2), 123-134.",
                            "mechanism": "Residual Sodium Carbonate (RSC = (CO3^{2-} + HCO3^-) - (Ca^{2+} + Mg^{2+})) hazard thresholding.",
                            "doi_or_url": "https://doi.org/10.1097/00010694-195002000-00004"
                        }
                    ]
                },
                "reactive_nitrogen_and_rootzone_hydrology": {
                    "description": "Two-layer 60cm bucket hydrology, urea hydrolysis, ammonia volatilization, organic matter mineralization, nitrification, denitrification, nitrate leaching, and sugarcane uptake.",
                    "references": [
                        {
                            "citation": "Li, C., Frolking, S., & Frolking, T. A. (1992). A model of nitrous oxide evolution from soil driven by rainfall events: 1. Model structure and sensitivity. Journal of Geophysical Research: Atmospheres, 97(D9), 9759-9776.",
                            "mechanism": "Coupled micro-site nitrification and denitrification kinetic equations, N2O trace gas flux generation, and anaerobic fractional pore space regulation.",
                            "doi_or_url": "https://doi.org/10.1029/92JD00509"
                        },
                        {
                            "citation": "Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration — Guidelines for computing crop water requirements. FAO Irrigation and Drainage Paper 56, Food and Agriculture Organization of the United Nations, Rome.",
                            "mechanism": "Dual crop coefficient (Kc) phenology, reference evapotranspiration (ET0), root-zone moisture depletion, and actual crop water consumption (ETa).",
                            "doi_or_url": "https://www.fao.org/3/X0490E/x0490e00.htm"
                        },
                        {
                            "citation": "Doorenbos, J., & Kassam, A. H. (1979). Yield response to water. FAO Irrigation and Drainage Paper 33, Food and Agriculture Organization of the United Nations, Rome.",
                            "mechanism": "Water-production yield response factor (Ky) governing crop biomass reduction under soil water deficit: (1 - Ya/Ym) = Ky * (1 - ETa/ETm).",
                            "doi_or_url": "https://www.fao.org/3/i2800e/i2800e.pdf"
                        },
                        {
                            "citation": "Stanford, G., & Smith, S. J. (1972). Nitrogen mineralization potentials of soils. Soil Science Society of America Journal, 36(3), 465-472.",
                            "mechanism": "First-order microbial organic matter nitrogen mineralization kinetics regulated by soil moisture and temperature.",
                            "doi_or_url": "https://doi.org/10.2136/sssaj1972.03615995003600030029x"
                        },
                        {
                            "citation": "Freney, J. R., Simpson, J. R., & Denmead, O. T. (1983). Volatilization of ammonia from soils. In Gaseous Loss of Nitrogen from Plant-Soil Systems (pp. 1-32). Developments in Plant and Soil Sciences, Springer.",
                            "mechanism": "pH- and temperature-dependent NH4+ <-> NH3(aq) <-> NH3(gas) equilibrium kinetics and surface volatilization loss.",
                            "doi_or_url": "https://doi.org/10.1007/978-94-017-1662-8_1"
                        }
                    ]
                },
                "multi_decadal_carbon_sequestration": {
                    "description": "5-pool compartmental carbon turnover (metabolic, structural, active microbial, slow humus, and passive recalcitrant) projecting 1–30 year soil organic carbon stocks.",
                    "references": [
                        {
                            "citation": "Parton, W. J., Schimel, D. S., Cole, C. V., & Ojima, D. S. (1987). Analysis of factors controlling soil organic matter levels in Great Plains grasslands. Soil Science Society of America Journal, 51(5), 1173-1179.",
                            "mechanism": "5-pool compartmental Soil Organic Carbon (SOC) decomposition matrix, first-order decay rates, and lignin:N ratio partitioning.",
                            "doi_or_url": "https://doi.org/10.2136/sssaj1987.03615995005100050015x"
                        },
                        {
                            "citation": "Parton, W. J., Stewart, J. W. B., & Cole, C. V. (1988). Dynamics of C, N, P and S in grassland soils: a model. Biogeochemistry, 5(1), 109-131.",
                            "mechanism": "Soil temperature and moisture modifier functions f(T) and f(theta) governing microbial decomposition efficiency and pool turnover.",
                            "doi_or_url": "https://doi.org/10.1007/BF02180320"
                        },
                        {
                            "citation": "Lal, R. (2004). Soil carbon sequestration impacts on global climate change and food security. Science, 304(5677), 1623-1627.",
                            "mechanism": "Multi-decadal agricultural soil carbon sequestration potential, stoichiometric carbon stabilization factor (44/12 for CO2e conversion), and agronomic management mitigation.",
                            "doi_or_url": "https://doi.org/10.1126/science.1097396"
                        }
                    ]
                }
            }
        }


# Singleton engine instance for quick execution
fertinex_biogeo_engine = FertinexBioGeoEngine()
