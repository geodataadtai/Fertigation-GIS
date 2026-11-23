"""Synthetic daily weather from monthly climate normals.

The farmer picks a climate preset; the generator turns monthly normals into a
reproducible daily series (Markov-chain rainfall, AR(1) temperature noise) and
computes reference evapotranspiration with the Hargreaves equation.

The seed is fixed per year-type so that every what-if scenario for a field is
run against *the same* weather, which keeps scenario comparisons fair.
"""
from __future__ import annotations

import calendar
import datetime as dt
import math
import random

# Monthly normals (Jan..Dec). Indicative values, not station data.
PRESETS = {
    "deccan": {
        "label": "Semi-arid monsoon (Deccan plateau, e.g. Pune, Nashik)",
        "lat": 18.5,
        "tmean": [22.5, 24.5, 27.5, 29.5, 29.5, 26.5, 25.0, 24.5, 25.0, 25.5, 24.0, 22.0],
        "dtr": [16, 16, 16, 15, 13, 9, 7, 7, 9, 12, 14, 15],
        "rain": [2, 1, 6, 18, 38, 120, 180, 120, 120, 80, 25, 5],
    },
    "nwindia": {
        "label": "Semi-arid north-west (Punjab, Haryana, Rajasthan)",
        "lat": 29.5,
        "tmean": [13, 16, 21, 28, 33, 34, 31, 30, 29, 25, 19, 14],
        "dtr": [14, 14, 15, 16, 15, 12, 9, 9, 12, 15, 15, 14],
        "rain": [22, 22, 15, 10, 12, 55, 190, 170, 65, 10, 5, 10],
    },
    "gangetic": {
        "label": "Humid sub-tropical monsoon (Bengal, Bihar, eastern UP)",
        "lat": 24.0,
        "tmean": [18, 21, 26, 30, 31, 30, 29, 29, 29, 27, 23, 19],
        "dtr": [14, 13, 13, 12, 10, 7, 6, 6, 7, 10, 13, 14],
        "rain": [12, 28, 36, 50, 120, 260, 330, 330, 290, 140, 20, 5],
    },
    "westcoast": {
        "label": "Humid tropical (Konkan, Kerala, west coast)",
        "lat": 12.0,
        "tmean": [27, 27.5, 28.5, 29, 28.5, 26, 25.5, 25.5, 26, 27, 27, 27],
        "dtr": [9, 9, 9, 8, 7, 5, 4, 4, 5, 7, 8, 9],
        "rain": [4, 3, 12, 60, 250, 700, 800, 500, 300, 250, 120, 25],
    },
    "temperate": {
        "label": "Temperate humid (e.g. north-west Europe)",
        "lat": 50.0,
        "tmean": [3, 4, 6.5, 9.5, 13, 16, 18, 18, 15, 11, 6.5, 4],
        "dtr": [5, 6, 7, 8, 9, 9, 9, 9, 8, 6, 5, 4],
        "rain": [60, 45, 48, 45, 55, 55, 60, 60, 55, 70, 70, 65],
    },
}

YEAR_FACTOR = {"dry": 0.65, "normal": 1.0, "wet": 1.35}
_YEAR_SEED = {"dry": 101, "normal": 202, "wet": 303}


def _interp(vals, d: dt.date) -> float:
    """Interpolate monthly values to a day (month centres are the anchors)."""
    dim = calendar.monthrange(d.year, d.month)[1]
    p = (d.month - 1) + (d.day - 0.5) / dim - 0.5
    i0 = math.floor(p)
    w = p - i0
    return vals[i0 % 12] * (1 - w) + vals[(i0 + 1) % 12] * w


def extraterrestrial_radiation(lat_deg: float, doy: int) -> float:
    """Ra in mm/day of equivalent evaporation (FAO-56, eq. 21)."""
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    x = -math.tan(phi) * math.tan(delta)
    ws = math.acos(max(-1.0, min(1.0, x)))
    ra = (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(ws)
    )
    return ra * 0.408


def hargreaves_et0(tmax: float, tmin: float, lat: float, doy: int) -> float:
    tmean = 0.5 * (tmax + tmin)
    et0 = 0.0023 * (tmean + 17.8) * math.sqrt(max(tmax - tmin, 0.5)) * extraterrestrial_radiation(lat, doy)
    return max(0.3, et0)


def generate(preset: str, start: dt.date, ndays: int, year_type: str = "normal",
             lat: float | None = None) -> dict:
    """Daily weather for ``ndays`` days beginning at ``start``."""
    p = PRESETS[preset]
    lat = p["lat"] if lat is None else lat
    rng = random.Random(_YEAR_SEED[year_type])
    yf = YEAR_FACTOR[year_type]
    end = start + dt.timedelta(days=ndays - 1)

    # Work over whole calendar months so monthly totals can be rescaled properly.
    first = dt.date(start.year, start.month, 1)
    last = dt.date(end.year, end.month, calendar.monthrange(end.year, end.month)[1])
    days = []
    d = first
    while d <= last:
        days.append(d)
        d += dt.timedelta(days=1)

    rain = {}
    wet_prev = False
    by_month = {}
    for d in days:
        by_month.setdefault((d.year, d.month), []).append(d)
    for (yr, mo), mdays in by_month.items():
        dim = len(mdays)
        target = p["rain"][mo - 1] * yf * math.exp(rng.gauss(0, 0.22) - 0.5 * 0.22 ** 2)
        mean_wet = 14.0 if p["rain"][mo - 1] > 200 else 10.0
        n_wet = min(max(target / mean_wet, 0.3), 0.85 * dim)
        pi = n_wet / dim
        p_ww = pi + 0.3 * (1 - pi)
        p_dw = pi * (1 - p_ww) / (1 - pi)
        vals = []
        for _ in mdays:
            wet = rng.random() < (p_ww if wet_prev else p_dw)
            wet_prev = wet
            vals.append(rng.gammavariate(0.8, (target / max(n_wet, 0.3)) / 0.8) if wet and target > 0.5 else 0.0)
        s = sum(vals)
        if target > 0.5 and s <= 0:
            vals[rng.randrange(dim)] = target
            s = target
        scale = target / s if s > 0 else 0.0
        for dd, v in zip(mdays, vals):
            rain[dd] = v * scale

    tmax, tmin, rn, et0, dates = [], [], [], [], []
    noise = 0.0
    # burn the noise state forward to the start date so results do not depend on it
    for d in days:
        noise = 0.7 * noise + rng.gauss(0, 1.8 * math.sqrt(1 - 0.49))
        if d < start or d > end:
            continue
        tm = _interp(p["tmean"], d) + noise
        dtr = _interp(p["dtr"], d)
        r = rain[d]
        if r > 5:
            dtr *= 0.7
        tx, tn = tm + dtr / 2, tm - dtr / 2
        dates.append(d)
        tmax.append(tx)
        tmin.append(tn)
        rn.append(r)
        et0.append(hargreaves_et0(tx, tn, lat, d.timetuple().tm_yday))
    return {"dates": dates, "tmax": tmax, "tmin": tmin, "rain": rn, "et0": et0}


def monthly_normals(preset: str, lat: float | None = None) -> dict:
    """Deterministic monthly mean temperature, rainfall and PET (mm)."""
    p = PRESETS[preset]
    lat = p["lat"] if lat is None else lat
    pet = []
    for m in range(12):
        tm = p["tmean"][m]
        dtr = p["dtr"][m]
        doy = int(15 + 30.4 * m)
        pet.append(hargreaves_et0(tm + dtr / 2, tm - dtr / 2, lat, doy) * calendar.monthrange(2023, m + 1)[1])
    return {"tmean": list(p["tmean"]), "rain": list(p["rain"]), "pet": pet}
