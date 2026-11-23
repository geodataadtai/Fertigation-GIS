"""
Fertigation Schedule Engine
===========================
Implements the KVK-prescribed DAP (Days After Planting) fertigation schedule
transcribed directly from the agronomists' official 'Fertigation .xlsx' guide.

Core Features:
1. Soil pH Adaptation:
   - pH 6.0–8.2: Standard/slightly alkaline schedule (e.g. Neem Cake, 24:24:00, 10:26:26, Phosphoric acid)
   - pH 8.3–8.9: High alkalinity schedule (e.g. Phosphogypsum, Ammonium Sulphate, 09:24:24)
2. Crop Season DAP Cutoff:
   - Adsali: up to 450 DAP (long duration crop)
   - Pre-Seasonal: up to 420 DAP
   - Suru: up to 360 DAP (annual crop)
3. Dynamic Nutrient Balancing (Least-Squares Optimization):
   Scales remaining doses so that:
   Nutrients Applied Till Date + Advised Upcoming Doses = Bounded STCR Targeted Yield Dose.
4. Farm Acreage Scaling:
   Computes both standard rate per acre and total scaled farm quantity (kg/liters) + drip water volume.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, List, Tuple, Optional
import numpy as np


# =============================================================================
# 1. Base Schedule Data (from Fertigation .xlsx)
# =============================================================================

# Schedule for Normal / Slightly Alkaline Soils (pH 6.0 - 8.2)
_SCHEDULE_PH_LOW: List[Dict[str, Any]] = [
    # 0 DAP (Planting / Basal Stage)
    {"dap": 0,   "fertilizer": "Neem Cake",                  "quantity": 40.0,  "unit": "kg",  "water": 0.0},
    {"dap": 0,   "fertilizer": "10:26:26",                   "quantity": 30.0,  "unit": "kg",  "water": None},
    {"dap": 0,   "fertilizer": "Organic Manure",             "quantity": 500.0, "unit": "kg",  "water": None},
    {"dap": 0,   "fertilizer": "Fertera",                    "quantity": 4.0,   "unit": "kg",  "water": None},
    # 5 DAP
    {"dap": 5,   "fertilizer": "Mycorrhiza",                 "quantity": 100.0, "unit": "gm",  "water": 200.0},
    {"dap": 5,   "fertilizer": "Humic + Fulvic Acid",        "quantity": 100.0, "unit": "gm",  "water": None},
    # 7 DAP
    {"dap": 7,   "fertilizer": "12:61:00",                   "quantity": 2.0,   "unit": "kg",  "water": 300.0},
    {"dap": 7,   "fertilizer": "Magnesium Sulphate",         "quantity": 1.0,   "unit": "kg",  "water": None},
    # 9 DAP
    {"dap": 9,   "fertilizer": "Corogen",                    "quantity": 150.0, "unit": "ml",  "water": 300.0},
    # 15 DAP
    {"dap": 15,  "fertilizer": "24:24:00",                   "quantity": 10.0,  "unit": "kg",  "water": 100.0},
    {"dap": 15,  "fertilizer": "Organic Carbon",             "quantity": 1.0,   "unit": "lit", "water": None},
    # 22 DAP
    {"dap": 22,  "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    # 30 DAP
    {"dap": 30,  "fertilizer": "Micronutrient Slurry",       "quantity": 10.0,  "unit": "lit", "water": 100.0},
    {"dap": 30,  "fertilizer": "Urea",                       "quantity": 10.0,  "unit": "kg",  "water": None},
    # 40 DAP
    {"dap": 40,  "fertilizer": "Calcium Nitrate",            "quantity": 2.0,   "unit": "kg",  "water": 100.0},
    # 45 DAP
    {"dap": 45,  "fertilizer": "Ammonium Sulphate",          "quantity": 25.0,  "unit": "kg",  "water": 0.0},
    {"dap": 45,  "fertilizer": "Diammonium Phosphate",       "quantity": 25.0,  "unit": "kg",  "water": None},
    {"dap": 45,  "fertilizer": "Urea",                       "quantity": 25.0,  "unit": "kg",  "water": None},
    {"dap": 45,  "fertilizer": "Poly Sulphate",              "quantity": 10.0,  "unit": "kg",  "water": None},
    # 50 DAP
    {"dap": 50,  "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 50,  "fertilizer": "24:24:00",                   "quantity": 10.0,  "unit": "kg",  "water": None},
    {"dap": 50,  "fertilizer": "Urea",                       "quantity": 15.0,  "unit": "kg",  "water": None},
    # 60 DAP
    {"dap": 60,  "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 60,  "fertilizer": "Ammonium Sulphate",          "quantity": 3.0,   "unit": "kg",  "water": None},
    # 75 DAP
    {"dap": 75,  "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 75,  "fertilizer": "24:24:00",                   "quantity": 5.0,   "unit": "kg",  "water": None},
    {"dap": 75,  "fertilizer": "Urea",                       "quantity": 15.0,  "unit": "kg",  "water": None},
    # 90 DAP
    {"dap": 90,  "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 90,  "fertilizer": "13:00:45",                   "quantity": 2.0,   "unit": "kg",  "water": None},
    # 110 DAP
    {"dap": 110, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 110, "fertilizer": "24:24:00",                   "quantity": 5.0,   "unit": "kg",  "water": None},
    {"dap": 110, "fertilizer": "Urea",                       "quantity": 15.0,  "unit": "kg",  "water": None},
    # 120 DAP
    {"dap": 120, "fertilizer": "Ammonium Sulphate",          "quantity": 50.0,  "unit": "kg",  "water": 0.0},
    {"dap": 120, "fertilizer": "09:24:24",                   "quantity": 30.0,  "unit": "kg",  "water": None},
    {"dap": 120, "fertilizer": "Muriate of Potash",          "quantity": 25.0,  "unit": "kg",  "water": None},
    {"dap": 120, "fertilizer": "Poly Sulphate",              "quantity": 25.0,  "unit": "kg",  "water": None},
    {"dap": 120, "fertilizer": "Fertera",                    "quantity": 4.0,   "unit": "kg",  "water": None},
    # 125 DAP
    {"dap": 125, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 125, "fertilizer": "Boron",                      "quantity": 250.0, "unit": "gm",  "water": None},
    # 140 DAP
    {"dap": 140, "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 140, "fertilizer": "Sulphate of Potash",         "quantity": 3.0,   "unit": "kg",  "water": None},
    {"dap": 140, "fertilizer": "Ammonium Sulphate",          "quantity": 3.0,   "unit": "kg",  "water": None},
    # 160 DAP
    {"dap": 160, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 160, "fertilizer": "Urea",                       "quantity": 15.0,  "unit": "kg",  "water": None},
    # 180 DAP
    {"dap": 180, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 180, "fertilizer": "13:00:45",                   "quantity": 2.0,   "unit": "kg",  "water": None},
    {"dap": 180, "fertilizer": "Boron",                      "quantity": 250.0, "unit": "gm",  "water": None},
    # 200 DAP
    {"dap": 200, "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 200, "fertilizer": "Sulphate of Potash",         "quantity": 3.0,   "unit": "kg",  "water": None},
    {"dap": 200, "fertilizer": "Ammonium Sulphate",          "quantity": 3.0,   "unit": "kg",  "water": None},
    # 210 DAP
    {"dap": 210, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 210, "fertilizer": "24:24:00",                   "quantity": 5.0,   "unit": "kg",  "water": None},
    {"dap": 210, "fertilizer": "Urea",                       "quantity": 10.0,  "unit": "kg",  "water": None},
    # 230 DAP
    {"dap": 230, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 230, "fertilizer": "13:00:45",                   "quantity": 3.0,   "unit": "kg",  "water": None},
    {"dap": 230, "fertilizer": "Boron",                      "quantity": 250.0, "unit": "gm",  "water": None},
    # 250 DAP
    {"dap": 250, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 250, "fertilizer": "24:24:00",                   "quantity": 5.0,   "unit": "kg",  "water": None},
    {"dap": 250, "fertilizer": "Urea",                       "quantity": 15.0,  "unit": "kg",  "water": None},
    # 280 DAP
    {"dap": 280, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 280, "fertilizer": "13:00:45",                   "quantity": 3.0,   "unit": "kg",  "water": None},
    {"dap": 280, "fertilizer": "Boron",                      "quantity": 250.0, "unit": "gm",  "water": None},
    # 290 DAP
    {"dap": 290, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    # 300 DAP
    {"dap": 300, "fertilizer": "24:24:00",                   "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 300, "fertilizer": "Urea",                       "quantity": 15.0,  "unit": "kg",  "water": None},
    {"dap": 300, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    {"dap": 300, "fertilizer": "13:00:45",                   "quantity": 3.0,   "unit": "kg",  "water": None},
    # 310 DAP
    {"dap": 310, "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 310, "fertilizer": "Muriate of Potash",          "quantity": 7.0,   "unit": "kg",  "water": None},
    {"dap": 310, "fertilizer": "Ammonium Sulphate",          "quantity": 3.0,   "unit": "kg",  "water": None},
    # 320 DAP
    {"dap": 320, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 320, "fertilizer": "Boron",                      "quantity": 100.0, "unit": "gm",  "water": None},
    # 330 DAP
    {"dap": 330, "fertilizer": "24:24:00",                   "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 330, "fertilizer": "Muriate of Potash",          "quantity": 7.0,   "unit": "kg",  "water": None},
    # 340 DAP
    {"dap": 340, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    # 350 DAP
    {"dap": 350, "fertilizer": "Urea",                       "quantity": 20.0,  "unit": "kg",  "water": 100.0},
    {"dap": 350, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    # 360 DAP
    {"dap": 360, "fertilizer": "24:24:00",                   "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 360, "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": None},
    # 370 DAP
    {"dap": 370, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 370, "fertilizer": "Boron",                      "quantity": 200.0, "unit": "gm",  "water": None},
    # 380 DAP
    {"dap": 380, "fertilizer": "Urea",                       "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 380, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    # 390 DAP
    {"dap": 390, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    # 400 DAP
    {"dap": 400, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    # 420 DAP
    {"dap": 420, "fertilizer": "Urea",                       "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 420, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    # 450 DAP
    {"dap": 450, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
]

# Schedule for Alkaline Soils (pH 8.3 - 8.9)
_SCHEDULE_PH_HIGH: List[Dict[str, Any]] = [
    # 0 DAP (Planting / Basal Stage)
    {"dap": 0,   "fertilizer": "Phosphogypsum",              "quantity": 250.0, "unit": "kg",  "water": 0.0},
    {"dap": 0,   "fertilizer": "Prome",                      "quantity": 200.0, "unit": "kg",  "water": None},
    {"dap": 0,   "fertilizer": "Fertera",                    "quantity": 4.0,   "unit": "kg",  "water": None},
    # 5 DAP
    {"dap": 5,   "fertilizer": "Mycorrhiza",                 "quantity": 100.0, "unit": "gm",  "water": 200.0},
    {"dap": 5,   "fertilizer": "Humic + Fulvic Acid",        "quantity": 100.0, "unit": "gm",  "water": None},
    # 7 DAP
    {"dap": 7,   "fertilizer": "12:61:00",                   "quantity": 2.0,   "unit": "kg",  "water": 200.0},
    {"dap": 7,   "fertilizer": "PlantoDrip",                 "quantity": 1.0,   "unit": "lit", "water": None},
    # 9 DAP
    {"dap": 9,   "fertilizer": "Corogen",                    "quantity": 150.0, "unit": "ml",  "water": 300.0},
    # 22 DAP
    {"dap": 22,  "fertilizer": "Phosphoric Acid",            "quantity": 2.5,   "unit": "kg",  "water": 100.0},
    {"dap": 22,  "fertilizer": "Ammonium Sulphate",          "quantity": 2.0,   "unit": "kg",  "water": None},
    # 30 DAP
    {"dap": 30,  "fertilizer": "Micronutrient Slurry",       "quantity": 10.0,  "unit": "lit", "water": 100.0},
    {"dap": 30,  "fertilizer": "Urea",                       "quantity": 25.0,  "unit": "kg",  "water": None},
    # 40 DAP
    {"dap": 40,  "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    # 45 DAP
    {"dap": 45,  "fertilizer": "Ammonium Sulphate",          "quantity": 25.0,  "unit": "kg",  "water": 0.0},
    {"dap": 45,  "fertilizer": "09:24:24",                   "quantity": 25.0,  "unit": "kg",  "water": None},
    {"dap": 45,  "fertilizer": "Promo",                      "quantity": 50.0,  "unit": "kg",  "water": None},
    {"dap": 45,  "fertilizer": "Poly Sulphate",              "quantity": 25.0,  "unit": "kg",  "water": None},
    # 50 DAP
    {"dap": 50,  "fertilizer": "24:24:00",                   "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 50,  "fertilizer": "Humistar",                   "quantity": 100.0, "unit": "gm",  "water": None},
    # 60 DAP
    {"dap": 60,  "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 60,  "fertilizer": "Ammonium Sulphate",          "quantity": 3.0,   "unit": "kg",  "water": None},
    # 75 DAP
    {"dap": 75,  "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 75,  "fertilizer": "Urea",                       "quantity": 25.0,  "unit": "kg",  "water": None},
    # 90 DAP
    {"dap": 90,  "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 90,  "fertilizer": "13:00:45",                   "quantity": 2.0,   "unit": "kg",  "water": None},
    # 110 DAP
    {"dap": 110, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 110, "fertilizer": "Urea",                       "quantity": 25.0,  "unit": "kg",  "water": None},
    # 120 DAP
    {"dap": 120, "fertilizer": "Ammonium Sulphate",          "quantity": 50.0,  "unit": "kg",  "water": 0.0},
    {"dap": 120, "fertilizer": "09:24:24",                   "quantity": 25.0,  "unit": "kg",  "water": None},
    {"dap": 120, "fertilizer": "Muriate of Potash",          "quantity": 45.0,  "unit": "kg",  "water": None},
    {"dap": 120, "fertilizer": "Poly Sulphate",              "quantity": 25.0,  "unit": "kg",  "water": None},
    {"dap": 120, "fertilizer": "Fertera",                    "quantity": 4.0,   "unit": "kg",  "water": None},
    # 125 DAP
    {"dap": 125, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 125, "fertilizer": "Boron",                      "quantity": 250.0, "unit": "gm",  "water": None},
    # 140 DAP
    {"dap": 140, "fertilizer": "Phosphoric Acid",            "quantity": 2.5,   "unit": "kg",  "water": 100.0},
    {"dap": 140, "fertilizer": "Muriate of Potash",          "quantity": 4.0,   "unit": "kg",  "water": None},
    {"dap": 140, "fertilizer": "Ammonium Sulphate",          "quantity": 3.0,   "unit": "kg",  "water": None},
    # 160 DAP
    {"dap": 160, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 160, "fertilizer": "Urea",                       "quantity": 25.0,  "unit": "kg",  "water": None},
    # 180 DAP
    {"dap": 180, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 180, "fertilizer": "Boron",                      "quantity": 250.0, "unit": "gm",  "water": None},
    # 200 DAP
    {"dap": 200, "fertilizer": "24:24:00",                   "quantity": 10.0,  "unit": "kg",  "water": 100.0},
    {"dap": 200, "fertilizer": "Humistar",                   "quantity": 100.0, "unit": "gm",  "water": None},
    # 210 DAP
    {"dap": 210, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 210, "fertilizer": "Urea",                       "quantity": 10.0,  "unit": "kg",  "water": None},
    # 220 DAP
    {"dap": 220, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 220, "fertilizer": "Boron",                      "quantity": 200.0, "unit": "gm",  "water": None},
    # 230 DAP
    {"dap": 230, "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 230, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    {"dap": 230, "fertilizer": "Urea",                       "quantity": 20.0,  "unit": "kg",  "water": None},
    # 250 DAP
    {"dap": 250, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    {"dap": 250, "fertilizer": "Urea",                       "quantity": 10.0,  "unit": "kg",  "water": None},
    # 280 DAP
    {"dap": 280, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    # 290 DAP
    {"dap": 290, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    # 300 DAP
    {"dap": 300, "fertilizer": "24:24:00",                   "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 300, "fertilizer": "Urea",                       "quantity": 15.0,  "unit": "kg",  "water": None},
    {"dap": 300, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    {"dap": 300, "fertilizer": "13:00:45",                   "quantity": 3.0,   "unit": "kg",  "water": None},
    # 310 DAP
    {"dap": 310, "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 310, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    {"dap": 310, "fertilizer": "Ammonium Sulphate",          "quantity": 3.0,   "unit": "kg",  "water": None},
    # 320 DAP
    {"dap": 320, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 320, "fertilizer": "Boron",                      "quantity": 100.0, "unit": "gm",  "water": None},
    # 330 DAP
    {"dap": 330, "fertilizer": "24:24:00",                   "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 330, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    # 340 DAP
    {"dap": 340, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    # 350 DAP
    {"dap": 350, "fertilizer": "Urea",                       "quantity": 20.0,  "unit": "kg",  "water": 100.0},
    {"dap": 350, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    # 360 DAP
    {"dap": 360, "fertilizer": "24:24:00",                   "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 360, "fertilizer": "Phosphoric Acid",            "quantity": 3.0,   "unit": "kg",  "water": None},
    # 370 DAP
    {"dap": 370, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    {"dap": 370, "fertilizer": "Boron",                      "quantity": 200.0, "unit": "gm",  "water": None},
    # 380 DAP
    {"dap": 380, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": 100.0},
    {"dap": 380, "fertilizer": "Urea",                       "quantity": 25.0,  "unit": "kg",  "water": None},
    # 390 DAP
    {"dap": 390, "fertilizer": "Micronutrient Slurry",       "quantity": 15.0,  "unit": "lit", "water": 100.0},
    # 400 DAP
    {"dap": 400, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
    # 420 DAP
    {"dap": 420, "fertilizer": "Urea",                       "quantity": 25.0,  "unit": "kg",  "water": 100.0},
    {"dap": 420, "fertilizer": "Muriate of Potash",          "quantity": 5.0,   "unit": "kg",  "water": None},
    # 450 DAP
    {"dap": 450, "fertilizer": "Calcium Nitrate",            "quantity": 3.0,   "unit": "kg",  "water": 100.0},
]

# Season maximum DAP cutoff limits
_SEASON_DAP_LIMITS: Dict[str, int] = {
    "adsali":       450,
    "pre-seasonal": 420,
    "preseasonal":  420,
    "suru":         360,
    "seasonal":     360,
}
_DEFAULT_DAP_LIMIT = 420


# =============================================================================
# 2. Nutrient Composition Database
# =============================================================================

_FERTILIZER_NUTRIENTS: Dict[str, Tuple[float, float, float]] = {
    # NPK Grades (% N, % P2O5, % K2O)
    "09::24::24": (9.0, 24.0, 24.0),
    "9::24::24":  (9.0, 24.0, 24.0),
    "09:24:24":   (9.0, 24.0, 24.0),
    "9:24:24":    (9.0, 24.0, 24.0),
    "10::26::26": (10.0, 26.0, 26.0),
    "10:26:26":   (10.0, 26.0, 26.0),
    "12::61::00": (12.0, 61.0, 0.0),
    "12:61:00":   (12.0, 61.0, 0.0),
    "13::00::45": (13.0, 0.0, 45.0),
    "13:00:45":   (13.0, 0.0, 45.0),
    "24::24::00": (24.0, 24.0, 0.0),
    "24:24:00":   (24.0, 24.0, 0.0),

    # Straight & Specialty Fertilizers
    "urea":                  (46.0, 0.0, 0.0),
    "ammonium sulphate":     (20.6, 0.0, 0.0),
    "muriate of potash":     (0.0, 0.0, 60.0),
    "sulphate of potash":    (0.0, 0.0, 50.0),
    "calcium nitrate":       (15.5, 0.0, 0.0),
    "daiammonium phosphate": (18.0, 46.0, 0.0),
    "diammonium phosphate":  (18.0, 46.0, 0.0),
    "dap":                   (18.0, 46.0, 0.0),
    "phosphoric acid":       (0.0, 52.0, 0.0),
    "poly sulphate":         (0.0, 0.0, 14.0),

    # Organics
    "neem cake":      (5.0, 1.0, 2.0),
    "organic manure": (0.5, 0.2, 0.5),
    "prome":          (0.4, 10.4, 0.0),
    "promo":          (0.4, 10.4, 0.0),
}

# =============================================================================
# Marathi Translation Maps Transcribed from 'Fertigation_Marathi.xlsx'
# =============================================================================

FERT_NAMES_MARATHI: Dict[str, str] = {
    # NPK compound grades (with both :: and : notations)
    "09::24::24": "०९:२४:२४",
    "9::24::24": "०९:२४:२४",
    "09:24:24": "०९:२४:२४",
    "9:24:24": "०९:२४:२४",
    "10::26::26": "१०:२६:२६",
    "10:26:26": "१०:२६:२६",
    "12::61::00": "१२:६१:००",
    "12:61:00": "१२:६१:००",
    "13::00::45": "१३:००:४५",
    "13:00:45": "१३:००:४५",
    "24::24::00": "२४:२४:००",
    "24:24:00": "२४:२४:००",
    "19:19:19": "१९:१९:१९",
    "19::19::19": "१९:१९:१९",
    "14:35:14": "१४:३५:१४",
    "14::35::14": "१४:३५:१४",
    "12:32:16": "१२:३२:१६",
    "12::32::16": "१२:३२:१६",
    "00:52:34": "००:५२:३४ (MKP)",
    "00:00:50": "००:००:५० (SOP)",

    # Straight & Mineral Fertilizers from Fertigation .xlsx
    "Urea": "युरिया",
    "urea": "युरिया",
    "Ammonium Sulphate": "अमोनियम सल्फेट",
    "ammonium sulphate": "अमोनियम सल्फेट",
    "Calcium Nitrate": "कॅल्शियम नायट्रेट",
    "Calcium nitrate": "कॅल्शियम नायट्रेट",
    "calcium nitrate": "कॅल्शियम नायट्रेट",
    "Muriate of Potash": "म्युरिएट ऑफ पोटॅश (MOP)",
    "muriate of potash": "म्युरिएट ऑफ पोटॅश (MOP)",
    "MOP": "म्युरिएट ऑफ पोटॅश (MOP)",
    "Sulphate of Potash": "सल्फेट ऑफ पोटॅश (SOP)",
    "sulphate of potash": "सल्फेट ऑफ पोटॅश (SOP)",
    "SOP": "सल्फेट ऑफ पोटॅश (SOP)",
    "Phosphoric Acid": "फॉस्फोरिक ॲसिड",
    "phosphoric acid": "फॉस्फोरिक ॲसिड",
    "Diammonium Phosphate": "डी.ए.पी. (डायअमोनियम फॉस्फेट १८:४६:००)",
    "Daiammonium Phosphate": "डी.ए.पी. (डायअमोनियम फॉस्फेट १८:४६:००)",
    "diammonium phosphate": "डी.ए.पी. (डायअमोनियम फॉस्फेट १८:४६:००)",
    "DAP": "डी.ए.पी. (१८:४६:००)",
    "Poly Sulphate": "पॉली सल्फेट",
    "poly sulphate": "पॉली सल्फेट",
    "Phospogypsum": "फॉस्फोजिप्सम",
    "Phosphogypsum": "फॉस्फोजिप्सम",
    "phosphogypsum": "फॉस्फोजिप्सम",
    "Magnesium Sulphate": "मॅग्नेशियम सल्फेट",
    "magnesium sulphate": "मॅग्नेशियम सल्फेट",
    "Ferrous Sulphate": "फेरस सल्फेट (हिराकस)",
    "Zinc Sulphate": "झिंक सल्फेट",
    "Boron": "बोरोन",
    "boron": "बोरोन",
    "Borax": "बोराक्स",
    "CAN": "कॅन (कॅल्शियम अमोनियम नायट्रेट)",
    "SSP": "सिंगल सुपर फॉस्फेट (SSP)",

    # Biologicals, Organics & Amendments from Fertigation .xlsx
    "Neem Cake": "निंबोळी पेंड",
    "neem cake": "निंबोळी पेंड",
    "Nimboli Pend": "निंबोळी पेंड",
    "Organic Manure": "सेंद्रिय खत (शेणखत / कंपोस्ट)",
    "organic manure": "सेंद्रिय खत (शेणखत / कंपोस्ट)",
    "Organic Corben": "सेंद्रिय कर्ब",
    "Organic Carbon": "सेंद्रिय कर्ब",
    "organic corben": "सेंद्रिय कर्ब",
    "organic carbon": "सेंद्रिय कर्ब",
    "Prome": "प्रोम (फॉस्फेट समृद्ध सेंद्रिय खत)",
    "Promo": "प्रोम (फॉस्फेट समृद्ध सेंद्रिय खत)",
    "prome": "प्रोम (फॉस्फेट समृद्ध सेंद्रिय खत)",
    "promo": "प्रोम (फॉस्फेट समृद्ध सेंद्रिय खत)",
    "Fertera": "फरटेरा",
    "fertera": "फरटेरा",
    "Mycorrizha": "मायकोरायझा",
    "Mycorrhiza": "मायकोरायझा",
    "mycorrizha": "मायकोरायझा",
    "mycorrhiza": "मायकोरायझा",
    "Humic + Fulvic Acid": "ह्युमिक + फुल्विक ॲसिड",
    "Humic+Fulvic Acid": "ह्युमिक + फुल्विक ॲसिड",
    "Humistar": "ह्युमिस्टार",
    "humistar": "ह्युमिस्टार",
    "Corogen": "कोराजन",
    "corogen": "कोराजन",
    "PlantoDrip": "प्लॅन्टोड्रिप",
    "plantodrip": "प्लॅन्टोड्रिप",
    "Micronutrient Slurry": "सूक्ष्म अन्नद्रव्य स्लरी",
    "Micronutrient slurry": "सूक्ष्म अन्नद्रव्य स्लरी",
    "micronutrient slurry": "सूक्ष्म अन्नद्रव्य स्लरी",
    "Azotobacter": "ॲझोटोबॅक्टर",
    "PSB": "पीएसबी (स्फुरद विरघळवणारे जिवाणू)",
}

UNIT_NAMES_MARATHI: Dict[str, str] = {
    "kg": "किलो",
    "Kg": "किलो",
    "gm": "ग्रॅम",
    "g": "ग्रॅम",
    "lit": "लिटर",
    "Lit": "लिटर",
    "ml": "मिली",
}


def _calculate_product_npk(fert_name: str, qty: float, unit: str) -> Tuple[float, float, float]:
    """Calculates kilograms of elemental N, P2O5, K2O delivered by a product quantity."""
    if not qty or qty <= 0:
        return 0.0, 0.0, 0.0

    unit_clean = str(unit or "").strip().lower()
    conversion_factor = 0.001 if unit_clean in ("gm", "g", "ml") else 1.0
    qty_kg = float(qty) * conversion_factor

    name_clean = str(fert_name or "").strip().lower()
    percentages = _FERTILIZER_NUTRIENTS.get(name_clean)

    if not percentages:
        # Match standard N:P:K format like 19:19:19 or 12-32-16
        match = re.match(r"(\d+)[:\.\-]+(\d+)[:\.\-]+(\d+)", name_clean)
        if match:
            percentages = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
        else:
            percentages = (0.0, 0.0, 0.0)

    n_kg = qty_kg * (percentages[0] / 100.0)
    p_kg = qty_kg * (percentages[1] / 100.0)
    k_kg = qty_kg * (percentages[2] / 100.0)
    return n_kg, p_kg, k_kg


# =============================================================================
# 3. Helper Functions
# =============================================================================

def _detect_season_key(season_value: Any, p_date: Optional[date] = None) -> str:
    """Normalizes season string or infers it from planting month."""
    s = str(season_value or "").strip().lower()
    if "adsali" in s or "आडसाली" in s:
        return "adsali"
    if "pre" in s or "preseasonal" in s or "पूर्वहंगामी" in s:
        return "pre-seasonal"
    if "suru" in s or "seasonal" in s or "सुरू" in s or "हंगामी" in s:
        return "suru"

    # Infer by plantation month according to Maharashtra sugarcane seasons
    if p_date:
        month = p_date.month
        if 6 <= month <= 9:      # June to September
            return "adsali"
        elif 10 <= month <= 11:  # October to November
            return "pre-seasonal"
        else:                    # December to May
            return "suru"

    return "pre-seasonal"


def _parse_plantation_date(plantation_date: Any) -> Optional[date]:
    """Parses date string or date object into Python date."""
    if plantation_date is None:
        return None
    if isinstance(plantation_date, date):
        return plantation_date

    s = str(plantation_date).strip()
    for fmt in ("%d %b %Y", "%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%B %d, %Y"):
        try:
            from datetime import datetime as _dt
            return _dt.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _select_schedule(ph: Any) -> List[Dict[str, Any]]:
    """Selects base schedule table according to soil pH."""
    try:
        ph_val = float(ph)
    except (TypeError, ValueError):
        ph_val = 7.0
    return _SCHEDULE_PH_HIGH if ph_val > 8.2 else _SCHEDULE_PH_LOW


def _ph_range_label(ph: Any) -> str:
    """Returns human-readable pH category label."""
    try:
        ph_val = float(ph)
    except (TypeError, ValueError):
        ph_val = 7.0
    return "8.3–8.9 (Alkaline)" if ph_val > 8.2 else "6.0–8.2 (Normal/Slightly Alkaline)"


# =============================================================================
# 4. Main Fertigation Engine Pipeline
# =============================================================================

def get_fertigation_schedule(
    ph: Any,
    season: Any,
    plantation_date: Any,
    to_be_applied_npk: Optional[Dict[str, float]] = None,
    login_dap: Optional[int] = None,
    is_farmer_view: bool = False,
    applied_till_date: Optional[Dict[str, float]] = None,
    area_acre: float = 1.0,
) -> Dict[str, Any]:
    """
    Computes a tailored DAP fertigation schedule.

    Balancing Strategy:
    Uses a 3x3 least-squares nutrient matrix solver across upcoming doses so that
    the cumulative total of all remaining fertigation doses perfectly meets the
    farmer's bounded STCR targeted nutrient requirements.
    """
    p_date = _parse_plantation_date(plantation_date)
    season_key = _detect_season_key(season, p_date)
    max_dap = _SEASON_DAP_LIMITS.get(season_key, _DEFAULT_DAP_LIMIT)
    base_schedule = _select_schedule(ph)
    today = date.today()

    days_since = (today - p_date).days if p_date else None
    current_dap = days_since

    # Standard seasonal target benchmarks (kg/acre)
    SEASON_TARGETS = {
        "adsali":       {"N": 160.0, "P": 68.0, "K": 68.0},
        "pre-seasonal": {"N": 136.0, "P": 68.0, "K": 68.0},
        "suru":         {"N": 100.0, "P": 46.0, "K": 46.0},
    }
    target_season_npk = SEASON_TARGETS.get(season_key, SEASON_TARGETS["pre-seasonal"])

    # 1. Filter schedule up to the crop season cutoff DAP
    filtered = [e for e in base_schedule if e["dap"] <= max_dap]

    # 2. Threshold to split past doses from upcoming doses
    split_dap = login_dap if login_dap is not None else current_dap

    if split_dap is not None:
        base_upcoming_entries = [e for e in filtered if e["dap"] > split_dap]
    else:
        base_upcoming_entries = filtered

    # 3. Target remaining nutrients to satisfy in remaining doses
    if to_be_applied_npk and isinstance(to_be_applied_npk, dict):
        rem_target_n = max(0.0, float(to_be_applied_npk.get("N", 0.0)))
        rem_target_p = max(0.0, float(to_be_applied_npk.get("P", 0.0)))
        rem_target_k = max(0.0, float(to_be_applied_npk.get("K", 0.0)))
    elif applied_till_date and isinstance(applied_till_date, dict):
        rem_target_n = max(0.0, target_season_npk["N"] - float(applied_till_date.get("N", 0.0)))
        rem_target_p = max(0.0, target_season_npk["P"] - float(applied_till_date.get("P", 0.0)))
        rem_target_k = max(0.0, target_season_npk["K"] - float(applied_till_date.get("K", 0.0)))
    else:
        rem_target_n = target_season_npk["N"]
        rem_target_p = target_season_npk["P"]
        rem_target_k = target_season_npk["K"]

    # 4. Least-squares matrix solver to determine optimal scaling factors
    scale_n, scale_p, scale_k = 1.0, 1.0, 1.0
    if base_upcoming_entries:
        A = np.zeros((3, 3), dtype=float)
        for entry in base_upcoming_entries:
            pn, pp, pk = _calculate_product_npk(entry["fertilizer"], entry["quantity"], entry["unit"])
            if pn == 0.0 and pp == 0.0 and pk == 0.0:
                continue

            # Assign product to dominant nutrient column (0: N, 1: P, 2: K)
            if pp > pn and pp >= pk:
                col = 1
            elif pk > pn and pk > pp:
                col = 2
            else:
                col = 0

            A[0, col] += pn
            A[1, col] += pp
            A[2, col] += pk

        b = np.array([rem_target_n, rem_target_p, rem_target_k], dtype=float)
        try:
            factors, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            factors = np.clip(factors, 0.05, 8.0)
            scale_n = float(factors[0])
            scale_p = float(factors[1])
            scale_k = float(factors[2])
        except Exception:
            scale_n, scale_p, scale_k = 1.0, 1.0, 1.0

    # 5. Group by DAP application points and calculate dates
    dap_groups: Dict[int, List[Dict[str, Any]]] = {}
    for entry in filtered:
        dap_groups.setdefault(entry["dap"], []).append(entry)

    area_multiplier = float(area_acre) if (area_acre is not None and float(area_acre) > 0) else 1.0
    application_groups: List[Dict[str, Any]] = []

    for dap_val in sorted(dap_groups.keys()):
        items = dap_groups[dap_val]
        is_upcoming_entry = (split_dap is None) or (dap_val > split_dap)

        if p_date:
            app_date = p_date + timedelta(days=dap_val)
            app_date_str = app_date.strftime("%d-%b-%Y")
            start_date_str = (app_date - timedelta(days=2)).strftime("%d-%b-%Y")
            end_date_str = (app_date + timedelta(days=5)).strftime("%d-%b-%Y")
            days_remaining = (app_date - today).days

            if days_remaining > 0:
                status = "upcoming"
            elif days_remaining == 0:
                status = "due_today"
            elif days_remaining >= -7:
                status = "overdue"
            else:
                status = "completed"
        else:
            app_date_str = None
            start_date_str = None
            end_date_str = None
            days_remaining = None
            status = "unknown"

        # Build scaled product list
        products = []
        products_mr = []
        rec_lines = []
        rec_lines_mr = []
        for entry in items:
            base_qty = float(entry["quantity"])
            unit_str = str(entry["unit"] or "").strip().lower()

            if is_upcoming_entry and to_be_applied_npk:
                n_pct, p_pct, k_pct = _calculate_product_npk(entry["fertilizer"], 100.0, "kg")
                if p_pct > n_pct and p_pct >= k_pct:
                    ratio = scale_p
                elif k_pct > n_pct and k_pct > p_pct:
                    ratio = scale_k
                elif n_pct > 0:
                    ratio = scale_n
                else:
                    ratio = 1.0  # Micronutrients/amendments stay unscaled
                scaled_qty = base_qty * ratio
            else:
                scaled_qty = base_qty

            # Per-acre quantity
            if unit_str in ("gm", "ml"):
                qty_per_acre = round(scaled_qty)
            else:
                qty_per_acre = round(scaled_qty, 1)
                if qty_per_acre == int(qty_per_acre):
                    qty_per_acre = float(int(qty_per_acre))

            # Total farm quantity scaled for farmer's acreage
            farm_qty = scaled_qty * area_multiplier
            if unit_str in ("gm", "ml"):
                qty_farm = round(farm_qty)
            else:
                qty_farm = round(farm_qty, 1)
                if qty_farm == int(qty_farm):
                    qty_farm = float(int(qty_farm))

            base_water = float(entry["water"]) if entry.get("water") is not None else None
            farm_water = round(base_water * area_multiplier) if base_water is not None else None

            fert_name = entry["fertilizer"]
            fert_name_mr = FERT_NAMES_MARATHI.get(fert_name, fert_name)
            unit_val = entry["unit"]
            unit_val_mr = UNIT_NAMES_MARATHI.get(unit_str, unit_val)

            products.append({
                "fertilizer": fert_name,
                "quantity": qty_farm,
                "quantity_per_acre": qty_per_acre,
                "unit": unit_val,
                "water_liters": farm_water,
                "water_liters_per_acre": base_water,
            })

            products_mr.append({
                "fertilizer": fert_name_mr,
                "quantity": qty_farm,
                "quantity_per_acre": qty_per_acre,
                "unit": unit_val_mr,
                "water_liters": farm_water,
                "water_liters_per_acre": base_water,
            })

            line = f"{fert_name}: {qty_farm} {unit_val}"
            if farm_water is not None and farm_water > 0:
                line += f" (Water: {farm_water} L)"
            rec_lines.append(line)

            line_mr = f"{fert_name_mr}: {qty_farm} {unit_val_mr}"
            if farm_water is not None and farm_water > 0:
                line_mr += f" (पाणी: {farm_water} लि.)"
            rec_lines_mr.append(line_mr)

        recommendation_text = "\n".join(rec_lines)
        recommendation_text_mr = "\n".join(rec_lines_mr)

        # Total NPK delivered in this dose
        total_farm_n = sum(_calculate_product_npk(p["fertilizer"], p["quantity"], p["unit"])[0] for p in products)
        total_farm_p = sum(_calculate_product_npk(p["fertilizer"], p["quantity"], p["unit"])[1] for p in products)
        total_farm_k = sum(_calculate_product_npk(p["fertilizer"], p["quantity"], p["unit"])[2] for p in products)

        per_acre_n = total_farm_n / area_multiplier if area_multiplier > 0 else total_farm_n
        per_acre_p = total_farm_p / area_multiplier if area_multiplier > 0 else total_farm_p
        per_acre_k = total_farm_k / area_multiplier if area_multiplier > 0 else total_farm_k

        application_groups.append({
            "dap": dap_val,
            "period": f"{dap_val} DAP",
            "period_mr": f"{dap_val} दिवस (DAP)",
            "application_date": app_date_str,
            "start_date": start_date_str or app_date_str,
            "end_date": end_date_str or app_date_str,
            "days_remaining": days_remaining,
            "status": status,
            "recommendation_text": recommendation_text,
            "recommendation_text_mr": recommendation_text_mr,
            "products": products,
            "products_mr": products_mr,
            "target_n": round(total_farm_n, 1) if round(total_farm_n, 1) != int(round(total_farm_n)) else int(round(total_farm_n)),
            "target_p": round(total_farm_p, 1) if round(total_farm_p, 1) != int(round(total_farm_p)) else int(round(total_farm_p)),
            "target_k": round(total_farm_k, 1) if round(total_farm_k, 1) != int(round(total_farm_k)) else int(round(total_farm_k)),
            "farm_target_n": round(total_farm_n, 2),
            "farm_target_p": round(total_farm_p, 2),
            "farm_target_k": round(total_farm_k, 2),
            "target_n_per_acre": round(per_acre_n, 1) if round(per_acre_n, 1) != int(round(per_acre_n)) else int(round(per_acre_n)),
            "target_p_per_acre": round(per_acre_p, 1) if round(per_acre_p, 1) != int(round(per_acre_p)) else int(round(per_acre_p)),
            "target_k_per_acre": round(per_acre_k, 1) if round(per_acre_k, 1) != int(round(per_acre_k)) else int(round(per_acre_k)),
            "exact_n": round(total_farm_n, 2),
            "exact_p": round(total_farm_p, 2),
            "exact_k": round(total_farm_k, 2),
            "is_upcoming": is_upcoming_entry,
        })

    # Find the current active or next upcoming dose whose date/deadline has not passed
    current_index = 0
    ref_dap = current_dap if current_dap is not None else split_dap
    if application_groups:
        # Prioritize upcoming doses where days_remaining >= 0 (target date has not passed yet)
        upcoming_indices = [
            i for i, g in enumerate(application_groups)
            if (g.get("days_remaining") is not None and g.get("days_remaining") >= 0)
            or (g.get("days_remaining") is None and ref_dap is not None and g["dap"] >= ref_dap)
        ]
        if upcoming_indices:
            current_index = upcoming_indices[0]
        else:
            current_index = len(application_groups) - 1

    current_group = application_groups[current_index] if application_groups else None

    # Farmer mobile view: Show 2 previous doses + 1 upcoming dose (max 3 periods)
    start_idx = max(0, current_index - 2)
    end_idx = min(len(application_groups), current_index + 1)
    farmer_periods = application_groups[start_idx:end_idx] if application_groups else []
    if not farmer_periods and application_groups:
        farmer_periods = [application_groups[-1]]

    season_display = {
        "adsali": "Adsali",
        "pre-seasonal": "Pre-Seasonal",
        "suru": "Suru",
    }.get(season_key, season_key.title())

    return {
        "ph_range": _ph_range_label(ph),
        "ph_range_mr": "८.३ - ८.९ (अति क्षारयुक्त जमीन)" if float(ph if ph is not None else 7) > 8.2 else "६.० - ८.२ (मध्यम/सामान्य जमीन)",
        "ph_sheet": "8.3-8.9" if float(ph if ph is not None else 7) > 8.2 else "6.0-8.2",
        "season": season_display,
        "season_mr": {
            "adsali": "आडसाली",
            "pre-seasonal": "पूर्वहंगामी",
            "suru": "सुरू",
        }.get(season_key, season_key),
        "season_key": season_key,
        "target_season_npk": target_season_npk,
        "max_dap": max_dap,
        "plantation_date_str": p_date.strftime("%d-%b-%Y") if p_date else None,
        "days_since_planting": days_since,
        "current_dap": current_dap,
        "login_dap": login_dap,
        "split_dap": split_dap,
        "current_group": current_group,
        "current_period": current_group,
        "farmer_periods": farmer_periods,
        "total_application_points": len(application_groups),
        "application_groups": application_groups,
        "area_acre": area_multiplier,
        "periods": application_groups,
        "cycle_status": (
            "completed" if (current_dap is not None and current_dap > max_dap)
            else "in_progress" if current_dap is not None and current_dap >= 0
            else "not_started"
        ),
    }
