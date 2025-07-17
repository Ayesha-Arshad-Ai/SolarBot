"""
solarbot.py
FastAPI backend for Pakistan-focused solar sizing.
"""

import logging
import math
import os
import json
import re
from functools import lru_cache
from typing import Any, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq

from solar_function import (
    get_lat_long,
    get_solar_data,
    PANEL_WATTAGE,
    SYSTEM_EFFICIENCY,    # 0.77 recommended in solar_function.py
    DAILY_BACKUP_HOURS,   # legacy constant; not used directly below
)

# ------------------------------------------------------------------
# User-configurable sizing knobs
# ------------------------------------------------------------------
# Hours of *average* load you want the battery to cover for each system type.
BACKUP_HOURS_GRID_PLUS_BATT = 4.0     # short outage coverage
BACKUP_HOURS_OFFGRID = 24.0           # 1 day autonomy baseline
# Minimum recommended usable battery capacity (kWh).
MIN_BATTERY_KWH = 2.0

# ------------------------------------------------------------------
# Env + logging
# ------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Groq client
# ------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")
groq = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-versatile"

# ------------------------------------------------------------------
# Appliance catalog (Pakistan‑tuned)
# NOTE: watts are "typical maximum running" (not surge) unless stated.
# ------------------------------------------------------------------
APPLIANCE_CATALOG: List[dict] = [
    {"name": "LED Bulb",                 "watt": 10,    "aliases": ["bulb", "light", "led lamp"]},
    {"name": "Tube Light",               "watt": 40,    "aliases": ["tubelight", "fluorescent", "tube light"]},
    {"name": "Ceiling Fan",              "watt": 70,    "aliases": ["fan", "ceiling fan"]},
    {"name": "Table Fan",                "watt": 60,    "aliases": ["desk fan", "table fan"]},
    {"name": "Inverter AC (1.5 Ton)",    "watt": 1200,  "aliases": ["inverter ac", "dc inverter"]},
    {"name": "Non-Inverter AC (1.5 Ton)","watt": 2000,  "aliases": ["window ac", "air conditioner"]},
    {"name": "Refrigerator (200 L)",     "watt": 725,   "aliases": ["fridge", "refrigerator"]},
    {"name": "Deep Freezer (200 L)",     "watt": 1080,  "aliases": ["freezer", "deep freezer"]},
    {"name": "Washing Machine",          "watt": 500,   "aliases": ["washer", "washing machine"]},
    {"name": "Clothes Iron",             "watt": 1400,  "aliases": ["iron", "clothes iron"]},
    {"name": "Vacuum Cleaner",           "watt": 1000,  "aliases": ["vacuum", "hoover"]},
    {"name": "Dishwasher",               "watt": 1800,  "aliases": ["dish washer", "dishwasher"]},
    {"name": "Microwave Oven",           "watt": 1000,  "aliases": ["microwave", "microwave oven"]},
    {"name": "Electric Kettle",          "watt": 1500,  "aliases": ["kettle", "electric kettle"]},
    {"name": "Rice Cooker",              "watt": 700,   "aliases": ["rice cooker"]},
    {"name": "Mixer Grinder",            "watt": 300,   "aliases": ["blender", "mixer"]},
    {"name": "Electric Stove",           "watt": 1500,  "aliases": ["cooktop", "hot plate"]},
    {"name": "Deep-Well Pump",           "watt": 400,   "aliases": ["submersible pump", "sump pump", "water motor"]},
    {"name": "Water Dispenser",          "watt": 100,   "aliases": ["water cooler", "dispenser"]},
    {"name": "Electric Water Heater",    "watt": 4500,  "aliases": ["geyser", "water heater"]},
    {"name": "LED TV (32\")",            "watt": 60,    "aliases": ["led tv", "tv"]},
    {"name": "Projector",                "watt": 170,   "aliases": ["projector"]},
    {"name": "Clock Radio",              "watt": 10,    "aliases": ["radio", "clock radio"]},
    {"name": "Set-Top Box",              "watt": 15,    "aliases": ["decoder", "stb"]},
    {"name": "Desktop Computer",         "watt": 300,   "aliases": ["pc", "desktop"]},
    {"name": "Laptop",                   "watt": 60,    "aliases": ["notebook", "laptop"]},
    {"name": "Wi-Fi Router",             "watt": 15,    "aliases": ["router", "modem"]},
    {"name": "UPS (Home Backup)",        "watt": 300,   "aliases": ["ups", "inverter"]},
    {"name": "Sewing Machine",           "watt": 100,   "aliases": ["sewing machine", "sewing"]},
    {"name": "Smartphone Charger",       "watt": 5,     "aliases": ["phone charger", "mobile charger"]},
]

def lookup_wattage(name: str) -> Optional[float]:
    key = name.strip().lower()
    for e in APPLIANCE_CATALOG:
        if e["name"].lower() == key:
            return e["watt"]
        if key in (alias.lower() for alias in e["aliases"]):
            return e["watt"]
    return None

def coerce_watt(val, default: float = 100.0) -> float:
    """
    Convert an arbitrary user/LLM wattage to a float.
    Accepts int, float, numeric string, or "500 W". Returns default on failure.
    """
    if val is None:
        return float(default)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r"[\d.]+", val)
        if m:
            try:
                return float(m.group())
            except Exception:
                pass
    return float(default)

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------
class Appliance(BaseModel):
    name: str
    wattage: Optional[float] = Field(None, ge=1, le=10000)
    quantity: int = Field(..., ge=1, le=100)
    hours: float = Field(..., ge=0.1, le=24)

class UserQuery(BaseModel):
    location: str = Field(..., min_length=3, example="Lahore, Pakistan")
    electricity_kwh_per_month: Optional[int] = Field(None, ge=100, le=5000, example=450)
    appliances: Optional[List[Appliance]] = Field(
        None,
        example=[
            {"name": "LED Bulb", "wattage": 10, "quantity": 5, "hours": 5},
            {"name": "Custom Device", "wattage": None, "quantity": 1, "hours": 2},
        ],
    )
    system_type: Literal["Grid-Tie", "Grid-Tie + Batteries", "Off-Grid"] = Field(
        ..., example="Grid-Tie + Batteries"
    )

class SystemMetric(BaseModel):
    name: str
    description: str
    value: Any
    unit: str

class SolarRecommendation(BaseModel):
    metrics: List[SystemMetric]

# ------------------------------------------------------------------
# Calculator
# ------------------------------------------------------------------
class SystemCalculator:
    @staticmethod
    def calculate(
        daily_kwh: float,
        sun_hours: float,
        backup_hours: float,
        min_battery_kwh: float = MIN_BATTERY_KWH,
    ):
        """
        Size core system components.

        daily_kwh: average energy use per day.
        sun_hours: location-specific peak sun hours.
        backup_hours: hours of average load desired from batteries.
        min_battery_kwh: enforce minimum recommended capacity.
        """
        if sun_hours <= 0:
            sun_hours = 1e-6  # avoid divide-by-zero

        # PV array sizing based on energy harvest
        system_kw = daily_kwh / (sun_hours * SYSTEM_EFFICIENCY)

        # count of panels
        panels = max(1, math.ceil((system_kw * 1000) / PANEL_WATTAGE)) if system_kw > 0 else 0

        # inverter oversize factor
        inverter = round(system_kw * 1.15, 2)

        # battery sizing: coverage of average load for backup_hours
        # avg_load_kW = daily_kwh / 24h
        avg_kw_load = daily_kwh / 24.0
        battery_kwh = avg_kw_load * backup_hours
        if backup_hours <= 0:
            battery_kwh = 0.0
        else:
            battery_kwh = max(battery_kwh, min_battery_kwh)
        battery_kwh = round(battery_kwh, 1)

        return {
            "system_kw": round(system_kw, 2),
            "panels": panels,
            "inverter": inverter,
            "battery": battery_kwh,
        }

# ------------------------------------------------------------------
# LLM wrapper
# ------------------------------------------------------------------
def generate_llm_response(prompt: str) -> str:
    resp = groq.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,   # lower = more deterministic for structure
        max_tokens=256,
        top_p=1,
    )
    return resp.choices[0].message.content.strip()

# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------
@app.post("/recommend", response_model=SolarRecommendation)
async def get_recommendation(query: UserQuery):
    try:
        # 1) Validate input exclusivity
        if bool(query.electricity_kwh_per_month) == bool(query.appliances):
            raise HTTPException(400, "Provide either electricity_kwh_per_month OR appliances list.")

        # 2) Location + sun hours
        lat, lon = get_lat_long(query.location)
        if not lat:
            raise HTTPException(400, "Location lookup failed.")
        solar = get_solar_data(query.location)
        if not solar:
            raise HTTPException(500, "Solar data unavailable.")
        sun_hours = solar["avg_daily_kwh"]

        # 3) Collect unknown appliance names
        missing: List[str] = []
        if query.appliances:
            for item in query.appliances:
                if item.wattage is None and lookup_wattage(item.name) is None:
                    missing.append(item.name)

        # 4) Resolve missing wattages via LLM (if needed)
        watt_map = {}
        if missing:
            wattage_prompt = (
                "You are an expert on Pakistani household electricity usage.\n"
                "For each appliance name in this JSON array, return a JSON object mapping the name "
                "to its TYPICAL MAXIMUM running wattage in watts (integer, no units).\n\n"
                f"Input: {json.dumps(missing)}\n\n"
                "Output JSON ONLY. No text."
            )
            wattage_reply = generate_llm_response(wattage_prompt)
            logger.debug(f"LLM wattage reply: {wattage_reply!r}")
            start = wattage_reply.find("{")
            end = wattage_reply.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    watt_map = json.loads(wattage_reply[start:end+1])
                except Exception as e:
                    logger.error(f"Failed to parse wattage JSON: {e}\n{wattage_reply}")
            else:
                logger.error(f"No JSON found in wattage reply:\n{wattage_reply}")

        # 4c) Finalize wattages (user → catalog → LLM → default)
        if query.appliances:
            lower_map = {k.lower(): v for k, v in watt_map.items()} if watt_map else {}
            for item in query.appliances:
                if item.wattage is not None:
                    item.wattage = coerce_watt(item.wattage)
                    continue
                cat_w = lookup_wattage(item.name)
                if cat_w is not None:
                    item.wattage = float(cat_w)
                    continue
                llm_val = lower_map.get(item.name.strip().lower())
                if llm_val is not None:
                    item.wattage = coerce_watt(llm_val)
                    continue
                logger.warning(f"No wattage for '{item.name}' – using 100 W default.")
                item.wattage = 100.0

        # 5) Compute daily energy
        if query.electricity_kwh_per_month:
            daily_kwh = query.electricity_kwh_per_month / 30.0
        else:
            daily_kwh = 0.0
            for item in query.appliances:
                w = coerce_watt(item.wattage)
                daily_kwh += (w * item.quantity * item.hours) / 1000.0  # Wh → kWh
                logger.debug(
                    f"Appliance '{item.name}': w={w} qty={item.quantity} h={item.hours} "
                    f"→ {(w * item.quantity * item.hours)/1000.0:.3f} kWh"
                )

        # 6) Choose battery backup target based on system type
        if query.system_type == "Grid-Tie":
            backup_hours = 0.0
        elif query.system_type == "Grid-Tie + Batteries":
            backup_hours = BACKUP_HOURS_GRID_PLUS_BATT
        else:  # Off-Grid
            backup_hours = BACKUP_HOURS_OFFGRID

        # 7) Deterministic sizing
        calc = SystemCalculator.calculate(
            daily_kwh=round(daily_kwh, 2),
            sun_hours=round(sun_hours, 2),
            backup_hours=backup_hours,
            min_battery_kwh=MIN_BATTERY_KWH,
        )
        logger.debug(f"CALC RESULTS: {calc}")

        # 8) Build metrics
        metrics: List[SystemMetric] = [
            SystemMetric(
                name="daily_consumption",
                description="Daily consumption",
                value=round(daily_kwh, 2),
                unit="kWh/day",
            ),
            SystemMetric(
                name="solar_hours",
                description="Peak sun hours",
                value=sun_hours,
                unit="h/day",
            ),
            SystemMetric(
                name="system_size",
                description="System size",
                value=calc["system_kw"],
                unit="kW",
            ),
            SystemMetric(
                name="panel_count",
                description="Panel count",
                value=calc["panels"],
                unit="panels",
            ),
            SystemMetric(
                name="inverter_size",
                description="Inverter size",
                value=calc["inverter"],
                unit="kW",
            ),
        ]
        if query.system_type in ["Grid-Tie + Batteries", "Off-Grid"]:
            metrics.append(
                SystemMetric(
                    name="battery_storage",
                    description="Battery storage",
                    value=calc["battery"],
                    unit="kWh",
                )
            )
        metrics.append(
            SystemMetric(
                name="system_type",
                description="System type",
                value=query.system_type,
                unit="",
            )
        )

        return SolarRecommendation(metrics=metrics)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /recommend")
        raise HTTPException(500, f"Server error: {e}")
