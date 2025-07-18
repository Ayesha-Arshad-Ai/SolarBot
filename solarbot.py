"""
solarbot.py
FastAPI backend for Pakistan-focused solar sizing.

This version:
- Extracts wattage from appliance *name label* (e.g., "AC (Window/Split) (1500W avg)").
- Uses user-provided wattage field if given.
- Falls back to catalog lookup, then LLM guess (Pakistan), then 100 W default.
"""

import logging
import math
import os
import json
import re
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
    SYSTEM_EFFICIENCY,    # e.g., 0.77 in solar_function.py
)

# ------------------------------------------------------------------
# User-configurable sizing knobs
# ------------------------------------------------------------------
BACKUP_HOURS_GRID_PLUS_BATT = 4.0     # Grid-Tie + Batteries backup coverage
BACKUP_HOURS_OFFGRID = 24.0           # Off-Grid 1 day autonomy baseline
MIN_BATTERY_KWH = 2.0                 # enforce a small minimum

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
# Appliance catalog (PASTE FULL LIST HERE)
# watts = typical max running (not surge)
# ------------------------------------------------------------------
APPLIANCE_CATALOG: List[dict] = [
    {"name": "LED Bulb",                 "watt": 10,    "aliases": ["bulb", "light", "led lamp"]},
    # TODO: paste remaining catalog rows you finalized earlier
]

# ------------------------------------------------------------------
# Helper: catalog lookup
# ------------------------------------------------------------------
def lookup_wattage(name: str) -> Optional[float]:
    key = name.strip().lower()
    for e in APPLIANCE_CATALOG:
        if e["name"].lower() == key:
            return e["watt"]
        for alias in e["aliases"]:
            if key == alias.lower():
                return e["watt"]
    return None

# ------------------------------------------------------------------
# Helper: parse watts from label text
# Looks for first numeric value followed by optional W/w within parentheses
# Works on: "AC (Window/Split) (1500W avg)", "LED Bulb (10W)", "Fan 75 w"
# Returns float or None
# ------------------------------------------------------------------
WATT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kK]?)[wW]")

def parse_watt_from_label(label: str) -> Optional[float]:
    if not label:
        return None
    m = WATT_RE.search(label)
    if not m:
        return None
    num = float(m.group(1))
    kilo = m.group(2)
    if kilo:  # 'kW' or 'Kw' etc.
        num *= 1000.0
    return num

# ------------------------------------------------------------------
# Helper: coerce user/LLM watt to float
# Accepts numeric, string like "500", "500W", "0.5kW"
# ------------------------------------------------------------------
def coerce_watt(val, default: float = 100.0) -> float:
    if val is None:
        return float(default)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Reuse parse_watt_from_label for convenience
        parsed = parse_watt_from_label(val)
        if parsed is not None:
            return parsed
        # fallback: just digits
        m = re.search(r"[\d.]+", val)
        if m:
            try:
                return float(m.group())
            except Exception:
                pass
    return float(default)

# ------------------------------------------------------------------
# Helper: LLM single-watt guess (Pakistan context)
# Returns float or None
# ------------------------------------------------------------------
def llm_guess_single_watt(name: str) -> Optional[float]:
    prompt = (
        "You are an expert in Pakistani household energy usage. "
        "Give the TYPICAL MAXIMUM RUNNING wattage for this appliance: "
        f"{name!r}. Reply with ONLY an integer number of watts (no units, no text)."
    )
    try:
        reply = groq.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=16,
            top_p=1,
        )
        text = reply.choices[0].message.content.strip()
        logger.debug(f"LLM watt guess raw reply for '{name}': {text!r}")
        # parse integer
        m = re.search(r"\d+", text)
        if m:
            return float(m.group())
    except Exception as e:
        logger.error(f"LLM single watt guess failed for '{name}': {e}")
    return None

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
    wattage: Optional[float] = Field(
        None,
        ge=1,
        le=10000,
        description="If omitted, backend will parse from name string or guess.",
    )
    quantity: int = Field(..., ge=1, le=100)
    hours: float = Field(..., ge=0.1, le=24)

class UserQuery(BaseModel):
    location: str = Field(..., min_length=3, example="Lahore, Pakistan")
    electricity_kwh_per_month: Optional[int] = Field(
        None, ge=100, le=5000, example=450,
        description="Provide this OR an appliances list (not both)."
    )
    appliances: Optional[List[Appliance]] = Field(
        None,
        example=[
            {"name": "AC (Window/Split) (1500W avg)", "quantity": 1, "hours": 4},
            {"name": "Light Bulb (LED) (10W avg)", "quantity": 10, "hours": 5}
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
        if sun_hours <= 0:
            sun_hours = 1e-6  # guard
        system_kw = daily_kwh / (sun_hours * SYSTEM_EFFICIENCY)
        panels = max(1, math.ceil((system_kw * 1000) / PANEL_WATTAGE)) if system_kw > 0 else 0
        inverter = round(system_kw * 1.15, 2)
        # battery sizing
        avg_kw_load = daily_kwh / 24.0
        if backup_hours <= 0:
            battery_kwh = 0.0
        else:
            battery_kwh = max(avg_kw_load * backup_hours, min_battery_kwh)
        battery_kwh = round(battery_kwh, 1)
        return {
            "system_kw": round(system_kw, 2),
            "panels": panels,
            "inverter": inverter,
            "battery": battery_kwh,
        }

# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------
@app.post("/recommend", response_model=SolarRecommendation)
async def get_recommendation(query: UserQuery):
    try:
        # 1) Validate mutual exclusivity
        if bool(query.electricity_kwh_per_month) == bool(query.appliances):
            raise HTTPException(
                400,
                "Provide either electricity_kwh_per_month OR appliances list (not both).",
            )

        # 2) Location + sun hours
        lat, lon = get_lat_long(query.location)
        if not lat:
            raise HTTPException(400, "Location lookup failed.")
        solar = get_solar_data(query.location)
        if not solar:
            raise HTTPException(500, "Solar data unavailable.")
        sun_hours = solar["avg_daily_kwh"]

        # 3) Daily consumption
        if query.electricity_kwh_per_month is not None:
            # monthly bill path
            daily_kwh = query.electricity_kwh_per_month / 30.0
        else:
            # appliance path
            daily_kwh = 0.0
            for ap in query.appliances:
                # Resolve wattage: label -> field -> catalog -> LLM -> default
                w = parse_watt_from_label(ap.name)
                if w is None and ap.wattage is not None:
                    w = coerce_watt(ap.wattage)
                if w is None:
                    cat_w = lookup_wattage(ap.name)
                    if cat_w is not None:
                        w = float(cat_w)
                if w is None:
                    w = llm_guess_single_watt(ap.name)
                if w is None:
                    logger.warning(f"No wattage resolved for '{ap.name}' → 100W default.")
                    w = 100.0
                ap.wattage = w
                kwh = (w * ap.quantity * ap.hours) / 1000.0
                daily_kwh += kwh
                logger.debug(
                    f"Appliance '{ap.name}' resolved_w={w} qty={ap.quantity} h={ap.hours} → {kwh:.3f} kWh"
                )

        # 4) Backup hours by system type
        if query.system_type == "Grid-Tie":
            backup_hours = 0.0
        elif query.system_type == "Grid-Tie + Batteries":
            backup_hours = BACKUP_HOURS_GRID_PLUS_BATT
        else:  # Off-Grid
            backup_hours = BACKUP_HOURS_OFFGRID

        # 5) Size system
        calc = SystemCalculator.calculate(
            daily_kwh=round(daily_kwh, 2),
            sun_hours=round(sun_hours, 2),
            backup_hours=backup_hours,
            min_battery_kwh=MIN_BATTERY_KWH,
        )
        logger.debug(f"CALC: {calc}")

         # 6) Build metrics
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
            # ←–– NEW METRIC: show wattage per panel
            SystemMetric(
                name="panel_size",
                description="Rated wattage per panel",
                value=PANEL_WATTAGE,
                unit="W",
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
