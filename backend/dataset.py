"""
Real Indian housing dataset loader.

Source: "Housing Prices in Metropolitan Areas of India" — original Kaggle
dataset by Ruchi Bhatia (Ruchi798). The six CSVs (Mumbai, Delhi, Bangalore,
Chennai, Hyderabad, Kolkata) are bundled in /app/data/raw/. Combined we have
~32,000 listings with Price (INR), Area (sqft), Location, BHK, Resale flag
and 35 binary amenity flags. The "9" sentinel used in the source for
"unknown amenity" is converted to NaN and median-imputed.

This loader produces the canonical /app/data/indian_housing.csv consumed by
the ML pipeline.
"""
from __future__ import annotations

import os
import logging
from typing import List
import numpy as np
import pandas as pd

log = logging.getLogger("dataset")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "..", "data", "raw")
CANONICAL_PATH = os.path.join(BASE_DIR, "..", "data", "indian_housing.csv")

CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata"]

# 35 binary amenity columns from the Kaggle dataset (after Price/Area/Location/
# BHK/Resale). We rename to snake_case for downstream use.
AMENITY_COLS_RENAME = {
    "MaintenanceStaff":   "amen_maintenance_staff",
    "Gymnasium":          "amen_gym",
    "SwimmingPool":       "amen_pool",
    "LandscapedGardens":  "amen_garden",
    "JoggingTrack":       "amen_jogging",
    "RainWaterHarvesting":"amen_rainwater",
    "IndoorGames":        "amen_indoor_games",
    "ShoppingMall":       "amen_mall",
    "Intercom":           "amen_intercom",
    "SportsFacility":     "amen_sports",
    "ATM":                "amen_atm",
    "ClubHouse":          "amen_clubhouse",
    "School":             "amen_school",
    "24X7Security":       "amen_security",
    "PowerBackup":        "amen_power_backup",
    "CarParking":         "amen_parking",
    "StaffQuarter":       "amen_staff_quarter",
    "Cafeteria":          "amen_cafeteria",
    "MultipurposeRoom":   "amen_multipurpose",
    "Hospital":           "amen_hospital",
    "WashingMachine":     "amen_washing_machine",
    "Gasconnection":      "amen_gas",
    "AC":                 "amen_ac",
    "Wifi":               "amen_wifi",
    "Children'splayarea": "amen_kids_play",
    "LiftAvailable":      "amen_lift",
    "BED":                "amen_bed",
    "VaastuCompliant":    "amen_vaastu",
    "Microwave":          "amen_microwave",
    "GolfCourse":         "amen_golf",
    "TV":                 "amen_tv",
    "DiningTable":        "amen_dining_table",
    "Sofa":               "amen_sofa",
    "Wardrobe":           "amen_wardrobe",
    "Refrigerator":       "amen_refrigerator",
}

AMENITY_COLS: List[str] = list(AMENITY_COLS_RENAME.values())


def _load_one(city: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{city}.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Price": "price_inr",
        "Area": "area_sqft",
        "Location": "locality",
        "No. of Bedrooms": "bhk",
        "Resale": "resale",
        **AMENITY_COLS_RENAME,
    })
    df["city"] = city
    return df


def build_canonical(force: bool = False) -> pd.DataFrame:
    """Load all 6 city CSVs, clean, combine and persist canonical file."""
    if not force and os.path.exists(CANONICAL_PATH):
        return pd.read_csv(CANONICAL_PATH)

    frames = [_load_one(c) for c in CITIES]
    df = pd.concat(frames, ignore_index=True)

    # 1. Sentinel "9" in amenity flags means unknown -> NaN
    for c in AMENITY_COLS:
        df[c] = df[c].replace(9, np.nan)

    # 2. Median impute amenity columns (per city for fidelity)
    df[AMENITY_COLS] = (
        df.groupby("city")[AMENITY_COLS]
          .transform(lambda s: s.fillna(s.median()))
    )
    # Any leftover -> 0
    df[AMENITY_COLS] = df[AMENITY_COLS].fillna(0).astype(int)

    # 3. Clean string columns
    df["locality"] = df["locality"].astype(str).str.strip()
    df = df.dropna(subset=["price_inr", "area_sqft", "bhk", "locality"])

    # 4. Basic sanity filters
    df = df[(df["area_sqft"] >= 200) & (df["area_sqft"] <= 15000)]
    df = df[(df["price_inr"] >= 1e6) & (df["price_inr"] <= 5e8)]
    df = df[df["bhk"].between(1, 8)]
    # Filter implausible price per sqft (handles dirty rows in the source data)
    psf = df["price_inr"] / df["area_sqft"]
    df = df[(psf >= 500) & (psf <= 60000)]

    # 5. Engineered features
    df["price_per_sqft"] = df["price_inr"] / df["area_sqft"]
    # Tier (all six are Tier-1 metros)
    df["tier"] = 1
    # Amenity score = sum of meaningful amenity flags
    df["amenity_score"] = df[AMENITY_COLS].sum(axis=1)
    # Ratio of bedrooms to area (sqft per BHK)
    df["sqft_per_bhk"] = df["area_sqft"] / df["bhk"]
    # Locality median price per sqft (target encoding to capture micro-market)
    loc_med = (
        df.groupby(["city", "locality"])["price_per_sqft"]
        .transform("median")
    )
    df["locality_psf_median"] = loc_med
    # City median ₹/sqft
    df["city_psf_median"] = df.groupby("city")["price_per_sqft"].transform("median")

    os.makedirs(os.path.dirname(CANONICAL_PATH), exist_ok=True)
    df.to_csv(CANONICAL_PATH, index=False)
    log.info("Wrote canonical dataset: %s rows=%d", CANONICAL_PATH, len(df))
    return df


def load() -> pd.DataFrame:
    return build_canonical(force=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_canonical(force=True)
    print(df.shape, df["price_inr"].describe()[["min", "50%", "max"]])
