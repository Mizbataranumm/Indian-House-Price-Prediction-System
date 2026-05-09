"""FastAPI server for the Indian House Price Prediction System."""
from __future__ import annotations

import os
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, APIRouter, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

import ml_pipeline
import dataset as ds_mod
import pdf_report

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("server")

MONGO_URL = os.environ.get("MONGO_URL")
DB_NAME = os.environ.get("DB_NAME")
client = None
db = None
_memory_predictions: List[Dict[str, Any]] = []

if MONGO_URL and DB_NAME:
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
else:
    log.warning(
        "MongoDB env vars are not fully configured. "
        "Prediction history will use in-memory storage."
    )

app = FastAPI(title="Indian House Price Prediction API")
api = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------
class PropertyInput(BaseModel):
    """Schema aligned with the real Kaggle Indian housing dataset."""
    model_config = ConfigDict(extra="allow")

    city: str = "Mumbai"
    locality: str = "Andheri West"
    bhk: int = Field(2, ge=1, le=10)
    area_sqft: float = Field(900, ge=150, le=20000)
    resale: int = Field(1, ge=0, le=1)
    tier: int = Field(1, ge=1, le=3)
    # Surfaced amenities (others default to dataset median at predict time)
    amen_gym: int = 0
    amen_pool: int = 0
    amen_garden: int = 0
    amen_security: int = 1
    amen_power_backup: int = 1
    amen_parking: int = 1
    amen_lift: int = 1
    amen_clubhouse: int = 0
    amen_kids_play: int = 0
    amen_intercom: int = 1
    amen_ac: int = 0
    amen_wifi: int = 0


class WhatIfBody(BaseModel):
    base: PropertyInput
    feature: str
    values: List[float]


class CompareBody(BaseModel):
    properties: List[PropertyInput]


class EmiBody(BaseModel):
    loan_amount: float = Field(..., gt=0)
    rate: float = Field(..., gt=0, le=30)
    tenure_years: int = Field(..., gt=0, le=40)


class RoiBody(BaseModel):
    purchase_price: float = Field(..., gt=0)
    monthly_rent: float = Field(0, ge=0)
    annual_appreciation: float = Field(6.0)
    holding_years: int = Field(5, ge=1, le=40)
    annual_expense: float = Field(0, ge=0)


class PdfBody(BaseModel):
    input: PropertyInput
    emi: Optional[EmiBody] = None


# ---------- Background warmup ----------
def _warmup():
    try:
        log.info("Starting background ML warmup")
        ml_pipeline.pipeline.ensure_trained()
        log.info("ML pipeline trained: best=%s",
                 ml_pipeline.pipeline.artifacts.best_model_name)
    except Exception as e:
        log.exception("Training failed: %s", e)


async def _save_prediction(rec: Dict[str, Any]) -> None:
    if db is not None:
        await db.predictions.insert_one(rec.copy())
        return
    _memory_predictions.append(rec.copy())


async def _list_predictions(session_id: str, limit: int) -> List[Dict[str, Any]]:
    if db is not None:
        cur = db.predictions.find(
            {"session_id": session_id}, {"_id": 0}
        ).sort("created_at", -1).limit(limit)
        return [r async for r in cur]

    items = [r for r in reversed(_memory_predictions)
             if r["session_id"] == session_id]
    return items[:limit]


async def _clear_predictions(session_id: str) -> int:
    if db is not None:
        res = await db.predictions.delete_many({"session_id": session_id})
        return int(res.deleted_count)

    before = len(_memory_predictions)
    _memory_predictions[:] = [
        r for r in _memory_predictions if r["session_id"] != session_id
    ]
    return before - len(_memory_predictions)


@app.on_event("startup")
async def on_start():
    threading.Thread(target=_warmup, daemon=True).start()


# ---------- Routes ----------
@app.get("/")
async def app_root():
    return {
        "status": "ok",
        "service": "indian-house-price-prediction",
        "docs_hint": "/api/",
    }


@app.head("/")
async def app_root_head():
    return


@api.get("/")
async def root():
    return {"status": "ok", "service": "indian-house-price-prediction"}


@api.get("/status")
async def status():
    p = ml_pipeline.pipeline
    fully_trained = bool(p.fitted) and bool(p.artifacts.best_model_name)
    return {
        "trained": fully_trained,
        "best_model": p.artifacts.best_model_name or None,
        "n_rows": int(len(p.df)) if p.df is not None else 0,
        "dataset": "Kaggle - Housing Prices in Metropolitan Areas of India",
        "storage": "mongodb" if db is not None else "memory",
    }


@api.get("/dataset/summary")
async def dataset_summary():
    ml_pipeline.pipeline.ensure_trained()
    p = ml_pipeline.pipeline
    df = p.df
    cities = sorted(df["city"].unique().tolist())
    # Top localities per city for the Predict dropdown
    top_localities: Dict[str, List[str]] = {}
    for c in cities:
        s = df[df["city"] == c]["locality"].value_counts().head(20)
        top_localities[c] = s.index.tolist()
    return {
        "cities": cities,
        "localities_by_city": top_localities,
        "ui_amenities": ml_pipeline.UI_AMENITIES,
        "amenity_labels": _amenity_label_map(),
        "n_rows": int(len(df)),
        "price_min": float(df["price_inr"].min()),
        "price_max": float(df["price_inr"].max()),
        "price_mean": float(df["price_inr"].mean()),
        "price_median": float(df["price_inr"].median()),
        "feature_means": p.artifacts.feature_means,
        "dataset_source": ("Kaggle - Housing Prices in Metropolitan Areas of "
                           "India (Ruchi Bhatia)"),
    }


def _amenity_label_map() -> Dict[str, str]:
    rev = {v: k for k, v in ds_mod.AMENITY_COLS_RENAME.items()}
    pretty = {
        "amen_gym": "Gym", "amen_pool": "Swimming pool",
        "amen_garden": "Garden", "amen_security": "24x7 Security",
        "amen_power_backup": "Power backup", "amen_parking": "Car parking",
        "amen_lift": "Lift", "amen_clubhouse": "Clubhouse",
        "amen_kids_play": "Kids play area", "amen_intercom": "Intercom",
        "amen_ac": "AC", "amen_wifi": "Wi-Fi",
    }
    for k, v in rev.items():
        pretty.setdefault(k, v)
    return pretty


@api.get("/analytics")
async def analytics():
    ml_pipeline.pipeline.ensure_trained()
    return ml_pipeline.pipeline.to_dict()


@api.post("/predict")
async def predict(body: PropertyInput, save: bool = Query(True),
                  session_id: str = Query("anonymous")):
    out = ml_pipeline.pipeline.predict(body.model_dump())
    rec = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "input": body.model_dump(),
        "output": out,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if save:
        await _save_prediction(rec)
    return rec


@api.post("/predict/shap")
async def predict_shap(body: PropertyInput):
    ml_pipeline.pipeline.ensure_trained()
    return ml_pipeline.pipeline.shap_explain(body.model_dump())


@api.post("/predict/whatif")
async def predict_whatif(body: WhatIfBody):
    return ml_pipeline.pipeline.whatif(
        body.base.model_dump(), body.feature, list(body.values))


@api.post("/predict/compare")
async def predict_compare(body: CompareBody):
    if not body.properties:
        raise HTTPException(400, "Provide at least one property")
    out = []
    for p in body.properties:
        r = ml_pipeline.pipeline.predict(p.model_dump())
        out.append({"input": p.model_dump(), "output": r})
    return {"results": out}


@api.get("/history")
async def history(session_id: str = "anonymous", limit: int = 50):
    return {"items": await _list_predictions(session_id, limit)}


@api.delete("/history")
async def clear_history(session_id: str = "anonymous"):
    return {"deleted": await _clear_predictions(session_id)}


@api.post("/emi")
async def emi(body: EmiBody):
    P, r, n = body.loan_amount, body.rate / 12 / 100, body.tenure_years * 12
    emi_val = P / n if r == 0 else P * r * (1 + r) ** n / ((1 + r) ** n - 1)
    total = emi_val * n
    schedule = []
    bal = P
    for m in range(1, n + 1):
        interest = bal * r
        principal = emi_val - interest
        bal = max(0, bal - principal)
        if m % 12 == 0 or m == n:
            schedule.append({
                "month": m, "principal": principal, "interest": interest,
                "balance": bal,
            })
    return {
        "loan_amount": P, "rate": body.rate, "tenure_years": body.tenure_years,
        "emi": emi_val, "total_payment": total, "total_interest": total - P,
        "schedule_yearly": schedule,
    }


@api.post("/roi")
async def roi(body: RoiBody):
    years = body.holding_years
    appreciated = body.purchase_price * (1 + body.annual_appreciation / 100) ** years
    rental = body.monthly_rent * 12 * years
    expense = body.annual_expense * years
    profit = (appreciated - body.purchase_price) + rental - expense
    cagr = ((appreciated + rental) / body.purchase_price) ** (1 / years) - 1
    return {
        "future_value": appreciated, "rental_income": rental,
        "total_expense": expense, "net_profit": profit,
        "roi_pct": profit / body.purchase_price * 100,
        "cagr_pct": cagr * 100,
    }


@api.post("/report/pdf")
async def report_pdf(body: PdfBody):
    pred = ml_pipeline.pipeline.predict(body.input.model_dump())
    shap = ml_pipeline.pipeline.shap_explain(body.input.model_dump())
    emi_data = None
    if body.emi:
        e = body.emi
        P, r, n = e.loan_amount, e.rate / 12 / 100, e.tenure_years * 12
        emi_val = P / n if r == 0 else P * r * (1 + r) ** n / ((1 + r) ** n - 1)
        total = emi_val * n
        emi_data = {
            "loan_amount": P, "rate": e.rate, "tenure_years": e.tenure_years,
            "emi": emi_val, "total_payment": total, "total_interest": total - P,
        }
    pdf_bytes = pdf_report.build_report(
        body.input.model_dump(), pred, shap, emi_data)
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=house_price_report.pdf"},
    )


app.include_router(api)
