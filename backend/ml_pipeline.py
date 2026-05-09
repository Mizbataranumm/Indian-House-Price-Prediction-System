"""
ML Pipeline for the Indian House Price Prediction System.

Trains 6 regressors (Linear / Ridge / Lasso / Random Forest / XGBoost / SVR)
on the real Kaggle "Housing Prices in Metropolitan Areas of India" dataset
(Ruchi Bhatia, ~32k rows across 6 metros). Models learn log-price for
stability. Produces the analytics artifacts consumed by the frontend.
"""
from __future__ import annotations

import os
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

import dataset as ds_mod

log = logging.getLogger("ml")
FAST_TRAINING = os.environ.get(
    "FAST_TRAINING",
    "1" if os.environ.get("RENDER") else "0",
) == "1"

# ---- Feature schema (post-engineering) ----
NUMERIC_FEATURES = [
    "area_sqft", "bhk", "resale", "tier",
    "amenity_score", "sqft_per_bhk",
    "locality_psf_median", "city_psf_median",
] + ds_mod.AMENITY_COLS  # all 35 amenity flags

CATEGORICAL_FEATURES = ["city", "locality"]

# Subset of amenities surfaced in the UI (others default to dataset median).
UI_AMENITIES = [
    "amen_gym", "amen_pool", "amen_garden", "amen_security",
    "amen_power_backup", "amen_parking", "amen_lift", "amen_clubhouse",
    "amen_kids_play", "amen_intercom", "amen_ac", "amen_wifi",
]

TARGET = "price_inr"


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        # min_frequency keeps the OHE matrix bounded (groups rare localities).
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25,
                              sparse_output=True),
         CATEGORICAL_FEATURES),
    ])


def _models() -> Dict[str, Any]:
    rf_estimators = 48 if FAST_TRAINING else 120
    rf_depth = 12 if FAST_TRAINING else 16
    xgb_estimators = 120 if FAST_TRAINING else 300
    xgb_depth = 4 if FAST_TRAINING else 6
    lasso_iters = 8000 if FAST_TRAINING else 20000
    svr_c = 4.0 if FAST_TRAINING else 5.0
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression":  Ridge(alpha=1.0, random_state=42),
        "Lasso Regression":  Lasso(alpha=0.001, random_state=42,
                                   max_iter=lasso_iters),
        "Random Forest":     RandomForestRegressor(
            n_estimators=rf_estimators, max_depth=rf_depth,
            min_samples_leaf=3,
            n_jobs=-1, random_state=42),
        "XGBoost":           xgb.XGBRegressor(
            n_estimators=xgb_estimators, max_depth=xgb_depth,
            learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
            tree_method="hist"),
        "SVR":               SVR(kernel="rbf", C=svr_c, epsilon=0.05,
                                 gamma="scale"),
    }


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _remove_outliers(df: pd.DataFrame, cols: List[str], k: float = 3.0) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        q1, q3 = out[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        out = out[(out[c] >= q1 - k * iqr) & (out[c] <= q3 + k * iqr)]
    return out


@dataclass
class TrainArtifacts:
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_model_name: str = ""
    feature_importance: List[Dict[str, Any]] = field(default_factory=list)
    correlation: Dict[str, Any] = field(default_factory=dict)
    price_histogram: List[Dict[str, Any]] = field(default_factory=list)
    price_vs_area: List[Dict[str, Any]] = field(default_factory=list)
    residuals: List[Dict[str, Any]] = field(default_factory=list)
    learning_curve: List[Dict[str, Any]] = field(default_factory=list)
    city_stats: List[Dict[str, Any]] = field(default_factory=list)
    feature_means: Dict[str, float] = field(default_factory=dict)
    train_size: int = 0
    test_size: int = 0
    trained_at: float = 0.0


class MLPipeline:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.df: pd.DataFrame | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.fitted: Dict[str, Pipeline] = {}
        self.test_preds: Dict[str, np.ndarray] = {}
        self.artifacts = TrainArtifacts()
        self._shap_explainer = None
        self._shap_feature_names: List[str] = []
        self.training_started_at = 0.0
        self.training_in_progress = False
        self.training_current_model = ""
        self.training_models_total = 0
        self.training_models_done = 0
        # Per-locality and city ₹/sqft medians used to fill engineered features
        # for new (user-input) rows.
        self._loc_psf: Dict[tuple, float] = {}
        self._city_psf: Dict[str, float] = {}

    # ---------- public API ----------
    def ensure_trained(self) -> None:
        with self._lock:
            if self.fitted:
                return
            self._train()

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.ensure_trained()
        X = self._payload_to_df(payload)
        best = self.fitted[self.artifacts.best_model_name]
        price = float(np.expm1(best.predict(X)[0]))

        rf = self.fitted.get("Random Forest")
        if rf is not None:
            preproc = rf.named_steps["pre"]
            estim = rf.named_steps["model"]
            Xt = preproc.transform(X)
            tree_log = np.array([t.predict(Xt)[0] for t in estim.estimators_])
            half_width = float(np.std(tree_log)) * 1.645  # 90% interval
            lo = float(price * np.exp(-half_width))
            hi = float(price * np.exp(half_width))
        else:
            lo, hi = price * 0.92, price * 1.08

        all_predictions = {n: float(np.expm1(m.predict(X)[0]))
                           for n, m in self.fitted.items()}
        # Clamp pathological predictions for the leaderboard display
        # (Linear/Lasso can be numerically unstable on high-cardinality OHE).
        for k, v in list(all_predictions.items()):
            if not np.isfinite(v) or v < 1e5 or v > 1e10:
                all_predictions[k] = None
        return {
            "predicted_price": price,
            "confidence_low": lo,
            "confidence_high": hi,
            "price_per_sqft": price / max(1, payload["area_sqft"]),
            "best_model": self.artifacts.best_model_name,
            "all_predictions": all_predictions,
        }

    def shap_explain(self, payload: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        self.ensure_trained()
        import shap
        if self._shap_explainer is None:
            xgb_pipe = self.fitted["XGBoost"]
            self._shap_preproc = xgb_pipe.named_steps["pre"]
            self._shap_feature_names = list(self._shap_preproc.get_feature_names_out())
            self._shap_explainer = shap.TreeExplainer(xgb_pipe.named_steps["model"])

        X = self._payload_to_df(payload)
        Xt = self._shap_preproc.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        sv = self._shap_explainer.shap_values(Xt)
        base_log = float(self._shap_explainer.expected_value)
        pred_price = float(np.expm1(
            self.fitted[self.artifacts.best_model_name].predict(X)[0]))
        items = sorted(
            [{"feature": f, "value": float(v) * pred_price}
             for f, v in zip(self._shap_feature_names, sv[0])],
            key=lambda d: abs(d["value"]), reverse=True,
        )[:top_k]
        return {"base_value": float(np.expm1(base_log)), "contributions": items}

    def whatif(self, base_payload: Dict[str, Any], feature: str,
               values: List[Any]) -> List[Dict[str, Any]]:
        self.ensure_trained()
        out, best = [], self.fitted[self.artifacts.best_model_name]
        for v in values:
            p = dict(base_payload)
            p[feature] = v
            pred = float(np.expm1(best.predict(self._payload_to_df(p))[0]))
            out.append({"value": v, "predicted_price": pred})
        return out

    # ---------- internals ----------
    def _payload_to_df(self, p: Dict[str, Any]) -> pd.DataFrame:
        """Build a single-row DF that matches NUMERIC + CATEGORICAL features.
        Engineered fields (locality_psf_median, city_psf_median, sqft_per_bhk,
        amenity_score) are computed here for new inputs."""
        city = p.get("city")
        locality = p.get("locality") or city
        # locality_psf_median fallback: city median, then global median
        loc_psf = self._loc_psf.get(
            (city, locality),
            self._city_psf.get(city, self.artifacts.feature_means.get(
                "locality_psf_median", 8000.0)),
        )
        city_psf = self._city_psf.get(
            city, self.artifacts.feature_means.get("city_psf_median", 8000.0))
        amen_total = sum(int(p.get(a, self.artifacts.feature_means.get(a, 0)))
                         for a in ds_mod.AMENITY_COLS)
        area = float(p.get("area_sqft", 1)) or 1.0
        bhk = max(1, int(p.get("bhk", 1)))

        row = {
            "area_sqft": area, "bhk": bhk,
            "resale": int(p.get("resale", 1)),
            "tier": int(p.get("tier", 1)),
            "amenity_score": float(amen_total),
            "sqft_per_bhk": area / bhk,
            "locality_psf_median": float(loc_psf),
            "city_psf_median": float(city_psf),
            "city": city, "locality": locality,
        }
        for a in ds_mod.AMENITY_COLS:
            row[a] = int(p.get(a, self.artifacts.feature_means.get(a, 0)))
        return pd.DataFrame([row])

    def _train(self) -> None:
        self.training_started_at = time.time()
        self.training_in_progress = True
        self.training_current_model = "Preparing dataset"
        self.training_models_done = 0
        df = ds_mod.build_canonical(force=False)
        df = _remove_outliers(df, ["price_per_sqft"], k=2.5)
        df = _remove_outliers(df, ["area_sqft"], k=3.0)
        self.df = df

        # Cache lookup tables for engineered features at predict-time
        self._loc_psf = (
            df.groupby(["city", "locality"])["price_per_sqft"].median().to_dict()
        )
        self._city_psf = (
            df.groupby("city")["price_per_sqft"].median().to_dict()
        )

        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        y = df[TARGET].values

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = Xtr, Xte, ytr, yte

        # Train all models on log(price); evaluate on raw INR.
        ytr_log = np.log1p(ytr)
        metrics: Dict[str, Dict[str, float]] = {}
        # SVR is O(n^2) so subsample heavily for fitting & CV
        rng = np.random.default_rng(0)
        svr_idx = rng.choice(
            len(Xtr),
            size=min(1500 if FAST_TRAINING else 3000, len(Xtr)),
            replace=False,
        )
        # For CV we use a sub-sample so total time stays bounded
        cv_idx = rng.choice(
            len(X),
            size=min(4000 if FAST_TRAINING else 8000, len(X)),
            replace=False,
        )
        cv_X_full = X.iloc[cv_idx]
        cv_y_full = np.log1p(y[cv_idx])

        models = _models()
        self.training_models_total = len(models)
        for name, est in models.items():
            self.training_current_model = name
            t0 = time.time()
            pipe = Pipeline([("pre", _build_preprocessor()), ("model", est)])
            if name == "SVR":
                pipe.fit(Xtr.iloc[svr_idx], ytr_log[svr_idx])
            else:
                pipe.fit(Xtr, ytr_log)
            pred = np.clip(np.expm1(pipe.predict(Xte)), 1e5, 5e8)
            pred = np.nan_to_num(pred, nan=1e6, posinf=5e8, neginf=1e5)
            try:
                if name == "SVR":
                    cv_X, cv_y = X.iloc[svr_idx], ytr_log[svr_idx]
                else:
                    cv_X, cv_y = cv_X_full, cv_y_full
                cv = cross_val_score(pipe, cv_X, cv_y, cv=3, scoring="r2", n_jobs=1)
                cv_mean = float(cv.mean())
                # Clamp pathological CV scores caused by collinear OHE in folds.
                if not np.isfinite(cv_mean) or abs(cv_mean) > 5:
                    cv_mean = float("nan")
            except Exception as e:
                log.warning("CV failed for %s: %s", name, e)
                cv_mean = float("nan")
            metrics[name] = {
                "r2":   float(r2_score(yte, pred)),
                "mae":  float(mean_absolute_error(yte, pred)),
                "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                "mape": _mape(yte, pred),
                "cv_r2": cv_mean,
                "train_time_sec": round(time.time() - t0, 3),
            }
            self.fitted[name] = pipe
            self.test_preds[name] = pred
            self.training_models_done += 1
            log.info("Trained %s in %.1fs r2=%.3f", name,
                     metrics[name]["train_time_sec"], metrics[name]["r2"])

        best = max(metrics, key=lambda k: metrics[k]["r2"])
        self.artifacts.metrics = metrics
        self.artifacts.best_model_name = best
        self.artifacts.train_size = len(Xtr)
        self.artifacts.test_size = len(Xte)
        self.artifacts.trained_at = time.time()

        # ---- analytics artifacts ----
        imp_pipe = self.fitted.get("XGBoost") or self.fitted[best]
        pre = imp_pipe.named_steps["pre"]
        names = list(pre.get_feature_names_out())
        model = imp_pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        else:
            vals = np.abs(getattr(model, "coef_", np.zeros(len(names))))
        order = np.argsort(vals)[::-1][:15]
        self.artifacts.feature_importance = [
            {"feature": names[i], "importance": float(vals[i])} for i in order
        ]

        # correlation (subset of meaningful numeric features only)
        corr_cols = [
            "area_sqft", "bhk", "resale", "amenity_score", "sqft_per_bhk",
            "locality_psf_median", "city_psf_median",
            "amen_gym", "amen_pool", "amen_security", "amen_lift", "amen_parking",
            TARGET,
        ]
        corr = df[corr_cols].corr(numeric_only=True).round(3)
        self.artifacts.correlation = {
            "labels": list(corr.columns),
            "matrix": corr.values.tolist(),
        }

        hist, edges = np.histogram(
            df[TARGET].clip(upper=df[TARGET].quantile(0.99)) / 1e5, bins=30)
        self.artifacts.price_histogram = [
            {"bin": float((edges[i] + edges[i + 1]) / 2), "count": int(hist[i])}
            for i in range(len(hist))
        ]

        s = df[["area_sqft", TARGET, "city"]].sample(min(800, len(df)), random_state=1)
        self.artifacts.price_vs_area = [
            {"area": int(r.area_sqft), "price": int(getattr(r, TARGET)), "city": r.city}
            for r in s.itertuples()
        ]

        pred_best = self.test_preds[best]
        resid = yte - pred_best
        idx = np.random.default_rng(0).choice(
            len(yte), size=min(400, len(yte)), replace=False)
        self.artifacts.residuals = [
            {"predicted": float(pred_best[i]), "residual": float(resid[i])}
            for i in idx
        ]

        try:
            # Use a sub-sample to keep learning_curve fast
            lc_idx = np.random.default_rng(1).choice(
                len(X),
                size=min(2500 if FAST_TRAINING else 6000, len(X)),
                replace=False,
            )
            sizes, train_scores, val_scores = learning_curve(
                self.fitted[best], X.iloc[lc_idx], np.log1p(y[lc_idx]), cv=3,
                train_sizes=np.linspace(0.25, 1.0, 4 if FAST_TRAINING else 5),
                n_jobs=1, scoring="r2")
            self.artifacts.learning_curve = [
                {"size": int(s),
                 "train_r2": float(train_scores[i].mean()),
                 "val_r2": float(val_scores[i].mean())}
                for i, s in enumerate(sizes)
            ]
        except Exception as e:
            log.warning("learning_curve failed: %s", e)

        agg = df.groupby("city").agg(
            mean_price=(TARGET, "mean"),
            median_price=(TARGET, "median"),
            mean_psf=("price_per_sqft", "mean"),
            count=(TARGET, "count"),
        ).reset_index()
        self.artifacts.city_stats = [
            {"city": r.city, "tier": 1,
             "mean_price": float(r.mean_price),
             "median_price": float(r.median_price),
             "mean_psf": float(r.mean_psf), "count": int(r.count)}
            for r in agg.itertuples()
        ]

        self.artifacts.feature_means = {
            f: float(df[f].mean()) for f in NUMERIC_FEATURES
            if f in df.columns
        }
        self.training_current_model = ""
        self.training_in_progress = False

    def to_dict(self) -> Dict[str, Any]:
        a = self.artifacts
        # Replace NaN/inf with None for JSON safety.
        def _safe(v):
            if isinstance(v, float) and not np.isfinite(v):
                return None
            return v
        metrics_safe = {
            k: {kk: _safe(vv) for kk, vv in m.items()}
            for k, m in a.metrics.items()
        }
        return {
            "metrics": metrics_safe,
            "best_model": a.best_model_name,
            "feature_importance": a.feature_importance,
            "correlation": a.correlation,
            "price_histogram": a.price_histogram,
            "price_vs_area": a.price_vs_area,
            "residuals": a.residuals,
            "learning_curve": a.learning_curve,
            "city_stats": a.city_stats,
            "feature_means": a.feature_means,
            "train_size": a.train_size,
            "test_size": a.test_size,
            "trained_at": a.trained_at,
            "n_rows": int(len(self.df)) if self.df is not None else 0,
            "features_numeric": NUMERIC_FEATURES,
            "features_categorical": CATEGORICAL_FEATURES,
            "ui_amenities": UI_AMENITIES,
            "dataset_source": "Kaggle: Housing Prices in Metropolitan Areas of India (Ruchi Bhatia)",
        }


pipeline = MLPipeline()
