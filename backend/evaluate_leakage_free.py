"""
evaluate_leakage_free.py
Re-evaluate the 6 Indian House Price Prediction models without target leakage.

Setup & Protocol:
- Set FAST_TRAINING=0 BEFORE importing ml_pipeline
- Reuse preprocessor, outlier cleaning, models, and feature schemas
- Build leakage-free features using 5-fold OOF medians on training data only
- Run experiments A through I, save results as CSV in backend/results/,
  and print a consolidated plain-text summary.
"""
from __future__ import annotations

import os
import sys
import time

# MUST set FAST_TRAINING="0" BEFORE importing ml_pipeline
os.environ["FAST_TRAINING"] = "0"

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import shap

# Local imports from backend
import dataset as ds_mod
import ml_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET = "price_inr"
NUMERIC_FEATURES = ml_pipeline.NUMERIC_FEATURES
CATEGORICAL_FEATURES = ml_pipeline.CATEGORICAL_FEATURES


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_lookup(df: pd.DataFrame):
    """Compute global, city, and city+locality median price_per_sqft."""
    global_med = float(df["price_per_sqft"].median())
    city_med = df.groupby("city")["price_per_sqft"].median().to_dict()
    loc_med = df.groupby(["city", "locality"])["price_per_sqft"].median().to_dict()
    return global_med, city_med, loc_med


def apply_lookup(df: pd.DataFrame, global_med: float, city_med: dict, loc_med: dict):
    """Apply median price_per_sqft lookup with city and global fallbacks."""
    city_psf = [city_med.get(c, global_med) for c in df["city"]]
    loc_psf = [
        loc_med.get((c, loc), city_med.get(c, global_med))
        for c, loc in zip(df["city"], df["locality"])
    ]
    return np.array(loc_psf, dtype=float), np.array(city_psf, dtype=float)


def build_features(train_df: pd.DataFrame, other_df: pd.DataFrame | None = None):
    """
    Compute locality_psf_median and city_psf_median without target leakage.
    - training rows: out-of-fold 5-fold KFold (shuffle=True, random_state=42)
    - other_df rows: from full train_df with the same fallback hierarchy
    """
    train_out = train_df.copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_loc = np.zeros(len(train_df), dtype=float)
    oof_city = np.zeros(len(train_df), dtype=float)

    for tr_idx, val_idx in kf.split(train_df):
        f_tr = train_df.iloc[tr_idx]
        f_val = train_df.iloc[val_idx]
        g_med, c_med, l_med = compute_lookup(f_tr)
        loc_v, city_v = apply_lookup(f_val, g_med, c_med, l_med)
        oof_loc[val_idx] = loc_v
        oof_city[val_idx] = city_v

    train_out["locality_psf_median"] = oof_loc
    train_out["city_psf_median"] = oof_city

    if other_df is None:
        return train_out

    other_out = other_df.copy()
    g_med, c_med, l_med = compute_lookup(train_df)
    loc_o, city_o = apply_lookup(other_df, g_med, c_med, l_med)
    other_out["locality_psf_median"] = loc_o
    other_out["city_psf_median"] = city_o

    return train_out, other_out


def get_original_feature(name: str) -> str:
    """Map one-hot encoded or transformed column name back to original feature name."""
    clean = name
    if clean.startswith("num__"):
        clean = clean[len("num__"):]
    elif clean.startswith("cat__"):
        clean = clean[len("cat__"):]

    if clean == "locality_psf_median":
        return "locality_psf_median"
    if clean == "city_psf_median":
        return "city_psf_median"

    if clean.startswith("city_") or clean == "city":
        return "city"
    if clean.startswith("locality_") or clean == "locality":
        return "locality"

    return clean


def main():
    start_total_time = time.time()
    print("=" * 80)
    print("BHARAT ESTATE: LEAKAGE-FREE EVALUATION PIPELINE")
    print("=" * 80)

    # 1. Dataset stats & verification
    print("\n[Step 1] Loading raw files & verifying dataset cleaning...")
    raw_row_counts = {}
    for c in ds_mod.CITIES:
        raw_p = os.path.join(ds_mod.RAW_DIR, f"{c}.csv")
        raw_row_counts[c] = len(pd.read_csv(raw_p))
    sum_raw_rows = sum(raw_row_counts.values())

    df_canonical = ds_mod.build_canonical(force=False)
    rows_canonical = len(df_canonical)

    df_out1 = ml_pipeline._remove_outliers(df_canonical, ["price_per_sqft"], k=2.5)
    rows_out1 = len(df_out1)

    df_clean = ml_pipeline._remove_outliers(df_out1, ["area_sqft"], k=3.0)
    rows_clean = len(df_clean)

    # Train-test split (exact same split as ml_pipeline._train)
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    train_size = len(train_df)
    test_size = len(test_df)

    print(f"  Sum of rows across 6 raw CSVs: {sum_raw_rows}")
    for c, cnt in raw_row_counts.items():
        print(f"    - {c}: {cnt}")
    print(f"  Rows in canonical CSV: {rows_canonical}")
    print(f"  Rows after price_per_sqft outlier filter (k=2.5): {rows_out1}")
    print(f"  Rows after area_sqft outlier filter (k=3.0): {rows_clean}")
    print(f"  Train size (80%): {train_size}")
    print(f"  Test size (20%): {test_size}")

    ytr = train_df[TARGET].values
    yte = test_df[TARGET].values
    ytr_log = np.log1p(ytr)

    # SVR 3000-row subsample (rng seed 0)
    svr_rng = np.random.default_rng(0)
    svr_idx = svr_rng.choice(train_size, size=min(3000, train_size), replace=False)

    # ---------------------------------------------------------
    # Experiment A: "Leaky baseline" (precomputed columns)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Experiment A] Evaluating 'Leaky Baseline' (original features with precomputed medians)...")
    Xtr_leaky = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    Xte_leaky = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    models_a = ml_pipeline._models()
    records_a = []
    preds_a = {}

    for name, est in models_a.items():
        t0 = time.time()
        pipe = Pipeline([("pre", ml_pipeline._build_preprocessor()), ("model", est)])
        if name == "SVR":
            pipe.fit(Xtr_leaky.iloc[svr_idx], ytr_log[svr_idx])
        else:
            pipe.fit(Xtr_leaky, ytr_log)
        
        pred = np.clip(np.expm1(pipe.predict(Xte_leaky)), 1e5, 5e8)
        pred = np.nan_to_num(pred, nan=1e6, posinf=5e8, neginf=1e5)
        preds_a[name] = pred

        r2 = float(r2_score(yte, pred))
        mae = float(mean_absolute_error(yte, pred) / 1e5)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)) / 1e5)
        mape = float(_mape(yte, pred))
        t_sec = round(time.time() - t0, 2)

        records_a.append({
            "Model": name,
            "Test R2": r2,
            "Test MAE (Lakh)": mae,
            "Test RMSE (Lakh)": rmse,
            "Test MAPE (%)": mape,
            "Fit Time (s)": t_sec,
        })
        print(f"  {name:<20}: Test R2={r2:.4f}, MAE={mae:.2f}L, RMSE={rmse:.2f}L, MAPE={mape:.2f}% ({t_sec}s)")

    df_table_a = pd.DataFrame(records_a)
    df_table_a.to_csv(os.path.join(RESULTS_DIR, "table_A_leaky_baseline.csv"), index=False)
    print("  -> Saved backend/results/table_A_leaky_baseline.csv")

    # ---------------------------------------------------------
    # Experiment B: Leakage-free (Table III)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Experiment B] Evaluating 'Leakage-Free' (OOF medians on train, full train medians on test)...")
    train_df_lf, test_df_lf = build_features(train_df, test_df)
    Xtr_lf = train_df_lf[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    Xte_lf = test_df_lf[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    models_b = ml_pipeline._models()
    records_b = []
    fitted_lf = {}
    preds_b = {}

    for name, est in models_b.items():
        t0 = time.time()
        pipe = Pipeline([("pre", ml_pipeline._build_preprocessor()), ("model", est)])
        if name == "SVR":
            pipe.fit(Xtr_lf.iloc[svr_idx], ytr_log[svr_idx])
        else:
            pipe.fit(Xtr_lf, ytr_log)
        
        # Train prediction for Train R2
        pred_tr = np.clip(np.expm1(pipe.predict(Xtr_lf)), 1e5, 5e8)
        pred_tr = np.nan_to_num(pred_tr, nan=1e6, posinf=5e8, neginf=1e5)
        tr_r2 = float(r2_score(ytr, pred_tr))

        # Test prediction
        pred_te = np.clip(np.expm1(pipe.predict(Xte_lf)), 1e5, 5e8)
        pred_te = np.nan_to_num(pred_te, nan=1e6, posinf=5e8, neginf=1e5)
        fitted_lf[name] = pipe
        preds_b[name] = pred_te

        te_r2 = float(r2_score(yte, pred_te))
        te_mae = float(mean_absolute_error(yte, pred_te) / 1e5)
        te_rmse = float(np.sqrt(mean_squared_error(yte, pred_te)) / 1e5)
        te_mape = float(_mape(yte, pred_te))
        t_sec = round(time.time() - t0, 2)

        records_b.append({
            "Model": name,
            "Train R2": tr_r2,
            "Test R2": te_r2,
            "Test MAE (Lakh)": te_mae,
            "Test RMSE (Lakh)": te_rmse,
            "Test MAPE (%)": te_mape,
            "Fit Time (s)": t_sec,
        })
        print(f"  {name:<20}: Train R2={tr_r2:.4f}, Test R2={te_r2:.4f}, MAE={te_mae:.2f}L, RMSE={te_rmse:.2f}L, MAPE={te_mape:.2f}% ({t_sec}s)")

    df_table_b = pd.DataFrame(records_b)
    df_table_b.to_csv(os.path.join(RESULTS_DIR, "table_III_leakage_free.csv"), index=False)
    print("  -> Saved backend/results/table_III_leakage_free.csv")

    best_model_name = df_table_b.sort_values(by="Test R2", ascending=False).iloc[0]["Model"]
    print(f"\n  ==> Best Model identified: {best_model_name}")

    # ---------------------------------------------------------
    # Experiment C: 95% Bootstrap CI for Best Model
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[Experiment C] 95% Bootstrap CI (1000 resamples) for Best Model: {best_model_name}...")
    best_pred_te = preds_b[best_model_name]
    rng_c = np.random.default_rng(42)
    boot_r2, boot_mae, boot_rmse, boot_mape = [], [], [], []

    for _ in range(1000):
        b_idx = rng_c.choice(test_size, size=test_size, replace=True)
        y_b = yte[b_idx]
        p_b = best_pred_te[b_idx]
        boot_r2.append(r2_score(y_b, p_b))
        boot_mae.append(mean_absolute_error(y_b, p_b) / 1e5)
        boot_rmse.append(np.sqrt(mean_squared_error(y_b, p_b)) / 1e5)
        boot_mape.append(_mape(y_b, p_b))

    point_est = {
        "R2": r2_score(yte, best_pred_te),
        "MAE (Lakh)": mean_absolute_error(yte, best_pred_te) / 1e5,
        "RMSE (Lakh)": np.sqrt(mean_squared_error(yte, best_pred_te)) / 1e5,
        "MAPE (%)": _mape(yte, best_pred_te),
    }

    records_c = []
    for metric, arr in [("R2", boot_r2), ("MAE (Lakh)", boot_mae), ("RMSE (Lakh)", boot_rmse), ("MAPE (%)", boot_mape)]:
        ci_lo = float(np.percentile(arr, 2.5))
        ci_hi = float(np.percentile(arr, 97.5))
        pt = float(point_est[metric])
        records_c.append({
            "Metric": metric,
            "Point Estimate": pt,
            "95% CI Lower": ci_lo,
            "95% CI Upper": ci_hi,
        })
        print(f"  {metric:<12}: Point={pt:.4f}, 95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

    df_table_c = pd.DataFrame(records_c)
    df_table_c.to_csv(os.path.join(RESULTS_DIR, "table_C_bootstrap_ci.csv"), index=False)
    print("  -> Saved backend/results/table_C_bootstrap_ci.csv")

    # ---------------------------------------------------------
    # Experiment D: Repeated 5-fold CV (on training set only)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Experiment D] Repeated 5-fold CV (3 repeats = 15 folds) on TRAINING SET ONLY...")
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    n_total_folds = 5 * 3

    # Storage for fold metrics per model
    cv_metrics = {name: {"r2": [], "mae": [], "rmse": [], "mape": []} for name in models_b.keys()}

    t_cv_start = time.time()
    for fold_num, (tr_idx, val_idx) in enumerate(rkf.split(train_df), 1):
        f_t0 = time.time()
        f_tr_raw = train_df.iloc[tr_idx]
        f_val_raw = train_df.iloc[val_idx]

        # Recompute leakage-free features strictly inside fold
        f_tr_lf, f_val_lf = build_features(f_tr_raw, f_val_raw)
        y_f_tr = f_tr_raw[TARGET].values
        y_f_tr_log = np.log1p(y_f_tr)
        y_f_val = f_val_raw[TARGET].values

        X_f_tr = f_tr_lf[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_f_val = f_val_lf[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        svr_fold_rng = np.random.default_rng(0)
        s_fold_idx = svr_fold_rng.choice(len(X_f_tr), size=min(3000, len(X_f_tr)), replace=False)

        fold_models = ml_pipeline._models()
        for name, est in fold_models.items():
            pipe = Pipeline([("pre", ml_pipeline._build_preprocessor()), ("model", est)])
            if name == "SVR":
                pipe.fit(X_f_tr.iloc[s_fold_idx], y_f_tr_log[s_fold_idx])
            else:
                pipe.fit(X_f_tr, y_f_tr_log)
            
            p_val = np.clip(np.expm1(pipe.predict(X_f_val)), 1e5, 5e8)
            p_val = np.nan_to_num(p_val, nan=1e6, posinf=5e8, neginf=1e5)

            cv_metrics[name]["r2"].append(r2_score(y_f_val, p_val))
            cv_metrics[name]["mae"].append(mean_absolute_error(y_f_val, p_val) / 1e5)
            cv_metrics[name]["rmse"].append(np.sqrt(mean_squared_error(y_f_val, p_val)) / 1e5)
            cv_metrics[name]["mape"].append(_mape(y_f_val, p_val))

        f_elapsed = time.time() - f_t0
        print(f"  Fold {fold_num:2d}/{n_total_folds} completed in {f_elapsed:.1f}s")

    print(f"  Repeated CV completed in {time.time() - t_cv_start:.1f}s")

    records_d = []
    for name in models_b.keys():
        r2_m, r2_s = np.mean(cv_metrics[name]["r2"]), np.std(cv_metrics[name]["r2"], ddof=1)
        mae_m, mae_s = np.mean(cv_metrics[name]["mae"]), np.std(cv_metrics[name]["mae"], ddof=1)
        rmse_m, rmse_s = np.mean(cv_metrics[name]["rmse"]), np.std(cv_metrics[name]["rmse"], ddof=1)
        mape_m, mape_s = np.mean(cv_metrics[name]["mape"]), np.std(cv_metrics[name]["mape"], ddof=1)
        records_d.append({
            "Model": name,
            "R2 Mean": r2_m,
            "R2 Std": r2_s,
            "MAE Mean (Lakh)": mae_m,
            "MAE Std (Lakh)": mae_s,
            "RMSE Mean (Lakh)": rmse_m,
            "RMSE Std (Lakh)": rmse_s,
            "MAPE Mean (%)": mape_m,
            "MAPE Std (%)": mape_s,
        })
        print(f"  {name:<20}: R2={r2_m:.4f}±{r2_s:.4f}, MAE={mae_m:.2f}±{mae_s:.2f}L, RMSE={rmse_m:.2f}±{rmse_s:.2f}L, MAPE={mape_m:.2f}±{mape_s:.2f}%")

    df_table_d = pd.DataFrame(records_d)
    df_table_d.to_csv(os.path.join(RESULTS_DIR, "table_IV_repeated_cv.csv"), index=False)
    print("  -> Saved backend/results/table_IV_repeated_cv.csv")

    # ---------------------------------------------------------
    # Experiment E: Corrected Resampled t-test (Nadeau & Bengio)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[Experiment E] Corrected Resampled t-test (test/train ratio = 1/4) on Fold-wise RMSE...")
    print(f"  Comparing Best Model ({best_model_name}) against each other model...")

    best_rmse_folds = np.array(cv_metrics[best_model_name]["rmse"])
    records_e = []
    J = n_total_folds  # 15
    ratio = 1.0 / 4.0  # test/train ratio = 0.25

    for name in models_b.keys():
        if name == best_model_name:
            continue
        other_rmse_folds = np.array(cv_metrics[name]["rmse"])
        # d_j = RMSE_other - RMSE_best (>0 means best model had lower RMSE)
        diff = other_rmse_folds - best_rmse_folds
        mean_d = float(np.mean(diff))
        var_d = float(np.var(diff, ddof=1))
        # Nadeau & Bengio corrected variance
        corrected_var = (1.0 / J + ratio) * var_d
        corrected_std = np.sqrt(max(1e-12, corrected_var))
        t_stat = mean_d / corrected_std
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df=J - 1))

        records_e.append({
            "Comparison": f"{best_model_name} vs {name}",
            "Mean RMSE Diff (Lakh)": mean_d,
            "Corrected Std Err": corrected_std,
            "t-statistic": t_stat,
            "p-value": p_val,
            "Significant (p<0.05)": p_val < 0.05,
        })
        print(f"  {best_model_name} vs {name:<18}: Mean Diff={mean_d:+.3f}L, t={t_stat:.4f}, p={p_val:.4e} (Sig: {p_val < 0.05})")

    df_table_e = pd.DataFrame(records_e)
    df_table_e.to_csv(os.path.join(RESULTS_DIR, "table_E_corrected_ttest.csv"), index=False)
    print("  -> Saved backend/results/table_E_corrected_ttest.csv")

    # ---------------------------------------------------------
    # Experiment F: Ablation (without locality_psf_median & city_psf_median)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[Experiment F] Ablation Study for {best_model_name}: removing locality_psf_median & city_psf_median...")
    ablation_num_features = [f for f in NUMERIC_FEATURES if f not in ("locality_psf_median", "city_psf_median")]
    preproc_ablation = ColumnTransformer([
        ("num", StandardScaler(), ablation_num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=True), CATEGORICAL_FEATURES),
    ])

    est_ablation = ml_pipeline._models()[best_model_name]
    pipe_ablation = Pipeline([("pre", preproc_ablation), ("model", est_ablation)])

    Xtr_abl = train_df_lf[ablation_num_features + CATEGORICAL_FEATURES]
    Xte_abl = test_df_lf[ablation_num_features + CATEGORICAL_FEATURES]

    if best_model_name == "SVR":
        pipe_ablation.fit(Xtr_abl.iloc[svr_idx], ytr_log[svr_idx])
    else:
        pipe_ablation.fit(Xtr_abl, ytr_log)

    pred_abl = np.clip(np.expm1(pipe_ablation.predict(Xte_abl)), 1e5, 5e8)
    pred_abl = np.nan_to_num(pred_abl, nan=1e6, posinf=5e8, neginf=1e5)

    abl_r2 = float(r2_score(yte, pred_abl))
    abl_mae = float(mean_absolute_error(yte, pred_abl) / 1e5)
    abl_rmse = float(np.sqrt(mean_squared_error(yte, pred_abl)) / 1e5)
    abl_mape = float(_mape(yte, pred_abl))

    # Compare with leakage-free baseline
    base_row = df_table_b[df_table_b["Model"] == best_model_name].iloc[0]
    records_f = [
        {
            "Configuration": "With Leakage-Free Medians",
            "Test R2": float(base_row["Test R2"]),
            "Test MAE (Lakh)": float(base_row["Test MAE (Lakh)"]),
            "Test RMSE (Lakh)": float(base_row["Test RMSE (Lakh)"]),
            "Test MAPE (%)": float(base_row["Test MAPE (%)"]),
        },
        {
            "Configuration": "Ablated (Without Medians)",
            "Test R2": abl_r2,
            "Test MAE (Lakh)": abl_mae,
            "Test RMSE (Lakh)": abl_rmse,
            "Test MAPE (%)": abl_mape,
        },
    ]
    df_table_f = pd.DataFrame(records_f)
    df_table_f.to_csv(os.path.join(RESULTS_DIR, "table_F_ablation.csv"), index=False)
    print(f"  Full Features : Test R2={base_row['Test R2']:.4f}, MAE={base_row['Test MAE (Lakh)']:.2f}L, RMSE={base_row['Test RMSE (Lakh)']:.2f}L, MAPE={base_row['Test MAPE (%)']:.2f}%")
    print(f"  Ablated       : Test R2={abl_r2:.4f}, MAE={abl_mae:.2f}L, RMSE={abl_rmse:.2f}L, MAPE={abl_mape:.2f}%")
    print("  -> Saved backend/results/table_F_ablation.csv")

    # ---------------------------------------------------------
    # Experiment G: Leave-One-City-Out (Table V)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"[Experiment G] Leave-One-City-Out Cross-Validation for Best Model ({best_model_name})...")
    records_g = []

    for held_out_city in ds_mod.CITIES:
        train_city_raw = df_clean[df_clean["city"] != held_out_city]
        test_city_raw = df_clean[df_clean["city"] == held_out_city]

        # Features built from training cities only (held-out city falls back to global median)
        train_c_lf, test_c_lf = build_features(train_city_raw, test_city_raw)

        y_c_tr = train_city_raw[TARGET].values
        y_c_tr_log = np.log1p(y_c_tr)
        y_c_te = test_city_raw[TARGET].values

        est_city = ml_pipeline._models()[best_model_name]
        pipe_city = Pipeline([("pre", ml_pipeline._build_preprocessor()), ("model", est_city)])

        X_c_tr = train_c_lf[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        X_c_te = test_c_lf[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        if best_model_name == "SVR":
            s_c_idx = svr_rng.choice(len(X_c_tr), size=min(3000, len(X_c_tr)), replace=False)
            pipe_city.fit(X_c_tr.iloc[s_c_idx], y_c_tr_log[s_c_idx])
        else:
            pipe_city.fit(X_c_tr, y_c_tr_log)

        pred_c = np.clip(np.expm1(pipe_city.predict(X_c_te)), 1e5, 5e8)
        pred_c = np.nan_to_num(pred_c, nan=1e6, posinf=5e8, neginf=1e5)

        c_r2 = float(r2_score(y_c_te, pred_c))
        c_mae = float(mean_absolute_error(y_c_te, pred_c) / 1e5)
        c_rmse = float(np.sqrt(mean_squared_error(y_c_te, pred_c)) / 1e5)
        c_mape = float(_mape(y_c_te, pred_c))

        records_g.append({
            "Held-Out City": held_out_city,
            "Test Rows": len(test_city_raw),
            "Test R2": c_r2,
            "Test MAE (Lakh)": c_mae,
            "Test RMSE (Lakh)": c_rmse,
            "Test MAPE (%)": c_mape,
        })
        print(f"  City: {held_out_city:<10} (N={len(test_city_raw):5d}) : Test R2={c_r2:.4f}, MAE={c_mae:.2f}L, RMSE={c_rmse:.2f}L, MAPE={c_mape:.2f}%")

    df_table_g = pd.DataFrame(records_g)
    df_table_g.to_csv(os.path.join(RESULTS_DIR, "table_V_leave_one_city_out.csv"), index=False)
    print("  -> Saved backend/results/table_V_leave_one_city_out.csv")

    # ---------------------------------------------------------
    # Experiment H: Prediction-Interval Coverage (Random Forest)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Experiment H] Prediction-Interval Coverage for Random Forest (replicating server.py)...")
    rf_pipe = fitted_lf["Random Forest"]
    rf_preproc = rf_pipe.named_steps["pre"]
    rf_estim = rf_pipe.named_steps["model"]

    Xt_test = rf_preproc.transform(Xte_lf)
    # Replicate server.py per-tree std:
    # tree_log = np.array([t.predict(Xt)[0] for t in estim.estimators_])
    # half_width = float(np.std(tree_log)) * 1.645
    tree_preds = np.array([t.predict(Xt_test) for t in rf_estim.estimators_])  # (n_trees, n_samples)
    tree_std = np.std(tree_preds, axis=0, ddof=0)
    hw = 1.645 * tree_std

    pred_rf = np.clip(np.expm1(rf_pipe.predict(Xte_lf)), 1e5, 5e8)
    lo = pred_rf * np.exp(-hw)
    hi = pred_rf * np.exp(hw)

    inside = (yte >= lo) & (yte <= hi)
    coverage_pct = float(np.mean(inside) * 100)
    rel_widths = (hi - lo) / pred_rf
    mean_rel_width = float(np.mean(rel_widths))

    print(f"  Nominal Coverage Level : 90.0%")
    print(f"  Empirical Test Coverage: {coverage_pct:.2f}%")
    print(f"  Mean Relative Interval Width ((hi-lo)/pred): {mean_rel_width:.4f}")

    df_table_h = pd.DataFrame([{
        "Nominal Coverage (%)": 90.0,
        "Empirical Coverage (%)": coverage_pct,
        "Mean Relative Width": mean_rel_width,
        "Total Test Rows": test_size,
        "Covered Rows": int(np.sum(inside)),
    }])
    df_table_h.to_csv(os.path.join(RESULTS_DIR, "table_H_coverage.csv"), index=False)
    print("  -> Saved backend/results/table_H_coverage.csv")

    # ---------------------------------------------------------
    # Experiment I: XGBoost Explainability (SHAP + Feature Importance)
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Experiment I] XGBoost Explainability (TreeExplainer on 2000 random test rows)...")
    xgb_pipe = fitted_lf["XGBoost"]
    xgb_preproc = xgb_pipe.named_steps["pre"]
    xgb_model = xgb_pipe.named_steps["model"]
    feature_names = list(xgb_preproc.get_feature_names_out())

    rng_shap = np.random.default_rng(42)
    sample_size = min(2000, test_size)
    shap_idx = rng_shap.choice(test_size, size=sample_size, replace=False)
    X_sample = Xte_lf.iloc[shap_idx]

    Xt_sample = xgb_preproc.transform(X_sample)
    if hasattr(Xt_sample, "toarray"):
        Xt_sample = Xt_sample.toarray()

    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(Xt_sample)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    # Aggregate back to original feature names
    shap_agg = {}
    for name, val in zip(feature_names, mean_abs_shap):
        orig = get_original_feature(name)
        shap_agg[orig] = shap_agg.get(orig, 0.0) + float(val)

    fi_agg = {}
    for name, val in zip(feature_names, xgb_model.feature_importances_):
        orig = get_original_feature(name)
        fi_agg[orig] = fi_agg.get(orig, 0.0) + float(val)

    top10_shap = sorted(shap_agg.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_fi = sorted(fi_agg.items(), key=lambda x: x[1], reverse=True)[:10]

    records_i = []
    for rank in range(10):
        records_i.append({
            "Rank": rank + 1,
            "Top Feature by Mean |SHAP|": top10_shap[rank][0],
            "Mean |SHAP| (log scale)": top10_shap[rank][1],
            "Top Feature by XGBoost FI": top10_fi[rank][0],
            "XGBoost Feature Importance": top10_fi[rank][1],
        })

    df_table_i = pd.DataFrame(records_i)
    df_table_i.to_csv(os.path.join(RESULTS_DIR, "table_I_shap_feature_importance.csv"), index=False)
    print("  -> Saved backend/results/table_I_shap_feature_importance.csv")

    total_time_sec = time.time() - start_total_time

    # ---------------------------------------------------------
    # Consolidated Plain-Text Summary of ALL Tables (A to I)
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print("CONSOLIDATED SUMMARY OF ALL TABLES (A TO I)")
    print("#" * 80)

    summary_text = f"""
================================================================================
DATASET SUMMARY & VERIFICATION
================================================================================
Sum of rows of six raw CSVs: {sum_raw_rows}
  - Mumbai: {raw_row_counts.get('Mumbai', 0)}, Delhi: {raw_row_counts.get('Delhi', 0)}, Bangalore: {raw_row_counts.get('Bangalore', 0)}
  - Chennai: {raw_row_counts.get('Chennai', 0)}, Hyderabad: {raw_row_counts.get('Hyderabad', 0)}, Kolkata: {raw_row_counts.get('Kolkata', 0)}
Rows in canonical CSV (after initial cleaning & price/area/BHK sanity): {rows_canonical}
Rows after price_per_sqft IQR filtering (k=2.5): {rows_out1}
Rows after area_sqft IQR filtering (k=3.0): {rows_clean}
Training set size (80% split): {train_size}
Test set size (20% split): {test_size}

================================================================================
TABLE A: LEAKY BASELINE (Original Features With Precomputed Medians)
================================================================================
{df_table_a.to_string(index=False)}

================================================================================
TABLE III: LEAKAGE-FREE EVALUATION (OOF Medians on Train, Clean Medians on Test)
================================================================================
{df_table_b.to_string(index=False)}

Best Model: {best_model_name}

================================================================================
TABLE C: 95% BOOTSTRAP CONFIDENCE INTERVALS (Best Model: {best_model_name}, 1000 Resamples)
================================================================================
{df_table_c.to_string(index=False)}

================================================================================
TABLE IV: REPEATED 5-FOLD CROSS-VALIDATION (Training Set Only, 3 Repeats = 15 Folds)
================================================================================
{df_table_d.to_string(index=False)}

================================================================================
TABLE E: CORRECTED RESAMPLED T-TEST ON FOLD-WISE RMSE (Nadeau & Bengio, ratio=1/4)
Reference Best Model: {best_model_name}
================================================================================
{df_table_e.to_string(index=False)}

================================================================================
TABLE F: ABLATION STUDY (Best Model: {best_model_name})
================================================================================
{df_table_f.to_string(index=False)}

================================================================================
TABLE V: LEAVE-ONE-CITY-OUT EVALUATION (Best Model: {best_model_name})
================================================================================
{df_table_g.to_string(index=False)}

================================================================================
TABLE H: PREDICTION-INTERVAL COVERAGE (Random Forest, Nominal 90%)
================================================================================
{df_table_h.to_string(index=False)}

================================================================================
TABLE I: TOP 10 FEATURES (SHAP vs. XGBoost Feature Importance)
================================================================================
{df_table_i.to_string(index=False)}

Total Execution Time: {total_time_sec:.1f} seconds ({total_time_sec / 60:.2f} minutes)
================================================================================
"""
    print(summary_text)

    # Save summary text as a file as well for easy access
    summary_path = os.path.join(RESULTS_DIR, "summary_all_tables.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Summary text saved to: {summary_path}")


if __name__ == "__main__":
    main()
