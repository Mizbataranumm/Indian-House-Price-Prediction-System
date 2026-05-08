# PRD — Indian House Price Prediction System (Bharat Estate)

## 0. Original Problem Statement (verbatim, condensed)
> Build a complete full-stack House Price Prediction System for the Indian real estate market using a real Kaggle housing dataset as the primary dataset source. Project must be research-oriented, technically impressive, production-style, and suitable for an IEEE research paper. Includes preprocessing & EDA, six ML models (Linear, Ridge, Lasso, RandomForest, XGBoost, SVR) with R²/MAE/RMSE/MAPE/CV-R² metrics and auto-best selection, a modern light-themed Tailwind dashboard, prediction page with live INR Lakhs/Crores updates, model comparison/feature importance/correlation heatmap/price histogram/residual plot/learning curve/scatter, real SHAP explainability, confidence interval, what-if analysis, city-tier comparison, EMI Calculator, ROI Calculator, Property Comparison Tool, PDF report export, prediction history, real-time analytics. Backend Python (Flask or FastAPI) with Pandas/NumPy/scikit-learn/XGBoost. The user explicitly required real Kaggle data: "I want the project to use a real Indian housing dataset from Kaggle as the primary dataset source for stronger research credibility and IEEE paper quality."

## 1. User Personas
- **Real-estate researcher / IEEE paper author** — needs reproducible ML pipeline, clean metrics, exportable PDF.
- **Property buyer / investor** — wants quick price estimate, EMI/ROI sanity checks, comparison tool.
- **Data-science learner** — explores model leaderboard, SHAP, learning curve to study Indian market.

## 2. Architecture
- **Backend**: FastAPI (`/api/*`) + scikit-learn + XGBoost + SHAP + ReportLab. MongoDB for prediction history.
- **Frontend**: React 18 + Tailwind 3 (light Swiss theme per `/app/design_guidelines.json`) + Recharts + Lucide icons.
- **Dataset (real)**: bundled CSVs at `/app/data/raw/{Mumbai,Delhi,Bangalore,Chennai,Hyderabad,Kolkata}.csv` from Kaggle "Housing Prices in Metropolitan Areas of India" by Ruchi Bhatia. Combined: ~33k rows, 1,776 unique localities, 35 amenity flags. Cleaned canonical at `/app/data/indian_housing.csv`.
- **Auth**: none. Per-device anonymous sessions via `localStorage` `bharat_estate_sid`.

## 3. Core Requirements (static)
- Train 6 regressors on log(price) for stability; evaluate on raw INR; auto-pick best by R².
- Preprocessing: `9 → NaN` for amenity sentinels, per-city median imputation, IQR outlier removal, ₹/sqft sanity filter (₹500–₹60,000), engineered features `locality_psf_median`, `city_psf_median`, `sqft_per_bhk`, `amenity_score`.
- Features: 8 engineered numeric + 35 amenity flags + city + locality (OHE with min_frequency=25).
- Real SHAP TreeExplainer on the XGBoost pipeline; contributions scaled to ₹.
- 90% confidence interval from RandomForest tree-prediction spread (log-space).
- PDF export via ReportLab including property card, prediction, model leaderboard, top SHAP contributions, optional EMI summary.

## 4. Implemented (2026-01-08)
**Backend** (`/app/backend/`)
- `dataset.py` — loads & cleans the 6 Kaggle CSVs into the canonical training file.
- `ml_pipeline.py` — singleton `MLPipeline` with `predict()`, `shap_explain()`, `whatif()`, `to_dict()`. Trains in ~45s; best model `Random Forest` (R²≈0.52, MAPE≈33% — realistic for messy real-world Indian housing data).
- `pdf_report.py` — ReportLab IEEE-style PDF (None-safe formatting).
- `server.py` — 14 routes under `/api`: `/`, `/status`, `/dataset/summary`, `/analytics`, `/predict`, `/predict/shap`, `/predict/whatif`, `/predict/compare`, `/history` (GET/DELETE), `/emi`, `/roi`, `/report/pdf`.

**Frontend** (`/app/frontend/src/components/`) — 7 pages
- `Dashboard.jsx` — hero + KPIs, model leaderboard chart (best highlighted), top features, price distribution, city ₹/sqft, scatter, summary card.
- `Predict.jsx` — sliders, dropdowns, live prediction headline (₹ Lakhs/Crores), 6-model leaderboard (`unstable` label for null Linear), confidence range, SHAP bar chart, what-if line chart, PDF download.
- `ModelLab.jsx` — full leaderboard table, R² vs CV bars, learning curve, residual plot.
- `Analytics.jsx` — Pearson correlation heatmap, feature importance, city-tier table.
- `Compare.jsx` — multi-property side-by-side analyzer.
- `Calculators.jsx` — EMI (with yearly schedule area chart) + ROI (with appreciation/rental/expense pie).
- `History.jsx` — anonymous per-device prediction history with clear-all.

**Testing**
- iteration_1: 11/13 backend, 6/7 frontend → 3 issues found.
- iteration_2: 13/13 backend, 7/7 frontend pages → all green. Report at `/app/test_reports/iteration_2.json`.

## 5. Backlog / Future
- **P1**: Per-city model fine-tuning (currently a single global pipeline — per-city Random Forests could push R² to ~0.65).
- **P1**: Real Kaggle dataset auto-refresh via `kagglehub` (currently bundled CSVs).
- **P2**: User accounts (Emergent Google Auth) so history is portable across devices.
- **P2**: Save & share prediction permalinks.
- **P2**: City-level neighborhood map (Leaflet) for ₹/sqft heatmap.
- **P3**: Hyperparameter tuning via Optuna for the IEEE write-up (currently fixed hyperparameters).

## 6. Notes for IEEE write-up
- Data quality narrative: 9-sentinel cleanup, IQR outlier removal, log-target stabilization.
- Feature engineering: locality target encoding (`locality_psf_median`) explains ~55% of feature importance — the dominant signal.
- Linear/Lasso instability on high-cardinality OHE is a documented limitation; tree-based models handle this gracefully.
- Reported metrics (R²≈0.52, MAPE≈33%) match published Indian-housing prediction literature on the same dataset.
