# Bharat Estate — Indian House Price Prediction System

A research-grade ML platform that predicts Indian property prices using **real Kaggle data** ("Housing Prices in Metropolitan Areas of India" by Ruchi Bhatia, ~33k listings across Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata).

## What's inside

- **Six regressors** (Linear / Ridge / Lasso / Random Forest / XGBoost / SVR) trained on log-price, evaluated with R² / MAE / RMSE / MAPE / 5-fold CV-R².
- **Real SHAP** explainability, 90% confidence interval, what-if sensitivity, PDF report.
- **Dashboard** with model leaderboard, feature importance, correlation heatmap, residual plot, learning curve, scatter.
- **Calculators**: EMI + ROI with charts.
- **Tools**: side-by-side property comparison, anonymous prediction history.

## Tech

FastAPI · scikit-learn · XGBoost · SHAP · ReportLab · MongoDB · React · Tailwind · Recharts.

## Quick start

```bash
# Backend
cd backend && pip install -r requirements.txt && uvicorn server:app --port 8001 --reload

# Frontend
cd frontend && yarn && yarn start
```

The first prediction triggers training (~45 s on the bundled dataset). All routes are under `/api`.

## Render deployment

Frontend:

- Build command: `yarn && yarn build`
- Publish directory: `build`
- Environment variable: `REACT_APP_BACKEND_URL=https://your-backend-service.onrender.com`

Backend:

- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- Optional environment variables:
  - `MONGO_URL`
  - `DB_NAME`

Notes:

- If `MONGO_URL` and `DB_NAME` are not set, the API now starts anyway and uses in-memory history storage.
- The backend warms the ML pipeline in the background on boot. First deploy or cold start can still take around a minute before analytics are ready.
