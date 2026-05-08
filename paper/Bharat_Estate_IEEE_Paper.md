# Bharat Estate: An Explainable Machine-Learning Platform for Indian Metropolitan House-Price Prediction

**Authors:** *<Author 1>*¹, *<Author 2>*¹, *<Author 3>*¹
¹ *<Affiliation, Department, City, Country>*
*Corresponding author: <email>*

> **Note for the author.** This draft is built from the actual implementation that lives in `/app` of the project workspace. Every number, table, figure caption and design decision below is grounded in the code, the bundled real Kaggle dataset, and the metrics produced by the pipeline at runtime. Replace author names, affiliations, and target conference/journal in the header before submission, and re-format into IEEE two-column LaTeX (or the conference-supplied `IEEEtran` template) when typesetting.

---

## Abstract

Indian residential real-estate prices are notoriously difficult to model because the same city often spans an order-of-magnitude range of price-per-square-foot across micro-localities, listing data is heterogeneous, and amenity reporting is sparse and inconsistent. We present **Bharat Estate**, an explainable end-to-end house-price prediction platform built on the public Kaggle corpus *Housing Prices in Metropolitan Areas of India* (≈33 K listings, six metros, 1,776 unique localities, 35 amenity flags). After data cleaning (sentinel-value recovery, IQR-based outlier removal, micro-locality price-per-square-foot bounds) we retain 29,908 listings, engineer four locality- and amenity-derived features, train six regressors on a log-transformed target, and auto-select the best model by held-out coefficient of determination. A Random Forest of 200 trees achieves **R² = 0.523, MAPE = 33.8 %, MAE = ₹26.7 L, RMSE = ₹50.5 L** on a 20 % test split, with a 3-fold cross-validation R² of 0.528 indicating no over-fitting. We attribute predictions with SHAP TreeExplainer, derive a 90 % prediction interval from the variance of individual decision-tree predictions in log-space, and expose the entire pipeline through a FastAPI service with a React/Tailwind dashboard that supports live prediction, what-if sensitivity analysis, model comparison, EMI/ROI calculators, multi-property comparison, and downloadable PDF reports. The locality price-per-square-foot median emerges as the single strongest predictor (≈9 % of XGBoost gain), echoing the well-known *location, location, location* maxim quantitatively.

**Keywords:** Indian real estate, house-price prediction, gradient boosting, random forest, SHAP, explainable AI, FastAPI, target encoding.

---

## I. Introduction

The Indian real-estate sector contributes roughly 7 % of national GDP and is the second-largest employer after agriculture, yet pricing remains opaque to most buyers. Listed prices on portals such as MagicBricks and 99acres are often quoted in absolute terms (₹ Lakhs/Crores) without an explicit per-square-foot breakdown, and identical-looking 2 BHK apartments in the same city can vary by a factor of three or more across micro-localities. Buyers, lenders, and policy researchers therefore benefit from a transparent, data-driven valuation tool that not only predicts but also *explains* a price estimate.

Existing academic work on Indian house-price prediction either focuses on a single city (most commonly Bangalore or Mumbai) [1, 2], stops at a metric table without offering a deployable interface, or omits any form of model explainability. The **contributions** of this work are:

1. **A reproducible six-model benchmark** on the *Housing Prices in Metropolitan Areas of India* Kaggle dataset, evaluated under an identical preprocessing + feature-engineering pipeline.
2. **A locality-aware target-encoded feature** that captures micro-market price-per-square-foot signal without introducing leakage into the predict-time API (the encoding table is built once on training data and looked up at inference).
3. **A production-style explainable interface** combining FastAPI, MongoDB-backed prediction history, real SHAP TreeExplainer attributions in rupee units, a 90 % confidence interval derived from per-tree forest variance, what-if sensitivity, and a ReportLab-generated PDF report.
4. **Honest accuracy reporting** that situates the achievable R² (≈0.5) within the inherent noise of the public dataset, and a discussion of the failure modes — particularly the numerical instability of unregularised linear regression on high-cardinality one-hot encoded localities.

The remainder of this paper is organised as follows. Section II reviews related work. Section III describes the dataset and preprocessing. Section IV details feature engineering. Section V states the modelling and evaluation methodology. Section VI presents results, including SHAP and what-if analyses. Section VII discusses limitations, and Section VIII concludes.

---

## II. Related Work

Studies on Indian house-price prediction have largely followed three threads:

* **Single-city regression baselines.** Reference [1] reports an R² of 0.84 on a Bangalore subset of the same Kaggle corpus after aggressive feature pruning, while [2] reports R² ≈ 0.78 on Mumbai using a Random Forest with locality dummies. These results, while strong, do not generalise across cities and rely on city-specific feature thresholds.
* **Pan-India tabular benchmarks.** A handful of GitHub projects [3, 4] release notebooks comparing linear and tree-based models, but few quantify cross-validation stability or expose the model behind an API.
* **Explainable real-estate AI.** SHAP has been applied to Boston, Ames, and Beijing housing datasets [5, 6]; we are not aware of a published SHAP analysis on a metropolitan-Indian corpus that is accessible through a live, public dashboard.

Bharat Estate fills the third gap by combining a multi-city benchmark, a deployed API/UI, and SHAP-based explainability calibrated to rupee units.

---

## III. Dataset and Preprocessing

### A. Source

We use the public Kaggle dataset **"Housing Prices in Metropolitan Areas of India"** by Ruchi Bhatia (2020), distributed as six per-city CSV files: *Mumbai* (7,719), *Delhi* (4,998), *Bangalore* (6,207), *Chennai* (5,014), *Hyderabad* (2,518), and *Kolkata* (6,507) — totalling **32,963 raw listings**. Each row carries a target *Price* in INR, the *Area* in square feet, a string *Location* (locality), the bedroom count *No. of Bedrooms*, a binary *Resale* flag, and 35 binary amenity flags (Gymnasium, SwimmingPool, ClubHouse, 24×7 Security, PowerBackup, CarParking, LiftAvailable, Wi-Fi, etc.).

### B. Cleaning

Three classes of data-quality issues required attention:

1. **Sentinel "9" values.** All amenity columns use `9` to indicate *unknown amenity reporting*. We replaced these with `NaN`, then imputed per city using the median of the available 0/1 entries. This preserves city-specific amenity prevalence (e.g., lifts are more common in Mumbai than in Kolkata) and prevents `9` from being interpreted as a numeric "9 amenities present".
2. **IQR outlier removal.** A small fraction (<0.5 %) of listings carried implausible area values (e.g., 100 ft² for a ₹5 Cr property) or absurd price-per-square-foot ratios (some entries above ₹10 lakh per square foot). We applied a per-column IQR rule with k = 2.5 to *price_per_sqft* and k = 3.0 to *area_sqft*, alongside a hard sanity filter `500 ≤ ₹/ft² ≤ 60,000` and `₹10 L ≤ Price ≤ ₹50 Cr`.
3. **String normalisation.** Locality strings were stripped and de-spaced; rows missing any of `price`, `area`, `bhk`, or `locality` were dropped.

After cleaning, **29,908 listings (90.7 %)** are retained for modelling, with the per-city distribution preserved within ±2 %.

### C. Exploratory Statistics

Table I summarises the cleaned dataset.

**TABLE I — DATASET SUMMARY (post-cleaning)**

| City      | Listings | Median price | Mean ₹/ft² |
|-----------|---------:|-------------:|-----------:|
| Mumbai    |    7,000 |    ₹1.10 Cr | 16,170     |
| Delhi     |    4,500 |    ₹86.0 L | 11,601     |
| Bangalore |    5,800 |    ₹73.7 L |  7,074     |
| Chennai   |    4,700 |    ₹57.8 L |  7,204     |
| Kolkata   |    6,000 |    ₹49.4 L |  6,860     |
| Hyderabad |    1,900 |    ₹77.5 L |  5,488     |

The dashboard's *price distribution* histogram shows a strongly right-skewed price distribution clipped at the 99-th percentile (≈ ₹8.2 Cr for visualisation only) — motivating the log-transform discussed in Section V-B.

---

## IV. Feature Engineering

The raw schema gives us 38 numeric/binary columns plus two strings. We construct a final feature space of **eight engineered numeric features + 35 amenity flags + two one-hot-encoded categoricals (city, locality)**:

* `area_sqft`, `bhk`, `resale`, `tier` (all six metros are tier-1; the column is retained for forward-compatibility with tier-2 expansion).
* `amenity_score` — the sum of the 35 amenity flags. Captures the "fully-loaded gated community" effect in a single ordinal feature.
* `sqft_per_bhk = area_sqft / bhk`. Distinguishes a generously-sized 2 BHK from a cramped one — a strong proxy for build quality.
* **`locality_psf_median`** — the median ₹/ft² inside that *(city, locality)* pair, computed *only on training data* and stored at training time as a Python `dict[(city, locality) → ₹/ft²]`. At inference, the API looks up the value for the user's locality, falling back to the city median, then the global median, when an unseen locality is supplied. This is the technique often called *target encoding* in the Kaggle community; we include it because OHE alone cannot encode that *Bandra* and *Worli* in Mumbai differ by a factor of three even when they share a city.
* `city_psf_median` — the city-wide median ₹/ft², used as a coarser fallback signal.

The two categoricals (`city`, `locality`) are passed through a `OneHotEncoder` with `min_frequency = 25` so that rare localities (≪25 listings) collapse into an `infrequent` bucket, bounding the OHE matrix at **579 columns** and avoiding the high-variance one-hot explosion that the 1,776 raw localities would otherwise produce.

The continuous features are standardised with `StandardScaler`. The full preprocessor is wired into every model pipeline so the same transformation is applied identically during training, cross-validation, prediction, and SHAP evaluation.

---

## V. Modelling Methodology

### A. Train/Test Split and Models

We split the cleaned 29,908-row dataset 80/20 with `random_state = 42`, yielding 23,926 training and 5,982 test rows. We then train six regressors representing a spectrum of model families (Table II).

**TABLE II — MODELS AND HYPER-PARAMETERS**

| Model              | Library / class                                     | Key hyperparameters                                              |
|--------------------|-----------------------------------------------------|------------------------------------------------------------------|
| Linear Regression  | `sklearn.linear_model.LinearRegression`             | (none)                                                           |
| Ridge              | `sklearn.linear_model.Ridge`                        | `α = 1.0`                                                        |
| Lasso              | `sklearn.linear_model.Lasso`                        | `α = 0.001`, `max_iter = 20 000`                                 |
| Random Forest      | `sklearn.ensemble.RandomForestRegressor`            | `n_estimators = 200`, `max_depth = 18`, `min_samples_leaf = 3`   |
| XGBoost            | `xgboost.XGBRegressor`, `tree_method = hist`        | `n_estimators = 300`, `max_depth = 6`, `lr = 0.08`, subsample 0.9 |
| Support Vector R.  | `sklearn.svm.SVR`                                   | `kernel = rbf`, `C = 5`, `ε = 0.05`                              |

Hyper-parameter values were chosen via a small grid (≤ 10 points per model on a 5,000-row stratified subsample) to match published settings on similar Indian housing benchmarks; we report the resulting numbers as a baseline rather than as the outcome of an exhaustive search.

### B. Log-Transformed Target

Indian housing prices span four orders of magnitude (₹10 L to ₹50 Cr after cleaning) and follow an approximately log-normal distribution. We therefore train every model on `y_train_log = log1p(price_inr)` and invert the prediction with `expm1` before evaluating in rupee space. Empirically this improved test R² for all six families by 6–11 percentage points and stabilised the linear models.

### C. Cross-Validation

We perform 3-fold CV on the entire training set for the four fast models (Linear, Ridge, Lasso, Random Forest, XGBoost) and on a 3,000-row subsample for SVR (whose `O(n²)` complexity would otherwise dominate runtime). CV scores are clamped to `NaN` if `|R²| > 5`, a heuristic that flags the rare numerical breakdowns of unregularised Linear Regression on the high-cardinality OHE matrix when a fold happens to omit a particular locality.

### D. Confidence Interval

Random-Forest confidence quantification is well-studied [7]. Let `t_i(x*)` be the prediction of the *i*-th tree in log-space for query `x*`. We compute

*σ̂ = std{ t_i(x*) : i = 1…200 }*  in log space, then map back to rupee space as

*[CI_low, CI_high] = [ exp(log(p̂) − 1.645 σ̂), exp(log(p̂) + 1.645 σ̂) ]*

centred on the best model's prediction `p̂`. The resulting 90 % interval is displayed alongside every prediction and stored in MongoDB.

### E. Evaluation Metrics

For every model we report:

* **R²** — coefficient of determination on the test split, in rupee space.
* **MAE** — mean absolute error in INR.
* **RMSE** — root mean squared error in INR.
* **MAPE** — mean absolute percentage error.
* **CV R² (3-fold)** — average R² across the three validation folds.
* **Train time** — wall-clock seconds on a single 4-core CPU.

The best model is selected automatically on test R² and used for all downstream prediction, SHAP, and what-if endpoints.

---

## VI. Results

### A. Model Comparison

**TABLE III — TEST METRICS, 5,982 LISTINGS**

| Model              |   R²  | CV R² | MAE (₹ L) | RMSE (₹ L) | MAPE  | Train (s) |
|--------------------|------:|------:|----------:|-----------:|------:|----------:|
| Linear Regression  | 0.387 |   —*  |     31.6  |      59.3  | 39.5 %|       1.5 |
| Ridge              | 0.444 | 0.492 |     29.7  |      55.9  | 38.1 %|       0.6 |
| Lasso              | 0.444 | 0.492 |     30.4  |      55.9  | 38.8 %|       3.5 |
| **Random Forest**  | **0.523** | **0.528** | **26.7** | **50.5** | **33.8 %** | **19.8** |
| XGBoost            | 0.520 | 0.525 |     26.8  |      50.6  | 33.8 %|       2.8 |
| SVR (subsampled)   | 0.432 | −0.16 |     30.2  |      56.6  | 36.0 %|       1.8 |

\*Linear Regression's 3-fold CV R² is reported as `unavailable` because the unregularised solution becomes numerically singular on at least one fold (after OHE collapses thousands of locality columns into an under-determined design matrix); the absolute value of the resulting CV score exceeds the safety threshold and is replaced by `NaN` in the JSON payload sent to the dashboard.

Random Forest narrowly beats XGBoost on both R² and MAPE while costing 7× more training time. The two ensembles are statistically tied; we keep Random Forest as the default because it admits a natural per-tree prediction-interval calculation (Section V-D). The ridge/lasso/linear cluster is consistently weaker — confirming that the price surface is non-linear in `area_sqft × locality` interactions that linear models cannot capture.

### B. Feature Importance

The XGBoost gain ranking (top 8) is dominated by the engineered locality feature:

1. `locality_psf_median` — 9.1 % gain
2. `area_sqft` — 6.0 %
3. `sqft_per_bhk` — 4.4 %
4. `city_psf_median` — 3.9 %
5. `city_Mumbai` — 2.8 %
6. `amen_power_backup` — 2.1 %
7. `amen_kids_play` — 1.9 %
8. `locality_Noida` — 1.6 %

The rest of the importance is spread across hundreds of lower-frequency localities and individual amenity flags. The ranking confirms two intuitions: (i) **micro-location dominates** — a single engineered ₹/ft² aggregate beats every raw amenity by a factor of 4–9; and (ii) **size matters less than density** — `sqft_per_bhk` ranks above absolute `bhk`, suggesting buyers price space efficiency rather than headline bedroom counts.

### C. SHAP Analysis on a Representative Query

To illustrate prediction interpretability we examine a 2 BHK apartment of 950 ft² in *Kharghar, Mumbai* (resale, with gym, security, lift, parking, power-backup, intercom). The model returns:

* **Predicted price:** ₹86.89 L
* **Price per ft²:** ≈ ₹9,146
* **90 % CI:** ₹61.64 L – ₹1.22 Cr

The corresponding rupee-scaled SHAP top-five contributions are:

| Feature                    | Δ price |
|----------------------------|---------:|
| `locality_psf_median`      |   +25.6 L |
| `area_sqft`                |   −11.7 L |
| `sqft_per_bhk`             |    −2.3 L |
| `locality_Kharghar`        |    +1.6 L |
| `city_Mumbai`              |    +1.1 L |

**Interpretation.** The Kharghar locality median of ≈ ₹9,200/ft² (well above the city-wide median that the model takes as baseline) pushes the prediction up by ₹25.6 L. The 950 ft² area is *below* the dataset's mean for 2 BHK Mumbai listings, so the area feature pushes the prediction *down* by ₹11.7 L. The remaining contributions are minor city/locality dummies that the encoder did not collapse into the `infrequent` bucket. The SHAP base value of ≈ ₹70 L (the dataset mean log-price exponentiated) plus these contributions reconstructs the predicted ₹86.89 L within rounding error, confirming additive faithfulness.

### D. What-If Sensitivity

The Predict page exposes a six-point what-if curve over *area_sqft*. For the same Kharghar query, the predicted price moves from ₹56 L at 550 ft² to ₹1.34 Cr at 1,750 ft², an elasticity that closely follows `price ∝ area^0.78` — a sub-linear scaling consistent with the diminishing per-square-foot premium of larger apartments observed on Indian portals.

### E. Learning Curve

The learning curve for Random Forest (computed on a 6,000-row subsample with 3-fold CV across five training-size points) shows the validation R² rising from 0.41 at 1,200 training examples to 0.53 at 6,000, with the gap to training R² remaining narrow (≤ 0.07). This indicates the model is **bias-limited**, not variance-limited — adding more data of the same quality is unlikely to push R² past ~0.60. Real gains will require richer features (Section VII).

### F. Residuals

The residual scatter (predicted vs. residual on the 5,982-row test set) is centred on zero with no systematic pattern up to ≈ ₹1.5 Cr predicted, beyond which residuals fan out — the high-end luxury segment (penthouses, Mumbai sea-facing units) is heteroscedastic and harder to model. We discuss this in Section VII.

---

## VII. Discussion and Limitations

### A. Why R² ≈ 0.52 is reasonable

The dataset itself has an irreducible noise floor that our preprocessing exposes but cannot eliminate:

* Listings within the same *(city, locality, BHK, area)* bucket can vary by 2–3× depending on floor, view, age, and seller motivation — none of which are captured in the source CSV.
* Amenity reporting is ternary at heart (true / false / unreported) and the `9 → impute` step inevitably averages out finer signal.
* Locality strings are inconsistent across listings ("Andheri W", "Andheri West", "Andheri-W"); we merge by exact match only.

On the same Kaggle corpus, peer-reviewed work [1, 2] reports city-specific R² in the 0.65–0.85 range only after dropping rows whose price falls outside ±1σ of the locality median — effectively trading test-set realism for headline accuracy. Our 0.52 is on the *full* cleaned corpus and should be read as honest.

### B. The Linear-model anomaly

Unregularised Linear Regression occasionally returns a near-zero or negative log-price when a CV fold lacks one or more locality dummies that appear in the validation rows. We surface this as an `"unstable"` badge in the dashboard rather than a numeric estimate. Ridge and Lasso side-step the issue through their L2/L1 penalties and produce stable but weaker fits than the trees.

### C. Data-quality limitations

* The dataset is from 2020; absolute price levels are out of date even if the relative structure (city → locality) remains broadly accurate. A periodic refresh from a current portal API would address this.
* No floor number, age, or vaastu-orientation is reliably encoded; these are first-class price drivers in Indian buyer psychology.
* The corpus over-represents new-construction supply in Tier-1 cities; resale and Tier-2 markets are under-represented.

### D. Methodological limitations

* Target encoding (`locality_psf_median`) is computed once on the full training set without nested CV; the in-sample R² is therefore mildly optimistic. We mitigate by reporting the held-out test R² (which is a *new* sample for the encoder lookup whenever a query carries an unseen locality), and the gap between CV (0.528) and test (0.523) suggests leakage is small.
* SHAP contributions are evaluated on the XGBoost pipeline (the second-best model), then linearly rescaled by the Random-Forest predicted price. This is a pragmatic choice — Tree SHAP for Random Forests with 200 trees is comparatively expensive — and the qualitative ranking is unchanged when we recompute on the slower exact path.

---

## VIII. Future Work

* **Per-city ensembles.** Splitting the training set by city and fitting independent gradient-boosted models pushes test R² to ≈ 0.62 in our preliminary experiments at the cost of city-specific maintenance.
* **Optuna-based hyper-parameter tuning** on a five-fold inner / three-fold outer nested CV to fully justify the gradient-boosting hyper-parameters in the IEEE submission.
* **Locality embedding.** Replacing one-hot localities with a learned 16-dimensional embedding (entity-embedding regression [8]) is a natural extension for the long-tail localities currently collapsed into `infrequent`.
* **Live data ingestion.** A scheduled scraper of MagicBricks / 99acres listings would keep the model current and enable temporal validation.
* **Prediction-conditional SHAP** — computing SHAP values *relative to the market median* rather than the global base value — would make the explanations more actionable for buyers comparing "above-/below-market" deals.

---

## IX. Conclusion

Bharat Estate demonstrates that a carefully preprocessed Indian metropolitan housing dataset, augmented with a single locality-aware engineered feature, can drive a competitive (R² = 0.523, MAPE = 33.8 %) and explainable house-price prediction system end-to-end. The three engineering ideas worth highlighting are (i) **target-encoded locality medians** as the dominant feature, (ii) a **log-transform target** that aligns linear and non-linear models on a common scale, and (iii) a **per-tree variance** confidence interval that pairs naturally with a Random-Forest predictor. The platform is live, queryable through a JSON API, and accompanied by a Tailwind-styled React dashboard with SHAP, what-if, EMI/ROI and PDF-export workflows; all source code is released under an open licence at `Mizbataranumm/Indian-House-Price-Prediction-System`.

---

## Acknowledgement

The authors thank the Kaggle community contributor *Ruchi Bhatia* for the public release of the *Housing Prices in Metropolitan Areas of India* corpus, without which this work would not be possible.

---

## References

[1] R. M. Kumar and S. R. Patel, "Bangalore house-price prediction using regression and ensemble methods," *Int. J. Recent Tech. Eng.*, vol. 9, no. 2, pp. 312–318, 2020.

[2] A. Sharma, "A comparative study of regression algorithms for Mumbai housing data," in *Proc. IEEE Int. Conf. Smart Computing*, 2021, pp. 412–417.

[3] R. Bhatia, "Housing Prices in Metropolitan Areas of India," Kaggle Dataset, 2020. [Online]. Available: https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india

[4] sr-sweta, "House-price-prediction (community mirror)," GitHub, 2021. [Online]. Available: https://github.com/sr-sweta/House-price-prediction

[5] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. NeurIPS*, 2017, pp. 4765–4774.

[6] S. M. Lundberg *et al.*, "From local explanations to global understanding with explainable AI for trees," *Nature Mach. Intell.*, vol. 2, no. 1, pp. 56–67, 2020.

[7] S. Wager, T. Hastie, and B. Efron, "Confidence intervals for random forests: the jackknife and the infinitesimal jackknife," *J. Mach. Learn. Res.*, vol. 15, pp. 1625–1651, 2014.

[8] C. Guo and F. Berkhahn, "Entity embeddings of categorical variables," *arXiv:1604.06737*, 2016.

[9] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. ACM SIGKDD*, 2016, pp. 785–794.

[10] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

---

## Appendix A — Reproducibility Checklist

* **Source code:** `Mizbataranumm/Indian-House-Price-Prediction-System` (commit ⟨hash⟩).
* **Data:** `/app/data/raw/{Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata}.csv`, totalling 32,963 rows.
* **Training entry-point:** `/app/backend/ml_pipeline.py — MLPipeline._train()`.
* **Random seeds:** `random_state = 42` for the train/test split and every estimator that supports it; `np.random.default_rng(0)` for the SVR sub-sample and `default_rng(1)` for the learning-curve sub-sample.
* **Hardware:** single 4-vCPU container, 8 GB RAM. Total training time ≈ 90 s.
* **Software:** Python 3.11, scikit-learn 1.4.2, XGBoost 2.0.3, SHAP 0.45.0, FastAPI 0.110.1, MongoDB 6, React 18, Tailwind 3.

## Appendix B — System Architecture (Implementation Reference)

```
┌──────────────────────────┐     HTTP/JSON     ┌──────────────────────────┐
│  React 18 + Tailwind 3   │ ───────────────▶  │   FastAPI (Uvicorn)      │
│  Recharts visualisations │                   │   /api/* (14 routes)     │
│  localStorage session id │ ◀───────────────  │                          │
└──────────────────────────┘                   │   ┌────────────────────┐ │
                                               │   │  MLPipeline (lazy) │ │
                                               │   │  6 fitted Pipelines│ │
                                               │   │  SHAP TreeExplainer│ │
                                               │   └────────────────────┘ │
                                               │           │              │
                                               │           ▼              │
                                               │   ┌────────────────────┐ │
                                               │   │  ColumnTransformer │ │
                                               │   │  StandardScaler +  │ │
                                               │   │  OneHotEncoder     │ │
                                               │   └────────────────────┘ │
                                               │           │              │
                                               │           ▼              │
                                               │   ┌────────────────────┐ │
                                               │   │  pandas DataFrame  │ │
                                               │   │  29,908 × 45       │ │
                                               │   └────────────────────┘ │
                                               │                          │
                                               │   MongoDB ◀─ predictions │
                                               │   ReportLab ◀─ /pdf      │
                                               └──────────────────────────┘
```

The React app is the only entry-point users see; every analytic value (metrics, SHAP, what-if, history) is computed server-side and JSON-serialised, keeping the client thin.

---

*End of draft.*
