"""Backend API tests for Indian House Price Prediction System."""
import os
import uuid
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://price-forecast-india-1.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"


@pytest.fixture(scope="module")
def session():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


@pytest.fixture(scope="module")
def session_id():
    return f"TEST_{uuid.uuid4().hex[:10]}"


# ---- Health & status ----
class TestHealth:
    def test_root(self, session):
        r = session.get(f"{API}/", timeout=15)
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"

    def test_status(self, session):
        r = session.get(f"{API}/status", timeout=30)
        assert r.status_code == 200
        d = r.json()
        assert d["trained"] is True
        assert d["best_model"]
        assert d["n_rows"] > 25000


# ---- Dataset summary ----
class TestDataset:
    def test_summary(self, session):
        r = session.get(f"{API}/dataset/summary", timeout=60)
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d["cities"], list) and len(d["cities"]) == 6
        assert "Mumbai" in d["cities"]
        assert isinstance(d["ui_amenities"], list) and len(d["ui_amenities"]) >= 6
        assert isinstance(d["localities_by_city"], dict)
        for c in d["cities"]:
            assert c in d["localities_by_city"]
            assert len(d["localities_by_city"][c]) > 0
        assert isinstance(d["amenity_labels"], dict)


# ---- Analytics ----
class TestAnalytics:
    def test_analytics_shape(self, session):
        r = session.get(f"{API}/analytics", timeout=120)
        assert r.status_code == 200
        d = r.json()
        # metrics: 6 models
        m = d.get("metrics") or d.get("model_metrics") or {}
        assert isinstance(m, dict)
        assert len(m) >= 6, f"expected >=6 models, got {list(m.keys())}"
        assert "best_model" in d
        for key in ["feature_importance", "correlation",
                    "price_histogram", "price_vs_area",
                    "residuals", "learning_curve", "city_stats"]:
            assert key in d, f"missing key: {key}"

    def test_analytics_json_clean(self, session):
        # ensure no NaN/Infinity tokens in raw JSON text
        r = session.get(f"{API}/analytics", timeout=120)
        assert r.status_code == 200
        text = r.text
        for bad in ["NaN", "Infinity", "-Infinity"]:
            assert bad not in text, f"JSON contains {bad}"


# ---- Predict ----
SAMPLE = {
    "city": "Mumbai", "locality": "Andheri West",
    "bhk": 2, "area_sqft": 900, "resale": 1, "tier": 1,
    "amen_gym": 1, "amen_pool": 0, "amen_garden": 0,
    "amen_security": 1, "amen_power_backup": 1, "amen_parking": 1,
    "amen_lift": 1, "amen_clubhouse": 0, "amen_kids_play": 0,
    "amen_intercom": 1, "amen_ac": 0, "amen_wifi": 0,
}


class TestPredict:
    def test_predict_basic(self, session, session_id):
        r = session.post(f"{API}/predict?save=true&session_id={session_id}",
                         json=SAMPLE, timeout=60)
        assert r.status_code == 200, r.text
        d = r.json()
        out = d["output"]
        assert out["predicted_price"] > 1e5
        assert out["confidence_low"] <= out["predicted_price"] <= out["confidence_high"]
        assert out["price_per_sqft"] > 0
        assert out["best_model"]
        assert isinstance(out["all_predictions"], dict) and len(out["all_predictions"]) >= 6

    def test_shap(self, session):
        r = session.post(f"{API}/predict/shap", json=SAMPLE, timeout=60)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "base_value" in d
        assert "contributions" in d
        assert isinstance(d["contributions"], (list, dict))
        assert len(d["contributions"]) > 0

    def test_whatif(self, session):
        body = {"base": SAMPLE, "feature": "area_sqft",
                "values": [500, 800, 1100, 1500, 2000]}
        r = session.post(f"{API}/predict/whatif", json=body, timeout=60)
        assert r.status_code == 200, r.text
        d = r.json()
        # accept list or {"results": [...]}
        items = d if isinstance(d, list) else d.get("results") or d.get("items") or []
        assert len(items) == 5
        for it in items:
            assert "value" in it and "predicted_price" in it

    def test_compare(self, session):
        p2 = dict(SAMPLE); p2["city"] = "Bangalore"; p2["locality"] = "Whitefield"
        r = session.post(f"{API}/predict/compare",
                         json={"properties": [SAMPLE, p2]}, timeout=60)
        assert r.status_code == 200, r.text
        d = r.json()
        assert "results" in d and len(d["results"]) == 2
        for entry in d["results"]:
            assert "input" in entry and "output" in entry
            assert entry["output"]["predicted_price"] > 0


# ---- EMI / ROI ----
class TestCalculators:
    def test_emi(self, session):
        r = session.post(f"{API}/emi", json={
            "loan_amount": 5000000, "rate": 8.5, "tenure_years": 20
        }, timeout=15)
        assert r.status_code == 200
        d = r.json()
        # expected ~43391
        assert 43000 <= d["emi"] <= 44000, f"emi={d['emi']}"
        assert d["total_payment"] > d["loan_amount"]
        assert isinstance(d["schedule_yearly"], list) and len(d["schedule_yearly"]) >= 1

    def test_roi(self, session):
        r = session.post(f"{API}/roi", json={
            "purchase_price": 10000000, "monthly_rent": 30000,
            "annual_appreciation": 7.0, "holding_years": 5,
            "annual_expense": 20000
        }, timeout=15)
        assert r.status_code == 200
        d = r.json()
        assert d["future_value"] > 10000000
        assert "roi_pct" in d and "cagr_pct" in d


# ---- History ----
class TestHistory:
    def test_history_flow(self, session):
        sid = f"TEST_{uuid.uuid4().hex[:10]}"
        # create 1
        r = session.post(f"{API}/predict?save=true&session_id={sid}",
                         json=SAMPLE, timeout=60)
        assert r.status_code == 200
        # list
        r = session.get(f"{API}/history?session_id={sid}", timeout=15)
        assert r.status_code == 200
        items = r.json()["items"]
        assert len(items) >= 1
        for it in items:
            assert "_id" not in it  # mongo _id excluded
        # delete
        r = session.delete(f"{API}/history?session_id={sid}", timeout=15)
        assert r.status_code == 200
        assert r.json()["deleted"] >= 1
        # verify empty
        r = session.get(f"{API}/history?session_id={sid}", timeout=15)
        assert len(r.json()["items"]) == 0


# ---- PDF ----
class TestPDF:
    def test_pdf(self, session):
        body = {"input": SAMPLE,
                "emi": {"loan_amount": 5000000, "rate": 8.5, "tenure_years": 20}}
        r = session.post(f"{API}/report/pdf", json=body, timeout=60)
        assert r.status_code == 200, r.text
        assert r.content[:4] == b"%PDF"
        assert len(r.content) > 1000
