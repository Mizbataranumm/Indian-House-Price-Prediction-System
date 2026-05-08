import React, { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  LineChart, Line, ReferenceLine, Cell,
} from "recharts";
import { Sparkles, BadgeCheck, Loader2, FileText } from "lucide-react";
import { Card, CardHeader, Badge } from "./ui/Primitives";
import { api, inr, sessionId } from "../lib/api";

const TICK = { fontSize: 11, fill: "#64748B" };

const DEFAULTS = {
  city: "Mumbai", locality: "Kharghar",
  bhk: 2, area_sqft: 950, resale: 1, tier: 1,
  amen_gym: 1, amen_pool: 0, amen_garden: 0, amen_security: 1,
  amen_power_backup: 1, amen_parking: 1, amen_lift: 1, amen_clubhouse: 0,
  amen_kids_play: 0, amen_intercom: 1, amen_ac: 0, amen_wifi: 0,
};

export default function Predict() {
  const [meta, setMeta] = useState(null);
  const [form, setForm] = useState(DEFAULTS);
  const [pred, setPred] = useState(null);
  const [shap, setShap] = useState(null);
  const [whatif, setWhatif] = useState(null);
  const [loading, setLoading] = useState(false);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    api.get("/dataset/summary").then(({ data }) => {
      setMeta(data);
      // Pick a sensible default locality for Mumbai
      const loc = data.localities_by_city.Mumbai?.[0] || DEFAULTS.locality;
      const init = { ...DEFAULTS, locality: loc };
      setForm(init);
      runPredict(init);
    });
  }, []);

  const update = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  async function runPredict(payload = form) {
    setLoading(true);
    try {
      const [p, s] = await Promise.all([
        api.post(`/predict?save=true&session_id=${sessionId()}`, payload),
        api.post("/predict/shap", payload),
      ]);
      setPred(p.data); setShap(s.data);
      const base = payload.area_sqft;
      const areaValues = [
        Math.max(250, base - 400), Math.max(300, base - 200), base,
        base + 200, base + 400, base + 800,
      ];
      const w = await api.post("/predict/whatif", {
        base: payload, feature: "area_sqft", values: areaValues,
      });
      setWhatif(w.data);
    } finally { setLoading(false); }
  }

  async function downloadPdf() {
    setDownloading(true);
    try {
      const res = await api.post("/report/pdf", { input: form }, { responseType: "blob" });
      const url = URL.createObjectURL(new Blob([res.data], { type: "application/pdf" }));
      const a = document.createElement("a");
      a.href = url; a.download = "house_price_report.pdf"; a.click();
      URL.revokeObjectURL(url);
    } finally { setDownloading(false); }
  }

  const allPreds = useMemo(() => {
    if (!pred) return [];
    return Object.entries(pred.output.all_predictions).map(([m, v]) => ({
      model: m.replace(" Regression", ""),
      price: v == null ? null : Math.round(v),
      best: m === pred.output.best_model,
    }));
  }, [pred]);

  const localities = meta?.localities_by_city?.[form.city] || [];
  const amenityList = meta?.ui_amenities || [];
  const amenityLabel = meta?.amenity_labels || {};

  return (
    <div className="grid lg:grid-cols-[420px_1fr] gap-6">
      {/* Left: form */}
      <Card data-testid="predict-form-card">
        <CardHeader
          eyebrow="Inputs"
          title="Property profile"
          sub="Adjust the form to see live ML predictions"
        />
        <div className="p-5 space-y-4 max-h-[78vh] overflow-y-auto">
          <Field label="City">
            <select
              data-testid="input-city"
              className="input"
              value={form.city}
              onChange={(e) => {
                const city = e.target.value;
                const newLoc = (meta?.localities_by_city?.[city] || ["Central"])[0];
                setForm((f) => ({ ...f, city, locality: newLoc }));
              }}
            >
              {(meta?.cities || ["Mumbai"]).map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </Field>

          <Field label="Locality (top by listing count)">
            <select
              data-testid="input-locality"
              className="input"
              value={form.locality}
              onChange={(e) => update("locality", e.target.value)}
            >
              {localities.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </Field>

          <RangeField label={`Carpet area  ·  ${form.area_sqft} sqft`} min={250} max={5000} step={25}
            value={form.area_sqft} onChange={(v) => update("area_sqft", v)}
            testid="input-area" />

          <RangeField label={`BHK  ·  ${form.bhk}`} min={1} max={6} step={1}
            value={form.bhk} onChange={(v) => update("bhk", v)} testid="input-bhk" />

          <Field label="Listing">
            <div className="flex gap-2" data-testid="input-resale">
              {[
                { v: 0, l: "New / Under-construction" },
                { v: 1, l: "Resale" },
              ].map((o) => (
                <button
                  key={o.v}
                  type="button"
                  className={`flex-1 px-3 py-2 rounded-md border text-sm font-medium transition-colors ${
                    form.resale === o.v
                      ? "bg-ink-900 text-white border-ink-900"
                      : "bg-white border-line text-ink-700 hover:bg-surface-50"
                  }`}
                  data-testid={`resale-${o.v}`}
                  onClick={() => update("resale", o.v)}
                >
                  {o.l}
                </button>
              ))}
            </div>
          </Field>

          <div>
            <div className="label">Amenities</div>
            <div className="grid grid-cols-2 gap-2">
              {amenityList.map((k) => (
                <label key={k} className="flex items-center gap-2 text-sm bg-surface-50 border border-line rounded-md px-3 py-2 cursor-pointer hover:bg-white">
                  <input
                    type="checkbox"
                    checked={!!form[k]}
                    data-testid={`amenity-${k}`}
                    onChange={(e) => update(k, e.target.checked ? 1 : 0)}
                  />
                  {amenityLabel[k] || k}
                </label>
              ))}
            </div>
          </div>

          <div className="flex gap-2 pt-2">
            <button
              onClick={() => runPredict(form)}
              disabled={loading}
              className="btn-primary flex-1"
              data-testid="predict-btn"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
              {loading ? "Predicting" : "Predict price"}
            </button>
            <button
              onClick={downloadPdf}
              disabled={downloading || !pred}
              className="btn-secondary"
              data-testid="download-pdf-btn"
              title="Download price report (PDF)"
            >
              {downloading ? <Loader2 size={16} className="animate-spin" /> : <FileText size={16} />}
              PDF
            </button>
          </div>
        </div>
      </Card>

      {/* Right: results */}
      <div className="space-y-6">
        {/* Headline result */}
        <Card data-testid="prediction-result-card">
          <div className="grid md:grid-cols-[1.2fr_1fr] gap-6 p-6">
            <div>
              <Badge tone="info">Predicted price · {pred?.output?.best_model || "—"}</Badge>
              <div className="kpi-num mt-3" data-testid="prediction-headline">
                {pred ? inr(pred.output.predicted_price) : "—"}
              </div>
              <div className="text-sm text-ink-500 mt-1 font-mono">
                ≈ ₹{Math.round(pred?.output?.price_per_sqft || 0).toLocaleString("en-IN")}/sqft · {form.locality}, {form.city}
              </div>
              {pred && (
                <div className="mt-4 bg-surface-50 border border-line rounded-md px-4 py-3 text-sm flex items-center gap-3">
                  <BadgeCheck size={16} className="text-emerald-600" />
                  <span>
                    90% confidence: <b>{inr(pred.output.confidence_low)}</b> – <b>{inr(pred.output.confidence_high)}</b>
                  </span>
                </div>
              )}
            </div>
            <div className="space-y-2">
              <div className="text-[10px] tracking-[0.18em] uppercase font-semibold text-ink-500">Across all models</div>
              <div className="space-y-1.5">
                {allPreds.map((r) => (
                  <div key={r.model} className="flex items-center justify-between text-sm bg-white border border-line rounded-md px-3 py-1.5">
                    <span className="flex items-center gap-2">
                      {r.model}
                      {r.best && <Badge tone="success">best</Badge>}
                    </span>
                    <span className="font-mono">
                      {r.price == null ? <span className="text-ink-300">unstable</span> : inr(r.price)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>

        {/* SHAP */}
        <Card>
          <CardHeader
            eyebrow="Explainability"
            title="SHAP contributions"
            sub="Top features pushing the prediction up or down (₹ scale)"
          />
          <div className="p-4 h-80">
            {shap ? (
              <ResponsiveContainer>
                <BarChart
                  data={shap.contributions
                    .slice(0, 10)
                    .map((c) => ({
                      f: c.feature.replace(/^.*?__/, "").slice(0, 24),
                      v: Math.round(c.value),
                    }))
                    .reverse()}
                  layout="vertical"
                  margin={{ top: 8, right: 16, left: 8, bottom: 8 }}
                >
                  <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                  <XAxis type="number" tick={TICK} tickFormatter={inrShort} />
                  <YAxis dataKey="f" type="category" tick={TICK} width={150} />
                  <Tooltip formatter={(v) => [inr(Math.abs(v)) + (v < 0 ? " ↓" : " ↑"), "SHAP"]} />
                  <ReferenceLine x={0} stroke="#94A3B8" />
                  <Bar dataKey="v" radius={[0, 6, 6, 0]}>
                    {shap.contributions.slice(0, 10).reverse().map((c, i) => (
                      <Cell key={i} fill={c.value >= 0 ? "#2563EB" : "#EF4444"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="grid place-items-center h-full text-ink-500 text-sm">Loading…</div>
            )}
          </div>
        </Card>

        {/* What-if */}
        <Card>
          <CardHeader
            eyebrow="What-if"
            title="Sensitivity to area"
            sub="How predicted price moves as carpet area changes"
          />
          <div className="p-4 h-72">
            {whatif ? (
              <ResponsiveContainer>
                <LineChart data={whatif} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                  <XAxis dataKey="value" tick={TICK} unit=" sqft" />
                  <YAxis tick={TICK} tickFormatter={inrShort} />
                  <Tooltip formatter={(v) => [inr(v), "Predicted"]} />
                  <Line type="monotone" dataKey="predicted_price" stroke="#2563EB" strokeWidth={2.5} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="grid place-items-center h-full text-ink-500 text-sm">Loading…</div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div>
      <span className="label">{label}</span>
      {children}
    </div>
  );
}

function RangeField({ label, min, max, step, value, onChange, testid }) {
  return (
    <div>
      <span className="label">{label}</span>
      <input
        type="range"
        className="w-full accent-blue-600"
        min={min} max={max} step={step}
        value={value}
        data-testid={testid}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function inrShort(v) {
  if (v == null) return "";
  if (Math.abs(v) >= 1e7) return `${(v / 1e7).toFixed(1)}Cr`;
  return `${Math.round(v / 1e5)}L`;
}
