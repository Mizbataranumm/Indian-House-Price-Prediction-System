import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  AreaChart, Area, LineChart, Line, ScatterChart, Scatter, ZAxis, Cell,
} from "recharts";
import { TrendingUp, Database, Sparkles, ArrowUpRight, Cpu } from "lucide-react";
import { Card, CardHeader, Stat, Spinner, Badge, ErrorState } from "./ui/Primitives";
import { api, inr, inrCompact, formatApiError } from "../lib/api";

const TICK = { fontSize: 11, fill: "#64748B" };

export default function Dashboard() {
  const [a, setA] = useState(null);
  const [s, setS] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

    (async () => {
      try {
        while (alive) {
          const { data } = await api.get("/status");
          if (!alive) return;
          setStatus(data);
          if (data.trained) {
            const [ar, sr] = await Promise.all([
              api.get("/analytics", { timeout: 180000 }),
              api.get("/dataset/summary", { timeout: 180000 }),
            ]);
            if (!alive) return;
            setA(ar.data);
            setS(sr.data);
            return;
          }
          await sleep(3000);
        }
      } catch (err) {
        if (!alive) return;
        setError(formatApiError(err));
      }
    })();
    return () => { alive = false; };
  }, []);

  if (error) {
    return (
      <div className="space-y-4">
        <ErrorState
          title="Dashboard data could not load"
          message={error}
        />
        <Card className="p-5 text-sm text-ink-500">
          Expected backend routes:
          <div className="font-mono mt-2">/api/status</div>
          <div className="font-mono">/api/analytics</div>
          <div className="font-mono">/api/dataset/summary</div>
        </Card>
      </div>
    );
  }

  if (!a || !s) {
    return (
      <div className="card p-10 grid place-items-center min-h-[60vh]">
        <Spinner
          label={
            status?.trained
              ? "Loading analytics"
              : "Warming ML pipeline on first boot"
          }
        />
      </div>
    );
  }

  const best = a.metrics[a.best_model];
  const cityBars = a.city_stats
    .map((c) => ({ city: c.city, psf: Math.round(c.mean_psf) }))
    .sort((x, y) => y.psf - x.psf);

  return (
    <div className="space-y-6">
      {/* Hero */}
      <section
        className="relative overflow-hidden rounded-xl border border-line bg-white"
        data-testid="hero-section"
      >
        <div className="absolute inset-0 grain opacity-60" />
        <div className="relative grid md:grid-cols-[1.4fr_1fr] gap-6 p-6 sm:p-8">
          <div>
            <Badge tone="info">Real Kaggle data · India 2026</Badge>
            <h1 className="font-display text-4xl sm:text-5xl font-extrabold tracking-tighter mt-3">
              Predict Indian property prices
              <br />
              with explainable machine learning.
            </h1>
            <p className="text-ink-500 mt-3 max-w-xl">
              A six-model ensemble trained on a metropolitan India housing
              corpus. Live predictions, confidence intervals, SHAP
              attributions and a downloadable price report.
            </p>
            <div className="flex flex-wrap gap-3 mt-5">
              <Link to="/predict" className="btn-primary" data-testid="cta-predict">
                <Sparkles size={16} /> Run a prediction
              </Link>
              <Link to="/models" className="btn-secondary" data-testid="cta-models">
                <Cpu size={16} /> Inspect models
                <ArrowUpRight size={14} />
              </Link>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <Stat
              label="Rows trained"
              value={s.n_rows.toLocaleString("en-IN")}
              sub="after IQR cleaning"
              testid="stat-rows"
            />
            <Stat
              label="Best R²"
              value={(best.r2 * 100).toFixed(1) + "%"}
              sub={a.best_model}
              accent="text-emerald-600"
              testid="stat-best-r2"
            />
            <Stat
              label="Median price"
              value={inrCompact(s.price_median)}
              sub="dataset median"
              testid="stat-median"
            />
            <Stat
              label="Cities covered"
              value={s.cities.length}
              sub="Tier-1 metros"
              testid="stat-cities"
            />
          </div>
        </div>
      </section>

      {/* Model snapshot */}
      <section className="grid lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader
            eyebrow="Model leaderboard"
            title="R² across six models"
            sub="Higher is better. Auto-selected winner highlighted."
          />
          <div className="p-4 h-72">
            <ResponsiveContainer>
              <BarChart
                data={Object.entries(a.metrics).map(([n, m]) => ({
                  name: n, r2: +(m.r2 * 100).toFixed(2),
                }))}
                margin={{ top: 8, right: 12, left: 0, bottom: 8 }}
              >
                <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={TICK} interval={0} angle={-12} dy={8} height={50} />
                <YAxis tick={TICK} unit="%" domain={[0, 100]} />
                <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #E2E8F0" }} />
                <Bar dataKey="r2" radius={[6, 6, 0, 0]}>
                  {Object.entries(a.metrics).map(([n], i) => (
                    <Cell key={i} fill={n === a.best_model ? "#2563EB" : "#94A3B8"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <CardHeader
            eyebrow="Top features"
            title="What drives price"
            sub="From the winning model"
          />
          <div className="p-4 h-72">
            <ResponsiveContainer>
              <BarChart
                data={a.feature_importance.slice(0, 8).map((d) => ({
                  feature: d.feature.replace(/^.*?__/, "").slice(0, 18),
                  v: +d.importance.toFixed(4),
                })).reverse()}
                layout="vertical"
                margin={{ top: 4, right: 12, left: 12, bottom: 4 }}
              >
                <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                <XAxis type="number" tick={TICK} />
                <YAxis dataKey="feature" type="category" tick={TICK} width={100} />
                <Tooltip contentStyle={{ borderRadius: 8, border: "1px solid #E2E8F0" }} />
                <Bar dataKey="v" fill="#2563EB" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </section>

      {/* Distribution + city */}
      <section className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader
            eyebrow="Market"
            title="Price distribution"
            sub="In ₹ Lakhs across the dataset"
          />
          <div className="p-4 h-72">
            <ResponsiveContainer>
              <AreaChart data={a.price_histogram}>
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#2563EB" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#2563EB" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                <XAxis dataKey="bin" tick={TICK} tickFormatter={(v) => `${Math.round(v)}L`} />
                <YAxis tick={TICK} />
                <Tooltip
                  formatter={(v) => [v, "Properties"]}
                  labelFormatter={(v) => `~₹${Math.round(v)} L`}
                />
                <Area type="monotone" dataKey="count" stroke="#2563EB" fill="url(#g1)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <CardHeader
            eyebrow="Geography"
            title="City-tier price per sqft"
            sub="Mean ₹/sqft across cities"
          />
          <div className="p-4 h-72">
            <ResponsiveContainer>
              <BarChart data={cityBars} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
                <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                <XAxis dataKey="city" tick={TICK} interval={0} angle={-25} dy={10} height={60} />
                <YAxis tick={TICK} tickFormatter={(v) => `₹${v / 1000}k`} />
                <Tooltip formatter={(v) => [`₹${v.toLocaleString("en-IN")}`, "Mean ₹/sqft"]} />
                <Bar dataKey="psf" fill="#0F172A" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </section>

      {/* Scatter + summary */}
      <section className="grid lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader
            eyebrow="Relationship"
            title="Price vs Carpet area"
            sub="Sampled 500 listings across cities"
          />
          <div className="p-4 h-80">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
                <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                <XAxis type="number" dataKey="area" tick={TICK} unit=" sqft" />
                <YAxis
                  type="number"
                  dataKey="price"
                  tick={TICK}
                  tickFormatter={(v) => (v >= 1e7 ? `${(v / 1e7).toFixed(1)}Cr` : `${Math.round(v / 1e5)}L`)}
                />
                <ZAxis range={[40, 40]} />
                <Tooltip
                  formatter={(v, n) => (n === "price" ? [inr(v), "Price"] : [v, "Area"])}
                />
                <Scatter data={a.price_vs_area} fill="#2563EB" fillOpacity={0.55} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <CardHeader
            eyebrow="Pipeline"
            title="Best model"
            sub={a.best_model}
            right={<Badge tone="success">Auto-selected</Badge>}
          />
          <div className="p-5 space-y-3 text-sm">
            <Row k="R² Score" v={(best.r2 * 100).toFixed(2) + "%"} />
            <Row k="MAE" v={inr(best.mae)} />
            <Row k="RMSE" v={inr(best.rmse)} />
            <Row k="MAPE" v={best.mape.toFixed(2) + "%"} />
            <Row k="CV R² (5-fold)" v={(best.cv_r2 * 100).toFixed(2) + "%"} />
            <div className="pt-3 mt-3 border-t border-line text-xs text-ink-500 flex items-center gap-2">
              <Database size={14} /> Train {a.train_size} · Test {a.test_size}
            </div>
          </div>
        </Card>
      </section>
    </div>
  );
}

function Row({ k, v }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-line/60 pb-2">
      <span className="text-ink-500">{k}</span>
      <span className="font-mono text-ink-900">{v}</span>
    </div>
  );
}
