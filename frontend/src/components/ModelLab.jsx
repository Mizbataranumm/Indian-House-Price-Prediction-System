import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  LineChart, Line, Legend,
} from "recharts";
import { Card, CardHeader, Spinner, Badge } from "./ui/Primitives";
import { api, inr } from "../lib/api";

const TICK = { fontSize: 11, fill: "#64748B" };

export default function ModelLab() {
  const [a, setA] = useState(null);
  useEffect(() => { api.get("/analytics").then(({ data }) => setA(data)); }, []);
  if (!a) return <div className="card p-10"><Spinner /></div>;

  const rows = Object.entries(a.metrics).map(([n, m]) => ({
    name: n, ...m, r2_pct: +(m.r2 * 100).toFixed(2),
    cv_pct: +(m.cv_r2 * 100).toFixed(2),
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          eyebrow="Leaderboard"
          title="Model comparison"
          sub="Six regressors trained on the same train/test split (80/20)."
          right={<Badge tone="success">Best · {a.best_model}</Badge>}
        />
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm" data-testid="models-table">
            <thead className="bg-surface-50 text-ink-500">
              <tr className="text-left">
                {["Model", "R²", "CV R² (5)", "MAE", "RMSE", "MAPE", "Train (s)"].map((h) => (
                  <th key={h} className="px-4 py-3 text-[11px] uppercase tracking-[0.14em] font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name} className={`border-t border-line hover:bg-surface-50 ${r.name === a.best_model ? "bg-blue-50/50" : ""}`}>
                  <td className="px-4 py-3 font-medium">
                    {r.name}
                    {r.name === a.best_model && <span className="ml-2"><Badge tone="success">best</Badge></span>}
                  </td>
                  <td className="px-4 py-3 font-mono">{r.r2_pct}%</td>
                  <td className="px-4 py-3 font-mono">{isNaN(r.cv_pct) ? "—" : r.cv_pct + "%"}</td>
                  <td className="px-4 py-3 font-mono">{inr(r.mae)}</td>
                  <td className="px-4 py-3 font-mono">{inr(r.rmse)}</td>
                  <td className="px-4 py-3 font-mono">{r.mape.toFixed(2)}%</td>
                  <td className="px-4 py-3 font-mono">{r.train_time_sec}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader eyebrow="Performance" title="R² vs CV R²"
            sub="Bias-variance signal" />
          <div className="p-4 h-72">
            <ResponsiveContainer>
              <BarChart data={rows.map(r => ({ name: r.name, R2: r.r2_pct, CV: r.cv_pct }))}>
                <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={TICK} interval={0} angle={-15} dy={8} height={50} />
                <YAxis tick={TICK} unit="%" />
                <Tooltip />
                <Legend />
                <Bar dataKey="R2" fill="#2563EB" radius={[6, 6, 0, 0]} />
                <Bar dataKey="CV" fill="#94A3B8" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <CardHeader eyebrow="Calibration" title="Learning curve"
            sub="Best model train vs validation R²" />
          <div className="p-4 h-72">
            <ResponsiveContainer>
              <LineChart data={a.learning_curve}>
                <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                <XAxis dataKey="size" tick={TICK} />
                <YAxis tick={TICK} domain={[0, 1]} />
                <Tooltip formatter={(v) => (v * 100).toFixed(1) + "%"} />
                <Legend />
                <Line dataKey="train_r2" stroke="#0F172A" strokeWidth={2} dot />
                <Line dataKey="val_r2" stroke="#2563EB" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card>
        <CardHeader eyebrow="Diagnostics" title="Residual plot"
          sub={`Predicted vs residuals · ${a.best_model}`} />
        <div className="p-4 h-80">
          <ResponsiveContainer>
            <BarChart data={a.residuals.slice(0, 200).map(r => ({
              x: Math.round(r.predicted / 1e5), y: Math.round(r.residual / 1e5),
            }))}>
              <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
              <XAxis dataKey="x" tick={TICK} unit="L" />
              <YAxis dataKey="y" tick={TICK} unit="L" />
              <Tooltip formatter={(v, n) => [`₹${v}L`, n === "y" ? "Residual" : "Predicted"]} />
              <Bar dataKey="y" fill="#94A3B8" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </div>
  );
}
