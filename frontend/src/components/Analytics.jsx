import React, { useEffect, useState } from "react";
import { Card, CardHeader, Spinner } from "./ui/Primitives";
import { api, inr } from "../lib/api";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
} from "recharts";

const TICK = { fontSize: 11, fill: "#64748B" };

function colorFor(v) {
  // -1..1 -> blue scale (negative: red, positive: blue, zero: white)
  const a = Math.max(-1, Math.min(1, v));
  if (a >= 0) {
    const t = a; // 0..1
    const r = Math.round(255 - t * (255 - 37));
    const g = Math.round(255 - t * (255 - 99));
    const b = Math.round(255 - t * (255 - 235));
    return `rgb(${r},${g},${b})`;
  }
  const t = -a;
  const r = Math.round(255 - t * (255 - 239));
  const g = Math.round(255 - t * (255 - 68));
  const b = Math.round(255 - t * (255 - 68));
  return `rgb(${r},${g},${b})`;
}

export default function Analytics() {
  const [a, setA] = useState(null);
  useEffect(() => { api.get("/analytics").then(({ data }) => setA(data)); }, []);
  if (!a) return <div className="card p-10"><Spinner /></div>;

  const labels = a.correlation.labels;
  const matrix = a.correlation.matrix;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader eyebrow="Statistics" title="Correlation heatmap"
          sub="Pearson correlation between numeric features and price" />
        <div className="p-4 overflow-x-auto" data-testid="corr-heatmap">
          <table className="text-[10px] sm:text-xs font-mono border-separate border-spacing-0">
            <thead>
              <tr>
                <th className="bg-white sticky left-0"></th>
                {labels.map((l) => (
                  <th key={l} className="px-1.5 py-1 text-ink-500 font-semibold whitespace-nowrap rotate-[-30deg] origin-bottom-left h-16 align-bottom">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {labels.map((row, i) => (
                <tr key={row}>
                  <td className="bg-white sticky left-0 px-2 py-1 text-ink-500 font-semibold whitespace-nowrap">{row}</td>
                  {labels.map((col, j) => {
                    const v = matrix[i][j];
                    return (
                      <td
                        key={col}
                        className="text-center w-9 h-9 border border-white"
                        style={{ background: colorFor(v), color: Math.abs(v) > 0.55 ? "white" : "#0F172A" }}
                        title={`${row} ↔ ${col}: ${v}`}
                      >
                        {v.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader eyebrow="Feature impact" title="Importance ranking"
            sub="From the gradient-boosted model" />
          <div className="p-4 h-96">
            <ResponsiveContainer>
              <BarChart
                data={a.feature_importance.slice(0, 14)
                  .map((d) => ({ feature: d.feature.replace(/^.*?__/, ""), v: +d.importance.toFixed(4) }))
                  .reverse()}
                layout="vertical"
                margin={{ top: 4, right: 16, left: 12, bottom: 4 }}
              >
                <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                <XAxis type="number" tick={TICK} />
                <YAxis dataKey="feature" type="category" tick={TICK} width={150} />
                <Tooltip />
                <Bar dataKey="v" fill="#0F172A" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card>
          <CardHeader eyebrow="Markets" title="City-tier comparison"
            sub="Mean ₹/sqft and median price by city" />
          <div className="p-4 overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-surface-50 text-ink-500">
                <tr className="text-left">
                  {["City", "Tier", "Mean ₹/sqft", "Median price", "Listings"].map((h) => (
                    <th key={h} className="px-3 py-2 text-[11px] uppercase tracking-[0.14em] font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {a.city_stats
                  .slice()
                  .sort((x, y) => y.mean_psf - x.mean_psf)
                  .map((c) => (
                    <tr key={c.city} className="border-t border-line hover:bg-surface-50">
                      <td className="px-3 py-2 font-medium">{c.city}</td>
                      <td className="px-3 py-2">T{c.tier}</td>
                      <td className="px-3 py-2 font-mono">₹{Math.round(c.mean_psf).toLocaleString("en-IN")}</td>
                      <td className="px-3 py-2 font-mono">{inr(c.median_price)}</td>
                      <td className="px-3 py-2 font-mono">{c.count}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </div>
  );
}
