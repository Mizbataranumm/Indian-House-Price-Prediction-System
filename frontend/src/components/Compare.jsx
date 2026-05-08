import React, { useEffect, useMemo, useState } from "react";
import { Card, CardHeader, Spinner } from "./ui/Primitives";
import { api, inr } from "../lib/api";
import { Plus, Trash2, Loader2 } from "lucide-react";

const DEFAULT_PROP = {
  city: "Mumbai", locality: "Kharghar",
  bhk: 2, area_sqft: 900, resale: 1, tier: 1,
  amen_security: 1, amen_power_backup: 1, amen_parking: 1, amen_lift: 1,
};

export default function Compare() {
  const [meta, setMeta] = useState(null);
  const [props, setProps] = useState([
    { ...DEFAULT_PROP, city: "Mumbai", area_sqft: 900 },
    { ...DEFAULT_PROP, city: "Bangalore", area_sqft: 1100 },
  ]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.get("/dataset/summary").then(({ data }) => {
      setMeta(data);
      // Update default localities
      setProps((p) =>
        p.map((x) => ({
          ...x,
          locality: data.localities_by_city[x.city]?.[0] || x.locality,
        }))
      );
    });
  }, []);

  const update = (i, k, v) =>
    setProps((p) => p.map((x, j) => (j === i ? { ...x, [k]: v } : x)));
  const remove = (i) => setProps((p) => p.filter((_, j) => j !== i));
  const add = () => setProps((p) => [...p, { ...DEFAULT_PROP }]);

  async function run() {
    setLoading(true);
    try {
      const { data } = await api.post("/predict/compare", { properties: props });
      setResults(data.results);
    } finally {
      setLoading(false);
    }
  }

  const best = useMemo(() => {
    if (!results) return -1;
    let mn = Infinity, idx = -1;
    results.forEach((r, i) => {
      if (r.output.predicted_price < mn) {
        mn = r.output.predicted_price;
        idx = i;
      }
    });
    return idx;
  }, [results]);

  if (!meta) return <div className="card p-10"><Spinner /></div>;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          eyebrow="Property comparison"
          title="Side-by-side analyzer"
          sub="Compare up to 5 properties on the real Kaggle dataset."
          right={
            <div className="flex gap-2">
              <button onClick={add} className="btn-secondary" disabled={props.length >= 5} data-testid="add-property">
                <Plus size={14} /> Add
              </button>
              <button onClick={run} className="btn-primary" data-testid="compare-run">
                {loading ? <Loader2 size={16} className="animate-spin" /> : null}
                Compare
              </button>
            </div>
          }
        />
        <div className="p-4 grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {props.map((p, i) => (
            <div key={i} className="border border-line rounded-md p-3 bg-surface-50">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm font-semibold">Property #{i + 1}</div>
                <button onClick={() => remove(i)} className="text-ink-500 hover:text-red-500" data-testid={`remove-${i}`}>
                  <Trash2 size={14} />
                </button>
              </div>
              <div className="space-y-2 text-sm">
                <select className="input" value={p.city}
                  onChange={(e) => {
                    const city = e.target.value;
                    const newLoc = meta.localities_by_city[city]?.[0] || "Central";
                    update(i, "city", city);
                    update(i, "locality", newLoc);
                  }}>
                  {meta.cities.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                <select className="input" value={p.locality}
                  onChange={(e) => update(i, "locality", e.target.value)}>
                  {(meta.localities_by_city[p.city] || []).map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
                <div className="grid grid-cols-2 gap-2">
                  <NumIn label="BHK" v={p.bhk} on={(v) => update(i, "bhk", v)} />
                  <NumIn label="Area" v={p.area_sqft} on={(v) => update(i, "area_sqft", v)} />
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {results && (
        <Card>
          <CardHeader title="Results" sub="Lowest predicted price highlighted" />
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-surface-50 text-ink-500">
                <tr className="text-left">
                  {["#", "City", "Locality", "BHK", "Area", "Predicted", "₹/sqft", "Confidence"].map((h) => (
                    <th key={h} className="px-4 py-3 text-[11px] uppercase tracking-[0.14em] font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className={`border-t border-line ${i === best ? "bg-emerald-50/60" : ""}`}>
                    <td className="px-4 py-3">{i + 1}{i === best && " · best deal"}</td>
                    <td className="px-4 py-3">{r.input.city}</td>
                    <td className="px-4 py-3">{r.input.locality}</td>
                    <td className="px-4 py-3">{r.input.bhk}</td>
                    <td className="px-4 py-3 font-mono">{r.input.area_sqft} sqft</td>
                    <td className="px-4 py-3 font-mono font-semibold">{inr(r.output.predicted_price)}</td>
                    <td className="px-4 py-3 font-mono">₹{Math.round(r.output.price_per_sqft).toLocaleString("en-IN")}</td>
                    <td className="px-4 py-3 font-mono text-ink-500">
                      {inr(r.output.confidence_low)} – {inr(r.output.confidence_high)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}

function NumIn({ label, v, on }) {
  return (
    <label className="text-xs text-ink-500">
      <span>{label}</span>
      <input
        type="number"
        value={v}
        onChange={(e) => on(Number(e.target.value))}
        className="input mt-0.5 text-sm"
      />
    </label>
  );
}
