import React, { useEffect, useState } from "react";
import { Card, CardHeader, Spinner } from "./ui/Primitives";
import { api, inr, sessionId } from "../lib/api";
import { Trash2 } from "lucide-react";

export default function History() {
  const [items, setItems] = useState(null);

  async function load() {
    const { data } = await api.get(`/history?session_id=${sessionId()}&limit=100`);
    setItems(data.items);
  }
  useEffect(() => { load(); }, []);

  async function clearAll() {
    await api.delete(`/history?session_id=${sessionId()}`);
    load();
  }

  if (!items) return <div className="card p-10"><Spinner /></div>;

  return (
    <Card>
      <CardHeader
        eyebrow="Session history"
        title="Your recent predictions"
        sub={`Stored locally per device · ${items.length} entries`}
        right={
          items.length > 0 && (
            <button className="btn-secondary" onClick={clearAll} data-testid="clear-history">
              <Trash2 size={14} /> Clear
            </button>
          )
        }
      />
      {items.length === 0 ? (
        <div className="p-10 text-center text-ink-500 text-sm">
          No predictions yet. Run a prediction to see history here.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm" data-testid="history-table">
            <thead className="bg-surface-50 text-ink-500">
              <tr className="text-left">
                {["When", "City", "Locality", "BHK", "Area", "Best model", "Predicted"].map((h) => (
                  <th key={h} className="px-4 py-3 text-[11px] uppercase tracking-[0.14em] font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.id} className="border-t border-line hover:bg-surface-50">
                  <td className="px-4 py-3 font-mono text-xs text-ink-500">
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                  <td className="px-4 py-3">{r.input.city}</td>
                  <td className="px-4 py-3">{r.input.locality}</td>
                  <td className="px-4 py-3">{r.input.bhk}</td>
                  <td className="px-4 py-3 font-mono">{r.input.area_sqft} sqft</td>
                  <td className="px-4 py-3 text-xs">{r.output.best_model}</td>
                  <td className="px-4 py-3 font-mono font-semibold">{inr(r.output.predicted_price)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
