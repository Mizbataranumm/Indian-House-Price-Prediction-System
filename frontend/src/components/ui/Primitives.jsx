import React from "react";

export function Card({ className = "", children, ...rest }) {
  return (
    <div
      className={`card ${className}`}
      {...rest}
    >
      {children}
    </div>
  );
}

export function CardHeader({ title, sub, right, eyebrow, className = "" }) {
  return (
    <div className={`flex items-start justify-between gap-4 p-5 border-b border-line ${className}`}>
      <div>
        {eyebrow && (
          <div className="text-[10px] tracking-[0.18em] uppercase font-semibold text-ink-500 mb-1">
            {eyebrow}
          </div>
        )}
        <h3 className="font-display text-lg sm:text-xl font-bold text-ink-900 tracking-tight">
          {title}
        </h3>
        {sub && <p className="text-sm text-ink-500 mt-1">{sub}</p>}
      </div>
      {right}
    </div>
  );
}

export function Stat({ label, value, sub, accent = "text-ink-900", testid }) {
  return (
    <Card className="p-5" data-testid={testid}>
      <div className="text-[10px] tracking-[0.18em] uppercase font-semibold text-ink-500">
        {label}
      </div>
      <div className={`kpi-num mt-2 ${accent}`}>{value}</div>
      {sub && <div className="text-xs text-ink-500 mt-1">{sub}</div>}
    </Card>
  );
}

export function Spinner({ label = "Loading" }) {
  return (
    <div className="flex items-center gap-3 text-ink-500" data-testid="spinner">
      <span className="inline-block w-4 h-4 border-2 border-line border-t-brand rounded-full animate-spin" />
      <span className="text-sm">{label}…</span>
    </div>
  );
}

export function Badge({ children, tone = "neutral" }) {
  const tones = {
    neutral: "bg-surface-100 text-ink-700 border-line",
    success: "bg-emerald-50 text-emerald-700 border-emerald-200",
    info: "bg-blue-50 text-blue-700 border-blue-200",
    warning: "bg-amber-50 text-amber-700 border-amber-200",
  };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 text-[10px] font-semibold tracking-[0.14em] uppercase border rounded ${tones[tone]}`}>
      {children}
    </span>
  );
}

export function ErrorState({ title = "Something went wrong", message }) {
  return (
    <div className="rounded-xl border border-rose-200 bg-rose-50 px-5 py-4 text-sm text-rose-900">
      <div className="font-semibold">{title}</div>
      <div className="mt-1 text-rose-800/90">{message}</div>
    </div>
  );
}
