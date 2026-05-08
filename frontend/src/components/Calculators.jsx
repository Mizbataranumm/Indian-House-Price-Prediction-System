import React, { useState } from "react";
import { Card, CardHeader } from "./ui/Primitives";
import { api, inr } from "../lib/api";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
  PieChart, Pie, Cell,
} from "recharts";

const TICK = { fontSize: 11, fill: "#64748B" };

export default function Calculators() {
  return (
    <div className="space-y-6">
      <EmiCalc />
      <RoiCalc />
    </div>
  );
}

function EmiCalc() {
  const [loan, setLoan] = useState(5000000);
  const [rate, setRate] = useState(8.5);
  const [years, setYears] = useState(20);
  const [data, setData] = useState(null);

  async function run() {
    const { data } = await api.post("/emi", { loan_amount: loan, rate, tenure_years: years });
    setData(data);
  }

  return (
    <Card>
      <CardHeader eyebrow="Mortgage" title="EMI Calculator" sub="Indian rupee monthly installment" />
      <div className="grid lg:grid-cols-[380px_1fr] gap-6 p-5">
        <div className="space-y-4">
          <Range label={`Loan amount · ${inr(loan)}`} min={500000} max={50000000} step={50000}
            value={loan} onChange={setLoan} testid="emi-loan" />
          <Range label={`Interest rate · ${rate}%`} min={6} max={15} step={0.05}
            value={rate} onChange={setRate} testid="emi-rate" />
          <Range label={`Tenure · ${years} years`} min={1} max={30} step={1}
            value={years} onChange={setYears} testid="emi-tenure" />
          <button className="btn-primary w-full" onClick={run} data-testid="emi-calc">Calculate EMI</button>
          {data && (
            <div className="grid grid-cols-3 gap-2 pt-2">
              <Mini label="Monthly EMI" v={inr(data.emi)} />
              <Mini label="Total interest" v={inr(data.total_interest)} />
              <Mini label="Total payable" v={inr(data.total_payment)} />
            </div>
          )}
        </div>
        <div className="h-80">
          {data ? (
            <ResponsiveContainer>
              <AreaChart data={data.schedule_yearly.map((s, i) => ({
                year: i + 1, balance: s.balance, principal: s.principal, interest: s.interest,
              }))}>
                <defs>
                  <linearGradient id="bal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#2563EB" stopOpacity={0.5} />
                    <stop offset="100%" stopColor="#2563EB" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#E5E7EB" strokeDasharray="3 3" />
                <XAxis dataKey="year" tick={TICK} unit="y" />
                <YAxis tick={TICK} tickFormatter={(v) => v >= 1e7 ? `${(v / 1e7).toFixed(1)}Cr` : `${Math.round(v / 1e5)}L`} />
                <Tooltip formatter={(v) => inr(v)} />
                <Legend />
                <Area type="monotone" dataKey="balance" stroke="#2563EB" fill="url(#bal)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="grid place-items-center h-full text-sm text-ink-500 border border-dashed border-line rounded-md">
              Enter loan parameters & calculate
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

function RoiCalc() {
  const [purchase, setPurchase] = useState(8000000);
  const [rent, setRent] = useState(20000);
  const [appr, setAppr] = useState(7);
  const [years, setYears] = useState(7);
  const [exp, setExp] = useState(20000);
  const [data, setData] = useState(null);

  async function run() {
    const { data } = await api.post("/roi", {
      purchase_price: purchase, monthly_rent: rent,
      annual_appreciation: appr, holding_years: years, annual_expense: exp,
    });
    setData(data);
  }

  const COLORS = ["#2563EB", "#10B981", "#F59E0B"];

  return (
    <Card>
      <CardHeader eyebrow="Investment" title="ROI Calculator"
        sub="Project rental income, appreciation and CAGR over a holding period." />
      <div className="grid lg:grid-cols-[380px_1fr] gap-6 p-5">
        <div className="space-y-4">
          <Range label={`Purchase price · ${inr(purchase)}`} min={1000000} max={50000000} step={100000}
            value={purchase} onChange={setPurchase} testid="roi-price" />
          <Range label={`Monthly rent · ${inr(rent)}`} min={0} max={300000} step={1000}
            value={rent} onChange={setRent} testid="roi-rent" />
          <Range label={`Appreciation · ${appr}% / yr`} min={0} max={15} step={0.25}
            value={appr} onChange={setAppr} testid="roi-appr" />
          <Range label={`Holding · ${years} years`} min={1} max={25} step={1}
            value={years} onChange={setYears} testid="roi-years" />
          <Range label={`Annual expense · ${inr(exp)}`} min={0} max={500000} step={1000}
            value={exp} onChange={setExp} testid="roi-exp" />
          <button className="btn-primary w-full" onClick={run} data-testid="roi-calc">Calculate ROI</button>
          {data && (
            <div className="grid grid-cols-2 gap-2 pt-2">
              <Mini label="Future value" v={inr(data.future_value)} />
              <Mini label="Net profit" v={inr(data.net_profit)} />
              <Mini label="ROI" v={data.roi_pct.toFixed(1) + "%"} />
              <Mini label="CAGR" v={data.cagr_pct.toFixed(2) + "%"} />
            </div>
          )}
        </div>
        <div className="h-80">
          {data ? (
            <ResponsiveContainer>
              <PieChart>
                <Pie
                  data={[
                    { name: "Appreciation", value: Math.max(0, data.future_value - purchase) },
                    { name: "Rental income", value: data.rental_income },
                    { name: "Expenses", value: data.total_expense },
                  ]}
                  innerRadius={60} outerRadius={110} paddingAngle={3} dataKey="value"
                >
                  {COLORS.map((c, i) => <Cell key={i} fill={c} />)}
                </Pie>
                <Tooltip formatter={(v) => inr(v)} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="grid place-items-center h-full text-sm text-ink-500 border border-dashed border-line rounded-md">
              Enter investment parameters & calculate
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}

function Range({ label, min, max, step, value, onChange, testid }) {
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

function Mini({ label, v }) {
  return (
    <div className="bg-surface-50 border border-line rounded-md p-2">
      <div className="text-[10px] tracking-[0.14em] uppercase text-ink-500">{label}</div>
      <div className="font-mono font-semibold">{v}</div>
    </div>
  );
}
