import axios from "axios";

const BASE = process.env.REACT_APP_BACKEND_URL;
export const api = axios.create({ baseURL: `${BASE}/api`, timeout: 60000 });

const SID_KEY = "bharat_estate_sid";
export function sessionId() {
  let s = localStorage.getItem(SID_KEY);
  if (!s) {
    s =
      "sid_" +
      Math.random().toString(36).slice(2, 10) +
      Date.now().toString(36);
    localStorage.setItem(SID_KEY, s);
  }
  return s;
}

export function inr(v, opts = {}) {
  if (v == null || isNaN(v)) return "—";
  if (v >= 1e7) return `₹${(v / 1e7).toFixed(opts.dp ?? 2)} Cr`;
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(opts.dp ?? 2)} L`;
  return `₹${Math.round(v).toLocaleString("en-IN")}`;
}

export function inrCompact(v) {
  if (v == null || isNaN(v)) return "—";
  if (v >= 1e7) return `${(v / 1e7).toFixed(2)} Cr`;
  if (v >= 1e5) return `${(v / 1e5).toFixed(2)} L`;
  return Math.round(v).toLocaleString("en-IN");
}
