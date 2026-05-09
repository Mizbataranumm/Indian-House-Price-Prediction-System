import axios from "axios";

const RAW_BASE = (process.env.REACT_APP_BACKEND_URL || "").trim().replace(/\/+$/, "");
export const backendConfigured = Boolean(RAW_BASE);
export const apiBase = backendConfigured ? `${RAW_BASE}/api` : "/api";
export const api = axios.create({ baseURL: apiBase, timeout: 60000 });

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
  if (v == null || isNaN(v)) return "--";
  if (v >= 1e7) return `Rs ${Number(v / 1e7).toFixed(opts.dp ?? 2)} Cr`;
  if (v >= 1e5) return `Rs ${Number(v / 1e5).toFixed(opts.dp ?? 2)} L`;
  return `Rs ${Math.round(v).toLocaleString("en-IN")}`;
}

export function inrCompact(v) {
  if (v == null || isNaN(v)) return "--";
  if (v >= 1e7) return `${(v / 1e7).toFixed(2)} Cr`;
  if (v >= 1e5) return `${(v / 1e5).toFixed(2)} L`;
  return Math.round(v).toLocaleString("en-IN");
}

export function formatApiError(error) {
  if (!backendConfigured) {
    return "Frontend deployment is missing REACT_APP_BACKEND_URL. Point it to your Render backend service and rebuild.";
  }
  if (error?.code === "ECONNABORTED") {
    return "The backend took too long to respond. On first boot, Render may need about a minute to warm the ML pipeline.";
  }
  if (error?.response?.status) {
    return `Backend request failed with status ${error.response.status}. Check the Render backend logs for the failing endpoint.`;
  }
  return "Could not reach the backend service. Verify the backend URL, CORS, and whether the Render API service is healthy.";
}
