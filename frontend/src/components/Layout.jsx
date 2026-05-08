import React from "react";
import { NavLink, Outlet } from "react-router-dom";
import {
  LayoutDashboard,
  LineChart,
  Sparkles,
  GitCompareArrows,
  Calculator,
  History,
  BarChart3,
} from "lucide-react";
import Logo from "./ui/Logo";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/predict", label: "Predict", icon: Sparkles },
  { to: "/models", label: "Model Lab", icon: BarChart3 },
  { to: "/analytics", label: "Analytics", icon: LineChart },
  { to: "/compare", label: "Compare", icon: GitCompareArrows },
  { to: "/calculators", label: "Calculators", icon: Calculator },
  { to: "/history", label: "History", icon: History },
];

export default function Layout() {
  return (
    <div className="min-h-screen bg-surface-50 text-ink-900">
      {/* Top bar */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-white/80 border-b border-line">
        <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Logo size={36} />
            <div className="leading-tight">
              <div className="font-display font-bold tracking-tight">Bharat Estate</div>
              <div className="text-[10px] tracking-[0.2em] uppercase text-ink-500">
                Indian Property Analytics
              </div>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-2 text-xs text-ink-500">
            <span className="font-mono">v1.0</span>
            <span className="w-1 h-1 rounded-full bg-line-strong" />
            <span>Indian Metropolitan Housing</span>
          </div>
        </div>
      </header>

      <div className="max-w-[1400px] mx-auto grid grid-cols-1 lg:grid-cols-[240px_1fr] gap-6 px-4 sm:px-6 lg:px-8 py-6">
        {/* Sidebar */}
        <aside className="hidden lg:block">
          <nav className="sticky top-24 space-y-1" data-testid="sidebar-nav">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                data-testid={`nav-${label.toLowerCase().replace(/\s/g, "-")}`}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-ink-900 text-white hover:bg-ink-900 hover:text-white"
                      : "text-ink-700 hover:bg-surface-100 hover:text-ink-900"
                  }`
                }
              >
                <Icon size={16} />
                <span>{label}</span>
              </NavLink>
            ))}
            <div className="mt-6 px-3 text-[10px] tracking-[0.18em] uppercase text-ink-500">
              v1.0 · Live data
            </div>
          </nav>
        </aside>

        {/* Mobile nav */}
        <nav className="lg:hidden -mx-4 px-4 overflow-x-auto flex gap-2 pb-3 scroll-fade">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `whitespace-nowrap flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium border ${
                  isActive
                    ? "bg-ink-900 text-white border-ink-900"
                    : "bg-white border-line text-ink-700 hover:bg-surface-50 hover:text-ink-900"
                }`
              }
            >
              <Icon size={14} /> {label}
            </NavLink>
          ))}
        </nav>

        <main className="min-w-0">
          <Outlet />
        </main>
      </div>

      <footer className="mt-12 border-t border-line bg-white">
        <div className="max-w-[1400px] mx-auto px-4 sm:px-6 lg:px-8 py-6 text-xs text-ink-500 flex flex-col sm:flex-row gap-2 sm:items-center justify-between">
          <span>© Bharat Estate — Indian metropolitan housing analytics.</span>
          <span className="font-mono">scikit-learn · XGBoost · SHAP · FastAPI</span>
        </div>
      </footer>
    </div>
  );
}
