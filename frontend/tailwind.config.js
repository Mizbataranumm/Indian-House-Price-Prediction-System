/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./public/index.html"],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Cabinet Grotesk"', '"Plus Jakarta Sans"', "ui-sans-serif", "system-ui"],
        sans: ['"IBM Plex Sans"', "ui-sans-serif", "system-ui"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      colors: {
        ink: { 900: "#0F172A", 700: "#334155", 500: "#475569", 300: "#94A3B8" },
        line: { DEFAULT: "#E2E8F0", strong: "#CBD5E1" },
        brand: { DEFAULT: "#2563EB", dark: "#1D4ED8" },
        surface: { 50: "#F8FAFC", 100: "#F1F5F9", 200: "#E2E8F0" },
      },
      boxShadow: {
        card: "0 1px 2px 0 rgba(15,23,42,0.04), 0 1px 1px 0 rgba(15,23,42,0.03)",
        cardHover: "0 8px 24px -8px rgba(15,23,42,0.12)",
      },
    },
  },
  plugins: [],
};
