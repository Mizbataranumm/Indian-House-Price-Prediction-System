import React from "react";

/**
 * Bharat Estate logomark.
 *
 * Concept: a minimalist roof-line that doubles as a rising data series.
 * The two stepped peaks form the silhouette of paired buildings; the small
 * dot is the "current prediction" point on the trend. Colours adapt to the
 * surrounding text colour via `currentColor`, so the same component works
 * on the dark tile in the header, on a white card, or inside the PDF.
 */
export default function Logo({ size = 36, className = "" }) {
  return (
    <span
      className={`inline-grid place-items-center rounded-md bg-ink-900 text-white ${className}`}
      style={{ width: size, height: size }}
      aria-label="Bharat Estate logo"
    >
      <svg
        viewBox="0 0 32 32"
        width={size * 0.62}
        height={size * 0.62}
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {/* roof / trend line */}
        <path d="M3 19 L11 11 L17 16 L23 8 L29 13" />
        {/* baseline */}
        <path d="M4 26 L28 26" opacity="0.35" />
        {/* prediction dot */}
        <circle cx="23" cy="8" r="1.8" fill="currentColor" stroke="none" />
        {/* tiny window/foundation tick */}
        <path d="M11 26 L11 22" opacity="0.55" />
        <path d="M23 26 L23 14" opacity="0.55" />
      </svg>
    </span>
  );
}
