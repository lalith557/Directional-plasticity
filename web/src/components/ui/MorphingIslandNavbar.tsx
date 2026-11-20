import { useEffect, useRef, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";

type NavItem = { label: string; to: string };
type NavSection = { label: string; items: NavItem[]; href?: string };

const SECTIONS: NavSection[] = [
  {
    label: "Studio",
    items: [
      { label: "Overview", to: "/" },
      { label: "Experiment Studio", to: "/studio" },
      { label: "Live Monitor", to: "/monitor" },
    ],
  },
  {
    label: "Results",
    items: [
      { label: "Results Explorer", to: "/explorer" },
      { label: "Method Comparison", to: "/compare" },
      { label: "R20 — Plasticity Stress", to: "/experiments/r20" },
      { label: "λ Sweep", to: "/experiments/lambda" },
      { label: "Radial Sweep", to: "/experiments/radial" },
      { label: "Width Study", to: "/experiments/width" },
      { label: "Rotated MNIST", to: "/experiments/rotated" },
      { label: "CNN Probe", to: "/experiments/cnn" },
    ],
  },
  {
    label: "Analysis",
    items: [
      { label: "Scientific Analysis", to: "/analysis" },
      { label: "Update Geometry", to: "/analysis/geometry" },
      { label: "Architecture Explorer", to: "/architectures" },
      { label: "Norm ↔ Plasticity", to: "/analysis/norm-plasticity" },
      { label: "Ablations", to: "/analysis/ablations" },
      { label: "Comparative View", to: "/analysis/comparative" },
    ],
  },
  {
    label: "Docs",
    items: [
      { label: "Documentation", to: "/docs" },
      { label: "System Architecture", to: "/system" },
      { label: "Configuration", to: "/reproducibility/config" },
      { label: "Datasets", to: "/reproducibility/datasets" },
      { label: "Checkpoints", to: "/reproducibility/checkpoints" },
      { label: "Raw Results", to: "/reproducibility/raw" },
    ],
  },
];

function sectionOfPath(path: string): string | null {
  // Overview ("/") is reachable from the brand — don't highlight any section there.
  if (path.startsWith("/studio") || path.startsWith("/monitor")) return "Studio";
  if (path.startsWith("/explorer") || path.startsWith("/compare") || path.startsWith("/experiments") || path.startsWith("/runs"))
    return "Results";
  if (path.startsWith("/analysis") || path.startsWith("/architectures")) return "Analysis";
  if (path.startsWith("/docs") || path.startsWith("/system") || path.startsWith("/reproducibility")) return "Docs";
  // "/paper-results" is a direct-link tab (NavLink handles its active state);
  // "/paper" belongs to the Paper pill at the end, as before.
  return null;
}

export function MorphingIslandNavbar() {
  const [open, setOpen] = useState<string | null>(null);
  const location = useLocation();
  const nav = useNavigate();
  const rootRef = useRef<HTMLDivElement>(null);
  const activeSection = sectionOfPath(location.pathname);

  useEffect(() => {
    setOpen(null);
  }, [location.pathname]);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(e.target as Node)) setOpen(null);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(null);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <div className="mi-wrap" ref={rootRef}>
      <nav className="mi-island" aria-label="Primary">
        <div className="mi-brand" onClick={() => nav("/")} style={{ cursor: "pointer" }}>
          <span className="mi-brand-dot" aria-hidden />
          <span>PLASTICITY LAB</span>
        </div>
        {SECTIONS.map((s) => (
          <div key={s.label} style={{ position: "relative" }}>
            <button
              className={`mi-tab ${activeSection === s.label ? "active" : ""} ${open === s.label ? "active" : ""}`}
              onClick={() => setOpen((o) => (o === s.label ? null : s.label))}
              aria-haspopup="menu"
              aria-expanded={open === s.label}
            >
              {s.label}
            </button>
            {open === s.label && (
              <div className="mi-panel" role="menu">
                {s.items.map((it) => (
                  <NavLink
                    key={it.to}
                    to={it.to}
                    end={it.to === "/"}
                    className={({ isActive }) => `mi-item ${isActive ? "current" : ""}`}
                    role="menuitem"
                  >
                    {it.label}
                  </NavLink>
                ))}
              </div>
            )}
          </div>
        ))}
        <NavLink
          to="/paper-results"
          className={({ isActive }) => `mi-tab mi-tab-link ${isActive ? "active" : ""}`}
        >
          Paper Results
        </NavLink>
      </nav>
      <NavLink
        to="/paper"
        className={({ isActive }) => `mi-paper ${isActive ? "active" : ""}`}
      >
        Paper
      </NavLink>
    </div>
  );
}
