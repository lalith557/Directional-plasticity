export interface NavItem {
  label: string;
  to: string;
  icon: string;
}
export interface NavSection {
  label: string;
  items: NavItem[];
}

export const NAV: NavSection[] = [
  {
    label: "Experiments",
    items: [
      { label: "Overview", to: "/", icon: "◉" },
      { label: "R20 — Plasticity Stress", to: "/experiments/r20", icon: "■" },
      { label: "λ Sweep", to: "/experiments/lambda", icon: "⤳" },
      { label: "Radial Sweep", to: "/experiments/radial", icon: "↗" },
      { label: "Width Study", to: "/experiments/width", icon: "≡" },
      { label: "Rotated MNIST", to: "/experiments/rotated", icon: "↻" },
      { label: "CNN Probe", to: "/experiments/cnn", icon: "▦" },
    ],
  },
  {
    label: "Analysis",
    items: [
      { label: "Norm ↔ Plasticity", to: "/analysis/norm-plasticity", icon: "⊕" },
      { label: "Update Geometry", to: "/analysis/geometry", icon: "∠" },
      { label: "Ablations", to: "/analysis/ablations", icon: "◑" },
      { label: "Comparative View", to: "/analysis/comparative", icon: "☷" },
    ],
  },
  {
    label: "Reproducibility",
    items: [
      { label: "Configuration", to: "/reproducibility/config", icon: "⚙" },
      { label: "Datasets", to: "/reproducibility/datasets", icon: "▤" },
      { label: "Checkpoints", to: "/reproducibility/checkpoints", icon: "⚑" },
      { label: "Raw Results", to: "/reproducibility/raw", icon: "⌸" },
    ],
  },
];

// Flat list used by search + topbar breadcrumb.
export const ALL_ROUTES: { label: string; to: string; kind: string }[] = [
  ...NAV.flatMap((s) => s.items.map((i) => ({ label: i.label, to: i.to, kind: s.label }))),
  { label: "Run Details", to: "/runs", kind: "Reproducibility" },
  { label: "Experiment Studio", to: "/studio", kind: "Studio" },
  { label: "Live Monitor", to: "/monitor", kind: "Studio" },
  { label: "Results Explorer", to: "/explorer", kind: "Results" },
  { label: "Method Comparison", to: "/compare", kind: "Results" },
  { label: "Scientific Analysis", to: "/analysis", kind: "Analysis" },
  { label: "Architecture Explorer", to: "/architectures", kind: "Analysis" },
  { label: "Paper Results", to: "/paper-results", kind: "Paper" },
  { label: "Documentation", to: "/docs", kind: "Docs" },
];

export function labelForPath(path: string): string {
  const hit = ALL_ROUTES.find((r) => r.to === path);
  return hit ? hit.label : "Plasticity Lab";
}
