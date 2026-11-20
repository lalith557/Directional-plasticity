import { useMemo, useState } from "react";
import { C } from "../../lib/colors";

/**
 * Interactive evaluation of the fixed-step identity
 *     ‖θ + ηd‖² = ‖θ‖² + 2η⟨θ,d⟩ + η²,   ‖d‖ = 1.
 * d is decomposed as d = a·(−θ̂) + √(1−a²)·t̂ with t̂ ⊥ θ̂ (a = inward radial share).
 * Pure mathematics — no model is trained and no measured value is shown here.
 */
type Mode = "orthogonal" | "inward" | "combined" | "gamma";

export function IdentityExplorer() {
  const [norm, setNorm] = useState(20.4);
  const [eta, setEta] = useState(0.33);
  const [mode, setMode] = useState<Mode>("combined");
  const [a, setA] = useState(0.2);
  const [gamma, setGamma] = useState(0.6);
  const [T, setT] = useState(6000);
  const [theta0, setTheta0] = useState(3.8);

  // inward share a for each mode; for the inward rule we assume the SGD step is tangential
  // (measured mean cos(Δθ_SGD, θ) ≈ 0.01 in e23), so d ∝ ĝ − γθ̂ gives a = γ/√(1+γ²).
  const share = mode === "orthogonal" ? 0 : mode === "inward" ? 1 : mode === "gamma" ? gamma / Math.sqrt(1 + gamma * gamma) : a;
  const tang = Math.sqrt(Math.max(0, 1 - share * share));
  const inner = -share * norm;                                  // ⟨θ, d⟩
  const newSq = norm * norm + 2 * eta * inner + eta * eta;
  const newNorm = Math.sqrt(Math.max(0, newSq));
  const threshold = eta / (2 * norm);                           // Cor. 2: minimum inward share to not grow
  const reduces = newNorm <= norm + 1e-15;

  // T-step projection with constant η and constant share a (illustration only)
  const proj = useMemo(() => {
    let sq = theta0 * theta0;
    const pts: { t: number; n: number }[] = [{ t: 0, n: theta0 }];
    const every = Math.max(1, Math.floor(T / 60));
    for (let t = 1; t <= T; t++) {
      const n = Math.sqrt(Math.max(0, sq));
      sq = Math.max(0, sq + 2 * eta * (-share * n) + eta * eta);
      if (t % every === 0 || t === T) pts.push({ t, n: Math.sqrt(sq) });
    }
    return pts;
  }, [theta0, eta, share, T]);

  // drawing (step drawn enlarged for visibility; the numbers are exact)
  const R = 120, O = { x: 40, y: 200 };
  const L = Math.max(36, Math.min(110, (eta / norm) * R * 40));
  const P = { x: O.x + R, y: O.y };                            // θ tip (θ̂ along +x)
  const radial = { x: P.x - share * L, y: P.y };
  const Q = { x: radial.x, y: radial.y - tang * L };            // θ' tip
  const enlarge = L / ((eta / norm) * R);
  const arrow = (x1: number, y1: number, x2: number, y2: number, col: string, dash?: string, w = 1.8) => (
    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={col} strokeWidth={w} strokeDasharray={dash} markerEnd={`url(#ie-${col.replace(/[^a-z0-9]/gi, "")})`} />
  );
  const cols = [C.blue, C.amber, C.red, C.green, C.ink3];

  return (
    <div className="identity-explorer">
      <div className="badge amber mb3">Mathematical illustration — not a training run</div>
      <div className="grid-2col">
        <div>
          <svg viewBox="0 0 330 240" width="100%" style={{ maxWidth: 380 }} role="img" aria-label="Fixed-step geometry">
            <defs>{cols.map((c) => (
              <marker key={c} id={`ie-${c.replace(/[^a-z0-9]/gi, "")}`} markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto">
                <path d="M0,0 L6,3 L0,6 Z" fill={c} /></marker>))}</defs>
            <path d={`M ${O.x + R} ${O.y} A ${R} ${R} 0 0 0 ${O.x} ${O.y - R}`} fill="none" stroke={C.border} strokeDasharray="3 3" />
            <text x={O.x + 4} y={O.y - R - 6} fontSize="9" fill={C.ink3}>sphere ‖θ‖ = const</text>
            {arrow(O.x, O.y, P.x, P.y, C.blue, undefined, 2.2)}
            <text x={O.x + R / 2 - 4} y={O.y + 16} fontSize="12" fill={C.blue} fontStyle="italic">θ</text>
            {share > 0.001 && arrow(P.x, P.y, radial.x, radial.y, C.amber, "4 3")}
            {tang > 0.001 && arrow(radial.x, radial.y, Q.x, Q.y, C.green, "4 3")}
            {arrow(P.x, P.y, Q.x, Q.y, C.red, undefined, 2.2)}
            <line x1={O.x} y1={O.y} x2={Q.x} y2={Q.y} stroke={C.ink3} strokeDasharray="2 3" />
            <text x={Q.x + 6} y={Q.y} fontSize="10" fill={C.red}>θ' = θ + ηd</text>
            {share > 0.001 && <text x={(P.x + radial.x) / 2 - 18} y={P.y + 14} fontSize="9" fill={C.amber}>radial (inward)</text>}
            {tang > 0.001 && <text x={radial.x + 6} y={(radial.y + Q.y) / 2} fontSize="9" fill={C.green}>tangential</text>}
          </svg>
          <div className="small muted">Step drawn enlarged ×{enlarge.toFixed(0)} for visibility; the values on the right are exact.</div>
        </div>
        <div>
          <div className="seg mb3">
            {([["orthogonal", "Orthogonal"], ["inward", "Pure inward"], ["combined", "Radial + tangential"], ["gamma", "Inward rule γ"]] as [Mode, string][]).map(([k, l]) => (
              <button key={k} className={`seg-btn ${mode === k ? "on" : ""}`} onClick={() => setMode(k)}>{l}</button>))}
          </div>
          {mode === "combined" && <Slider label="Inward radial share a = ⟨d, −θ̂⟩" v={a} set={setA} min={-1} max={1} step={0.001} />}
          {mode === "gamma" && <Slider label="γ of the inward rule (assumes a tangential SGD step)" v={gamma} set={setGamma} min={0} max={1} step={0.01} />}
          <Slider label="‖θ‖ (current global norm)" v={norm} set={setNorm} min={0.5} max={40} step={0.1} />
          <Slider label="η (step length ‖Δθ‖)" v={eta} set={setEta} min={0.01} max={3} step={0.01} />
          <div className="kv mt3">
            <div className="kv-k">Inward share a / tangential √(1−a²)</div><div className="kv-v mono">{share.toFixed(4)} / {tang.toFixed(4)}</div>
            <div className="kv-k">⟨θ, d⟩ = −a‖θ‖</div><div className="kv-v mono">{inner.toFixed(4)}</div>
            <div className="kv-k">‖θ'‖² = ‖θ‖² + 2η⟨θ,d⟩ + η²</div><div className="kv-v mono">{(norm * norm).toFixed(3)} {inner * 2 * eta >= 0 ? "+" : "−"} {Math.abs(2 * eta * inner).toFixed(3)} + {(eta * eta).toFixed(4)} = {newSq.toFixed(4)}</div>
            <div className="kv-k">Change in squared norm</div><div className="kv-v mono">{(newSq - norm * norm >= 0 ? "+" : "") + (newSq - norm * norm).toFixed(4)}</div>
            <div className="kv-k">Norm-reduction threshold (Cor. 2)</div><div className="kv-v mono">a ≥ η/(2‖θ‖) = {threshold.toFixed(4)}</div>
            <div className="kv-k">Result</div><div className="kv-v">{share === 0 ? <>orthogonal: squared norm rises by exactly η² = {(eta * eta).toFixed(4)} (Cor. 1)</> :
              reduces ? <>norm does not grow — {(100 * (1 - tang)).toFixed(2)}% less of the unit step remains for descent</> :
              <>norm still grows (a below the threshold)</>}</div>
          </div>
        </div>
      </div>
      <div className="section-label mt4">Projection over T steps (constant η and a)</div>
      <div className="form-grid">
        <Slider label="‖θ₀‖" v={theta0} set={setTheta0} min={0.5} max={20} step={0.1} />
        <Slider label="T (steps)" v={T} set={setT} min={1} max={12000} step={1} />
      </div>
      <div className="small mono">‖θ_T‖ = {proj[proj.length - 1].n.toFixed(3)} after {T} steps{share === 0 ? ` (= √(‖θ₀‖² + Tη²) = ${Math.sqrt(theta0 * theta0 + T * eta * eta).toFixed(3)})` : ""}</div>
      <div className="small muted mt2">
        For reference only: the measured orthogonal arm (Cor. 1 audit, 6000 steps, varying η_t) went from a global norm of about 3.8 to 25.9,
        and ‖θ_T‖² − ‖θ₀‖² matched Σ η_t² to a relative error of 3×10⁻⁸ — see Paper Results, Finding 2.
      </div>
    </div>
  );
}

function Slider({ label, v, set, min, max, step }: { label: string; v: number; set: (x: number) => void; min: number; max: number; step: number }) {
  return (
    <label className="field">
      <span className="row between"><span className="field-label">{label}</span><span className="mono small">{v}</span></span>
      <input className="slider" type="range" min={min} max={max} step={step} value={v} onChange={(e) => set(parseFloat(e.target.value))} />
    </label>
  );
}
