import { BlurReveal } from "../components/ui/BlurReveal";
import { ScrollReveal } from "../components/ui/ScrollReveal";

const PDF_URL = import.meta.env.BASE_URL + "paper/plasticity-paper.pdf";

export function Paper() {
  const onDownload = () => {
    const a = document.createElement("a");
    a.href = PDF_URL;
    a.download = "plasticity-paper.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  return (
    <div className="paper-root">
      <div className="paper-hero">
        <span className="paper-tag">RESEARCH PAPER · 2026</span>
        <h1 className="paper-title">
          <BlurReveal as="span">
            Directional Plasticity under Permutation and Rotation Stress
          </BlurReveal>
        </h1>
        <p className="paper-sub">
          <BlurReveal as="span" delay={400} stagger={30}>
            Empirical study of norm, geometry and plasticity under extended continual-learning streams on MLPs and a small CNN.
          </BlurReveal>
        </p>

        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", justifyContent: "center" }}>
          <button className="dl-btn" onClick={onDownload} aria-label="Download paper as PDF">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Download PDF
          </button>
          <a className="dl-btn" href={PDF_URL} target="_blank" rel="noreferrer" style={{ textDecoration: "none" }}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
              <polyline points="15 3 21 3 21 9" />
              <line x1="10" y1="14" x2="21" y2="3" />
            </svg>
            Open in new tab
          </a>
        </div>

        <div className="paper-meta">
          <span><b>Lalith Gona</b><br/>Author</span>
          <span><b>IEEE ICA 2026</b><br/>Venue</span>
          <span><b>MLP · CNN</b><br/>Architectures</span>
          <span><b>35 regime arms</b><br/>Across 5 regimes</span>
        </div>
      </div>

      <ScrollReveal>
        <section style={{ maxWidth: 760, margin: "40px auto", padding: "0 20px" }}>
          <h2 style={{ fontSize: 22, letterSpacing: "-0.01em", marginBottom: 14 }}>Abstract</h2>
          <p style={{ color: "var(--ink-2)", lineHeight: 1.75, fontSize: 15 }}>
            This work investigates whether weight-norm growth is causally linked to plasticity loss
            under non-stationary task streams. Across five regimes — width 16/64/128 permuted MNIST,
            rotated MNIST, and a small CNN — we evaluate seven methods including SGD, L2-init,
            Shrink &amp; Perturb, CBP, ANR, and a radial update-direction family. The paper reports
            the correlation structure between final norm and plasticity decay without asserting
            causality, and documents when the relationship reverses sign across width and task
            geometry.
          </p>
        </section>
      </ScrollReveal>

      <ScrollReveal delay={120}>
        <section style={{ maxWidth: 760, margin: "40px auto 80px", padding: "0 20px" }}>
          <h2 style={{ fontSize: 22, letterSpacing: "-0.01em", marginBottom: 14 }}>What is released</h2>
          <ul style={{ color: "var(--ink-2)", lineHeight: 1.9, fontSize: 15, paddingLeft: 20 }}>
            <li>Full PDF manuscript (above)</li>
            <li>Eight per-run experiment summary JSONs — aggregate metrics and per-seed results</li>
            <li>Training scripts, model definitions, dataset loaders, and random seeds</li>
            <li>The dashboard generator (<span className="mono">web/scripts/build_data.py</span>) that produces every number in this UI</li>
          </ul>
          <p style={{ color: "var(--ink-3)", fontSize: 13, marginTop: 20 }}>
            Not released: trained model weights, optimizer states, or intermediate checkpoints.
          </p>
        </section>
      </ScrollReveal>
    </div>
  );
}
