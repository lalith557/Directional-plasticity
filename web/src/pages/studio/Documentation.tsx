import { Link } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Card, CardHead, Observation } from "../../components/ui";
import { Architecture2D } from "../../components/arch/Architecture2D";

const DEFS: [string, string][] = [
  ["Task accuracy a_t", "Test accuracy on task t immediately after training on it (full 10 000-image test set for permuted MNIST; fixed 2 000-image subset for rotated MNIST)."],
  ["Plasticity decay", "max_t a_t − mean(a_t over the last ⌈10 %⌉ tasks). run_arm also returns a different “plasticity_decay” (first 20 % − last 20 %); the Studio stores it separately as decay_first_last_20pct and never uses it for paper metrics."],
  ["Global weight norm ‖θ‖", "MLP tables: Euclidean norm of the backbone (hidden-layer weights and biases, head excluded). CNN: all parameters. The matched step and the Cor. 1 identity act on the full parameter vector."],
  ["Matched-step deviation", "max over steps of |‖Δθ'‖ − η| / η, relative to the plain-SGD step length η at the same parameters. Shown only for rules bound by the constraint; N/A (not 0) for magnitude/reset methods."],
  ["Dormant fraction", "Share of hidden ReLU units whose mean |activation| on the 1 000-image probe batch is below 10⁻³ (MLP: pooled over hidden layers; CNN: FC layer)."],
  ["Effective rank", "Stable rank ‖F‖²_F / ‖F‖²₂ of mean-centred penultimate features on the probe batch."],
  ["Radial / tangential component", "Derived per step from what the loop logs: radial = ‖Δθ'‖·cos(Δθ', θ), tangential = ‖Δθ'‖·√(1 − cos²). Not an extra measurement."],
  ["Update–gradient alignment", "cos(Δθ', Δθ_SGD) per step."],
];

export function Documentation() {
  return (
    <>
      <PageHeader title="Documentation" subtitle="How the Studio connects to the research code, what each number means, and what it cannot do." />
      <div className="grid" style={{ gap: 16 }}>
        <Card pad>
          <CardHead title="Architecture" sub="Hover or click any part for its files, endpoints and role" />
          <Architecture2D compact />
          <div className="mt3 small">
            <Link className="linkish" to="/system">Open the interactive 3D workflow and the step-by-step tour →</Link>
          </div>
        </Card>

        <Card pad>
          <CardHead title="Starting the Studio" />
          <div className="small mb2">Backend (from the project root):</div>
          <pre className="code-block">python -m uvicorn studio.api.server:app --host 127.0.0.1 --port 8765</pre>
          <div className="small mb2 mt3">Frontend (proxies /api to the backend):</div>
          <pre className="code-block">npm --prefix web run dev</pre>
          <div className="small mb2 mt3">Tests:</div>
          <pre className="code-block">python -m pytest studio/tests -q</pre>
          <div className="small muted mt3">Full written guides: <span className="mono">README.md</span> and <span className="mono">docs/STUDIO_GUIDE.md</span> (setup, protocol reference, API and known limitations).</div>
        </Card>

        <Card>
          <CardHead title="Metric definitions (as implemented)" />
          <div className="tbl-wrap"><table className="tbl tbl-fixed"><tbody>
            {DEFS.map(([k, v]) => <tr key={k}><td style={{ width: "26%" }}><b>{k}</b></td><td className="small">{v}</td></tr>)}
          </tbody></table></div>
        </Card>

        <Card pad>
          <CardHead title="Scientific-integrity rules enforced by the Studio" />
          <ul className="doc-list">
            <li>Existing artifacts are read, never written; every displayed number names its source file.</li>
            <li>Every run has a new ID and a read-only configuration snapshot whose SHA-256 is verified before execution.</li>
            <li>Only parameters implemented by <span className="mono">run_arm</span> / <span className="mono">run_arm_cnn</span> can be configured; unsupported keys are rejected.</li>
            <li>Non-finite values are shown as “Not available”, never 0. Diverged arms are excluded from correlations and listed as excluded.</li>
            <li>Correlations across methods are labelled method-level with their n; no confidence interval below n = 10.</li>
            <li>Paired statistics only for identical protocols with shared seeds; descriptive differences and tests are reported separately.</li>
            <li>A run is labelled “reproduced” only if its configuration equals a per-seed legacy record and its metrics match it.</li>
            <li>Live values come only from the running process; nothing is simulated. Long submissions require explicit confirmation.</li>
          </ul>
        </Card>

        <Card pad>
          <CardHead title="Known limitations" />
          <ul className="doc-list">
            <li><b>No resume.</b> run_arm keeps CBP utilities, the global RNG state and per-step lists internally; a resumed run would not equal an uninterrupted one. Cancelled or failed runs are re-run under a new ID.</li>
            <li><b>No pause</b> (the training loop cannot pause).</li>
            <li><b>No legacy per-task curves.</b> The original scripts did not persist them; curves exist only for Studio runs.</li>
            <li><b>Legacy configurations are reconstructed</b> from script constants; only e20 and e27 have per-seed values to validate against.</li>
            <li><b>CNN effective rank</b> jitters by ~10⁻⁵ between CUDA runs (cuDNN forward is not bitwise deterministic); training itself is reproducible.</li>
            <li><b>Single GPU queue</b>: one run at a time; GPU utilization is device-wide (NVML), not per run.</li>
            <li><b>No A100 job</b> is defined: the documentation names no experiment that requires one.</li>
          </ul>
        </Card>
        <Observation label="See also" tone="neutral">
          <Link className="linkish" to="/paper-results">Paper Results</Link> maps every paper claim to its artifact;{" "}
          <Link className="linkish" to="/explorer">Results Explorer</Link> lists all records with their status.
        </Observation>
      </div>
    </>
  );
}
