import { useState } from "react";
import { Link } from "react-router-dom";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { Card, CardHead, Observation, IntegrityBanner, StatLine } from "../components/ui";
import { GeometryDiagram } from "../components/GeometryDiagram";
import { SweepSlider } from "../components/SweepSlider";
import { acc, num, timestamp } from "../data/format";
import { IdentityExplorer } from "../components/studio/IdentityExplorer";

const gfmt = (v: number) => v.toFixed(1);

function SciVal({ v }: { v: number | null | undefined }) {
  if (v === null || v === undefined || Number.isNaN(v)) return <span className="na">N/A</span>;
  if (v === 0) return <span>0</span>;
  const s = v.toExponential(1);
  const [mantissa, exp] = s.split("e");
  return (
    <span className="mono">
      {mantissa}×10<sup>{exp.replace("+", "")}</sup>
    </span>
  );
}

export function UpdateGeometry() {
  const [gi, setGi] = useState(3); // gamma = 0.6 (the inward arm of R20)
  return (
    <DataGate>
      {(d) => {
        const cur = d.radial.points[gi];
        const orth = d.r20.arms.find((a) => a.key === "dir_control_c1.0");
        const inward = d.r20.arms.find((a) => a.key === "dir_inward_g0.6");
        const sgd = d.r20.arms.find((a) => a.key === "sgd");
        return (
          <>
            <PageHeader
              title="Update Geometry"
              subtitle="Fixed-step decomposition of the update into radial (norm-changing) and tangential (task-descent) components."
            />

            <div className="grid-side mb4">
              <Card>
                <CardHead title="Direction Decomposition" sub={`γ = ${gfmt(cur.gamma)}`} />
                <div className="card-pad" style={{ display: "grid", placeItems: "center" }}>
                  <GeometryDiagram gamma={cur.gamma} />
                  <div style={{ width: "100%", marginTop: 18 }}>
                    <SweepSlider values={d.radial.gammas} index={gi} onChange={setGi} label="Radial coefficient γ" format={gfmt} />
                  </div>
                  <div style={{ width: "100%", marginTop: 16 }}>
                    <StatLine
                      stats={[
                        { label: "Measured accuracy", value: acc(cur.acc) },
                        { label: "Measured final norm", value: num(cur.norm, 2) },
                      ]}
                    />
                  </div>
                </div>
              </Card>

              <div className="grid" style={{ gap: 16 }}>
                <Observation label="Fixed-step norm identity" tone="neutral">
                  For a unit direction d and step η, ‖θ + ηd‖² = ‖θ‖² + 2η⟨θ,d⟩ + η². An
                  orthogonal step (⟨θ,d⟩ = 0) raises the squared norm by exactly η²
                  (Cor. 1), and reducing the norm requires ⟨θ,d⟩ ≤ −η/2, i.e. an inward radial component
                  of at least η/(2‖θ‖) of the unit direction, drawn from the update budget (Cor. 2). This
                  constrains fixed-step, global-norm directional interventions — not all directional or
                  geometric methods.
                </Observation>

                <Card>
                  <CardHead title="Matched-step directional arms (R20)" />
                  <div className="tbl-wrap">
                    <table className="tbl">
                      <thead>
                        <tr>
                          <th>Arm</th>
                          <th className="num">Accuracy</th>
                          <th className="num">Final Norm</th>
                          <th className="num">Match Dev.</th>
                          <th>Regime</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sgd && (
                          <tr>
                            <td>SGD (reference)</td>
                            <td className="num">{acc(sgd.acc)}</td>
                            <td className="num">{num(sgd.norm)}</td>
                            <td className="num na">ref</td>
                            <td className="muted">baseline</td>
                          </tr>
                        )}
                        {orth && (
                          <tr>
                            <td>Directional, orthogonal</td>
                            <td className="num">{acc(orth.acc)}</td>
                            <td className="num">{num(orth.norm)}</td>
                            <td className="num"><SciVal v={orth.matchDev} /></td>
                            <td className="muted">Cor. 1 — cannot reduce norm</td>
                          </tr>
                        )}
                        {inward && (
                          <tr>
                            <td>Directional, inward</td>
                            <td className="num">{acc(inward.acc)}</td>
                            <td className="num">{num(inward.norm)}</td>
                            <td className="num"><SciVal v={inward.matchDev} /></td>
                            <td className="muted">Cor. 2 — reduces norm, collapses accuracy</td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            </div>

            <Card className="mb4">
              <CardHead title="Fixed-step identity explorer" sub="adjust the radial coefficient and see the exact geometric consequence" />
              <div className="card-pad"><IdentityExplorer /></div>
            </Card>

            <Observation label="Reading">
              Orthogonal steering attains the largest norm ({num(orth?.norm ?? null)}) and cannot
              reduce it; inward steering drives the norm down to {num(inward?.norm ?? null)} but
              collapses accuracy to {acc(inward?.acc ?? null)}. Direction <b>can</b> reduce the global
              norm — but only by spending the fixed update budget on the radial component. The full
              trade-off is swept in the{" "}
              <Link className="linkish" to="/experiments/radial">Radial Sweep</Link>.
            </Observation>

            <IntegrityBanner
              source="results/tables/e20_summary.json · e27_radial_sweep_summary.json"
              extra={`matched step verified to ${d.meta.matchedStepTolerance}`}
              loaded={timestamp(d.generatedAt)}
            />
          </>
        );
      }}
    </DataGate>
  );
}
