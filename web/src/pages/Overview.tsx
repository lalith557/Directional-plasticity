import { Link, useNavigate } from "react-router-dom";
import { DataGate } from "../components/DataGate";
import { PageHeader } from "../components/PageHeader";
import { MetricCard } from "../components/MetricCard";
import { Card, Observation, IntegrityBanner } from "../components/ui";
import { BarSci } from "../components/charts";
import { C } from "../lib/colors";
import { acc, num, signed, timestamp } from "../data/format";
import { StudioStatus } from "../components/studio/StudioStatus";

export function Overview() {
  const nav = useNavigate();
  return (
    <DataGate>
      {(d) => {
        // Sort by acc desc for chart display — existing convention
        const arms = [...d.r20.arms].sort((a, b) => b.acc - a.acc);
        const chartData = arms.map((a) => ({
          short: a.short,
          name: a.name,
          acc: a.acc,
          accStd: a.accStd,
          decay: a.decay,
          decayStd: a.decayStd,
          norm: a.norm,
          normStd: a.normStd,
        }));

        // Derive actual winners from source data — do not hardcode
        const bestAcc = arms[0]; // sorted desc by acc
        const bestDecay = [...d.r20.arms].sort((a, b) => a.decay - b.decay)[0];
        const bestNorm = [...d.r20.arms].sort((a, b) => a.norm - b.norm)[0];

        const r = d.normPlasticity.primary.pearson;
        const n = d.normPlasticity.primary.n;

        // Regime correlations (index order verified against source: width16, width64, width128, rot_mlp, rot_cnn)
        const rW16 = d.regimes[0]?.pearson ?? null;
        const rW64 = d.regimes[1]?.pearson ?? null;
        const rW128 = d.regimes[2]?.pearson ?? null;
        const rRotMlp = d.regimes[3]?.pearson ?? null;

        return (
          <>
            {/* Section A: Page title */}
            <PageHeader
              title="Overview"
              subtitle="Weight norm, update geometry, and plasticity in continual learning."
            />

            {/* Studio: live registry / GPU status (backend) */}
            <StudioStatus />

            {/* Section B: Five summary metric cards */}
            <div className="metrics mb4">
              <MetricCard value={d.meta.arms} label="Experimental Arms" />
              <MetricCard value={d.meta.seedsPerArm} label="Seeds per Arm" />
              <MetricCard value={d.meta.tasks} label="Tasks (R20)" />
              <MetricCard value={d.meta.architectures} label="Architectures" />
              <MetricCard value={d.meta.datasets} label="Datasets" />
            </div>

            {/* Section C: Main Finding (2/3) + Boundary Condition (1/3) — 2:1 desktop ratio */}
            <div className="grid-21 mb4">
              {/* Main Finding */}
              <Card className="feature accent">
                <span className="eyebrow">PRIMARY FINDING</span>
                <div className="finding-title">Norm growth and plasticity decay</div>
                <div className="finding-r">r&nbsp;=&nbsp;{signed(r, 2)}</div>
                <div className="finding-qualifier">
                  Pearson correlation · {n} method arms · descriptive, not causal
                </div>
                <div className="headline">
                  Strong positive association in the width-16 Permuted-MNIST stress regime.
                </div>
                <div>
                  <Link className="btn" to="/analysis/norm-plasticity">
                    Inspect norm–plasticity analysis &rarr;
                  </Link>
                </div>
              </Card>

              {/* Boundary Condition */}
              <Card className="feature accent-amber">
                <span className="eyebrow">BOUNDARY CONDITION</span>
                <div className="finding-title">Regime dependence</div>
                <div className="headline" style={{ fontSize: 13 }}>
                  The association weakens at width&nbsp;64 and reverses at width&nbsp;128 and on rotated MNIST.
                </div>
                <div className="regime-list">
                  <div className="regime-row">
                    <span className="regime-label">Width 16</span>
                    <span className="regime-r" style={{ color: "var(--blue)" }}>{signed(rW16, 2)}</span>
                  </div>
                  <div className="regime-row">
                    <span className="regime-label">Width 64</span>
                    <span className="regime-r" style={{ color: "var(--blue)", opacity: 0.75 }}>{signed(rW64, 2)}</span>
                  </div>
                  <div className="regime-row">
                    <span className="regime-label">Width 128</span>
                    <span className="regime-r" style={{ color: "var(--red)" }}>{signed(rW128, 2)}</span>
                  </div>
                  <div className="regime-row">
                    <span className="regime-label">Rotated MNIST</span>
                    <span className="regime-r" style={{ color: "var(--red)" }}>{signed(rRotMlp, 2)}</span>
                  </div>
                </div>
                <div>
                  <Link className="linkish" to="/experiments/width">
                    Explore regimes &rarr;
                  </Link>
                </div>
              </Card>
            </div>

            {/* Section D: R20 Main Results — three independent cards */}
            <div className="row between aic mb3">
              <div className="section-label" style={{ marginBottom: 0 }}>R20 — Main Results</div>
              <div className="card-sub">Mean ± SD · {d.meta.seedsPerArm} seeds</div>
            </div>
            <div className="grid-111 mb4">
              {/* Accuracy card */}
              <Card pad>
                <div className="card-title mb2">
                  Accuracy <span className="muted">&uarr;</span>
                </div>
                <div className="result-winner">
                  <span className="result-method">{bestAcc.name}</span>
                  <span className="result-value">{acc(bestAcc.acc)}</span>
                  <span className="result-std">±&nbsp;{acc(bestAcc.accStd)}</span>
                </div>
                <BarSci
                  data={chartData}
                  xKey="short"
                  yKey="acc"
                  stdKey="accStd"
                  color={C.blue}
                  yDomain={[0, 1]}
                  height={170}
                  valueFmt={(v) => acc(v)}
                  metricLabel="Mean acc"
                  seedN={d.meta.seedsPerArm}
                  onBarClick={() => nav("/experiments/r20")}
                />
                <Link className="linkish small result-link" to="/experiments/r20">
                  Full R20 comparison &rarr;
                </Link>
              </Card>

              {/* Plasticity Decay card */}
              <Card pad>
                <div className="card-title mb2">
                  Plasticity Decay <span className="muted">&darr;</span>
                </div>
                <div className="result-winner">
                  <span className="result-method">{bestDecay.name}</span>
                  <span className="result-value">{acc(bestDecay.decay)}</span>
                  <span className="result-std">±&nbsp;{acc(bestDecay.decayStd)}</span>
                </div>
                <BarSci
                  data={chartData}
                  xKey="short"
                  yKey="decay"
                  stdKey="decayStd"
                  color={C.red}
                  yDomain={[0, 0.24]}
                  height={170}
                  valueFmt={(v) => acc(v)}
                  metricLabel="Decay"
                  seedN={d.meta.seedsPerArm}
                  onBarClick={() => nav("/experiments/r20")}
                />
                <Link className="linkish small result-link" to="/experiments/r20">
                  Full R20 comparison &rarr;
                </Link>
              </Card>

              {/* Final Weight Norm card */}
              <Card pad>
                <div className="card-title mb2">
                  Final Weight Norm <span className="muted">&darr;</span>
                </div>
                <div className="result-winner">
                  <span className="result-method">{bestNorm.name}</span>
                  <span className="result-value">{num(bestNorm.norm)}</span>
                  {bestNorm.normStd != null && (
                    <span className="result-std">±&nbsp;{num(bestNorm.normStd)}</span>
                  )}
                </div>
                <BarSci
                  data={chartData}
                  xKey="short"
                  yKey="norm"
                  stdKey="normStd"
                  color={C.navy}
                  yDomain={[0, 28]}
                  height={170}
                  valueFmt={(v) => num(v)}
                  metricLabel="Final norm"
                  seedN={d.meta.seedsPerArm}
                  onBarClick={() => nav("/experiments/r20")}
                />
                <Link className="linkish small result-link" to="/experiments/r20">
                  Full R20 comparison &rarr;
                </Link>
              </Card>
            </div>

            {/* Section E: Scientific interpretation */}
            <Observation tone="neutral" label="INTERPRETATION">
              Magnitude-control methods perform strongly in the primary stress regime. However, the
              radial sweep shows that reducing parameter norm alone does not guarantee good accuracy.
              Results across wider and rotated-task regimes establish important limits on
              generalization.
            </Observation>

            {/* Section F: Provenance footer */}
            <IntegrityBanner
              source={d.r20.source}
              seeds={d.meta.seedsPerArm}
              loaded={timestamp(d.generatedAt)}
            />
          </>
        );
      }}
    </DataGate>
  );
}
