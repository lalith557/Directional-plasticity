/* Small presentational pieces shared by the 2D diagram, the 3D workflow and the tour. */
import { COMP, FLOWS, FLOW_KINDS, type Detail, type Flow } from "./archModel";

export const FLOW: Record<string, Flow> = Object.fromEntries(FLOWS.map((f) => [f.id, f]));

export function DetailCard({ d }: { d: Detail }) {
  return (
    <>
      <div className="a3-pop-title">{d.title}{d.sub && <span> · {d.sub}</span>}</div>
      <div className="a3-pop-body">{d.body}</div>
      {d.api && <div className="a3-pop-list">{d.api.map((a) => <code key={a}>{a}</code>)}</div>}
      {d.code && <div className="a3-pop-list">{d.code.map((a) => <code key={a} className="muted">{a}</code>)}</div>}
      {d.facts && <ul className="a3-pop-facts">{d.facts.map((f) => <li key={f}>{f}</li>)}</ul>}
    </>
  );
}

export function FlowCard({ f }: { f: Flow }) {
  return (
    <>
      <div className="a3-pop-title"><i className="a3-dot" style={{ background: FLOW_KINDS[f.kind].color }} />{f.label}</div>
      <div className="a3-pop-sub">{COMP[f.from].label} → {COMP[f.to].label} · {FLOW_KINDS[f.kind].label}</div>
      <div className="a3-pop-body">{f.detail}</div>
    </>
  );
}

export function FlowLegend({ className = "" }: { className?: string }) {
  return (
    <div className={className}>
      {Object.entries(FLOW_KINDS).map(([k, v]) => (
        <div key={k} title={v.hint}><i className="a3-dot" style={{ background: v.color }} />{v.label}</div>
      ))}
    </div>
  );
}
