/* 2D system-architecture diagram (laid out like the reference poster). Every tile is a real
   component from archModel; hover or click one for its details. Dots moving along the
   connectors show which way data flows. A tour step highlights its parts and connections. */
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type FocusEvent, type MouseEvent, type ReactNode, type Ref } from "react";
import { COMP, FLOW_KINDS, LAYER, METHOD_IDS, type FlowKind, type LayerId, type TourStep } from "./archModel";
import { DetailCard, FLOW, FlowLegend } from "./archUi";
import { Icon } from "./icons";
import "./arch.css";

const PIPE: Record<string, string> = { f_p1: "p1", f_p2: "p2", f_p3: "p3", f_p4: "p4", f_next: "loop", f_methods: "methods" };
/** Which drawn connector carries a given flow. */
function connectorOf(flowId: string): string {
  if (PIPE[flowId]) return PIPE[flowId];
  const f = FLOW[flowId];
  const pair = [COMP[f.from].layer, COMP[f.to].layer].sort().join("|");
  switch (pair) {
    case "frontend|researcher": return "u-fe";
    case "backend|researcher": return "u-be";
    case "backend|frontend": return "fe-be";
    case "backend|engine": return "be-eng";
    case "datasets|engine": return "ds-eng";
    case "engine|metrics": return "eng-met";
    default: return "st";                      // anything touching storage & compute
  }
}

type Lane = { kind: FlowKind; dir: "down" | "up" | "right" | "left" };
const cssVars = (v: Record<string, string>) => v as CSSProperties;
const laneStyle = (k: FlowKind) => cssVars({ "--c": FLOW_KINDS[k].color });

function MiniNet() {
  const counts = [4, 3, 3, 2], xs = [8, 30, 52, 74];
  const pts = counts.map((n, li) => Array.from({ length: n }, (_, k) => ({ x: xs[li], y: 26 + (k - (n - 1) / 2) * 12 })));
  return (
    <svg className="a2-net" viewBox="0 0 82 52" aria-hidden="true">
      {pts.slice(0, -1).map((layer, li) => layer.flatMap((a, i) => pts[li + 1].map((b, j) =>
        <line key={`${li}-${i}-${j}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} />)))}
      {pts.map((layer, li) => layer.map((p, i) => <circle key={`${li}-${i}`} className={`n${li}`} cx={p.x} cy={p.y} r={4.2} />))}
    </svg>
  );
}

type Props = { tour?: TourStep | null; compact?: boolean };

export function Architecture2D({ tour = null, compact = false }: Props) {
  const root = useRef<HTMLDivElement>(null);
  const top = useRef<HTMLDivElement>(null);
  const userRef = useRef<HTMLButtonElement>(null);
  const feRef = useRef<HTMLElement>(null);
  const beRef = useRef<HTMLElement>(null);
  const pop = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ id: string; el: HTMLElement } | null>(null);
  const [pinned, setPinned] = useState<{ id: string; el: HTMLElement } | null>(null);
  const [paths, setPaths] = useState<{ ufe: string; ube: string } | null>(null);
  const shown = hover ?? pinned;

  const active = useMemo(() => {
    if (!tour) return null;
    const comps = new Set<string>(), conns = new Set<string>();
    for (const fid of tour.flows) { const f = FLOW[fid]; comps.add(f.from); comps.add(f.to); conns.add(connectorOf(fid)); }
    return { comps, conns, layer: tour.layer };
  }, [tour]);
  const tileState = (id: string) => (active ? (active.comps.has(id) ? "is-on" : "is-dim") : "");
  const connState = (k: string) => (active ? (active.conns.has(k) ? "is-on" : "is-off") : "");
  const bandState = (l: LayerId) => (active ? (active.layer === l ? "is-on" : "is-dim") : "");

  // researcher → frontend / backend arrows, measured from the laid-out boxes
  useLayoutEffect(() => {
    const measure = () => {
      const t = top.current, u = userRef.current, fe = feRef.current, be = beRef.current;
      if (!t || !u || !fe || !be) return;
      const T = t.getBoundingClientRect(), U = u.getBoundingClientRect(), F = fe.getBoundingClientRect(), B = be.getBoundingClientRect();
      if (F.left - U.right < 20) { setPaths(null); return; }              // stacked layout: no side arrows
      const uy = U.top + U.height / 2 - T.top, ux = U.right - T.left + 4;
      const fy = Math.min(Math.max(uy, F.top - T.top + 30), F.bottom - T.top - 20);
      const cx = U.left + U.width / 2 - T.left, by = B.top + B.height / 2 - T.top, bx = B.left - T.left - 6;
      setPaths({
        ufe: `M ${ux} ${uy} C ${ux + 30} ${uy} ${F.left - T.left - 36} ${fy} ${F.left - T.left - 6} ${fy}`,
        ube: `M ${cx} ${U.bottom - T.top + 4} V ${by - 14} Q ${cx} ${by} ${cx + 14} ${by} H ${bx}`,
      });
    };
    measure();
    const ro = new ResizeObserver(measure);
    if (root.current) ro.observe(root.current);
    return () => ro.disconnect();
  }, []);

  // details popover: under the tile, kept inside the diagram
  useLayoutEffect(() => {
    const el = pop.current, r = root.current;
    if (!el || !r || !shown) return;
    const R = r.getBoundingClientRect(), E = shown.el.getBoundingClientRect();
    let left = E.left + E.width / 2 - R.left - el.offsetWidth / 2;
    left = Math.min(Math.max(8, left), R.width - el.offsetWidth - 8);
    let y = E.bottom - R.top + 10;
    if (y + el.offsetHeight > R.height - 8 && E.top - R.top - el.offsetHeight - 10 > 8) y = E.top - R.top - el.offsetHeight - 10;
    el.style.transform = `translate(${left}px, ${y}px)`;
  }, [shown]);

  useEffect(() => {
    if (!pinned) return;
    const off = (e: Event) => { if (!(e.target as HTMLElement).closest?.(".a2-tile, .a2-chip, .a2-band-head, .a2-user")) setPinned(null); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") setPinned(null); };
    document.addEventListener("mousedown", off);
    document.addEventListener("keydown", esc);
    return () => { document.removeEventListener("mousedown", off); document.removeEventListener("keydown", esc); };
  }, [pinned]);

  const handlers = (id: string) => ({
    onMouseEnter: (e: MouseEvent<HTMLElement>) => setHover({ id, el: e.currentTarget }),
    onMouseLeave: () => setHover(null),
    onFocus: (e: FocusEvent<HTMLElement>) => setHover({ id, el: e.currentTarget }),
    onBlur: () => setHover(null),
    onClick: (e: MouseEvent<HTMLElement>) => { const el = e.currentTarget; setPinned((p) => (p?.id === id ? null : { id, el })); },
    "aria-pressed": pinned?.id === id,
    "aria-label": id.startsWith("layer:") ? LAYER[id.slice(6) as LayerId].label : COMP[id].label,
  });

  const tile = (id: string, variant: "col" | "row" | "stage" = "row", extra?: ReactNode) => {
    const c = COMP[id];
    return (
      <button key={id} type="button" className={`a2-tile ${variant} ${tileState(id)}`} style={cssVars({ "--c": LAYER[c.layer].color })} {...handlers(id)}>
        {extra ?? <span className="a2-ico"><Icon name={c.icon} size={variant === "row" ? 18 : 21} /></span>}
        <span className="a2-txt"><b>{c.label}</b>{c.sub && <small>{c.sub}</small>}</span>
      </button>
    );
  };
  const band = (layer: LayerId, children: ReactNode, cls = "", ref?: Ref<HTMLElement>) => {
    const L = LAYER[layer];
    return (
      <section ref={ref} className={`a2-band ${cls} ${bandState(layer)}`} style={cssVars({ "--c": L.color })}>
        <button type="button" className="a2-band-head" {...handlers(`layer:${layer}`)}>
          <Icon name={L.icon} size={15} stroke={2} />{L.label}{L.tag && <span>{L.tag}</span>}
        </button>
        {children}
      </section>
    );
  };
  const vconn = (k: string, lanes: Lane[], label: string, cls = "") => (
    <div className={`a2-vc ${connState(k)} ${cls}`}>
      <div className="a2-lanes">{lanes.map((l, i) => <span key={i} className={`a2-lane v ${l.dir}`} style={laneStyle(l.kind)} />)}</div>
      <span className="a2-conn-label">{label}</span>
    </div>
  );
  const hconn = (k: string, lane: Lane, label: string) => (
    <div className={`a2-hc ${connState(k)}`}>
      <span className={`a2-lane h ${lane.dir}`} style={laneStyle(lane.kind)} />
      <span className="a2-lane v down" style={laneStyle(lane.kind)} />
      <span className="a2-conn-label">{label}</span>
    </div>
  );
  const arrow = (k: string, kind: FlowKind) => (
    <span className={`a2-arrow ${connState(k)}`} aria-hidden="true"><span className="a2-lane h right" style={laneStyle(kind)} /></span>
  );

  const detail = shown
    ? shown.id.startsWith("layer:") ? LAYER[shown.id.slice(6) as LayerId].detail : COMP[shown.id].detail
    : null;

  return (
    <div className={`a2 ${compact ? "compact" : ""}`} ref={root}>
      {!compact && (
        <div className="a2-title">
          <h3>Continual Learning Experimental Studio</h3>
          <span>System architecture · hover or click any part for details</span>
        </div>
      )}

      <div className="a2-top" ref={top}>
        <button ref={userRef} type="button" className={`a2-user ${tileState("researcher")}`} {...handlers("researcher")}>
          <span className="a2-avatar"><Icon name="user" size={30} stroke={1.9} /></span>
          <b>Researcher</b><small>User</small>
        </button>
        {vconn("u-fe", [{ kind: "control", dir: "down" }], "configure · run · inspect", "a2-user-vc")}
        {band("frontend", <div className="a2-grid-fe">
          {["fe_overview", "fe_studio", "fe_monitor", "fe_analysis", "fe_compare", "fe_geometry", "fe_explorer", "fe_paper", "fe_docs"].map((id) => tile(id, "col"))}
        </div>, "a2-fe", feRef)}
        {vconn("fe-be", [{ kind: "request", dir: "down" }, { kind: "live", dir: "up" }], "REST API · WebSocket (live updates)")}
        {band("backend", <div className="a2-grid-be">
          {["svc_experiment", "svc_config", "svc_registry", "svc_metrics", "svc_resource", "svc_log", "svc_result", "svc_validation"].map((id) => tile(id))}
        </div>, "a2-be", beRef)}
        {paths && (
          <svg className="a2-overlay" aria-hidden="true">
            <defs>
              {(["control", "request"] as const).map((k) => (
                <marker key={k} id={`a2-head-${k}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill={FLOW_KINDS[k].color} />
                </marker>
              ))}
            </defs>
            <path className={`base ${connState("u-fe")}`} d={paths.ufe} stroke={FLOW_KINDS.control.color} markerEnd="url(#a2-head-control)" />
            <path className={connState("u-fe")} d={paths.ufe} stroke={FLOW_KINDS.control.color} />
            <path className={`base ${connState("u-be")}`} d={paths.ube} stroke={FLOW_KINDS.request.color} markerEnd="url(#a2-head-request)" />
            <path className={connState("u-be")} d={paths.ube} stroke={FLOW_KINDS.request.color} />
          </svg>
        )}
      </div>

      {vconn("be-eng", [{ kind: "control", dir: "down" }, { kind: "live", dir: "up" }], "one subprocess per run · progress events back")}

      <div className="a2-mid">
        {band("datasets", <div className="a2-stack">{tile("ds_permuted")}{tile("ds_rotated")}</div>)}
        {hconn("ds-eng", { kind: "data", dir: "right" }, "task stream")}
        {band("engine", (
          <>
            <div className="a2-pipe">
              {tile("eng_input", "stage")}
              {arrow("p1", "data")}
              {tile("eng_mlp", "stage", <MiniNet />)}
              {arrow("p2", "update")}
              {tile("eng_opt", "stage")}
              {arrow("p3", "update")}
              {tile("eng_train", "stage")}
              {arrow("p4", "data")}
              {tile("eng_eval", "stage")}
            </div>
            <div className={`a2-loop ${connState("loop")}`} aria-hidden="true">
              <svg viewBox="0 0 100 26" preserveAspectRatio="none">
                <path d="M 97 0 V 14 Q 97 22 89 22 H 11 Q 3 22 3 14 V 1" vectorEffect="non-scaling-stroke" />
              </svg>
              <span>next task t + 1</span>
            </div>
            <div className={`a2-methods ${connState("methods") === "is-on" ? "is-on" : ""}`}>
              <button type="button" className={`a2-methods-head ${tileState("eng_methods")}`} {...handlers("eng_methods")}>
                <Icon name="layers" size={16} /> <span><b>Update methods</b><small>shared step size → Optimization</small></span>
              </button>
              <div className="a2-chips">
                {METHOD_IDS.map((id) => (
                  <button key={id} type="button" className={`a2-chip ${COMP[id].accent ? "accent" : ""} ${tileState(id)}`} {...handlers(id)}>
                    {COMP[id].label}
                  </button>
                ))}
              </div>
            </div>
          </>
        ), "a2-engine")}
        {hconn("eng-met", { kind: "metrics", dir: "right" }, "per-task diagnostics")}
        {band("metrics", <div className="a2-stack">
          {["met_norm", "met_decay", "met_dormant", "met_rank", "met_geometry"].map((id) => tile(id))}
        </div>)}
      </div>

      {vconn("st", [{ kind: "storage", dir: "down" }, { kind: "storage", dir: "up" }, { kind: "data", dir: "up" }], "config snapshots · results & logs · checkpoints · CUDA")}

      {band("storage", <div className="a2-storage">
        {["st_checkpoints", "st_results", "st_registry", "st_compute"].map((id) => tile(id))}
      </div>)}

      <FlowLegend className="a2-legend" />

      <div className="a3-pop a2-pop" ref={pop} hidden={!detail}>{detail && <DetailCard d={detail} />}</div>
    </div>
  );
}
