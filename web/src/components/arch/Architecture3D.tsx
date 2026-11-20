import { useEffect, useRef, useState } from "react";
import { useTheme } from "../ThemeProvider";
import { COMP, LAYER, type LayerId, type TourStep } from "./archModel";
import { DetailCard, FLOW, FlowCard, FlowLegend } from "./archUi";
import { ArchScene, type Pick } from "./scene3d";

type Props = {
  /** Active tour step: raises its tier and highlights its connections. */
  tour?: TourStep | null;
  /** Called when the user takes over (clicks a part), so the page can leave the tour. */
  onUserSelect?: () => void;
};

export default function Architecture3D({ tour = null, onUserSelect }: Props) {
  const { theme } = useTheme();
  const stage = useRef<HTMLDivElement>(null);
  const pop = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<ArchScene | null>(null);
  const userSelect = useRef(onUserSelect);
  userSelect.current = onUserSelect;
  const [hover, setHover] = useState<Pick | null>(null);
  const [selected, setSelected] = useState<LayerId | null>(null);
  const [interacted, setInteracted] = useState(false);
  const [rotate, setRotate] = useState(true);
  const [flowLabels, setFlowLabels] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const host = stage.current!;
    let s: ArchScene;
    try {
      s = new ArchScene(host, theme, {
        onHover: setHover,
        onPointer: (x, y) => {
          const el = pop.current, r = host.getBoundingClientRect();
          if (!el) return;
          let left = x - r.left + 18, top = y - r.top + 18;
          if (left + el.offsetWidth > r.width - 8) left = x - r.left - el.offsetWidth - 18;
          if (top + el.offsetHeight > r.height - 8) top = Math.max(8, r.height - el.offsetHeight - 8);
          el.style.transform = `translate(${Math.max(8, left)}px, ${top}px)`;
        },
        onSelect: (l) => { setSelected(l); userSelect.current?.(); },
        onComp: () => {},
        onFlow: () => {},
        onInteract: () => { setInteracted(true); setRotate(false); },
      });
    } catch (e) {
      setError(String(e));
      return;
    }
    sceneRef.current = s;
    const ro = new ResizeObserver(() => s.resize());
    ro.observe(host);
    const io = new IntersectionObserver(([en]) => s.setActive(en.isIntersecting));
    io.observe(host);
    return () => { ro.disconnect(); io.disconnect(); s.dispose(); sceneRef.current = null; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { sceneRef.current?.setTheme(theme); }, [theme]);
  useEffect(() => {
    if (tour) { setSelected(null); setRotate(false); setInteracted(true); }
    sceneRef.current?.setTour(tour);
  }, [tour]);
  // keep scene and UI in sync (e.g. after a remount); never clear an active tour from here
  useEffect(() => { if (selected !== null || !tour) sceneRef.current?.select(selected); }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { sceneRef.current?.setAutoRotate(rotate); }, [rotate]);
  useEffect(() => { sceneRef.current?.setFlowLabels(flowLabels); }, [flowLabels]);

  let content: JSX.Element | null = null;
  if (hover?.kind === "comp") content = <DetailCard d={COMP[hover.id].detail} />;
  else if (hover?.kind === "layer")
    content = hover.elevated ? <DetailCard d={LAYER[hover.id].detail} />
      : <div className="a3-pop-hint"><b>{LAYER[hover.id].label}</b> — click to raise it</div>;
  else if (hover?.kind === "flow") content = <FlowCard f={FLOW[hover.id]} />;

  return (
    <div className="a3-stage" ref={stage}>
      {error && <div className="studio-note a3-fallback">3D view unavailable in this browser (WebGL): {error}</div>}
      <div className="a3-toolbar">
        <button className="btn sm ghost" onClick={() => { sceneRef.current?.resetView(); setSelected(null); userSelect.current?.(); }}>Reset view</button>
        <button className={`btn sm ghost ${rotate ? "on" : ""}`} onClick={() => setRotate((r) => !r)}>{rotate ? "Stop rotation" : "Auto-rotate"}</button>
        <button className={`btn sm ghost ${flowLabels ? "on" : ""}`} onClick={() => setFlowLabels((v) => !v)}>{flowLabels ? "Hide flow labels" : "Show flow labels"}</button>
        {selected && <button className="btn sm ghost on" onClick={() => { sceneRef.current?.select(null); setSelected(null); }}>Lower {LAYER[selected].label}</button>}
      </div>
      {!interacted && <div className="a3-hint">Drag to orbit 360° · scroll to zoom · click a part to raise it, then hover it for details</div>}
      <FlowLegend className="a3-legend" />
      <div className="a3-pop" ref={pop} hidden={!content}>{content}</div>
    </div>
  );
}
