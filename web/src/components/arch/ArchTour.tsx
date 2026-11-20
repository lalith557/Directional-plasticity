/* "Follow one experiment run": a ten-step guided walk through the architecture. The page passes
   the active step to whichever view is showing (3D raises the tier, 2D highlights the parts). */
import { useEffect } from "react";
import { COMP, FLOW_KINDS, LAYER, TOUR } from "./archModel";
import { FLOW } from "./archUi";
import "./arch.css";

export const TOUR_STEP_MS = 8000;

type Props = {
  index: number | null;
  onChange: (i: number | null) => void;
  playing: boolean;
  onPlaying: (p: boolean) => void;
};

export function ArchTour({ index, onChange, playing, onPlaying }: Props) {
  const step = index !== null ? TOUR[index] : null;
  const last = TOUR.length - 1;

  useEffect(() => {                                   // autoplay
    if (!playing || index === null) return;
    const t = window.setTimeout(() => (index < last ? onChange(index + 1) : onPlaying(false)), TOUR_STEP_MS);
    return () => window.clearTimeout(t);
  }, [playing, index]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {                                   // ← / → / Esc while touring
    if (index === null) return;
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement | null)?.closest?.("input, textarea, select")) return;
      if (e.key === "ArrowRight" && index < last) { onPlaying(false); onChange(index + 1); }
      else if (e.key === "ArrowLeft" && index > 0) { onPlaying(false); onChange(index - 1); }
      else if (e.key === "Escape") { onPlaying(false); onChange(null); }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [index]); // eslint-disable-line react-hooks/exhaustive-deps

  const go = (i: number) => { onPlaying(false); onChange(i); };

  return (
    <section className={`tour ${step ? "is-active" : ""}`} aria-label="Guided tour">
      <div className="tour-head">
        <div>
          <div className="tour-title">Follow one experiment run</div>
          <div className="tour-sub">Ten steps from the first click to a validated result. Each step highlights the parts and the data flowing between them.</div>
        </div>
        <div className="tour-ctrl">
          {index === null ? (
            <button className="btn sm" onClick={() => { onChange(0); onPlaying(true); }}>▶ Start the tour</button>
          ) : (
            <>
              <button className="btn sm ghost" disabled={index === 0} onClick={() => go(index - 1)}>← Previous</button>
              <button className="btn sm ghost" onClick={() => onPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
              <button className="btn sm ghost" disabled={index === last} onClick={() => go(index + 1)}>Next →</button>
              <button className="btn sm ghost" onClick={() => { onPlaying(false); onChange(null); }}>Exit tour</button>
            </>
          )}
        </div>
      </div>

      <ol className="tour-steps">
        {TOUR.map((s, i) => (
          <li key={s.title}>
            <button type="button" aria-current={i === index ? "step" : undefined}
                    className={`tour-step ${i === index ? "on" : ""} ${index !== null && i < index ? "done" : ""}`} onClick={() => go(i)}>
              <span>{i + 1}</span>{s.title}
            </button>
          </li>
        ))}
      </ol>

      {step && index !== null && (
        <div className="tour-body">
          <div className="tour-text">
            <div className="tour-kicker" style={{ color: LAYER[step.layer].color }}>Step {index + 1} of {TOUR.length} · {LAYER[step.layer].label}</div>
            <h4>{step.title}</h4>
            <p>{step.text}</p>
            {playing && <div className="tour-progress"><span key={index} style={{ animationDuration: `${TOUR_STEP_MS}ms` }} /></div>}
          </div>
          <ul className="tour-flows">
            {step.flows.map((fid) => {
              const f = FLOW[fid];
              return (
                <li key={fid}>
                  <i className="a3-dot" style={{ background: FLOW_KINDS[f.kind].color }} />
                  <div><code>{f.label}</code><small>{COMP[f.from].label} → {COMP[f.to].label}</small></div>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}
