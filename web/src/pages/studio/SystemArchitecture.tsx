import { lazy, Suspense, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { PageHeader } from "../../components/PageHeader";
import { Architecture2D } from "../../components/arch/Architecture2D";
import { ArchTour } from "../../components/arch/ArchTour";
import { TOUR } from "../../components/arch/archModel";

const Architecture3D = lazy(() => import("../../components/arch/Architecture3D"));

export function SystemArchitecture() {
  const [params, setParams] = useSearchParams();
  const view = params.get("view") === "2d" ? "2d" : "3d";
  const setView = (v: "2d" | "3d") => setParams(v === "2d" ? { view: "2d" } : {}, { replace: true });
  const [tourIdx, setTourIdx] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const step = tourIdx !== null ? TOUR[tourIdx] : null;

  return (
    <>
      <PageHeader title="System Architecture"
        subtitle="How a run travels through the Studio — from the researcher's click to the GPU and back. Explore it in 3D or as a diagram, or follow one run step by step." />
      <div className="arch-bar">
        <div className="seg" role="tablist" aria-label="View">
          <button role="tab" aria-selected={view === "3d"} className={`seg-btn ${view === "3d" ? "on" : ""}`} onClick={() => setView("3d")}>3D workflow</button>
          <button role="tab" aria-selected={view === "2d"} className={`seg-btn ${view === "2d" ? "on" : ""}`} onClick={() => setView("2d")}>2D diagram</button>
        </div>
        <span className="small muted">
          {view === "3d" ? "Drag to orbit 360° · click a part to raise it · hover it for details" : "Hover or click any part for details · dots show which way data moves"}
        </span>
      </div>

      {/* the 3D scene stays mounted (and paused) while the diagram is shown, so its camera is kept */}
      <div hidden={view !== "3d"}>
        <Suspense fallback={<div className="studio-note">Loading the 3D view…</div>}>
          <Architecture3D tour={view === "3d" ? step : null} onUserSelect={() => { setTourIdx(null); setPlaying(false); }} />
        </Suspense>
      </div>
      {view === "2d" && <Architecture2D tour={step} />}

      <ArchTour index={tourIdx} onChange={setTourIdx} playing={playing} onPlaying={setPlaying} />
    </>
  );
}
