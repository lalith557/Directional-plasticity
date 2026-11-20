import { useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ALL_ROUTES, labelForPath } from "../lib/nav";
import { useData } from "../data/DataContext";

interface Hit {
  title: string;
  sub: string;
  to: string;
}

export function TopBar({ onMenu }: { onMenu: () => void }) {
  const loc = useLocation();
  const nav = useNavigate();
  const { data } = useData();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);

  const index = useMemo<Hit[]>(() => {
    const hits: Hit[] = ALL_ROUTES.map((r) => ({ title: r.label, sub: r.kind, to: r.to }));
    if (data) {
      data.r20.arms.forEach((a) =>
        hits.push({ title: a.name, sub: `Method · ${a.tag}`, to: "/experiments/r20" })
      );
      data.regimes.forEach((r) =>
        hits.push({ title: r.short, sub: `Regime · ${r.name}`, to: "/analysis/norm-plasticity" })
      );
      data.r20.seeds.forEach((s) =>
        hits.push({ title: `SGD · seed ${s}`, sub: "Run", to: "/runs" })
      );
    }
    return hits;
  }, [data]);

  const results = useMemo(() => {
    const term = q.trim().toLowerCase();
    if (!term) return [];
    const seen = new Set<string>();
    return index
      .filter((h) => (h.title + " " + h.sub).toLowerCase().includes(term))
      .filter((h) => {
        const k = h.title + h.to;
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      })
      .slice(0, 8);
  }, [q, index]);

  const go = (to: string) => {
    setQ("");
    setOpen(false);
    nav(to);
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (!open || !results.length) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => (a + 1) % results.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => (a - 1 + results.length) % results.length);
    } else if (e.key === "Enter") {
      go(results[active].to);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <header className="topbar">
      <button className="menu-btn" onClick={onMenu} aria-label="Toggle navigation">
        &#9776;
      </button>
      <div className="topbar-context">
        <span>Plasticity Lab</span>
        <span className="sep">/</span>
        <span className="cur">{labelForPath(loc.pathname)}</span>
      </div>
      <div className="topbar-spacer" />
      <div className="search" ref={boxRef}>
        <span className="search-ico" aria-hidden>
          &#9906;
        </span>
        <input
          value={q}
          placeholder="Search experiments..."
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
            setActive(0);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={onKey}
          aria-label="Search experiments"
        />
        {open && q.trim() && (
          <div className="search-results">
            {results.length === 0 ? (
              <div className="search-empty">No matches for &ldquo;{q}&rdquo;</div>
            ) : (
              results.map((h, i) => (
                <div
                  key={h.title + h.to + i}
                  className={`search-result ${i === active ? "active" : ""}`}
                  onMouseDown={() => go(h.to)}
                  onMouseEnter={() => setActive(i)}
                >
                  <span className="r-title">{h.title}</span>
                  <span className="r-sub">{h.sub}</span>
                </div>
              ))
            )}
          </div>
        )}
      </div>
      <div className="avatar" title="Researcher">
        R
      </div>
    </header>
  );
}
