import { useEffect, useRef } from "react";
import { useTheme } from "../ThemeProvider";

/**
 * WakeGlyph — ASCII character field that reacts to cursor movement. Lightweight
 * background decoration using canvas-rendered monospace glyphs. Behind content.
 */
export default function WakeGlyph({ className = "" }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { theme } = useTheme();

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const CHARS = "·.•+*×o◦○●".split("");
    const COLS = 90;
    const ROWS = 50;

    let w = (canvas.width = window.innerWidth * devicePixelRatio);
    let h = (canvas.height = window.innerHeight * devicePixelRatio);
    canvas.style.width = window.innerWidth + "px";
    canvas.style.height = window.innerHeight + "px";

    const mouse = { x: -9999, y: -9999, vx: 0, vy: 0 };
    let lastX = 0;
    let lastY = 0;

    const onMove = (e: MouseEvent) => {
      mouse.vx = e.clientX - lastX;
      mouse.vy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      mouse.x = e.clientX * devicePixelRatio;
      mouse.y = e.clientY * devicePixelRatio;
    };
    const onResize = () => {
      w = canvas.width = window.innerWidth * devicePixelRatio;
      h = canvas.height = window.innerHeight * devicePixelRatio;
      canvas.style.width = window.innerWidth + "px";
      canvas.style.height = window.innerHeight + "px";
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("resize", onResize);

    const trail: { x: number; y: number; t: number }[] = [];
    let raf = 0;
    let frame = 0;
    const draw = () => {
      frame++;
      ctx.clearRect(0, 0, w, h);
      const dark = theme === "dark";
      ctx.font = `${11 * devicePixelRatio}px "IBM Plex Mono", monospace`;

      if (mouse.x > 0 && (Math.abs(mouse.vx) > 1 || Math.abs(mouse.vy) > 1)) {
        trail.push({ x: mouse.x, y: mouse.y, t: performance.now() });
      }
      mouse.vx *= 0.9;
      mouse.vy *= 0.9;

      const cellW = w / COLS;
      const cellH = h / ROWS;
      const now = performance.now();

      // prune trail
      for (let i = trail.length - 1; i >= 0; i--) {
        if (now - trail[i].t > 1200) trail.splice(i, 1);
      }

      for (let cx = 0; cx < COLS; cx++) {
        for (let cy = 0; cy < ROWS; cy++) {
          const px = cx * cellW + cellW / 2;
          const py = cy * cellH + cellH / 2;

          // base noise
          let intensity = 0.08 + 0.04 * Math.sin((cx * 0.3 + cy * 0.2 + frame * 0.015));

          // trail influence
          for (const t of trail) {
            const dx = px - t.x;
            const dy = py - t.y;
            const d2 = dx * dx + dy * dy;
            const age = (now - t.t) / 1200;
            const reach = (120 * devicePixelRatio) * (1 - age);
            if (reach > 0 && d2 < reach * reach) {
              const f = 1 - Math.sqrt(d2) / reach;
              intensity += f * (1 - age) * 0.9;
            }
          }

          if (intensity < 0.06) continue;
          const idx = Math.min(CHARS.length - 1, Math.floor(intensity * CHARS.length));
          const alpha = Math.min(0.75, intensity);
          ctx.fillStyle = dark
            ? `rgba(190,195,205,${alpha * 0.55})`
            : `rgba(80,80,80,${alpha * 0.4})`;
          ctx.fillText(CHARS[idx], px, py);
        }
      }

      raf = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(raf);
    };
  }, [theme]);

  return (
    <div className={`bg-layer ${className}`} aria-hidden>
      <canvas ref={ref} />
    </div>
  );
}
