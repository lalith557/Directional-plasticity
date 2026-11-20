import { useEffect, useRef, ReactNode } from "react";
import { useTheme } from "../ThemeProvider";

/**
 * KineticGrid — interactive dot/line grid that warps toward the pointer and
 * ripples on click. Pure canvas, no deps. Scientific-neutral: it is background
 * decoration only and renders behind content.
 */
export default function KineticGrid({ children }: { children?: ReactNode }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { theme } = useTheme();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let w = (canvas.width = window.innerWidth * devicePixelRatio);
    let h = (canvas.height = window.innerHeight * devicePixelRatio);
    canvas.style.width = window.innerWidth + "px";
    canvas.style.height = window.innerHeight + "px";

    const SPACING = 40 * devicePixelRatio;
    const RADIUS = 180 * devicePixelRatio;
    const RADIUS_SQ = RADIUS * RADIUS;

    const mouse = { x: -9999, y: -9999 };
    const ripples: { x: number; y: number; t: number }[] = [];

    const onMove = (e: MouseEvent) => {
      mouse.x = e.clientX * devicePixelRatio;
      mouse.y = e.clientY * devicePixelRatio;
    };
    const onLeave = () => {
      mouse.x = -9999;
      mouse.y = -9999;
    };
    const onClick = (e: MouseEvent) => {
      ripples.push({
        x: e.clientX * devicePixelRatio,
        y: e.clientY * devicePixelRatio,
        t: performance.now(),
      });
    };
    const onResize = () => {
      w = canvas.width = window.innerWidth * devicePixelRatio;
      h = canvas.height = window.innerHeight * devicePixelRatio;
      canvas.style.width = window.innerWidth + "px";
      canvas.style.height = window.innerHeight + "px";
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseleave", onLeave);
    window.addEventListener("click", onClick);
    window.addEventListener("resize", onResize);

    let raf = 0;
    const draw = () => {
      ctx.clearRect(0, 0, w, h);
      const dark = theme === "dark";
      const now = performance.now();

      // prune old ripples
      for (let i = ripples.length - 1; i >= 0; i--) {
        if (now - ripples[i].t > 1800) ripples.splice(i, 1);
      }

      for (let x = SPACING / 2; x < w; x += SPACING) {
        for (let y = SPACING / 2; y < h; y += SPACING) {
          const dx = x - mouse.x;
          const dy = y - mouse.y;
          const d2 = dx * dx + dy * dy;

          let ox = 0;
          let oy = 0;
          let intensity = 0;

          if (d2 < RADIUS_SQ) {
            const f = 1 - d2 / RADIUS_SQ;
            ox = -dx * f * 0.22;
            oy = -dy * f * 0.22;
            intensity = f * 0.6;
          }

          // ripple contributions
          for (const r of ripples) {
            const age = (now - r.t) / 1800;
            const ringR = age * 500 * devicePixelRatio;
            const rdx = x - r.x;
            const rdy = y - r.y;
            const rd = Math.sqrt(rdx * rdx + rdy * rdy);
            const diff = Math.abs(rd - ringR);
            if (diff < 60 * devicePixelRatio) {
              const amp = (1 - age) * (1 - diff / (60 * devicePixelRatio));
              const theta = Math.atan2(rdy, rdx);
              ox += Math.cos(theta) * amp * 14 * devicePixelRatio;
              oy += Math.sin(theta) * amp * 14 * devicePixelRatio;
              intensity += amp * 0.8;
            }
          }

          const alpha = Math.min(1, 0.08 + intensity);
          ctx.fillStyle = dark
            ? `rgba(200,210,230,${alpha})`
            : `rgba(49,90,115,${alpha * 0.6})`;
          ctx.beginPath();
          ctx.arc(x + ox, y + oy, 1.3 * devicePixelRatio, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      raf = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseleave", onLeave);
      window.removeEventListener("click", onClick);
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(raf);
    };
  }, [theme]);

  return (
    <>
      <div className="bg-layer" aria-hidden>
        <canvas ref={canvasRef} />
      </div>
      {children}
    </>
  );
}
