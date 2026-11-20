import { useEffect } from "react";
import { useLocation } from "react-router-dom";

/**
 * useScrollAnim — attaches an IntersectionObserver to elements matching the
 * SELECTOR list after every route change. Mirrors the "animate on enter"
 * behavior of GSAP ScrollTrigger (once-only) without the dependency weight,
 * and respects prefers-reduced-motion.
 */
const SELECTOR = ".card, .observation, .page-head, section, .integrity-banner, .stat-line";

// Studio pages are working tools: content is shown immediately, never hidden behind a reveal.
const STATIC_ROUTE = /^\/(studio|monitor|explorer|compare|architectures|docs|paper-results|system)(\/|$)|^\/analysis$/;

export function useScrollAnim() {
  const { pathname } = useLocation();
  useEffect(() => {
    if (STATIC_ROUTE.test(pathname)) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      document
        .querySelectorAll<HTMLElement>(SELECTOR)
        .forEach((el) => el.classList.add("scroll-reveal", "is-visible"));
      return;
    }

    const seen = new WeakSet<Element>();
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting && !seen.has(e.target)) {
            seen.add(e.target);
            const idx = Number((e.target as HTMLElement).dataset.srIdx || 0);
            (e.target as HTMLElement).style.transitionDelay = `${Math.min(idx, 6) * 60}ms`;
            (e.target as HTMLElement).classList.add("is-visible");
            io.unobserve(e.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -60px 0px" }
    );

    // allow route render to settle before observing
    const raf = requestAnimationFrame(() => {
      document.querySelectorAll<HTMLElement>(SELECTOR).forEach((el, i) => {
        if (!el.classList.contains("scroll-reveal")) el.classList.add("scroll-reveal");
        el.dataset.srIdx = String(i);
        io.observe(el);
      });
    });

    return () => {
      cancelAnimationFrame(raf);
      io.disconnect();
    };
  }, [pathname]);
}
