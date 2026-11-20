import { ReactNode, useEffect, useRef } from "react";

/**
 * ScrollReveal — IntersectionObserver-based entrance. Adds `is-visible` to the
 * wrapper once the element enters the viewport. Mirrors a GSAP ScrollTrigger
 * "once" pattern without the dependency weight.
 */
export function ScrollReveal({
  children,
  as: As = "div",
  delay = 0,
  threshold = 0.15,
  className = "",
}: {
  children: ReactNode;
  as?: any;
  delay?: number;
  threshold?: number;
  className?: string;
}) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setTimeout(() => el.classList.add("is-visible"), delay);
            io.unobserve(el);
          }
        });
      },
      { threshold, rootMargin: "0px 0px -40px 0px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [delay, threshold]);
  return (
    <As ref={ref} className={`scroll-reveal ${className}`}>
      {children}
    </As>
  );
}
