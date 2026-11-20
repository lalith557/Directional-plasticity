import { ElementType, ReactNode, useId } from "react";

/**
 * BlurReveal — splits children (string) into words and animates each with a
 * blur-to-focus entrance. Mounts once per key change. Non-string children are
 * passed through unmodified.
 */
export function BlurReveal({
  children,
  as: As = "div",
  className = "",
  delay = 0,
  stagger = 60,
}: {
  children: ReactNode;
  as?: ElementType;
  className?: string;
  delay?: number;
  stagger?: number;
}) {
  const id = useId();
  if (typeof children !== "string") {
    return <As className={`blur-reveal-root ${className}`}>{children}</As>;
  }
  const words = children.split(/(\s+)/);
  return (
    <As className={`blur-reveal-root ${className}`}>
      {words.map((w, i) =>
        /^\s+$/.test(w) ? (
          <span key={`${id}-${i}`}>{w}</span>
        ) : (
          <span
            key={`${id}-${i}`}
            className="blur-reveal-word"
            style={{ animationDelay: `${delay + (i / 2) * stagger}ms` }}
          >
            {w}
          </span>
        )
      )}
    </As>
  );
}
