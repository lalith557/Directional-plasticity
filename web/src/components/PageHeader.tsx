import { ReactNode } from "react";
import { useLocation } from "react-router-dom";
import { BlurReveal } from "./ui/BlurReveal";

export function PageHeader({
  title,
  subtitle,
  right,
  badge,
}: {
  title: string;
  subtitle?: string;
  right?: ReactNode;
  badge?: ReactNode;
}) {
  const key = useLocation().pathname;
  return (
    <div className="page-head" key={key}>
      <div className="row between aic wrap" style={{ gap: 12 }}>
        <div>
          <div className="page-title-row">
            <h1 className="page-title">
              <BlurReveal as="span">{title}</BlurReveal>
            </h1>
            {badge}
          </div>
          {subtitle && (
            <p className="page-sub">
              <BlurReveal as="span" delay={200} stagger={30}>
                {subtitle}
              </BlurReveal>
            </p>
          )}
        </div>
        {right}
      </div>
    </div>
  );
}
