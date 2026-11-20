import { Outlet, useLocation } from "react-router-dom";
import { MorphingIslandNavbar } from "./ui/MorphingIslandNavbar";
import { ThemeToggle } from "./ThemeProvider";
import KineticGrid from "./ui/KineticGrid";
import WakeGlyph from "./ui/WakeGlyph";
import { useScrollAnim } from "./ui/useScrollAnim";

export function AppShell() {
  const { pathname } = useLocation();
  const isHome = pathname === "/";
  const isPaper = pathname.startsWith("/paper");
  // Studio pages are working tools: no decorative animated backgrounds.
  const isStudio = /^\/(studio|monitor|explorer|compare|architectures|docs)(\/|$)/.test(pathname) || pathname === "/analysis";
  useScrollAnim();

  return (
    <div className="shell">
      {isHome ? <KineticGrid /> : isPaper || isStudio ? null : <WakeGlyph />}
      <MorphingIslandNavbar />
      <ThemeToggle />
      <div className="main">
        <div className="content">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
