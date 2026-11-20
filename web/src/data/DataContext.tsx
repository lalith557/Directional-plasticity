import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import type { PlasticityData } from "./types";

interface State {
  data: PlasticityData | null;
  loading: boolean;
  error: { source: string; reason: string } | null;
}

const Ctx = createContext<State>({ data: null, loading: true, error: null });

const SOURCE = `${import.meta.env.BASE_URL}data/plasticity.json`;

export function DataProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<State>({ data: null, loading: true, error: null });

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await fetch(SOURCE, { cache: "no-cache" });
        if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
        const text = await res.text();
        let json: PlasticityData;
        try {
          json = JSON.parse(text);
        } catch (e) {
          throw new Error(`JSON parse error: ${(e as Error).message}`);
        }
        if (alive) setState({ data: json, loading: false, error: null });
      } catch (e) {
        if (alive)
          setState({
            data: null,
            loading: false,
            error: { source: SOURCE, reason: (e as Error).message },
          });
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  return <Ctx.Provider value={state}>{children}</Ctx.Provider>;
}

export function useData() {
  return useContext(Ctx);
}
