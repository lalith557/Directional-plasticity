import { ReactNode } from "react";
import { useData } from "../data/DataContext";
import type { PlasticityData } from "../data/types";
import { Loading, ErrorBox } from "./ui";

/** Render-prop gate: handles loading + error before exposing real data. */
export function DataGate({ children }: { children: (d: PlasticityData) => ReactNode }) {
  const { data, loading, error } = useData();
  if (loading)
    return (
      <div className="grid" style={{ gap: 16 }}>
        <Loading lines={2} height={40} />
        <Loading lines={4} height={80} />
      </div>
    );
  if (error) return <ErrorBox source={error.source} reason={error.reason} />;
  if (!data) return <ErrorBox source="plasticity.json" reason="Dataset is empty." />;
  return <>{children(data)}</>;
}
