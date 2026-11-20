import { Routes, Route, Navigate } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Overview } from "./pages/Overview";
import { R20 } from "./pages/R20";
import { LambdaSweep } from "./pages/LambdaSweep";
import { RadialSweep } from "./pages/RadialSweep";
import { WidthStudy } from "./pages/WidthStudy";
import { RotatedMNIST } from "./pages/RotatedMNIST";
import { CNNProbe } from "./pages/CNNProbe";
import { NormPlasticity } from "./pages/NormPlasticity";
import { UpdateGeometry } from "./pages/UpdateGeometry";
import { Ablations } from "./pages/Ablations";
import { Comparative } from "./pages/Comparative";
import { RunDetails } from "./pages/RunDetails";
import { Configuration } from "./pages/Configuration";
import { Datasets } from "./pages/Datasets";
import { Checkpoints } from "./pages/Checkpoints";
import { RawResults } from "./pages/RawResults";
import { Paper } from "./pages/Paper";
import { ExperimentStudio } from "./pages/studio/ExperimentStudio";
import { LiveMonitor } from "./pages/studio/LiveMonitor";
import { ResultsExplorer } from "./pages/studio/ResultsExplorer";
import { ExperimentDetail } from "./pages/studio/ExperimentDetail";
import { MethodComparison } from "./pages/studio/MethodComparison";
import { ScientificAnalysis } from "./pages/studio/ScientificAnalysis";
import { ArchitectureExplorer } from "./pages/studio/ArchitectureExplorer";
import { PaperResults } from "./pages/studio/PaperResults";
import { Documentation } from "./pages/studio/Documentation";
import { SystemArchitecture } from "./pages/studio/SystemArchitecture";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Overview />} />
        <Route path="/experiments/r20" element={<R20 />} />
        <Route path="/experiments/lambda" element={<LambdaSweep />} />
        <Route path="/experiments/radial" element={<RadialSweep />} />
        <Route path="/experiments/width" element={<WidthStudy />} />
        <Route path="/experiments/rotated" element={<RotatedMNIST />} />
        <Route path="/experiments/cnn" element={<CNNProbe />} />
        <Route path="/analysis/norm-plasticity" element={<NormPlasticity />} />
        <Route path="/analysis/geometry" element={<UpdateGeometry />} />
        <Route path="/analysis/ablations" element={<Ablations />} />
        <Route path="/analysis/comparative" element={<Comparative />} />
        <Route path="/runs" element={<RunDetails />} />
        <Route path="/reproducibility/config" element={<Configuration />} />
        <Route path="/reproducibility/datasets" element={<Datasets />} />
        <Route path="/reproducibility/checkpoints" element={<Checkpoints />} />
        <Route path="/reproducibility/raw" element={<RawResults />} />
        <Route path="/paper" element={<Paper />} />
        <Route path="/studio" element={<ExperimentStudio />} />
        <Route path="/monitor" element={<LiveMonitor />} />
        <Route path="/monitor/:id" element={<LiveMonitor />} />
        <Route path="/explorer" element={<ResultsExplorer />} />
        <Route path="/explorer/:id" element={<ExperimentDetail />} />
        <Route path="/compare" element={<MethodComparison />} />
        <Route path="/analysis" element={<ScientificAnalysis />} />
        <Route path="/architectures" element={<ArchitectureExplorer />} />
        <Route path="/paper-results" element={<PaperResults />} />
        <Route path="/docs" element={<Documentation />} />
        <Route path="/system" element={<SystemArchitecture />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
