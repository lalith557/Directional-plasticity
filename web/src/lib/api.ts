// Studio backend client. All calls go through the Vite dev proxy at /api
// (→ http://127.0.0.1:8765). Nothing here produces data on its own: if the backend
// is offline, callers render an explicit "backend not running" state.
import { useCallback, useEffect, useRef, useState } from "react";

export const API_BASE = "/api";

export const REPO_URL = "https://github.com/lalith557/Directional-plasticity";

/**
 * True when the app is served as a static site (e.g. GitHub Pages) instead of from a local
 * dev/preview server. There is no Studio backend in that case, so no request is attempted.
 */
export const IS_HOSTED =
  typeof window !== "undefined" && !["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);

export class ApiError extends Error {
  status: number;
  detail: any;
  constructor(status: number, detail: any) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
    this.detail = detail;
  }
}

const HOSTED_ERROR = "The Studio backend is not part of the hosted site.";

export async function api<T = any>(path: string, init?: RequestInit & { json?: any }): Promise<T> {
  if (IS_HOSTED) throw new ApiError(503, HOSTED_ERROR);
  const { json, ...rest } = init ?? {};
  const res = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers: { ...(json !== undefined ? { "Content-Type": "application/json" } : {}), ...(rest.headers ?? {}) },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  const text = await res.text();
  let body: any = text;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    /* plain-text endpoints (reports, config downloads) */
  }
  if (!res.ok) throw new ApiError(res.status, body?.detail ?? body ?? res.statusText);
  return body as T;
}

export async function apiText(path: string, init?: RequestInit & { json?: any }): Promise<string> {
  if (IS_HOSTED) throw new ApiError(503, HOSTED_ERROR);
  const { json, ...rest } = init ?? {};
  const res = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers: json !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.text();
}

/** Fetch (and optionally poll) an endpoint. `path = null` disables the request. */
export function useApi<T = any>(path: string | null, pollMs?: number) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [loading, setLoading] = useState<boolean>(!!path);
  const alive = useRef(true);

  const load = useCallback(async () => {
    if (!path) return;
    try {
      const d = await api<T>(path);
      if (alive.current) {
        setData(d);
        setError(null);
      }
    } catch (e) {
      if (alive.current) setError(e as Error);
    } finally {
      if (alive.current) setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    alive.current = true;
    setLoading(!!path);
    load();
    if (!pollMs || !path) return () => { alive.current = false; };
    const t = window.setInterval(load, pollMs);
    return () => {
      alive.current = false;
      window.clearInterval(t);
    };
  }, [path, pollMs, load]);

  return { data, error, loading, reload: load };
}

/** Backend reachability (polled). null = still checking. */
export function useBackend(pollMs = 5000) {
  const { data, error } = useApi<any>(IS_HOSTED ? null : "/health", pollMs);
  if (IS_HOSTED) return { online: false as boolean | null, health: null };
  return { online: data ? true : error ? false : null, health: data };
}

export function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${API_BASE}${path}`;
}

/** Trigger a browser download of text content (config files, reports). */
export function downloadText(filename: string, text: string, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ------------------------------------------------------------------ types
export type Status =
  | "proposed" | "queued" | "initializing" | "running" | "completed" | "failed" | "cancelled"
  | "completed_diverged" | "superseded" | "incomplete";

export interface MetricStat { mean: number | null; std: number | null; n?: number }

export interface RecordSummary {
  id: string;
  source: "legacy_artifact" | "studio" | "studio_group";
  name: string;
  category: string;
  category_label?: string;
  experiment?: string;
  status: Status | string;
  validation?: { status: string; detail?: string; against?: string; max_abs_diff?: number };
  dataset?: string | null;
  architecture?: any;
  method?: string | null;
  method_label?: string;
  family?: string;
  matched_step?: "reference" | "applicable" | "na";
  params?: Record<string, any>;
  seeds?: number[];
  n_seeds?: number;
  group?: string;
  metrics?: Record<string, MetricStat>;
  paper_ref?: string | null;
  artifact?: string | null;
  created_at?: string | null;
  has_per_task?: boolean;
  config_hash?: string;
  duration_s?: number | null;
  device_name?: string | null;
  arm_label?: string;
  sweep?: { param: string; value: number };
  verdict?: string;
  error?: string | null;
  notes?: string[];
}

export const TERMINAL = new Set(["completed", "failed", "cancelled"]);

export function archLabel(a: any): string {
  if (!a) return "—";
  if (a.name === "mlp") return `MLP h=${a.hidden}${a.depth && a.depth !== 2 ? ` d=${a.depth}` : ""}`;
  if (a.name === "cnn") return `CNN ${a.base}/${2 * a.base}/fc${a.fc}`;
  return a.name ?? "—";
}

export const DATASET_LABEL: Record<string, string> = {
  permuted_mnist: "Permuted MNIST",
  rotated_mnist: "Rotated MNIST",
  split_cifar10: "Split-CIFAR-10",
};
