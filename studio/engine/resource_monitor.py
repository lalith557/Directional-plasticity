"""GPU / CPU resource monitoring (read-only) and compute estimates."""
from __future__ import annotations

import time

# Measured reference: the Cor. 1 audit ran five 100-task × 60-step width-16 MLP runs
# on the RTX 4070 Laptop GPU in 27–32 s each (results/tables/cor1_orthogonal_audit.json
# timing printed by the audit script). 30 s / 6000 steps, evaluation included.
REFERENCE = {"arch": "mlp", "max_hidden": 16, "device_substr": "RTX 4070",
             "sec_per_step": 30.0 / 6000,
             "source": "Cor. 1 audit timing (RTX 4070 Laptop, MLP h=16, 100×60 steps ≈ 30 s)"}


def _nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def gpu_status() -> dict:
    """Current GPU state from torch + NVML; fields are omitted (not faked) if unavailable."""
    out: dict = {"timestamp": time.time()}
    try:
        import torch
        out["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            out.update(device=torch.cuda.get_device_name(0),
                       torch_free_gb=round(free / 1e9, 3), torch_total_gb=round(total / 1e9, 3))
    except Exception as e:  # pragma: no cover
        out["cuda_available"] = False
        out["torch_error"] = str(e)
    nv = _nvml()
    if nv is not None:
        try:
            h = nv.nvmlDeviceGetHandleByIndex(0)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            out.update(utilization_pct=int(util.gpu), memory_util_pct=int(util.memory),
                       memory_used_mb=round(mem.used / 2**20, 1),
                       memory_total_mb=round(mem.total / 2**20, 1))
            try:
                out["temperature_c"] = int(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            try:
                out["power_w"] = round(nv.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                pass
            out["source"] = "nvml"
        except Exception as e:
            out["nvml_error"] = str(e)
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["cpu_percent"] = psutil.cpu_percent(interval=None)
        out["ram_used_gb"] = round(vm.used / 1e9, 2)
        out["ram_total_gb"] = round(vm.total / 1e9, 2)
    except Exception:
        pass
    return out


def estimate(cfg: dict, n_seeds: int, history: list[dict], device_name: str | None) -> dict:
    """Wall-clock estimate from *measured* timings only.

    Uses completed Studio runs with the same architecture (and MLP width / CNN size)
    on the same device; falls back to the documented reference measurement only when it
    applies (MLP, width ≤ 16, RTX 4070). Otherwise reports 'not measurable yet'.
    """
    steps = cfg["dataset"]["n_tasks"] * cfg["optimization"]["steps_per_task"]
    ar = cfg["architecture"]
    size_key = (ar["name"], ar.get("hidden"), ar.get("depth"), ar.get("base"), ar.get("fc"))
    samples = []
    for h in history:
        hc = h.get("config") or {}
        ha = hc.get("architecture") or {}
        hk = (ha.get("name"), ha.get("hidden"), ha.get("depth"), ha.get("base"), ha.get("fc"))
        steps_done = h.get("steps_done")
        if hk == size_key and h.get("device_name") == device_name and steps_done and h.get("duration_s"):
            samples.append((h["duration_s"], steps_done))
    total_steps = steps * n_seeds
    if samples:
        # Pool total time over total steps: short runs carry fixed start-up/evaluation overhead
        # per step, so a per-run median would over-estimate long runs.
        sec, stp = sum(x[0] for x in samples), sum(x[1] for x in samples)
        sps = sec / stp
        basis = (f"pooled timing of {len(samples)} completed Studio run(s) with this architecture on "
                 f"{device_name} ({stp:,} steps in {sec:.1f} s)")
    elif (ar["name"] == REFERENCE["arch"] and (ar.get("hidden") or 0) <= REFERENCE["max_hidden"]
          and device_name and REFERENCE["device_substr"] in device_name):
        sps = REFERENCE["sec_per_step"]
        basis = REFERENCE["source"]
    else:
        return {"measurable": False, "total_steps": total_steps, "steps_per_run": steps,
                "basis": "no measured timing for this architecture/device yet — run a short "
                         "pilot (e.g. 5 tasks) to calibrate"}
    secs = sps * total_steps
    return {"measurable": True, "seconds": round(secs, 1), "seconds_per_run": round(sps * steps, 1),
            "sec_per_step": sps, "total_steps": total_steps, "steps_per_run": steps,
            "gpu_hours": round(secs / 3600, 4), "basis": basis}
