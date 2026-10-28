#!/usr/bin/env python3
"""The machine a batch runs on (docs/RUNNING.md "Results"): the CPU model, the CPUs this process may use (`nproc`), the
memory, every GPU nvidia-smi lists (index, name, driver, memory) and the docker and harbor versions. The batch runner
logs it when it starts and writes it into every job folder as host.json; write_task_results.py copies it into each
trial's config.json as `host`. A tool that is missing or fails leaves its field null (no GPU: an empty list), so it
never stops a run.

    host_info.py            # one JSON object on one line
    host_info.py --lines    # host:, GPUs: and software: lines, as MANIFEST.md writes them (+ each GPU's memory)
"""
import json, os, re, subprocess, sys


def run(*cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else None


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            m = re.search(r"^model name\s*:\s*(.+)$", f.read(), re.M)
    except OSError:
        return None
    return m.group(1).strip() if m else None


def memory_gb():
    """MemTotal in whole GiB, as `free -g` prints it."""
    try:
        with open("/proc/meminfo") as f:
            m = re.search(r"^MemTotal:\s*(\d+) kB", f.read(), re.M)
    except OSError:
        return None
    return int(m.group(1)) // (1 << 20) if m else None


def gpus():
    out = run("nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits") or ""
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            rows.append({"index": p[0], "name": p[1], "driver": p[2], "memory_mib": int(p[3]) if p[3].isdigit() else None})
    return rows


def info():
    return {"cpu_model": cpu_model(), "cpus": len(os.sched_getaffinity(0)), "memory_gb": memory_gb(), "gpus": gpus(),
            "docker": run("docker", "--version"), "harbor": run("harbor", "--version")}


def lines(h):
    gpu = "; ".join(f"{g['index']}, {g['name']}, {g['driver']}, {g['memory_mib']} MiB" for g in h["gpus"])
    return [f"host:       {h['cpu_model'] or 'unknown CPU'}, {h['cpus']} CPUs, {h['memory_gb'] if h['memory_gb'] is not None else '?'} GB RAM",
            f"GPUs:       {gpu or 'none (nvidia-smi lists no GPU)'}",
            f"software:   {h['docker'] or 'docker: not found'}; harbor {h['harbor'] or 'not found'}"]


if __name__ == "__main__":
    h = info()
    print("\n".join(lines(h)) if sys.argv[1:] == ["--lines"] else json.dumps(h))
