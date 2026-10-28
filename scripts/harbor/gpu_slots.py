#!/usr/bin/env python3
"""The host's trial slots: which GPU a trial runs on, and whether the host has the CPUs and memory its task declares
(scripts/harbor/run_agent_batch.sh with GPU_SCHED=1, docs/RUNNING.md "GPUs and lanes"). One ledger per host, shared by
every lane and every batch that use the same SLOTS_DIR.

    gpu_slots.py classify <task dir> [--deformable "<family> ..."] [--share 4] [--cpu-scale 1] [--mem-scale 1]
        -> "<units> <cpus> <memory GB> <family>"
    gpu_slots.py acquire --dir D --job J --runner-pid P --units U --cpus C --mem-gb M --gpus 0,1 [--share 4]
                         [--host-cpus N] [--host-mem-gb G] [--headroom-gb H]
        -> the GPU index on stdout ("" for a trial without a GPU), exit 0; exit 3: it does not fit now (why on stderr)
    gpu_slots.py release --dir D (--job J | --runner-pid P)
    gpu_slots.py status --dir D

The rule. A GPU has SHARE units (4). A trial of a task whose compose files reserve no GPU takes none; of a deformable
task (its [metadata] family in --deformable) all SHARE units, a GPU of its own; of any other task 1 unit, so up to SHARE
such trials share a GPU. A trial reserves the most CPUs and memory its task declares for the agent's or the verifier's
phase ([metadata.resources] min_cpus / verifier_min_cpus, min_memory_gb / verifier_min_memory_gb). A trial starts when
  - the reserved CPUs plus its own are at most the host's (--host-cpus, default the CPUs this process may use),
  - the reserved memory plus its own is at most the host's total less a headroom (default the larger of 4 GB and 10 %),
  - and a GPU has the units free: a 1-unit trial goes to the fullest GPU that still has a unit (packing, so that whole
    GPUs stay free for deformable trials), never to a GPU being drained; a deformable trial to an empty GPU.
When nothing else is held the CPU and memory checks are skipped: a task larger than the host must not wait forever. A
deformable trial that does not fit is recorded as waiting. While deformable trials wait, the empty GPUs are kept for
them (no 1-unit trial goes there), and for each waiting trial beyond the empty GPUs a busy GPU is drained: the one with
the fewest units in use takes no new 1-unit trial, so a deformable trial gets it once it empties. A waiting record
expires after WAIT_TTL_S unless a lane tries that trial again. Holdings of a runner whose process has ended (pid and
start time) are dropped on every call, so a killed lane frees its slots. Standard library only (Linux: fcntl, /proc).
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path

WAIT_TTL_S = 30 * 60
COMPOSE_FILES = ("docker-compose.yaml", "docker-compose.yml", "gpu.docker-compose.yaml", "gpu.docker-compose.yml")
DEFORMABLE = ("blanket_flatten blanket_fold fiber_bundling knot_untying orbits rope_threading slotted_post_ring "
              "wire_splicing")


# ---------- classify: what a task's trial needs ----------

def toml_section(text: str, name: str) -> str:
    """The lines of one [name] table of a task.toml (flat keys only; no TOML parser needed, any Python 3)."""
    m = re.search(r"^\[" + re.escape(name) + r"\][ \t]*\n(.*?)(?=^\[|\Z)", text, re.M | re.S)
    return m.group(1) if m else ""


def toml_value(section: str, key: str) -> str | None:
    m = re.search(r"^" + re.escape(key) + r'[ \t]*=[ \t]*(?:"([^"]*)"|([^#\n]*))', section, re.M)
    if not m:
        return None
    return m.group(1) if m.group(1) is not None else m.group(2).strip()


def uses_gpu(task: Path) -> bool:
    """Whether any compose file of the task (environment/, tests/) reserves an NVIDIA device."""
    for d in ("environment", "tests"):
        for f in COMPOSE_FILES:
            p = task / d / f
            if p.is_file() and re.search(r"^[ \t]*-?[ \t]*driver:[ \t]*nvidia\b", p.read_text(), re.M):
                return True
    return False


def classify(task: Path, deformable: set[str], share: int, cpu_scale: float = 1.0, mem_scale: float = 1.0):
    """-> (units, cpus, memory GB, family) of one trial of the task."""
    text = (task / "task.toml").read_text()
    family = toml_value(toml_section(text, "metadata"), "family") or ""
    res = toml_section(text, "metadata.resources")

    def num(key: str) -> float:
        v = toml_value(res, key)
        try:
            return float(v) if v not in (None, "") else 0.0
        except ValueError:
            return 0.0

    cpus = max(num("min_cpus"), num("verifier_min_cpus")) * cpu_scale
    mem = max(num("min_memory_gb"), num("verifier_min_memory_gb")) * mem_scale
    units = 0 if not uses_gpu(task) else (share if family in deformable else 1)
    return units, round(cpus, 3), round(mem, 3), family or "-"


# ---------- the ledger ----------

def proc_start(pid: int) -> str | None:
    """A process's start time (clock ticks since boot, /proc/<pid>/stat field 22): tells a live runner from a reused pid."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def alive(rec: dict) -> bool:
    start = proc_start(int(rec.get("pid", 0)))
    return start is not None and start == rec.get("start")


def host_mem_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024 / 1024
    raise SystemExit("no MemTotal in /proc/meminfo")


def host_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class Ledger:
    """state.json in the slots directory, read and written under an exclusive lock on its lock file."""

    def __init__(self, d: str):
        self.dir = Path(d)
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.dir / "state.json"

    def __enter__(self):
        self.lock = open(self.dir / "lock", "w")
        fcntl.flock(self.lock, fcntl.LOCK_EX)
        try:
            self.state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.state = {}
        for k in ("holds", "waiting"):
            self.state.setdefault(k, {})
        self.state.setdefault("draining", [])
        return self

    def save(self):
        tmp = self.path.with_name(f"state.json.{os.getpid()}")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True) + "\n")
        tmp.replace(self.path)

    def __exit__(self, *exc):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()

    def prune(self, now: float):
        """Drop the holdings and waiting records of runners that have ended, and waiting records past their TTL."""
        h, w = self.state["holds"], self.state["waiting"]
        for job in [j for j, r in h.items() if not alive(r)]:
            del h[job]
        for job in [j for j, r in w.items() if not alive(r) or now - r.get("t", 0) > WAIT_TTL_S]:
            del w[job]


def used_units(holds: dict, gpus: list[str]) -> dict[str, int]:
    u = {g: 0 for g in gpus}
    for r in holds.values():
        if r.get("gpu") is not None:
            u[r["gpu"]] = u.get(r["gpu"], 0) + int(r.get("units", 0))
    return u


def decide(state: dict, job: str, units: int, cpus: float, mem: float, gpus: list[str], share: int,
           cap_cpus: float, cap_mem: float):
    """-> (gpu or None, None) if the trial fits now, else (None, why). Updates state["draining"] (pure otherwise)."""
    holds, waiting = state["holds"], state["waiting"]
    use = used_units(holds, gpus)
    empty = sorted((g for g in gpus if use[g] == 0), key=int)
    # deformable trials waiting for a GPU of their own (this one included, if it is one): the empty GPUs are kept for
    # them, and for each one beyond those a busy GPU is drained (no new 1-unit trial), the one with the fewest units in use
    waits = set(waiting) | ({job} if units >= share else set())
    reserved = empty[:len(waits)]
    need = max(0, len(waits) - len(empty))
    drains = [g for g in state.get("draining", []) if g in gpus and use[g] > 0][:need]
    while len(drains) < need:
        rest = [g for g in gpus if use[g] > 0 and g not in drains]
        if not rest:
            break
        drains.append(min(rest, key=lambda g: (use[g], int(g))))
    state["draining"] = drains

    others = [r for j, r in holds.items() if j != job]
    if others:                                # nothing else held: no CPU or memory check (a trial larger than the host must not wait forever)
        r_cpus = sum(float(r.get("cpus", 0)) for r in others)
        r_mem = sum(float(r.get("mem_gb", 0)) for r in others)
        if r_cpus + cpus > cap_cpus + 1e-9:
            return None, f"CPUs: needs {cpus:g}, {r_cpus:g} of {cap_cpus:g} reserved"
        if r_mem + mem > cap_mem + 1e-9:
            return None, f"memory: needs {mem:g} GB, {r_mem:.1f} of {cap_mem:.1f} GB reserved"
    if units == 0:
        return None, None
    if units >= share:
        if not empty:
            return None, "GPU: needs a GPU of its own, none is empty (" + gpu_line(use, share, drains) + ")"
        return empty[0], None
    fits = [g for g in gpus if use[g] + units <= share and g not in drains and g not in reserved]
    if not fits:
        kept = f"; kept for {len(waits)} waiting deformable trial(s)" if waits else ""
        return None, f"GPU: needs {units} of {share} units, no GPU has them (" + gpu_line(use, share, drains) + kept + ")"
    gpu = min(fits, key=lambda g: (-use[g], int(g)))   # the fullest that still fits: packing
    return gpu, None


def gpu_line(use: dict, share: int, drains: list) -> str:
    return ", ".join(f"GPU {g}: {u}/{share}{' draining' if g in drains else ''}" for g, u in use.items())


def cmd_acquire(a) -> int:
    gpus = [g for g in a.gpus.split(",") if g != ""] if a.gpus else []
    if any(not g.isdigit() for g in gpus):
        sys.exit(f"--gpus {a.gpus}: GPU indices, comma-separated (e.g. 0,1,2)")
    if a.units and not gpus:
        sys.exit("a trial that needs a GPU, and no --gpus")
    total = a.host_mem_gb if a.host_mem_gb else host_mem_gb()
    headroom = a.headroom_gb if a.headroom_gb is not None else max(4.0, 0.1 * total)
    cap_cpus = a.host_cpus if a.host_cpus else host_cpus()
    now = time.time()
    start = proc_start(a.runner_pid)
    if start is None:
        sys.exit(f"--runner-pid {a.runner_pid}: no such process")
    with Ledger(a.dir) as L:
        L.prune(now)
        gpu, why = decide(L.state, a.job, a.units, a.cpus, a.mem_gb, gpus, a.share, cap_cpus, total - headroom)
        if why:
            if a.units >= a.share:
                L.state["waiting"][a.job] = {"pid": a.runner_pid, "start": start, "t": now}
            L.save()
            print(why, file=sys.stderr)
            return 3
        L.state["holds"][a.job] = {"pid": a.runner_pid, "start": start, "gpu": gpu, "units": a.units, "cpus": a.cpus,
                                   "mem_gb": a.mem_gb, "since": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        L.state["waiting"].pop(a.job, None)
        L.save()
    print(gpu if gpu is not None else "")
    return 0


def cmd_release(a) -> int:
    with Ledger(a.dir) as L:
        L.prune(time.time())
        if a.job:
            L.state["holds"].pop(a.job, None)
            L.state["waiting"].pop(a.job, None)
        else:
            for k in ("holds", "waiting"):
                for job in [j for j, r in L.state[k].items() if int(r.get("pid", 0)) == a.runner_pid]:
                    del L.state[k][job]
        L.save()
    return 0


def cmd_status(a) -> int:
    with Ledger(a.dir) as L:
        L.prune(time.time())
        L.save()
        s = L.state
    for job, r in sorted(s["holds"].items(), key=lambda x: (str(x[1].get("gpu")), x[0])):
        print(f"held   {job}: GPU {r.get('gpu') if r.get('gpu') is not None else '-'}, {r.get('units')} unit(s), "
              f"{r.get('cpus')} CPUs, {r.get('mem_gb')} GB, since {r.get('since')} (runner {r.get('pid')})")
    for job, r in sorted(s["waiting"].items()):
        print(f"waits  {job} (deformable, runner {r.get('pid')})")
    if s["draining"]:
        print("draining GPU " + ", ".join(s["draining"]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("task")
    c.add_argument("--deformable", default=DEFORMABLE)
    c.add_argument("--share", type=int, default=4)
    c.add_argument("--cpu-scale", type=float, default=1.0)
    c.add_argument("--mem-scale", type=float, default=1.0)
    q = sub.add_parser("acquire")
    q.add_argument("--dir", required=True)
    q.add_argument("--job", required=True)
    q.add_argument("--runner-pid", type=int, required=True)
    q.add_argument("--units", type=int, required=True)
    q.add_argument("--cpus", type=float, required=True)
    q.add_argument("--mem-gb", type=float, required=True)
    q.add_argument("--gpus", default="")
    q.add_argument("--share", type=int, default=4)
    q.add_argument("--host-cpus", type=float, default=0)
    q.add_argument("--host-mem-gb", type=float, default=0)
    q.add_argument("--headroom-gb", type=float, default=None)
    r = sub.add_parser("release")
    r.add_argument("--dir", required=True)
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--job")
    g.add_argument("--runner-pid", type=int)
    s = sub.add_parser("status")
    s.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "classify":
        t = Path(a.task)
        if not (t / "task.toml").is_file():
            sys.exit(f"{a.task}: not a task directory (no task.toml)")
        if a.share < 1 or a.cpu_scale <= 0 or a.mem_scale <= 0:
            sys.exit("--share must be at least 1 and the scales positive")
        units, cpus, mem, fam = classify(t, set(a.deformable.split()), a.share, a.cpu_scale, a.mem_scale)
        print(f"{units} {cpus:g} {mem:g} {fam}")
        return 0
    if a.cmd == "acquire":
        if a.share < 1 or a.units < 0 or a.units > a.share:
            sys.exit("--units must be between 0 and --share")
        return cmd_acquire(a)
    return cmd_release(a) if a.cmd == "release" else cmd_status(a)


if __name__ == "__main__":
    sys.exit(main())
