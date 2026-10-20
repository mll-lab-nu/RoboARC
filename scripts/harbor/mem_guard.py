#!/usr/bin/env python3
"""Host memory red line (opt-in): keep at least --min-gb (default 2) GiB available. While MemAvailable is below it, the
trial container using the most memory is killed (docker kill), one every --cooldown s, so the machine never thrashes
into the kernel OOM killer. Candidates: running containers whose NAME matches --match (default: RoboARC trial
containers, whose compose project is the task name `roboarc-...`), except --protect; no other container is ever
touched unless you widen --match yourself. Every kill (with the top containers at that moment) goes to --log, plus a
heartbeat line every --heartbeat s. A runner-side tool: scripts/harbor/run_agent_batch.sh starts one with MEM_GUARD=1.
  python3 scripts/harbor/mem_guard.py [--min-gb 2] [--match REGEX] [--log FILE] [--pidfile FILE]"""
import argparse, os, re, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--min-gb", type=float, default=2.0)
ap.add_argument("--match", default=r"^roboarc-")
ap.add_argument("--protect", default="rcb-log-mirror")
ap.add_argument("--interval", type=float, default=1.0)
ap.add_argument("--cooldown", type=float, default=15.0)
ap.add_argument("--heartbeat", type=float, default=600.0)
ap.add_argument("--log", default=os.path.join(ROOT, "jobs/agents/mem_guard.log"))
ap.add_argument("--pidfile", default=os.path.join(ROOT, "jobs/agents/.mem_guard.pid"))
a = ap.parse_args()
os.makedirs(os.path.dirname(a.log), exist_ok=True)
with open(a.pidfile, "w") as f:
    f.write(str(os.getpid()))
match, protect, floor = re.compile(a.match), set(a.protect.split(",")), int(a.min_gb * 2**30)


def log(msg: str) -> None:
    with open(a.log, "a") as f:
        f.write(f"{time.strftime('%F %T')} {msg}\n")


def available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def containers() -> list[tuple[int, str, str]]:
    """(bytes in use, name, image) of every running container, largest first."""
    out = subprocess.run(["docker", "ps", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}"],
                         capture_output=True, text=True, timeout=30).stdout
    rows = []
    for line in out.splitlines():
        cid, name, image = line.split("\t")
        try:
            with open(f"/sys/fs/cgroup/system.slice/docker-{cid}.scope/memory.current") as f:
                rows.append((int(f.read()), name, image))
        except OSError:
            pass
    return sorted(rows, reverse=True)


gb = lambda b: f"{b / 2**30:.1f} GiB"  # noqa: E731
top = lambda rows: ", ".join(f"{n} {gb(m)}" for m, n, _ in rows[:4]) or "none"  # noqa: E731
log(f"start pid {os.getpid()}: red line {a.min_gb} GiB available, candidates /{a.match}/ except {sorted(protect)}")
last_beat = last_kill = 0.0
while True:
    now, avail = time.time(), available()
    try:
        if avail < floor and now - last_kill >= a.cooldown:
            last_kill = now
            rows = containers()
            cands = [r for r in rows if r[1] not in protect and match.search(r[1])]
            if cands:
                m, name, image = cands[0]
                rc = subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=60).returncode
                log(f"KILL {name} ({image}, {gb(m)}): available {gb(avail)} < {a.min_gb} GiB; docker kill rc={rc}; containers: {top(rows)}")
            else:
                log(f"LOW available {gb(avail)} < {a.min_gb} GiB and no candidate container; containers: {top(rows)}")
        if now - last_beat >= a.heartbeat:
            last_beat = now
            log(f"ok available {gb(avail)}; containers: {top(containers())}")
    except Exception as e:  # a docker hiccup must not end the guard
        log(f"error {type(e).__name__}: {e}")
    time.sleep(a.interval)
