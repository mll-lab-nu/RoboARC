#!/usr/bin/env bash
# Shared-source rules (PROTOCOL §2.1, D9: copies of shared sources are byte-identical). Two parts:
# 1. Within the benchmark, every copy of a shared file is byte-identical (the generator, scripts/roboarc/mk_tasks.py,
#    copies them from scripts/roboarc/shared/ unchanged):
#    - the tooling modules rcb_topogym.py, render_bridge.py and nyx_direct.py wherever they sit (privileged environment/
#      and tests/, standard environment/sim/);
#    - the standard agent side environment/rcb_client.py + rcb1.py and the sim service environment/sim/ (server.py,
#      finalize.sh, run.sh, rcb1.py, rcb_limited.py);
#    - the graders, tests/grade.py (one version per mode).
#    A change to one of them is rolled out to every task at once. The IKEA MuJoCo tasks (roboarc-ikea-*) are a group
#    of their own (ikea/scripts/ikea_mujoco/: rcb_ikea.py, ikea_nyx.py, ikea_rig.py, their nyx_direct.py, graders and
#    sim service), and their standard layer is the live-grading pair scripts/eai_standard/v2_4_live/.
# 2. The eai-standard/2.4 layer (every task whose metadata.protocol is eai-standard/2.4): the agent's client
#    environment/rcb_client.py + rcb1.py and the service's environment/sim/rcb_limited.py + rcb1.py + run.sh are
#    byte-identical to their source (scripts/eai_standard/v2_4/rcb_client.py and rcb_limited.py, scripts/eai_standard/
#    rcb1.py and run.sh; docs/STANDARD_MODE.md), and the sidecar is set up for crash recovery: `command: sh /srv/run.sh`
#    (run.sh exits non-zero when the service crashes) and `restart: on-failure:N` with N >= 3 (PROTOCOL §4.1, §4.7).
#    A standard task that declares another protocol fails: this repository ships the 2.4 layer only.
# Usage: check-tooling-identity.sh [tasks dir]   (repo-level; run_all.sh calls it once)
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "${1:-$root/tasks}" "$root/scripts/eai_standard" <<'PY'
import collections, hashlib, os, re, sys, tomllib
from pathlib import Path
tasks, src = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(src))); import task_conventions as tc   # the mode registry (§2.16)
PREFIX = "roboarc-"
MODULES = {"rcb_topogym.py", "render_bridge.py", "nyx_direct.py",
           "rcb_ikea.py", "ikea_nyx.py", "ikea_rig.py", "render_video.py"}   # the last four: IKEA's
IKEA = "roboarc-ikea-"   # IKEA MuJoCo: its own group of copies, and its live-grading 2.4 pair (v2_4_live/)
SHARED = {"environment/rcb_client.py", "environment/rcb1.py", "environment/sim/server.py", "environment/sim/finalize.sh",
          "environment/sim/run.sh", "environment/sim/rcb1.py", "environment/sim/rcb_limited.py", "tests/grade.py"}
LAYER = {"environment/rcb_client.py": "v2_4/rcb_client.py", "environment/rcb1.py": "rcb1.py",
         "environment/sim/rcb_limited.py": "v2_4/rcb_limited.py", "environment/sim/rcb1.py": "rcb1.py",
         "environment/sim/run.sh": "run.sh"}
LIVE = {**LAYER, "environment/rcb_client.py": "v2_4_live/rcb_client.py",
        "environment/sim/rcb_limited.py": "v2_4_live/rcb_limited.py"}   # IKEA: graded by the live episode
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]  # noqa: E731
canon = {f: sha(os.path.join(src, f)) for f in set(LAYER.values()) | set(LIVE.values()) if os.path.isfile(os.path.join(src, f))}
seen = collections.defaultdict(dict)   # role -> {sha: first copy}
n_tasks = 0; bad = []; n_layer = 0
bad += [f"scripts/eai_standard/{f} missing (the eai-standard/2.4 layer)" for f in sorted((set(LAYER.values()) | set(LIVE.values())) - set(canon))]


def sim_service(compose: str) -> str:
    """The `sim:` service block of a compose file (indentation-based, no YAML dependency)."""
    out, inside = [], False
    for line in open(compose).read().splitlines():
        ind = len(line) - len(line.lstrip())
        if line.strip() and ind <= 2 and inside:
            break
        if re.match(r"^  sim:\s*$", line):
            inside = True; continue
        if inside:
            out.append(line)
    return "\n".join(out)


for t in sorted(os.listdir(tasks)):
    tt = os.path.join(tasks, t, "task.toml")
    if not os.path.isfile(tt) or not t.startswith(PREFIX):
        continue
    n_tasks += 1
    meta = tomllib.load(open(tt, "rb")).get("metadata", {})
    proto = str(meta.get("protocol", ""))
    mode = tc.mode_of(Path(t))
    if tc.MODE_REGISTRY[mode]["sidecar"] and proto != "eai-standard/2.4":
        bad.append(f"{t}: protocol {proto!r}; this repository ships the eai-standard/2.4 layer only")
    group = "ikea " if t.startswith(IKEA) else ""
    if proto == "eai-standard/2.4":
        n_layer += 1
        for rel, f in sorted((LIVE if group else LAYER).items()):
            p = os.path.join(tasks, t, rel)
            if not os.path.isfile(p):
                bad.append(f"{t}: {rel} missing (eai-standard/2.4 layer, scripts/eai_standard/{f})")
            elif sha(p) != canon.get(f):
                bad.append(f"{t}: {rel} differs from scripts/eai_standard/{f} (regenerate the task)")
        cf = os.path.join(tasks, t, "environment/docker-compose.yaml")
        blk = sim_service(cf) if os.path.isfile(cf) else ""
        m = re.search(r"^\s*restart:\s*[\"']?on-failure:(\d+)", blk, re.M)
        if not m or int(m.group(1)) < 3:
            bad.append(f"{t}: the sim service needs `restart: on-failure:N` with N >= 3 (crash recovery)")
        if not re.search(r"^\s*command:\s*[\"']?sh /srv/run\.sh[\"']?\s*$", blk, re.M):
            bad.append(f"{t}: the sim service must run `sh /srv/run.sh` (it exits non-zero when the service crashes)")
    for dp, dns, fns in os.walk(os.path.join(tasks, t)):
        dns[:] = [d for d in dns if d not in ("results", "__pycache__")]
        for f in fns:
            rel = os.path.relpath(os.path.join(dp, f), os.path.join(tasks, t)).replace(os.sep, "/")
            role = f if f in MODULES else f"{mode} {rel}" if rel in SHARED else None
            role = group + role if role else None
            if role:
                seen[role].setdefault(sha(os.path.join(dp, f)), f"{t}/{rel}")
bad += [f"{len(v)} versions of {role}: " + ", ".join(f"{h} ({p})" for h, p in sorted(v.items()))
        for role, v in sorted(seen.items()) if len(v) > 1]
if bad:
    print("FAIL shared sources (byte-identical copies; the eai-standard/2.4 layer as in scripts/eai_standard/):")
    for b in bad[:60]:
        print("  -", b)
    if len(bad) > 60:
        print(f"  ... {len(bad) - 60} more")
    sys.exit(1)
print(f"ok shared sources: {n_tasks} task dirs, {len(seen)} shared files; the eai-standard/2.4 layer in {n_layer} task dirs")
PY
