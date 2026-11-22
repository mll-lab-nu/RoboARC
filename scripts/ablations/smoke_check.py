#!/usr/bin/env python3
"""The smoke's check, about ten minutes after bash-runners/ablations/smoke.sh started (docs/ABLATIONS.md,
step 2), and again when its trials have ended:

    python3 scripts/ablations/smoke_check.py ablation-smoke-e1 [--jobs jobs/smoke]

Per trial: the runner's settings line, the model calls through the usage gateway and their HTTP statuses, requests the
gateway refused, tokens in / out, the reasoning effort Harbor was given (the job's config.json) and, with Codex, the one
its session ran with, and whether the agent's transcript already uses the condition's new interface (meta["objects"],
r.geometry(), the deformable state; informational: an agent may not have looked yet). Ends with OK or with what to
send us. Standard library only."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE)]
import make_variants as mv  # noqa: E402

INTERFACE = {   # what the agent's own code would contain once it uses the condition's addition
    "q1c10": r"""meta\[\s*["']objects["']\s*\]""",
    "q1c0g": r"\.geometry\(\s*\)|/geometry",
    "q1c1g": r"""\.geometry\(\s*\)|/geometry|meta\[\s*["']objects["']\s*\]""",
}


def interface(token: str) -> str | None:
    if token in INTERFACE:
        return INTERFACE[token]
    try:   # a condition module may name its own (E3: the deformable state's field)
        return getattr(mv.module(mv.REGISTRY[token]), "SMOKE_PATTERN", None)
    except SystemExit:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batch")
    ap.add_argument("--jobs", type=pathlib.Path, default=REPO / "jobs" / "smoke")
    a = ap.parse_args(argv)
    log = a.jobs / f"{a.batch}.log"
    problems = []
    runner = [l.strip() for l in log.read_text().splitlines() if re.search(r"runner \d+: ", l)] if log.is_file() else []
    print("runner:", runner[0].split("] ", 1)[-1] if runner else "(no runner line yet)")
    if runner and "usage gateway off" in runner[0]:
        problems.append("the usage gateway is off")
    if log.is_file() and "STOP lane" in log.read_text():
        problems.append("a lane was stopped (STOP lane in the batch log)")
    jobs = sorted(p for p in a.jobs.glob(f"{a.batch}-*") if p.is_dir())
    if not jobs:
        problems.append(f"no job folders {a.jobs}/{a.batch}-* yet")
    for job in jobs:
        token = next((t for t in mv.REGISTRY if f"-{t}" in job.name), None)
        cfg = json.loads((job / "config.json").read_text()) if (job / "config.json").is_file() else {}
        effort = re.findall(r'"reasoning_effort":\s*"?([a-z]+)', json.dumps(cfg))
        for trial in sorted(p for p in job.iterdir() if p.is_dir() and "__" in p.name) or [job]:
            recs = []
            for f in glob.glob(str(trial / "gateway" / "usage.jsonl")):
                recs += [json.loads(l) for l in open(f) if l.strip()]
            calls = [r for r in recs if not r.get("refused")]
            status = collections.Counter(r.get("status") for r in calls)
            refused = [(r.get("method"), r.get("path"), r.get("refused")) for r in recs if r.get("refused")]
            tok = [sum((r.get("usage") or {}).get(k, 0) or 0 for r in calls) for k in ("input", "output")]
            seen = set()
            for f in glob.glob(str(trial / "agent" / "sessions" / "*" / "*" / "*" / "rollout-*.jsonl")):
                for l in open(f):
                    if '"turn_context"' in l:
                        seen.add(json.loads(l)["payload"].get("effort"))
            text = ""
            for f in glob.glob(str(trial / "agent" / "**" / "*"), recursive=True):
                if f.endswith((".jsonl", ".json", ".txt", ".py", ".log")) and pathlib.Path(f).is_file():
                    try:
                        text += pathlib.Path(f).read_text(errors="replace")[:5_000_000]
                    except OSError:
                        pass
            pat = interface(token) if token else None
            used = "-" if pat is None else ("yes" if re.search(pat, text) else "not yet")
            print(f"{trial.name if trial != job else job.name}: condition {token}; model calls {len(calls)}, statuses "
                  f"{dict(status)}; refused {refused or 'none'}; tokens in/out {tok[0]}/{tok[1]}; effort given "
                  f"{sorted(set(effort)) or '-'}{', Codex ran with ' + str(sorted(x for x in seen if x)) if seen else ''}; "
                  f"uses the condition's interface: {used}")
            if trial == job:
                continue
            if not calls:
                problems.append(f"{trial.name}: no model call through the gateway yet")
            bad = {s: n for s, n in status.items() if s not in (200, 201)}
            if bad and sum(bad.values()) > max(2, len(calls) // 10):
                problems.append(f"{trial.name}: many model calls answered {bad}")
            if refused:
                problems.append(f"{trial.name}: the gateway refused {refused[:3]}")
            if effort and set(effort) != {"high"}:
                problems.append(f"{trial.name}: effort {sorted(set(effort))}, not high")
            if seen - {None, "high"}:
                problems.append(f"{trial.name}: Codex ran with effort {sorted(x for x in seen if x)}")
    if problems:
        print("\nNOT OK:\n  " + "\n  ".join(problems))
        print(f"Stop the smoke (docs/ABLATIONS.md, step 2) and send us this output and {log}.")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
