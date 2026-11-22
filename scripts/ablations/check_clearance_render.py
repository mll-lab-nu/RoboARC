#!/usr/bin/env python3
"""The GPU part of the E4 checks (ablation experiments): the relaxed scenes as the cameras see them. CPU checks cannot
show it: the oracle and nop runs of the variants pass with rendering off too, and the visuals come from the relaxed
source in each variant's image, not from the frozen MuJoCo scene that scripts/ablations/check_clr{2,4}x.py measure.

    python3 scripts/ablations/check_clearance_render.py [--tasks tasks] [--out jobs/ablations/checks/e4_render] [--gpu 0]
                                                      [variant ...]     # default: every E4 privileged variant

For each privileged variant <base>-clr<k>x-privileged and its base task <base>-privileged: the task's own agent image
(docker build of environment/, FROM the pinned release image), then in it, on one GPU with Nyx rendering on, a Session on
the staged instance replays the task's reference solution and saves the camera grid at the start, at the middle and at
the end of the reference (rcb_topogym Session.render_review_frames). Written to <out>/<variant>/: step_<n>.png of the
variant and base_step_<n>.png of the base task, and render.json (frames, each frame's mean intensity, the mean absolute
difference to the base task's frame at the same step, the replay's status). Fails (exit 1) when a variant or base frame
cannot be rendered or is black (mean intensity < 2 of 255), or when the variant's reference does not succeed in that
replay. The pictures are for a person to look at: the relaxed bore or the wider gap between the pieces must be what the
cameras show (<out>/report.md lists them). Needs docker, the NVIDIA Container Toolkit, an RTX-class GPU (Nyx) and the
pinned images (images/pull_all.sh). Standard library only on the host."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import cond_clearance as cc  # noqa: E402

JOB = r'''
import json, os, sys, numpy as np
sys.path.insert(0, "/opt/tools")
from rcb_topogym import Session, load_trajectory
task, out, prefix, owner = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = Session(task, instance="/app/instance.json")
acts, ticks = load_trajectory("/solution/trajectory.npz", task, s.action_dim, trusted=True, layout=s.action_layout)
steps = [1, max(1, len(acts) // 2), len(acts)]
files = s.render_review_frames(acts, [{"step": k} for k in steps], out, physics_steps=ticks, offsets=(0,))
from PIL import Image
frames = {}
for f in files:
    p = f.rsplit("/", 1)[0] + "/" + prefix + f.rsplit("/", 1)[1]
    os.replace(f, p)
    frames[p.rsplit("/", 1)[1]] = float(np.asarray(Image.open(p)).mean())
diff = {}
if prefix:      # the base task's frames, rendered after the variant's into the same directory: their difference
    for name in frames:
        other = out + "/" + name[len(prefix):]
        if os.path.isfile(other):
            a = np.asarray(Image.open(out + "/" + name), dtype=np.float64)
            b = np.asarray(Image.open(other), dtype=np.float64)
            diff[name[len(prefix):]] = float(np.abs(a - b).mean()) if a.shape == b.shape else -1.0
st = s.replay(acts, physics_steps=ticks, grading=True)
uid, gid = map(int, owner.split(":"))       # the frames belong to the caller, not to the container's root
for name in os.listdir(out):
    os.chown(os.path.join(out, name), uid, gid)
print("RENDER " + json.dumps(dict(frames=frames, diff=diff, success=bool(st["success"]), episode_end=st.get("episode_end"),
                                  cameras=list(s.VIDEO_CAMERAS))), flush=True)
'''


def sh(cmd: list[str], log: pathlib.Path) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True)
    with log.open("a") as f:
        f.write("$ " + " ".join(cmd) + "\n" + p.stdout + p.stderr + "\n")
    return p


def render(task_dir: pathlib.Path, family: str, out: pathlib.Path, prefix: str, gpu: str, log: pathlib.Path) -> dict:
    tag = f"rcb-e4-render/{task_dir.name}:local"
    if sh(["docker", "build", "-q", "-t", tag, str(task_dir / "environment")], log).returncode:
        return dict(error=f"docker build of {task_dir.name}/environment failed (log {log})")
    (out / "job.py").write_text(JOB)
    p = sh(["docker", "run", "--rm", "--network", "none", "--gpus", f"device={gpu}", "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
            "--shm-size", "2g", "-e", "RCB_SESSION_LOG=", "--user", "0:0",
            "-v", f"{task_dir / 'solution'}:/solution:ro", "-v", f"{out}:/out", tag,
            "python", "/out/job.py", family, "/out", prefix, f"{os.getuid()}:{os.getgid()}"], log)
    sh(["docker", "image", "rm", tag], log)
    lines = [x for x in p.stdout.splitlines() if x.startswith("RENDER ")]
    if p.returncode or not lines:
        return dict(error=f"rendering {task_dir.name} failed (log {log})")
    return json.loads(lines[-1][len("RENDER "):])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", type=pathlib.Path, default=REPO / "tasks")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "jobs" / "ablations" / "checks" / "e4_render")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("variants", nargs="*")
    a = ap.parse_args(argv)
    names = a.variants or [f"{b}-clr{k}x-privileged" for b in cc.BASES for k in cc.FACTORS]
    a.out.mkdir(parents=True, exist_ok=True)
    rows, bad = [], 0
    for name in names:
        base, _ = name.split("-clr")
        r = cc.RELAXATIONS[base]
        out = (a.out / name).resolve()
        out.mkdir(parents=True, exist_ok=True)
        log = out / "render.log"
        log.write_text("")
        got = render((a.tasks / name).resolve(), r.task, out, "", a.gpu, log)
        ref = render((a.tasks / f"{base}-privileged").resolve(), r.task, out, "base_", a.gpu, log)
        problems = [x["error"] for x in (got, ref) if "error" in x]
        if not problems:
            from_base = {k[len("base_"):]: v for k, v in ref["frames"].items()}
            problems += [f"{k}: black frame" for k, v in {**got["frames"], **ref["frames"]}.items() if v < 2.0]
            if sorted(got["frames"]) != sorted(from_base):
                problems.append("the variant and its base task rendered other steps")
            if not got["success"]:
                problems.append(f"the reference does not succeed in the rendered replay ({got.get('episode_end')})")
            got["diff_to_base"] = ref.get("diff") or {}
        rec = dict(variant=name, base=f"{base}-privileged", parameter=r.parameter, variant_render=got, base_render=ref,
                   problems=problems, ok=not problems)
        (out / "render.json").write_text(json.dumps(rec, indent=1) + "\n")
        rows.append(rec)
        bad += bool(problems)
        print(("ok " if not problems else "FAIL ") + name + ("" if not problems else ": " + "; ".join(problems)), flush=True)
    lines = ["# E4 render check", "", "| variant | ok | frames (variant / base) | mean abs difference to the base | problems |",
             "|---|---|---|---|---|"]
    for rec in rows:
        frames = sorted((rec["variant_render"].get("frames") or {}))
        diff = rec["variant_render"].get("diff_to_base") or {}
        lines.append(f"| {rec['variant']} | {'yes' if rec['ok'] else 'NO'} | "
                     + ", ".join(f"[{f}]({rec['variant']}/{f}) / [base]({rec['variant']}/base_{f})" for f in frames)
                     + f" | {', '.join(f'{v:.2f}' for v in diff.values())} | {'; '.join(rec['problems'])} |")
    (a.out / "report.md").write_text("\n".join(lines) + "\n")
    (a.out / "render.json").write_text(json.dumps(rows, indent=1) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
