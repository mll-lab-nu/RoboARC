#!/usr/bin/env python3
"""Harbor oracle / nop checks of RoboARC tasks with rendering off (also step 5 of tools/release/release_rebuild.py): a
docker host, no GPU. Python >= 3.11, standard library only; docker and harbor on the host.

usage: python3 scripts/roboarc/harbor_check.py --instances <dir> --references <dir> --run <dir> [options] base ...
       python3 scripts/roboarc/harbor_check.py --tasks <dir> --image <image> --run <dir> [options] base ...
  base          task bases under their current names, e.g. interlocked-claws-03-i01 (scripts/roboarc/release.json)
  --instances   one directory per base: a frozen MuJoCo instance (instance.json + instance_assets/)
  --references  one directory per base holding privileged.npz, the reference trajectory (the standard oracle streams
                its rows, scripts/roboarc/mk_tasks.py)
  --run         the run directory (created): dev.json, tasks/ (the dev task dirs, as mk_tasks.py --dev renders them),
                tasks_off/ (their rendering-off copies), jobs/ (Harbor), logs/, results.json, results.md
  --image       the image the tasks use (default ghcr.io/embodied-first-exam/roboarc:<images/roboarc/VERSION>); --build
                builds it first from this checkout (source/images/robotangle/Dockerfile.mujoco in source/, with
                --instances as its instances; images/roboarc/build.sh is the release build)
  --lanes       concurrent Harbor trials (default 3; a MuJoCo Session uses about one core)
  --agents, --modes   default oracle,nop and privileged,standard
  --harbor      the harbor command (default: harbor)
  --tasks       run on task dirs that exist already (<dir>/<bench>-<base>-<mode>, e.g. the repo's tasks/ as
                scripts/roboarc/mk_tasks.py renders them from release.json) instead of rendering dev task dirs; --image
                names the image they are built FROM, for the record
  --report      only (re)write results.md of an existing run directory
  --bench       the benchmark whose task dirs these are: <bench>-<base>-<mode> (default roboarc)
  --gpu-bases   bases whose physics needs the GPU (a GPU physics profile, e.g. the blanket's blanket_vbd_gpu): their
                rendering-off copies keep the device reservation (one GPU; rendering is still off)

Rendering off (docs/simulator/RENDER_BRIDGE.md, "Rendering off"): in tasks_off/ every compose service that held the GPU
(privileged agent and verifier, the standard sim) has RCB_RENDER=off instead, and no device reservation (a --gpu-bases
base keeps the reservation and NVIDIA_DRIVER_CAPABILITIES, and gets RCB_RENDER=off too). The tasks are
otherwise the generated ones: graders, videos (black), runtime bindings and frozen instances included. Each trial is
`harbor run -p tasks_off/<task> -a <agent> -o <run>/jobs --job-name <task>__<agent> -n 1 -k 1 --delete`; results.md has
the reward against the expected one (oracle 1, nop 0; a pass also needs infra_error 0 and the service log present), the
trial's wall time and the image id.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
GPU_BLOCK = ("    deploy:\n      resources:\n        reservations:\n          devices:\n            - driver: nvidia\n"
             "              count: 1\n              capabilities: [gpu]\n")
EXPECTED = {"oracle": 1.0, "nop": 0.0}
EVIDENCE = ("all_success", "live_success", "replay_success", "deterministic", "instance_exact", "runtime_binding_ok",
            "review_events", "missing_trajectory", "infra_error", "grader_error")   # reward fields shown in results.md
COST = {("standard", "oracle"): 2.5, ("privileged", "oracle"): 2.0}   # passes over the reference: the slowest start first
RUNTIME = ("import json, sys; sys.path.insert(0, '/tooling'); import rcb_topogym as rt; "
           "print(json.dumps(dict(runtime=rt.installed_runtime())))")


def sh(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def dev_json(a, bases: list[str]) -> pathlib.Path:
    """The image's identity and installed runtime (what each Session checks), and the bases' instances and references."""
    image_id = sh(["docker", "image", "inspect", a.image, "--format", "{{.Id}}"]).strip()
    out = sh(["docker", "run", "--rm", "--network", "none", "-v", f"{ROOT / 'scripts/roboarc/shared'}:/tooling:ro",
              a.image, "python", "-c", RUNTIME])
    runtime = json.loads(out.strip().splitlines()[-1])
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dev = dict(image=dict(reference=a.image, id=image_id, repo_digests=[]), **runtime, commit=commit or None,
               gpu_bases=[g for g in a.gpu_bases.split(",") if g],
               bases={b: dict(instance=str((a.instances / b).resolve()),
                              reference=str((a.references / b / "privileged.npz").resolve())) for b in bases})
    path = a.run / "dev.json"
    path.write_text(json.dumps(dev, indent=1) + "\n")
    return path


def render_off(src: pathlib.Path, dst: pathlib.Path, keep_gpu: bool = False) -> None:
    """A rendering-off copy of a task dir: RCB_RENDER=off instead of the GPU in every compose service that held one
    (keep_gpu: a GPU physics base keeps its one-GPU reservation, with RCB_RENDER=off beside it)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for compose in sorted(dst.glob("*/docker-compose.yaml")):
        s = compose.read_text()
        n, caps = s.count(GPU_BLOCK), s.count("      NVIDIA_DRIVER_CAPABILITIES: all\n")
        if n:   # the services that render (main / sim: NVIDIA_DRIVER_CAPABILITIES) and main's resource monitor (NVML)
            assert 1 <= caps <= n, compose
            if keep_gpu:
                s = s.replace("      NVIDIA_DRIVER_CAPABILITIES: all\n", '      NVIDIA_DRIVER_CAPABILITIES: all\n      RCB_RENDER: "off"\n')
                head = "# harbor_check.py: rendering off (RCB_RENDER=off); the GPU reservations stay (GPU physics)"
            else:
                s = s.replace(GPU_BLOCK, "").replace("      NVIDIA_DRIVER_CAPABILITIES: all\n", '      RCB_RENDER: "off"\n')
                head = "# harbor_check.py: rendering off (RCB_RENDER=off) and no GPU reservation (the monitor reads no GPU)"
            compose.write_text(head + "; the comments below are the generated task's.\n" + s)
    left = "".join(p.read_text() for p in dst.glob("*/docker-compose.yaml"))
    if keep_gpu:
        assert not re.search(r"(?m)^\s*runtime: nvidia", left) and left.count(GPU_BLOCK) >= 1, dst
    else:
        assert not re.search(r"(?m)^\s*(deploy:|- driver: nvidia|capabilities: \[gpu\]|runtime: nvidia)", left), dst


def trial(a, task: pathlib.Path, agent: str) -> dict:
    job = f"{task.name}__{agent}"
    if (a.run / "jobs" / job).exists():
        shutil.rmtree(a.run / "jobs" / job)
    cmd = [*a.harbor.split(), "run", "-p", str(task), "-a", agent, "-o", str(a.run / "jobs"), "--job-name", job,
           "-n", "1", "-k", "1", "--delete", "-y"]   # -y: no prompt (the verifier env, e.g. RCB_LIMITS_ENFORCE)
    t0 = time.time()
    with open(a.run / "logs" / f"{job}.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return dict(job=job, task=task.name, agent=agent, rc=rc, wall_s=round(time.time() - t0, 1), cmd=" ".join(cmd))


def reward_of(job_dir: pathlib.Path) -> dict:
    """The trial's reward (the verifier's reward.txt = reward.json `success`), all reward fields and the exception, from
    Harbor's trial result.json (one trial per job)."""
    found = {}
    for p in sorted(job_dir.glob("*/result.json")):
        r = json.loads(p.read_text())
        rewards = (r.get("verifier_result") or {}).get("rewards") or {}
        txt = p.parent / "verifier/reward.txt"
        found = dict(reward=rewards.get("success", rewards.get("reward")), rewards=rewards, trial=p.parent.name,
                     reward_txt=txt.read_text().strip() if txt.is_file() else None,
                     exception=(r.get("exception_info") or {}).get("exception_type"))
    return found


def end_states(job_dir: pathlib.Path, trial: str | None) -> list:
    """The oracle's end state hashes: both grader replays (privileged verdict.json) or the live episode and its fresh
    replay (standard final.json)."""
    t = job_dir / (trial or "-")
    if (t / "verifier/verdict.json").is_file():
        return [r.get("state_sha") for r in json.loads((t / "verifier/verdict.json").read_text())["evidence"].get("replays", [])]
    if (t / "artifacts/data/final.json").is_file():
        f = json.loads((t / "artifacts/data/final.json").read_text())
        return [f.get("live_sha"), f.get("replay_sha")]
    return []


def report(run: pathlib.Path) -> bool:
    """results.md from results.json: one row per task (oracle and nop), then one per trial with its evidence."""
    results = sorted(json.loads((run / "results.json").read_text()), key=lambda r: r["job"])
    dev = json.loads((run / "dev.json").read_text())
    by = {(r["task"], r["agent"]): r for r in results}
    gpu = dev.get("gpu_bases") or []
    lines = [f"Commit `{dev.get('commit')}`, image `{dev['image']['reference']}` = `{dev['image']['id']}`, rendering off "
             "(RCB_RENDER=off, no GPU" + (f"; GPU physics, one GPU reserved: {', '.join(gpu)}" if gpu else "") +
             "). Expected: oracle 1, nop 0.", "",
             "| task | oracle | nop | pass | oracle wall s | nop wall s | oracle end state |", "|---|---|---|---|---|---|---|"]
    for task in sorted({t for t, _ in by}):
        o, n = by.get((task, "oracle"), {}), by.get((task, "nop"), {})
        states = [x for x in end_states(run / "jobs" / f"{task}__oracle", o.get("trial")) if x]
        ok = all(r.get("pass") for r in (o, n) if r)
        lines.append(f"| {task} | {o.get('reward', '')} | {n.get('reward', '')} | {'yes' if ok else 'NO'} | "
                     f"{o.get('wall_s', 0):.0f} | {n.get('wall_s', 0):.0f} | "
                     f"{' / '.join(sorted(set(states))) + f' (x{len(states)})' if states else ''} |")
    lines += ["", "| trial | reward | expected | pass | wall s | evidence (reward.json) | Harbor rc | exception |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['job']} | {r.get('reward')} | {r['expected']:g} | {'yes' if r['pass'] else 'NO'} | {r['wall_s']:.0f} | "
              f"{' '.join(f'{k}={v}' for k, v in (r.get('rewards') or {}).items() if k in EVIDENCE)} | {r['rc']} | "
              f"{r.get('exception') or ''} |" for r in results]
    lines += ["", "Each trial: `" + (results[0]["cmd"] if results else "") + "` (task and agent vary)."]
    (run / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return bool(results) and all(r["pass"] for r in results)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", type=pathlib.Path)
    ap.add_argument("--references", type=pathlib.Path)
    ap.add_argument("--tasks", type=pathlib.Path, help="existing task dirs (see usage)")
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--image", default="ghcr.io/embodied-first-exam/roboarc:" + (ROOT / "images/roboarc/VERSION").read_text().strip())
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--lanes", type=int, default=3)
    ap.add_argument("--agents", default="oracle,nop")
    ap.add_argument("--modes", default="privileged,standard")
    ap.add_argument("--harbor", default="harbor")
    ap.add_argument("--report", action="store_true", help="only write results.md from the run's results.json")
    ap.add_argument("--bench", default="roboarc")
    ap.add_argument("--gpu-bases", default="", help="comma-separated bases whose physics needs the GPU (see usage)")
    ap.add_argument("bases", nargs="*")
    a = ap.parse_args(argv)
    a.run = a.run.resolve()
    if a.report:
        return 0 if report(a.run) else 1
    if not a.bases or (a.tasks is None and (a.instances is None or a.references is None)):
        ap.error("name the bases, and --tasks or --instances and --references")
    for d in ("tasks", "tasks_off", "jobs", "logs"):
        (a.run / d).mkdir(parents=True, exist_ok=True)
    if a.build:
        subprocess.run(["docker", "buildx", "build", "-f", "images/robotangle/Dockerfile.mujoco", "--build-context",
                        f"instances={a.instances.resolve()}", "-t", a.image, "."], cwd=ROOT / "source", check=True,
                       stdout=open(a.run / "logs/build.log", "w"), stderr=subprocess.STDOUT)
    if a.tasks is None:
        dev = dev_json(a, a.bases)
        subprocess.run([sys.executable, str(ROOT / "scripts/roboarc/mk_tasks.py"), "--dev", str(dev), "--out",
                        str(a.run / "tasks"), *a.bases], check=True)
        tasks = a.run / "tasks"
    else:       # the record report() reads: the checkout (with its local changes, if any) and the image
        git = lambda *x: subprocess.run(["git", "-C", str(ROOT), *x], capture_output=True, text=True).stdout.strip()  # noqa: E731
        dev = a.run / "dev.json"
        dev.write_text(json.dumps(dict(image=dict(reference=a.image, id=sh(["docker", "image", "inspect", a.image, "--format",
                                                                             "{{.Id}}"]).strip(), repo_digests=[]),
                                       commit=git("rev-parse", "HEAD") + (" + local changes" if git("status", "--porcelain") else ""),
                                       tasks=str(a.tasks.resolve()), gpu_bases=[g for g in a.gpu_bases.split(",") if g]),
                                  indent=1) + "\n")
        tasks = a.tasks.resolve()
    image_id = json.loads(dev.read_text())["image"]["id"]
    jobs = []
    for b in a.bases:
        ref = (tasks / f"{a.bench}-{b}-privileged/solution/trajectory.npz" if a.tasks is not None
               else a.references / b / "privileged.npz")
        size = zipfile.ZipFile(ref).getinfo("actions.npy").file_size if ref.is_file() else 1   # ~ rows; no reference: nop only
        for mode in a.modes.split(","):
            src = tasks / f"{a.bench}-{b}-{mode}"
            render_off(src, a.run / "tasks_off" / src.name, keep_gpu=b in a.gpu_bases.split(","))
            jobs += [(size * COST.get((mode, agent), 0.1), a.run / "tasks_off" / src.name, agent) for agent in a.agents.split(",")]
    results = []
    with cf.ThreadPoolExecutor(max_workers=a.lanes) as pool:
        futures = [pool.submit(trial, a, task, agent) for _, task, agent in sorted(jobs, key=lambda j: -j[0])]
        for f in cf.as_completed(futures):
            r = f.result()
            r.update(reward_of(a.run / "jobs" / r["job"]), expected=EXPECTED[r["agent"]], image_id=image_id)
            # a trial passes only with the expected reward AND a service that ran: a nop also scores 0 when its service
            # never started (reward.json infra_error=1, missing_log=1), so reward 0 alone proves nothing
            rw = r.get("rewards") or {}
            r["pass"] = (r.get("reward") is not None and float(r["reward"]) == r["expected"]
                         and int(rw.get("infra_error") or 0) == 0 and int(rw.get("missing_log") or 0) == 0)
            results.append(r)
            print(f"{r['job']}: reward {r.get('reward')} (expected {r['expected']}), {r['wall_s']} s", flush=True)
            (a.run / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    return 0 if report(a.run) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
