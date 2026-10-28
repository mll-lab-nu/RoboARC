#!/usr/bin/env python3
"""A base's appearance seeds keep its physics: the end-state identity evidence of RoboARC 1.2.0 (CHANGELOG.md,
"End-state identity"), for one base's seed 0 and seeds S made again (a release whose source changes the looks or a
frozen scene).

In the release image (or the image with a preview runtime mounted at /opt/topogym), the shared tooling at --tooling:
  python check_seed_physics.py run <base dir> --task <family> --reference <privileged.npz> [--seeds 1,2] [--fresh 2]
      [--tooling /tooling] [--out <json>]
    <base dir>  one base in the ROBOARC_INSTANCES layout: instance.json + instance_assets/ (seed 0), seeds/<S>/ (seed S)
  For seed 0 and each seed S, on a private copy of its instance (no runtime binding beside it): the reference (seed 0's
  privileged.npz; for seed S the same file re-stamped with seed S's birth, restamp.py; a reference without a birth stamp
  as it is) replayed strictly as the graders replay it, in N fresh processes (state_parts.py replay). Checks:
    birth     the reference was recorded on seed 0's birth (a re-made seed 0 must keep the birth of its reference)
    derived   seed S's derived_from names this seed 0 (its canonical sha256): not a seed of an earlier seed 0
    exact     every replay's instance is exact (Session strict: its state and assets reproduce)
    success   every replay of every seed succeeds; deterministic: the replays of a seed end on one state
    identity  each seed's end equals seed 0's in every part (world, robot, targets, coverage, state, ...) and in the
              task part but its instance id (state_parts.py: parts, task_but_instance_id)
  One line `SEED {json}` per seed, then `SEEDPHYS {json}`; exit 0 when every check holds, 1 when one fails, 2 when a
  replay could not run. For example, on the host (one base, CPU; the MuJoCo tasks' SIMD masking):

    docker run --rm --network none -e OMP_NUM_THREADS=1 -e CUDA_VISIBLE_DEVICES= -e RCB_RENDER=off \\
      -e NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR \\
      -e OPENBLAS_CORETYPE=Haswell -e HOME=/tmp -u $(id -u):$(id -g) -v $PWD:/repo:ro -v <instances>:/inst:ro \\
      -v <work>:/work -w /tmp ghcr.io/mll-lab-nu/roboarc:<v> python /repo/tools/release/check_seed_physics.py run \\
      /inst/cast-keyhole-03-i01 --task cast_keyhole --tooling /repo/scripts/roboarc/shared \\
      --reference /repo/scripts/roboarc/oracles/cast-keyhole-03-i01/privileged.npz --out /work/cast-keyhole-03-i01.json

  A blanket base (GPU physics) needs one GPU: --gpus device=<i> -e NVIDIA_DRIVER_CAPABILITIES=all instead of
  CUDA_VISIBLE_DEVICES=.

On the host (standard library only):
  python3 check_seed_physics.py harbor-list [--changed <base or family, ...; @file: one base per line>] [--seeds 0,1,2]
      [--tree <render tree>] [--image <image>] [--out <dir>] [--json]
    the Harbor oracle / nop trials of a release that changes the seeds: one base per family (the first, by name, with a
    reference) and every base --changed names, each at every seed, both modes, oracle and nop (nop only for a base
    without a reference, release.json oracle "none"), on scripts/task_seed.py materialize copies of the render tree's
    task dirs, run by scripts/roboarc/harbor_check.py --tasks (rendering off; blanket bases keep their GPU): the
    commands, one harbor_check.py run per seed and agent set, or (--json) one row per trial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def stamp(npz: pathlib.Path) -> str | None:
    """The birth a reference was recorded on (instance_sha.npy), None without one (release_rebuild.npz_text)."""
    with zipfile.ZipFile(npz) as z:
        if "instance_sha.npy" not in z.namelist():
            return None
        raw = z.read("instance_sha.npy")
    size = 4 if raw[6] >= 2 else 2
    return raw[8 + size + int.from_bytes(raw[8:8 + size], "little"):].decode("utf-32-le").rstrip("\0")


def replay(task: str, instance: pathlib.Path, traj: pathlib.Path, fresh: int, tooling: str) -> dict:
    """state_parts.py replay: the trajectory replayed strictly in `fresh` fresh processes; its PARTS record."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (tooling, os.environ.get("PYTHONPATH")) if p))
    p = subprocess.run([sys.executable, str(HERE / "state_parts.py"), "replay", "--task", task, "--instance",
                        str(instance), "--traj", str(traj), "--fresh", str(fresh)], capture_output=True, text=True,
                       env=env, cwd=instance.parent)
    line = [x for x in p.stdout.splitlines() if x.startswith("PARTS ")]
    if p.returncode or not line:
        sys.stderr.write(f"{instance}: the replay did not run (rc {p.returncode}):\n{(p.stdout + p.stderr)[-3000:]}\n")
        raise SystemExit(2)
    return json.loads(line[-1][6:])


def run(a) -> int:
    sys.path.insert(0, str(HERE))
    import restamp
    base = a.base.resolve()
    seeds = [0] + sorted({int(x) for x in a.seeds.split(",") if x.strip()} - {0})
    inst0 = json.loads((base / "instance.json").read_text())
    birth0, recorded = inst0["state_sha"], stamp(a.reference)
    checks = dict(birth=recorded in (None, birth0))
    rows, work = {}, pathlib.Path(tempfile.mkdtemp(prefix=f"seedphys-{base.name}-"))
    try:
        for s in seeds:
            src = base if s == 0 else base / "seeds" / str(s)
            inst = json.loads((src / "instance.json").read_text())
            d = work / f"s{s}"
            d.mkdir()
            shutil.copyfile(src / "instance.json", d / "instance.json")
            shutil.copytree(src / "instance_assets", d / "instance_assets")
            traj = a.reference
            if s and recorded is not None:       # seed S's reference: the rows with seed S's birth (release_rebuild)
                traj = d / f"privileged.s{s}.npz"
                restamp.restamp(a.reference, inst["state_sha"], traj)
            t0 = time.time()
            r = replay(a.task, d / "instance.json", traj, a.fresh, a.tooling)
            first = r["replays"][0]
            rows[s] = dict(seed=s, birth=inst["state_sha"], end=first["state_sha"],
                           derived=s == 0 or (inst.get("derived_from") or {}).get("instance_sha256") == canonical(inst0),
                           exact=all(x["instance_exact"] and x["birth"] == inst["state_sha"] for x in r["replays"]),
                           success=r["all_success"], deterministic=r["deterministic"],
                           replays=[dict(success=x["success"], rows=x["rows"], end=x["state_sha"]) for x in r["replays"]],
                           parts=first["parts"], task_but_instance_id=first.get("task_but_instance_id"),
                           seconds=round(time.time() - t0, 1))
            print("SEED " + json.dumps({k: v for k, v in rows[s].items() if k != "parts"}), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    p0 = rows[0]
    for s, r in rows.items():
        r["parts_differing"] = sorted(k for k in set(p0["parts"]) | set(r["parts"])
                                      if p0["parts"].get(k) != r["parts"].get(k))
        r["identity"] = (set(r["parts_differing"]) <= {"task"} and r["task_but_instance_id"] is not None
                         and r["task_but_instance_id"] == p0["task_but_instance_id"])
    for k in ("derived", "exact", "success", "deterministic", "identity"):
        checks[k] = all(r[k] for r in rows.values())
    out = dict(base=base.name, task=a.task, seeds=seeds, reference=str(a.reference), reference_birth=recorded,
               seed0_birth=birth0, ok=all(checks.values()), checks=checks,
               seeds_detail={str(s): {k: v for k, v in r.items() if k != "parts"} for s, r in rows.items()})
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")
    print("SEEDPHYS " + json.dumps(dict(base=base.name, ok=out["ok"], checks=checks,
                                        ends={s: r["end"] for s, r in rows.items()},
                                        parts_differing={s: r["parts_differing"] for s, r in rows.items() if s})),
          flush=True)
    return 0 if out["ok"] else 1


def harbor_list(a) -> int:
    rel = json.loads((a.tree / "scripts/roboarc/release.json").read_text())["bases"]
    named = set()
    for x in (a.changed or "").split(","):
        x = x.strip()
        if x.startswith("@"):
            named |= {n.strip() for n in pathlib.Path(x[1:]).read_text().split() if n.strip()}
        elif x:
            named.add(x)
    fams = sorted({r["family"] for r in rel.values()})
    unknown = named - set(rel) - set(fams)
    if unknown:
        raise SystemExit(f"--changed: neither a base of release.json nor a family: {sorted(unknown)}")
    changed = {b for b, r in rel.items() if b in named or r["family"] in named}
    picked = set(changed)
    for f in fams:                          # one base per family: the first by name with a reference
        mine = sorted(b for b, r in rel.items() if r["family"] == f)
        if not picked & set(mine):
            picked.add(next((b for b in mine if rel[b].get("oracle") != "none"), mine[0]))
    seeds = sorted({int(x) for x in a.seeds.split(",")})
    nop_only = sorted(b for b in picked if rel[b].get("oracle") == "none")
    full = sorted(picked - set(nop_only))
    gpu = sorted(b for b in picked if rel[b].get("gpu_physics"))
    trials = [dict(base=b, mode=m, seed=s, agent=g, task=f"roboarc-{b}-{m}")
              for s in seeds for b in sorted(picked) for m in ("privileged", "standard")
              for g in (("nop",) if b in nop_only else ("oracle", "nop"))]
    if a.json:
        print(json.dumps(dict(bases=sorted(picked), seeds=seeds, nop_only=nop_only, gpu_bases=gpu, trials=trials),
                         indent=1))
        return 0
    image = a.image or "ghcr.io/mll-lab-nu/roboarc:" + (a.tree / "images/roboarc/VERSION").read_text().strip()
    print(f"# Harbor oracle / nop, rendering off: {len(picked)} bases ({len(changed)} changed, one base of each other "
          f"family) x seeds {','.join(map(str, seeds))} x privileged, standard: "
          f"{len(trials)} trials ({sum(t['agent'] == 'oracle' for t in trials)} oracle, "
          f"{sum(t['agent'] == 'nop' for t in trials)} nop); expected oracle 1, nop 0")
    print(f"# bases: {' '.join(sorted(picked))}")
    for s in seeds:
        d = a.out / f"s{s}"
        print(f"\n# seed {s}: materialized copies of {a.tree}/tasks, then Harbor on them")
        print(f"mkdir -p {d}/images && cp {a.tree}/images/pins.json {d}/images/")
        print(f"for t in {' '.join(f'roboarc-{b}-{m}' for b in sorted(picked) for m in ('privileged', 'standard'))}; do "
              f"python3 scripts/task_seed.py materialize {a.tree}/tasks/$t --seed {s} --out {d}/tasks; done")
        for agents, sel, run_dir in (("oracle,nop", full, "harbor"), ("nop", nop_only, "harbor_nop")):
            if sel:
                g = [b for b in sel if b in gpu]
                print(f"python3 scripts/roboarc/harbor_check.py --tasks {d}/tasks --image {image} --run {d}/{run_dir} "
                      f"--agents {agents}" + (f" --gpu-bases {','.join(g)}" if g else "") + " " + " ".join(sel))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("base", type=pathlib.Path)
    r.add_argument("--task", required=True)
    r.add_argument("--reference", type=pathlib.Path, required=True)
    r.add_argument("--seeds", default="1,2")
    r.add_argument("--fresh", type=int, default=2)
    r.add_argument("--tooling", default="/tooling")
    r.add_argument("--out", type=pathlib.Path)
    h = sub.add_parser("harbor-list")
    h.add_argument("--changed")
    h.add_argument("--seeds", default="0,1,2")
    h.add_argument("--tree", type=pathlib.Path, default=ROOT)
    h.add_argument("--image")
    h.add_argument("--out", type=pathlib.Path, default=pathlib.Path("seed-harbor"))
    h.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else harbor_list(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
