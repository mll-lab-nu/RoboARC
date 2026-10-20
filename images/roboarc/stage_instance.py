#!/usr/bin/env python3
"""Stage one RoboARC task's frozen instance from the image into a task directory, and check it against the task's
runtime binding. Standard library only.

usage: python /opt/robotangle/stage_instance.py <base> <dest> [--binding <runtime_binding.json>] [--seed-file <seed>]
                                                [--goal <task goal dir> <goal dest>] [--purge]

The image (images/roboarc/Dockerfile) carries every task's frozen instance, instance.json + instance_assets/, under
/opt/robotangle/instances/<base>/. The files come from the private Hugging Face dataset eai-v2-internal/robotangle-assets
at the revision pinned in scripts/roboarc/assets.json, and the image build checks each file's sha256 there. A task
directory holds only runtime_binding.json, which names the instance's canonical sha256 and the sha256 of every asset.
A task that allows more than its default seed (`[metadata.seeds] build = "runtime"`, scripts/task_seed.py) has each other
seed's instance in the image too, under /opt/robotangle/instances/<base>/seeds/<seed>/, and next to every copy of its
runtime_binding.json that seed's binding, runtime_binding.s<seed>.json. The runner's copy of the task for seed S carries a
file `seed` (the integer S) in every place the task runs and grades from; --seed-file names it. No such file, or seed 0:
the default instance and runtime_binding.json, as before.
A task whose instruction shows goal pictures (environment/goal/*.png, at /app/goal/ in the agent's container) has them
rendered in each seed's looks: seed 0's ship in the task directory, seed S's in the image beside that seed's instance,
/opt/robotangle/instances/<base>/seeds/<S>/goal/ (pinned in scripts/roboarc/assets.json like every image file).
This script:
  - copies the instance into <dest>, next to <dest>/runtime_binding.json (--binding copies that file there first; for a
    seed S other than 0, runtime_binding.s<S>.json from the same directory as that file, or from <dest>);
  - checks both against the binding, by the rule of topogym.simulator.frozen_binding.check_inputs, which every session
    applies again at start;
  - with --goal, puts the goal pictures of the task's seed into <goal dest>: seed 0's from <task goal dir> (the task's
    own, which the Dockerfile copies there), seed S's from the image; seed S must have exactly the task's file names;
  - with --purge, removes /opt/robotangle/instances afterwards, so a task image keeps only its own instance.
Privileged tasks run it in environment/Dockerfile (with --goal when they have goal pictures) and tests/Dockerfile.
Standard tasks run it in the sim service's entrypoint, before `sh /srv/run.sh`, and RCB_INSTANCE points into <dest>;
their agent's container takes nothing from the sim service or its image (PROTOCOL H.1, H.3), so a seed's goal pictures
reach it in the runner's copy of the task instead (scripts/task_seed.py materialize, from tests/seeds/<S>/goal/).
"""
import hashlib
import json
import os
import pathlib
import shutil
import sys

SRC = pathlib.Path("/opt/robotangle/instances")


def file_sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    purge = "--purge" in argv
    argv = [a for a in argv if a != "--purge"]
    binding = seed_file = None
    if "--binding" in argv:
        i = argv.index("--binding")
        binding = pathlib.Path(argv[i + 1])
        del argv[i:i + 2]
    if "--seed-file" in argv:
        i = argv.index("--seed-file")
        seed_file = pathlib.Path(argv[i + 1])
        del argv[i:i + 2]
    goal = None
    if "--goal" in argv:
        i = argv.index("--goal")
        goal = pathlib.Path(argv[i + 1]), pathlib.Path(argv[i + 2])
        del argv[i:i + 3]
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    base, dest = argv[0], pathlib.Path(argv[1])
    seed = 0
    if seed_file is not None and seed_file.is_file():
        text = seed_file.read_text().strip()
        if not text.isdigit():
            print(f"FAIL stage_instance {base}: the seed file {seed_file} holds {text!r}, not a seed", file=sys.stderr)
            return 1
        seed = int(text)
    src = SRC / base if seed == 0 else SRC / base / "seeds" / str(seed)
    if not (src / "instance.json").is_file():
        print(f"FAIL stage_instance: no frozen instance for {base}{f' seed {seed}' if seed else ''} in this image ({src})",
              file=sys.stderr)
        return 1
    if seed:                                  # that seed's binding, next to the task's runtime_binding.json
        binding = (binding.parent if binding is not None else dest) / f"runtime_binding.s{seed}.json"
        if not binding.is_file():
            print(f"FAIL stage_instance {base}: seed {seed} has no runtime binding ({binding})", file=sys.stderr)
            return 1
    dest.mkdir(parents=True, exist_ok=True)
    if binding is not None:
        shutil.copyfile(binding, dest / "runtime_binding.json")
    lock = json.loads((dest / "runtime_binding.json").read_text())
    shutil.copyfile(src / "instance.json", dest / "instance.json")
    if (dest / "instance_assets").exists():
        shutil.rmtree(dest / "instance_assets")
    shutil.copytree(src / "instance_assets", dest / "instance_assets", symlinks=True)
    bad = []
    inst = json.loads((dest / "instance.json").read_text())
    canon = hashlib.sha256(json.dumps(inst, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if canon != lock["instance_sha256"]:
        bad.append(f"instance.json {canon[:12]} is not the bound instance {lock['instance_sha256'][:12]}")
    assets = {}
    for p in sorted((dest / "instance_assets").rglob("*")):
        if "__pycache__" in p.parts:
            continue
        if p.is_symlink():
            bad.append(f"symlink in the instance assets: {p}")
        elif p.is_file():
            assets["instance_assets/" + p.relative_to(dest / "instance_assets").as_posix()] = file_sha(p)
    if assets != lock["asset_files"]:
        diff = sorted(set(assets.items()) ^ set(lock["asset_files"].items()))
        bad.append(f"instance_assets differ from the binding ({len(diff)} entries, e.g. {diff[:2]})")
    for p in [dest / "instance.json", *(dest / "instance_assets").rglob("*")]:
        os.chmod(p, 0o755 if p.is_dir() else 0o644)
    if goal is not None and not bad:          # the goal pictures of the task's seed, by the task's own file names
        own, out = goal
        names = sorted(p.name for p in own.iterdir() if p.is_file()) if own.is_dir() else []
        pics = own if seed == 0 else src / "goal"
        have = sorted(p.name for p in pics.iterdir() if p.is_file()) if pics.is_dir() else []
        if not names:
            bad.append(f"no goal pictures in {own}")
        elif have != names:
            bad.append(f"seed {seed}'s goal pictures in {pics} are {have}, not the task's {names}")
        else:
            out.mkdir(parents=True, exist_ok=True)
            for n in names:
                shutil.copyfile(pics / n, out / n)
                os.chmod(out / n, 0o644)
    for b in bad:
        print(f"FAIL stage_instance {base}: {b}", file=sys.stderr)
    if bad:
        return 1
    if purge:
        shutil.rmtree(SRC)
    pictures = f", {len(names)} goal pictures -> {goal[1]}" if goal is not None else ""
    print(f"ok stage_instance {base}{f' seed {seed}' if seed else ''}: instance {canon[:12]}, {len(assets)} asset files "
          f"-> {dest}{pictures}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
