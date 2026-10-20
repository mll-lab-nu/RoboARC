#!/usr/bin/env python3
"""Goal pictures of appearance seeds (owner, 2026-10-20: each seed's pictures rendered in its looks and kept beside its
instance in the image), without real seed instances or pictures:
- images/roboarc/stage_instance.py --goal: no seed file or seed 0 puts the task's own pictures in place; seed S puts
  /opt/robotangle/instances/<base>/seeds/<S>/goal/'s; a seed whose pictures are missing, or not exactly the task's file
  names, fails, and so does a task without pictures; without --goal nothing is placed;
- scripts/roboarc/mk_tasks.py on a base with goal pictures given two fake seeds in assets.json (instance and picture
  pins): the privileged environment copies the pictures next to the bindings, not to /app/goal/, and stages the seed's
  (check_image_files' COPY mapping); the standard agent image is unchanged (check-standard-agent-image.sh) and its task's
  tests/seeds/<S>/goal/ holds the pinned bytes; a picture that is not the pinned one, or a seed without picture pins,
  stops the render;
- scripts/task_seed.py materialize: a standard seed-S copy has seed S's pictures in environment/goal/ and no
  tests/seeds/; a seed-0 copy keeps the default's; a privileged copy keeps its pictures (the image has the seed's).
Standard library only (python >= 3.11); no docker, no image."""
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts" / "roboarc")]
import check_image_files  # noqa: E402
import task_seed  # noqa: E402

fails = []
BASE = "blanket-fold-01-i01"
NAMES = ("target_cam_high.png", "target_cam_low.png")


def check(cond, what):
    if not cond:
        fails.append(what)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def canon(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def instance(d: Path, tag: int) -> dict:
    """A fake frozen instance under d; its binding."""
    (d / "instance_assets").mkdir(parents=True)
    inst = {"seed": tag}
    (d / "instance.json").write_text(json.dumps(inst))
    (d / "instance_assets" / "x.bin").write_bytes(b"asset %d" % tag)
    return {"schema_version": 1, "instance_sha256": canon(inst),
            "asset_files": {"instance_assets/x.bin": hashlib.sha256(b"asset %d" % tag).hexdigest()}}


def picture(seed: int, name: str) -> bytes:
    return f"seed {seed} {name}".encode()


def test_stage_goal():
    si = load("stage_instance", ROOT / "images" / "roboarc" / "stage_instance.py")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        si.SRC = td / "instances"
        b0 = instance(si.SRC / BASE, 0)
        b1 = instance(si.SRC / BASE / "seeds" / "1", 1)
        b2 = instance(si.SRC / BASE / "seeds" / "2", 2)
        img = si.SRC / BASE / "seeds" / "1" / "goal"
        img.mkdir()
        for n in NAMES:
            (img / n).write_bytes(picture(1, n))
        task = td / "task"
        (task / "goal").mkdir(parents=True)
        for n in NAMES:
            (task / "goal" / n).write_bytes(picture(0, n))
        for name, b in (("runtime_binding.json", b0), ("runtime_binding.s1.json", b1), ("runtime_binding.s2.json", b2)):
            (task / name).write_text(json.dumps(b))

        def stage(seed_text, own=task / "goal", goal=True, *extra):
            dest = td / f"dest{len(list(td.iterdir()))}"
            sf = task / "seed"
            sf.unlink(missing_ok=True)
            if seed_text is not None:
                sf.write_text(seed_text)
            args = [BASE, str(dest), "--binding", str(task / "runtime_binding.json"), "--seed-file", str(sf), *extra]
            rc = si.main(args + (["--goal", str(own), str(dest / "goal")] if goal else []))
            out = dest / "goal"
            return rc, {p.name: p.read_bytes() for p in out.iterdir()} if out.is_dir() else None

        own = {n: picture(0, n) for n in NAMES}
        for txt in (None, "0\n"):
            rc, got = stage(txt)
            check(rc == 0 and got == own, f"seed file {txt!r}: the task's own pictures ({rc}, {got})")
        rc, got = stage("1\n")
        check(rc == 0 and got == {n: picture(1, n) for n in NAMES}, f"seed 1: the image's seed-1 pictures ({rc}, {got})")
        rc, got = stage("1\n", goal=False)
        check(rc == 0 and got is None, "without --goal no pictures are placed")
        rc, got = stage("2\n")
        check(rc == 1 and got is None, "seed 2 has no pictures in the image: must fail")
        (img / NAMES[1]).unlink()
        rc, got = stage("1\n")
        check(rc == 1 and got is None, "seed 1 missing a picture: must fail")
        (img / NAMES[1]).write_bytes(picture(1, NAMES[1]))
        (img / "extra.png").write_bytes(b"x")
        rc, got = stage("1\n")
        check(rc == 1 and got is None, "seed 1 with a picture the task does not have: must fail")
        (img / "extra.png").unlink()
        (td / "empty").mkdir()
        rc, got = stage("0\n", td / "empty")
        check(rc == 1, "--goal on a task without pictures must fail")
        rc, got = stage("1\n", task / "goal", True, "--purge")
        check(rc == 0 and got == {n: picture(1, n) for n in NAMES} and not si.SRC.exists(),
              "--purge after the seed's pictures are placed")


def test_render_and_materialize():
    mk = load("mk_tasks_goal", ROOT / "scripts" / "roboarc" / "mk_tasks.py")
    check(mk.seeds_of_base(BASE) == [0], "assets.json as committed: no appearance seeds yet")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pkg = td / "packages"
        for m in ("privileged", "standard"):
            shutil.copytree(mk.PKG / f"{BASE}-{m}", pkg / f"{BASE}-{m}")
        check(sorted(p.name for p in (pkg / f"{BASE}-standard" / "environment" / "goal").iterdir()) == list(NAMES),
              f"{BASE} ships the pictures {NAMES}")
        lock0 = json.loads((pkg / f"{BASE}-privileged" / "runtime_binding.upstream.json").read_text())
        assets = copy.deepcopy(mk.ASSETS)
        pre = f"instances/{BASE}/"
        for s in (1, 2):                       # seed 0's lock and asset pins under another instance hash; fake pictures
            fake = hashlib.sha256(f"fake instance {s}".encode()).hexdigest()
            (pkg / f"{BASE}-privileged" / f"runtime_binding.s{s}.upstream.json").write_text(
                json.dumps({**lock0, "instance_sha256": fake}, indent=2))
            assets["instances"][BASE].setdefault("seeds", {})[str(s)] = {"instance_sha256": fake}
            assets["files"].update({f"{pre}seeds/{s}/{k[len(pre):]}": v for k, v in mk.ASSETS["files"].items()
                                    if k.startswith(pre + "instance_assets/")})
            for n in NAMES:
                assets["files"][f"{pre}seeds/{s}/goal/{n}"] = hashlib.sha256(picture(s, n)).hexdigest()
                p = pkg / f"{BASE}-standard" / "tests" / "seeds" / str(s) / "goal" / n
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(picture(s, n))
        mk.PKG, mk.ASSETS = pkg, assets
        priv, std = mk.render(BASE, td / "out")
        tmp = "/opt/robotangle/task"
        s = (priv / "environment" / "Dockerfile").read_text()
        check("COPY goal/ /app/goal/" not in s and f"COPY goal/ {tmp}/goal/\n" in s
              and s.index(f"COPY goal/ {tmp}/goal/") < s.index("stage_instance.py")
              and f"--seed-file {tmp}/seed --goal {tmp}/goal /app/goal --purge" in s,
              f"{priv.name}/environment/Dockerfile does not stage the seed's pictures")
        exp = check_image_files.expected(str(priv / "environment"), str(priv / "environment" / "Dockerfile"))
        check(all(exp.get(f"{tmp}/goal/{n}") == f"goal/{n}" for n in NAMES) and not any(k.startswith("/app/goal") for k in exp),
              f"privileged COPY mapping: {sorted(k for k in exp if 'goal' in k)}")
        check("COPY goal/ /app/goal/\n" in (std / "environment" / "Dockerfile").read_text(),
              f"{std.name}: the agent image copies goal/ to /app/goal/")
        r = subprocess.run(["bash", str(ROOT / "ci_checks" / "check-standard-agent-image.sh"), str(std)],
                           capture_output=True, text=True)
        check(r.returncode == 0, f"check-standard-agent-image: {r.stdout.strip()} {r.stderr.strip()}")
        for s_ in (1, 2):
            got = {p.name: p.read_bytes() for p in (std / "tests" / "seeds" / str(s_) / "goal").iterdir()}
            check(got == {n: picture(s_, n) for n in NAMES}, f"{std.name}/tests/seeds/{s_}/goal: {sorted(got)}")
        check(not (priv / "tests" / "seeds").exists(), f"{priv.name}: no committed pictures (the image has them)")
        cs = load("check_seeds", ROOT / "ci_checks" / "check-seeds.py")
        ct = load("check_twins", ROOT / "ci_checks" / "check-twins.py")
        for t in (priv, std):
            check(cs.check(t) == [], f"{t.name}: check-seeds {cs.check(t)}")
        check(ct.errors([priv, std]) == [], f"check-twins: {ct.errors([priv, std])}")
        default = {n: (std / "environment" / "goal" / n).read_bytes() for n in NAMES}
        for seed in (0, 1, 2):
            out = td / f"copy{seed}"
            c = task_seed.materialize(std, seed, out)
            got = {n: (c / "environment" / "goal" / n).read_bytes() for n in NAMES}
            want = default if seed == 0 else {n: picture(seed, n) for n in NAMES}
            check(got == want and not (c / "tests" / "seeds").exists(), f"standard seed {seed} copy: goal pictures")
            c = task_seed.materialize(priv, seed, out)
            got = {n: (c / "environment" / "goal" / n).read_bytes() for n in NAMES}
            check(got == default and (c / "environment" / "seed").is_file() == (seed != 0),
                  f"privileged seed {seed} copy: its own pictures and the seed file")
        # a committed picture that is not the pinned one, and a seed without picture pins, stop the render
        (pkg / f"{BASE}-standard" / "tests" / "seeds" / "2" / "goal" / NAMES[0]).write_bytes(b"other")
        try:
            mk.render(BASE, td / "bad1")
            check(False, "a standard picture that is not the pinned one must stop the render")
        except AssertionError as e:
            check("tests/seeds/2/goal" in str(e), f"wrong failure for a changed picture: {e}")
        (pkg / f"{BASE}-standard" / "tests" / "seeds" / "2" / "goal" / NAMES[0]).write_bytes(picture(2, NAMES[0]))
        del assets["files"][f"{pre}seeds/2/goal/{NAMES[1]}"]
        try:
            mk.render(BASE, td / "bad2")
            check(False, "a seed without a picture pin must stop the render")
        except AssertionError as e:
            check("pins no goal picture" in str(e), f"wrong failure for a missing pin: {e}")


if __name__ == "__main__":
    test_stage_goal()
    test_render_and_materialize()
    if fails:
        print("\n".join(f"FAIL {f}" for f in fails))
        print(f"{len(fails)} failure(s)")
        sys.exit(1)
    print("ok seed goal images: stage_instance --goal (the task's own for seed 0, the image's for seed S, failures closed); "
          "mk_tasks on fake seed and picture pins: the privileged environment stages the seed's pictures, the standard "
          "task commits the pinned bytes under tests/seeds/<S>/goal/; materialize puts them in a standard copy")
