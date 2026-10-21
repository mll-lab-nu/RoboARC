#!/usr/bin/env python3
"""Goal pictures of appearance seeds (owner, 2026-10-20: each seed's pictures rendered in its looks, committed in the
task dir as tests/seeds/<S>/goal/, sha256 in assets.json, and copied over environment/goal/ by scripts/task_seed.py
materialize in both modes), without real seed instances or pictures:
- scripts/roboarc/mk_tasks.py on a base with goal pictures given two fake seeds in assets.json (instance and picture
  pins, re-stamped references): both twins commit tests/seeds/<S>/goal/ with the pinned bytes; their agent images copy
  goal/ to /app/goal/ as before (nothing from the simulator's image; check-standard-agent-image.sh) and check-seeds,
  check-twins pass; a picture that is not the pinned one, or pins that do not name the task's pictures, stop the render;
- scripts/task_seed.py materialize: a seed-S copy of either twin has seed S's pictures in environment/goal/ and no
  tests/seeds/; a seed-0 copy keeps the default's.
Standard library only (python >= 3.11); no docker, no image."""
import copy
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts" / "roboarc")]
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


def restamp(data: bytes, birth: str) -> bytes:
    """A reference with instance_sha.npy's text replaced (release_rebuild.restamp's result), every other entry as it is."""
    zin = zipfile.ZipFile(io.BytesIO(data))
    raw = zin.read("instance_sha.npy")
    head = 10 + int.from_bytes(raw[8:10], "little")
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for n in zin.namelist():
            z.writestr(n, raw[:head] + birth.encode("utf-32-le").ljust(len(raw) - head, b"\0") if n == "instance_sha.npy"
                       else zin.read(n))
    return out.getvalue()


def picture(seed: int, name: str) -> bytes:
    return f"seed {seed} {name}".encode()


def test_render_and_materialize():
    mk = load("mk_tasks_goal", ROOT / "scripts" / "roboarc" / "mk_tasks.py")
    check(mk.seeds_of_base(BASE) in ([0], [0, 1, 2]), f"assets.json as committed: seeds {mk.seeds_of_base(BASE)}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pkg, orc = td / "packages", td / "oracles"
        for m in ("privileged", "standard"):
            shutil.copytree(mk.PKG / f"{BASE}-{m}", pkg / f"{BASE}-{m}")
            check(sorted(p.name for p in (pkg / f"{BASE}-{m}" / "environment" / "goal").iterdir()) == list(NAMES),
                  f"{BASE}-{m} ships the pictures {NAMES}")
        shutil.copytree(mk.ORC / BASE, orc / BASE)
        lock0 = json.loads((pkg / f"{BASE}-privileged" / "runtime_binding.upstream.json").read_text())
        assets = copy.deepcopy(mk.ASSETS)
        assets["instances"][BASE].pop("seeds", None)   # the release's own seed pins give way to the fake ones
        pre = f"instances/{BASE}/"
        for s in (1, 2):                       # seed 0's lock and asset pins under another instance hash; fake pictures
            fake = hashlib.sha256(f"fake instance {s}".encode()).hexdigest()
            (pkg / f"{BASE}-privileged" / f"runtime_binding.s{s}.upstream.json").write_text(
                json.dumps({**lock0, "instance_sha256": fake}, indent=2))
            assets["instances"][BASE].setdefault("seeds", {})[str(s)] = {
                "instance_sha256": fake, "goal": {n: hashlib.sha256(picture(s, n)).hexdigest() for n in NAMES}}
            assets["files"].update({f"{pre}seeds/{s}/{k[len(pre):]}": v for k, v in mk.ASSETS["files"].items()
                                    if k.startswith(pre + "instance_assets/")})
            for m in ("privileged", "standard"):
                for n in NAMES:
                    p = pkg / f"{BASE}-{m}" / "tests" / "seeds" / str(s) / "goal" / n
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(picture(s, n))
            (orc / BASE / f"privileged.s{s}.npz").write_bytes(restamp((orc / BASE / "privileged.npz").read_bytes(),
                                                                      hashlib.sha256(f"birth {s}".encode()).hexdigest()[:16]))
        mk.PKG, mk.ASSETS, mk.ORC = pkg, assets, orc
        priv, std = mk.render(BASE, td / "out")
        for t in (priv, std):
            check("COPY goal/ /app/goal/\n" in (t / "environment" / "Dockerfile").read_text()
                  and "--goal" not in (t / "environment" / "Dockerfile").read_text(),
                  f"{t.name}: the agent image copies goal/ to /app/goal/, nothing else")
            for s_ in (1, 2):
                got = {p.name: p.read_bytes() for p in (t / "tests" / "seeds" / str(s_) / "goal").iterdir()}
                check(got == {n: picture(s_, n) for n in NAMES}, f"{t.name}/tests/seeds/{s_}/goal: {sorted(got)}")
        r = subprocess.run(["bash", str(ROOT / "ci_checks" / "check-standard-agent-image.sh"), str(std)],
                           capture_output=True, text=True)
        check(r.returncode == 0, f"check-standard-agent-image: {r.stdout.strip()} {r.stderr.strip()}")
        cs = load("check_seeds", ROOT / "ci_checks" / "check-seeds.py")
        ct = load("check_twins", ROOT / "ci_checks" / "check-twins.py")
        for t in (priv, std):
            check(cs.check(t) == [], f"{t.name}: check-seeds {cs.check(t)}")
        check(ct.errors([priv, std]) == [], f"check-twins: {ct.errors([priv, std])}")
        for t in (priv, std):
            default = {n: (t / "environment" / "goal" / n).read_bytes() for n in NAMES}
            for seed in (0, 1, 2):
                c = task_seed.materialize(t, seed, td / f"copy{seed}")
                got = {n: (c / "environment" / "goal" / n).read_bytes() for n in NAMES}
                want = default if seed == 0 else {n: picture(seed, n) for n in NAMES}
                check(got == want and not (c / "tests" / "seeds").exists(), f"{t.name} seed {seed} copy: goal pictures")
        # a committed picture that is not the pinned one, and pins that do not name the task's pictures, stop the render
        bad = pkg / f"{BASE}-privileged" / "tests" / "seeds" / "2" / "goal" / NAMES[0]
        bad.write_bytes(b"other")
        try:
            mk.render(BASE, td / "bad1")
            check(False, "a picture that is not the pinned one must stop the render")
        except AssertionError as e:
            check("tests/seeds/2/goal" in str(e), f"wrong failure for a changed picture: {e}")
        bad.write_bytes(picture(2, NAMES[0]))
        del assets["instances"][BASE]["seeds"]["2"]["goal"][NAMES[1]]
        try:
            mk.render(BASE, td / "bad2")
            check(False, "a seed without a picture pin must stop the render")
        except AssertionError as e:
            check("pins the goal pictures" in str(e), f"wrong failure for a missing pin: {e}")


if __name__ == "__main__":
    test_render_and_materialize()
    if fails:
        print("\n".join(f"FAIL {f}" for f in fails))
        print(f"{len(fails)} failure(s)")
        sys.exit(1)
    print("ok seed goal images: mk_tasks on fake seed and picture pins: both twins commit the pinned bytes under "
          "tests/seeds/<S>/goal/, the agent images copy goal/ as before; materialize puts seed S's pictures in "
          "environment/goal/ of either twin's copy")
