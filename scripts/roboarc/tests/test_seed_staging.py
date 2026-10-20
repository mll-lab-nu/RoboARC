#!/usr/bin/env python3
"""Appearance seeds on the task side (owner, 2026-10-20), without real seed instances:
- images/roboarc/stage_instance.py --seed-file: no file or seed 0 stages the default instance as before; seed S stages
  /opt/robotangle/instances/<base>/seeds/<S>/ with runtime_binding.s<S>.json as the task's runtime_binding.json; a seed
  without an instance or a binding, a binding of another instance, or a seed file that is no number fails;
- scripts/roboarc/mk_tasks.py on a base given two fake seeds in assets.json (seed 0's lock and asset pins under another
  instance hash): [metadata.seeds] allowed = oracle = [0, 1, 2], build = "runtime" in both twins (ci_checks/check-seeds.py
  and check-twins.py pass), runtime_binding.s<S>.json beside every runtime_binding.json, the Dockerfiles and the sim
  entrypoint read the `seed` file scripts/task_seed.py materialize writes (check_image_files' COPY mapping of a seed-1
  copy holds it; a seed-0 copy has none), and the standard verifier's shell step picks the seed's binding.
Standard library only (python >= 3.11); no docker, no image."""
import copy
import hashlib
import importlib.util
import json
import re
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
BASE = "blanket-flatten-01-i01"


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


def test_stage_instance():
    si = load("stage_instance", ROOT / "images" / "roboarc" / "stage_instance.py")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        si.SRC = td / "instances"
        b0 = instance(si.SRC / BASE, 0)
        b1 = instance(si.SRC / BASE / "seeds" / "1", 1)
        task = td / "task"
        task.mkdir()
        (task / "runtime_binding.json").write_text(json.dumps(b0))
        (task / "runtime_binding.s1.json").write_text(json.dumps(b1))

        def stage(seed_text, *extra):
            dest = td / f"dest{len(list(td.iterdir()))}"
            sf = task / "seed"
            sf.unlink(missing_ok=True)
            if seed_text is not None:
                sf.write_text(seed_text)
            rc = si.main([BASE, str(dest), "--binding", str(task / "runtime_binding.json"), "--seed-file", str(sf), *extra])
            got = json.loads((dest / "instance.json").read_text()) if (dest / "instance.json").is_file() else None
            bnd = json.loads((dest / "runtime_binding.json").read_text()) if (dest / "runtime_binding.json").is_file() else None
            return rc, got, bnd

        for txt in (None, "0\n"):
            rc, got, bnd = stage(txt)
            check(rc == 0 and got == {"seed": 0} and bnd == b0, f"seed file {txt!r}: the default instance ({rc}, {got})")
        rc = si.main([BASE, str(td / "plain"), "--binding", str(task / "runtime_binding.json")])
        check(rc == 0 and json.loads((td / "plain" / "instance.json").read_text()) == {"seed": 0}, "no --seed-file: as before")
        rc, got, bnd = stage("1\n")
        check(rc == 0 and got == {"seed": 1} and bnd == b1, f"seed 1: its instance and binding ({rc}, {got}, {bnd})")
        rc, _, _ = stage("2\n")
        check(rc == 1, "seed 2 has no instance in the image: must fail")
        rc, _, _ = stage("x\n")
        check(rc == 1, "a seed file that is no number must fail")
        (task / "runtime_binding.s1.json").write_text(json.dumps(b0))
        rc, _, _ = stage("1\n")
        check(rc == 1, "seed 1 with a binding of another instance must fail")
        (task / "runtime_binding.s1.json").unlink()
        rc, _, _ = stage("1\n")
        check(rc == 1, "seed 1 without runtime_binding.s1.json must fail")
        (task / "runtime_binding.s1.json").write_text(json.dumps(b1))
        rc, got, _ = stage("1\n", "--purge")
        check(rc == 0 and got == {"seed": 1} and not si.SRC.exists(), "--purge removes every instance, the seeds' too")


def test_render():
    mk = load("mk_tasks_seeds", ROOT / "scripts" / "roboarc" / "mk_tasks.py")
    check(mk.seeds_of_base(BASE) == [0], "assets.json as committed: no appearance seeds yet")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pkg = td / "packages"
        for m in ("privileged", "standard"):
            shutil.copytree(mk.PKG / f"{BASE}-{m}", pkg / f"{BASE}-{m}")
        lock0 = json.loads((pkg / f"{BASE}-privileged" / "runtime_binding.upstream.json").read_text())
        assets = copy.deepcopy(mk.ASSETS)
        fake = {}
        for s in (1, 2):                       # seed 0's lock and asset pins under another instance hash
            fake[s] = hashlib.sha256(f"fake instance {s}".encode()).hexdigest()
            (pkg / f"{BASE}-privileged" / f"runtime_binding.s{s}.upstream.json").write_text(
                json.dumps({**lock0, "instance_sha256": fake[s]}, indent=2))
            assets["instances"][BASE].setdefault("seeds", {})[str(s)] = {"instance_sha256": fake[s]}
            pre = f"instances/{BASE}/"
            assets["files"].update({f"{pre}seeds/{s}/{k[len(pre):]}": v for k, v in mk.ASSETS["files"].items()
                                    if k.startswith(pre + "instance_assets/")})
        mk.PKG, mk.ASSETS = pkg, assets
        check(mk.seeds_of_base(BASE) == [0, 1, 2], f"seeds from assets.json: {mk.seeds_of_base(BASE)}")
        priv, std = mk.render(BASE, td / "out")
        cs = load("check_seeds", ROOT / "ci_checks" / "check-seeds.py")
        ct = load("check_twins", ROOT / "ci_checks" / "check-twins.py")
        for t in (priv, std):
            sd = task_seed.seeds(t)
            check(sd == {"default": 0, "allowed": [0, 1, 2], "oracle": [0, 1, 2] if sd["oracle"] else [], "build": "runtime"},
                  f"{t.name}: [metadata.seeds] {sd}")
            check(cs.check(t) == [], f"{t.name}: check-seeds {cs.check(t)}")
        check(ct.errors([priv, std]) == [], f"check-twins: {ct.errors([priv, std])}")
        places = {priv: ("environment", "tests"), std: ("environment/sim", "tests")}
        for t, ds in places.items():
            for d in ds:
                b = json.loads((t / d / "runtime_binding.json").read_text())
                for s in (1, 2):
                    bs = json.loads((t / d / f"runtime_binding.s{s}.json").read_text())
                    check(bs == {**b, "instance_sha256": fake[s]}, f"{t.name}/{d}/runtime_binding.s{s}.json is not seed {s}'s binding")
        for n in ("environment/Dockerfile", "tests/Dockerfile"):
            s = (priv / n).read_text()
            check("COPY runtime_binding*.json see[d] /opt/robotangle/task/\n" in s and "--seed-file /opt/robotangle/task/seed" in s
                  and "COPY runtime_binding.json" not in s, f"{priv.name}/{n}: not seed-aware")
        compose = (std / "environment" / "docker-compose.yaml").read_text()
        check("--binding /srv/runtime_binding.json --seed-file /srv/seed && exec" in compose, "the sim entrypoint ignores /srv/seed")
        # what the image builds hold: a seed-1 copy (scripts/task_seed.py materialize) brings its seed file, a seed-0 copy none
        for seed in (0, 1):
            out = td / f"copy{seed}"
            c = task_seed.materialize(priv, seed, out)
            for n, dst in (("environment", "/opt/robotangle/task"), ("tests", "/opt/robotangle/task")):
                exp = check_image_files.expected(str(c / n), str(c / n / "Dockerfile"))
                check((exp.get(f"{dst}/seed") == "seed") == (seed == 1) and exp.get(f"{dst}/runtime_binding.s2.json")
                      == "runtime_binding.s2.json", f"seed {seed} copy, {n}/Dockerfile maps {sorted(k for k in exp if dst in k)}")
            cs_ = task_seed.materialize(std, seed, out)
            check(((cs_ / "environment/sim/seed").is_file() and (cs_ / "tests/seed").is_file()) == (seed == 1),
                  f"standard seed {seed} copy: seed files")
        # the standard verifier's step that picks the seed's binding (its python image has no stage_instance.py)
        run = next(l for l in (std / "tests" / "Dockerfile").read_text().splitlines() if l.startswith("RUN s=$(cat "))
        cmd = run[len("RUN "):].split("   # ")[0]
        for seed_text, want in ((None, "runtime_binding.json"), ("0", "runtime_binding.json"), ("2\n", "runtime_binding.s2.json"),
                                ("x", None), ("3", None)):
            w = td / f"pick{len(list(td.iterdir()))}"
            shutil.copytree(std / "tests", w / "ctx")
            (w / "tests").mkdir()
            if seed_text is not None:
                (w / "ctx" / "seed").write_text(seed_text)
            r = subprocess.run(["sh", "-c", cmd.replace("/tmp/rcb-binding", str(w / "ctx")).replace("/tests/", f"{w}/tests/")],
                               capture_output=True, text=True)
            got = (w / "tests" / "runtime_binding.json").read_bytes() if (w / "tests" / "runtime_binding.json").is_file() else None
            ok = r.returncode != 0 and got is None if want is None else \
                r.returncode == 0 and got == (std / "tests" / want).read_bytes() and not (w / "ctx").exists()
            check(ok, f"standard verifier binding step, seed file {seed_text!r}: rc {r.returncode} {r.stderr.strip()}")


if __name__ == "__main__":
    test_stage_instance()
    test_render()
    if fails:
        print("\n".join(f"FAIL {f}" for f in fails))
        print(f"{len(fails)} failure(s)")
        sys.exit(1)
    print("ok seed staging: stage_instance --seed-file (default unchanged, seed S from seeds/<S>/ with its binding, failures "
          "closed); mk_tasks on fake seed pins: seeds block in both twins, per-seed bindings, Dockerfiles and the sim "
          "entrypoint read the runner's seed file, the standard verifier picks the seed's binding")
