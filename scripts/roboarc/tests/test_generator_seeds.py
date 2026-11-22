"""The generator's appearance seeds (scripts/roboarc/mk_tasks.py, docs/RUNNING.md "Seeds") on fixture seeds: two
committed bases (a rigid one, and the cloth one with goal images) given seeds 1 and 2 in a copy of their packages, pins
and references. Standard library only, Python >= 3.11, a few seconds:
  1. a seeded render passes ci_checks/check-seed-files.py and task_seed.py materialize, in both twins: `seed` files,
     runtime_binding.s<S>.json (the seed's pinned instance), Dockerfiles that copy them by name and stage with
     --seed-file, the standard grader's binding choice, per-seed goal images and re-stamped references;
  2. `[metadata.seeds]`: allowed [0, 1, 2], build "runtime", oracle = 0 and the seeds whose Harbor oracle passed;
  3. everything else is the unseeded render's (seeds add files and edit only the staging lines and task.toml);
  4. fail closed: a seeded goal base without a seed's goal images, a seed without its upstream binding, a pin that is
     not the binding's.
usage: python3 scripts/roboarc/tests/test_generator_seeds.py"""
import copy, hashlib, json, pathlib, shutil, sys, tempfile, zipfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import mk_tasks as M  # noqa: E402
import task_seed  # noqa: E402
sys.path.insert(0, str(HERE.parents[2] / "ci_checks"))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("check_seed_files", HERE.parents[2] / "ci_checks" / "check-seed-files.py")
CSF = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(CSF)
sys.path.insert(0, str(HERE.parents[2] / "tools" / "release"))
import restamp  # noqa: E402

RIGID, GOAL = "interlocked-claws-08-i01", "blanket-fold-01-i01"


def tree(d: pathlib.Path) -> dict:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def fixture(tmp: pathlib.Path, base: str, goal_images: bool = True) -> None:
    """Seeds 1, 2 of `base` in M's packages, pins and references (copies under tmp): each seed's upstream lock is seed
    0's with its own instance hash and one changed visual asset; its reference is seed 0's re-stamped."""
    for m in ("privileged", "standard"):
        shutil.copytree(ORIG_PKG / f"{base}-{m}", M.PKG / f"{base}-{m}")
        shutil.rmtree(M.PKG / f"{base}-{m}" / "tests" / "seeds", ignore_errors=True)   # the release's own seed pictures
    up = json.loads((M.PKG / f"{base}-privileged" / "runtime_binding.upstream.json").read_text())
    rec = M.ASSETS["instances"][base]
    files = M.ASSETS["files"]
    prefix = f"instances/{base}/"
    for s in (1, 2):
        lock = copy.deepcopy(up)
        lock["instance_sha256"] = hashlib.sha256(f"{base} seed {s}".encode()).hexdigest()
        rec.setdefault("seeds", {})[str(s)] = {"instance_sha256": lock["instance_sha256"]}
        for k, v in list(files.items()):
            if k.startswith(prefix) and "/seeds/" not in k:
                files[f"{prefix}seeds/{s}/{k[len(prefix):]}"] = v
        visual = sorted(k for k in lock["asset_files"] if not k.startswith("instance_assets/mujoco"))
        if visual:
            lock["asset_files"][visual[0]] = hashlib.sha256(f"visual {s}".encode()).hexdigest()
            files[f"{prefix}seeds/{s}/{visual[0]}"] = lock["asset_files"][visual[0]]
        (M.PKG / f"{base}-privileged" / f"runtime_binding.s{s}.upstream.json").write_text(json.dumps(lock, indent=2))
        for m in ("privileged", "standard"):
            g = M.PKG / f"{base}-{m}" / M.GOAL_DIR
            if g.is_dir() and goal_images:
                d = M.PKG / f"{base}-{m}" / "tests" / "seeds" / str(s) / "goal"
                shutil.rmtree(d, ignore_errors=True)
                d.mkdir(parents=True)
                for p in g.iterdir():
                    (d / p.name).write_bytes(p.read_bytes() + f"seed {s}".encode())
                    rec["seeds"][str(s)].setdefault("goal", {})[p.name] = hashlib.sha256((d / p.name).read_bytes()).hexdigest()
    (M.ORC / base).mkdir(parents=True, exist_ok=True)
    for f in (ORIG_ORC / base).iterdir():
        shutil.copy(f, M.ORC / base / f.name)
    if (M.ORC / base / "privileged.npz").is_file():
        for s in (1, 2):
            restamp.restamp(M.ORC / base / "privileged.npz", f"{s:x}" * 16, M.ORC / base / f"privileged.s{s}.npz")


ORIG_PKG, ORIG_ORC, ORIG_ASSETS = M.PKG, M.ORC, copy.deepcopy(M.ASSETS)
for _r in ORIG_ASSETS["instances"].values():     # the unseeded state: the release's own seed pins dropped (fixtures add theirs)
    _r.pop("seeds", None)
ORIG_ASSETS["files"] = {k: v for k, v in ORIG_ASSETS["files"].items() if "/seeds/" not in k}


def fresh(tmp: pathlib.Path) -> None:
    M.PKG, M.ORC = tmp / "packages", tmp / "oracles"
    shutil.rmtree(M.PKG, ignore_errors=True); shutil.rmtree(M.ORC, ignore_errors=True)
    M.PKG.mkdir(); M.ORC.mkdir()
    M.ASSETS.clear(); M.ASSETS.update(copy.deepcopy(ORIG_ASSETS))


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = pathlib.Path(t)
        fresh(tmp)
        for b in (RIGID, GOAL):
            for m in ("privileged", "standard"):
                shutil.copytree(ORIG_PKG / f"{b}-{m}", M.PKG / f"{b}-{m}")
            shutil.copytree(ORIG_ORC / b, M.ORC / b)
        plain = {b: {d.name: tree(d) for d in M.render(b, tmp / "plain")} for b in (RIGID, GOAL)}
        fresh(tmp)
        for b in (RIGID, GOAL):
            fixture(tmp, b)
        M.REL["bases"][RIGID] = dict(M.REL["bases"][RIGID], seeds={"1": {"harbor": "passed"}, "2": {"harbor": "failed"}})
        M.REL["bases"][GOAL] = {k: v for k, v in M.REL["bases"][GOAL].items() if k != "seeds"}   # no Harbor record here
        for b in (RIGID, GOAL):
            assert M.seeds_of_base(b) == [1, 2]
            for d in M.render(b, tmp / "seeded"):
                assert CSF.check(d) == [], (d.name, CSF.check(d))
                sd = task_seed.seeds(d)
                assert (sd["allowed"], sd["build"]) == ([0, 1, 2], "runtime"), sd
                assert sd["oracle"] == ([0, 1] if b == RIGID else [0] if sd["oracle"] else []), (b, sd)
                mode = "privileged" if d.name.endswith("-privileged") else "standard"
                for place in M.SEED_PLACES[mode]:
                    for s in (1, 2):
                        got = json.loads((d / place / f"runtime_binding.s{s}.json").read_text())
                        assert got["instance_sha256"] == M.ASSETS["instances"][b]["seeds"][str(s)]["instance_sha256"]
                texts = "".join((d / n).read_text() for n in ("environment/Dockerfile", "tests/Dockerfile",
                                                              "environment/docker-compose.yaml"))
                assert "--seed-file" in texts and not any("*" in l for l in texts.splitlines() if l.startswith("COPY"))
                if mode == "standard":
                    assert M.STD_BINDING_CHOICE in (d / "tests/Dockerfile").read_text()
                cp = task_seed.materialize(d, 2, tmp / "mat" / mode / b)
                assert (cp / M.SEED_PLACES[mode][0] / "seed").read_text() == "2\n"
                if (d / M.GOAL_DIR).is_dir():
                    assert all(p.read_bytes().endswith(b"seed 2") for p in (cp / M.GOAL_DIR).iterdir())
                if (d / "solution/trajectory.npz").is_file():
                    assert restamp.check(d / "solution/trajectory.npz", d / "solution/trajectory.s1.npz", "1" * 16) == []
                # everything but the seed additions and the edited texts is the unseeded render's
                was, now = plain[b][d.name], tree(d)
                edited = {"task.toml", "README.md", "DIGEST", "environment/Dockerfile", "tests/Dockerfile",
                          "environment/docker-compose.yaml", "solution/solve.sh"}
                added = {k for k in now if k not in was}
                assert all(k.endswith("/seed") or ".s1." in k or ".s2." in k or k.startswith("tests/seeds/") for k in added), added
                assert not set(was) - set(now), set(was) - set(now)
                assert all(was[k] == now[k] for k in was if k not in edited), [k for k in was if k not in edited and was[k] != now[k]]
        # fail closed
        fresh(tmp)
        fixture(tmp, GOAL, goal_images=False)
        try:
            M.render(GOAL, tmp / "nogoal"); raise SystemExit("a seeded goal base without seed goal images rendered")
        except AssertionError as e:
            assert "goal images of seed" in str(e), e
        fresh(tmp)
        fixture(tmp, RIGID)
        (M.PKG / f"{RIGID}-privileged" / "runtime_binding.s2.upstream.json").unlink()
        try:
            M.render(RIGID, tmp / "nolock"); raise SystemExit("a seed without its upstream binding rendered")
        except FileNotFoundError:
            pass
        fresh(tmp)
        fixture(tmp, RIGID)
        M.ASSETS["instances"][RIGID]["seeds"]["1"]["instance_sha256"] = "0" * 64
        try:
            M.render(RIGID, tmp / "badpin"); raise SystemExit("a seed whose pin is not its binding rendered")
        except AssertionError as e:
            assert "seed 1" in str(e), e
    print("ok mk_tasks seeds: seeded twins pass check-seed-files and materialize; only the seed files and staging lines differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
