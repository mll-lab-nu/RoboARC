#!/usr/bin/env python3
"""The release files of appearance seeds (scripts/robotangle/release_rebuild.py --appearance-seeds), on fake instances,
without docker, a GPU or numpy:
  - restamp: only instance_sha.npy changes (the birth, under the same .npy header); a birth longer than it holds fails;
  - write_release with seeds: per seed S the lock of instances/<base>/seeds/<S>/ (runtime_binding.s<S>.upstream.json),
    the reference re-stamped with the seed's birth (oracles/<base>/privileged.s<S>.npz) and the seeds' pins in
    assets.json; a previous release's seed files of the base are removed; without seeds it writes what it wrote before
    (the same tree as with seeds, less the seed files and pins);
  - seed_goal_pictures: pictures from --goal-pictures or rendered (the docker command, faked here) go to both packages'
    tests/seeds/<S>/goal/ and are pinned under instances[<base>].seeds[<S>].goal; a missing picture, a record of another
    birth or a family without a renderer stops the run;
  - assets.py pin keeps the seeds (and a seed's goal pins while its instance is unchanged); check verifies them;
  - images/robotangle/stage_instance.py --seed-file stages seeds/<S>/ with runtime_binding.s<S>.json, else the default."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OLD, NEW = "blanket-fold-easy-i00", "blanket-fold-01-i01"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rr = load("release_rebuild", ROOT / "scripts/robotangle/release_rebuild.py")


def npy_text(s: str, width: int = 16) -> bytes:
    """A 0-d '<U{width}' .npy (what np.save writes for a str of that dtype)."""
    head = "{'descr': '<U%d', 'fortran_order': False, 'shape': (), }" % width
    head += " " * (64 - (10 + len(head) + 1) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + len(head).to_bytes(2, "little") + head.encode() + s.encode("utf-32-le").ljust(4 * width, b"\0")


def reference(birth: str) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("actions.npy", b"rows " * 40)
        z.writestr("instance_sha.npy", npy_text(birth))
        z.writestr("meta.npy", npy_text("{}", 2))
    return out.getvalue()


def instance(d: Path, birth: str, look: str) -> dict:
    (d / "instance_assets").mkdir(parents=True)
    inst = {"state_sha": birth, "look": look, "topogym_source": {"sha256": "5" * 64},
            "simulator": {"physics_engine": "mujoco"}}
    (d / "instance.json").write_text(json.dumps(inst, indent=1))
    (d / "instance_assets" / "cloth.png").write_bytes(f"texture {look}".encode())
    return inst


def tree(td: Path, seeds: list[int]):
    """A render tree (scripts/roboarc/ of a fake RoboARC), the head's released texts, a work dir with the made instances
    (the seeds' too), and the run's arguments."""
    out, repo, work = td / "out", td / "repo", td / "work"
    rb = out / "scripts/roboarc"
    (rb / "shared").mkdir(parents=True)
    (rb / "shared/rcb_topogym.py").write_text("# tooling\n")
    (rb / "release.json").write_text(json.dumps({"bases": {}, "runtime": {}, "uipc_builds": {}}))
    (rb / "assets.json").write_text(json.dumps({"files": {}, "instances": {}, "revision": "r0"}))
    (out / "images/roboarc").mkdir(parents=True)
    (repo / "images/robotangle").mkdir(parents=True)
    (repo / "images/robotangle/Dockerfile.mujoco").write_text("RUN --mount=type=bind,from=instances,target=/tmp/mj-instances\n")
    for m in ("privileged", "standard"):
        p = repo / "scripts/robotangle/packages" / f"{OLD}-{m}"
        (p / "environment/goal").mkdir(parents=True)
        (p / "task.toml").write_text(f'[metadata]\nfamily = "blanket_fold"\nname = "{OLD}"\n')
        (p / "environment/goal/target_cam_high.png").write_bytes(b"\x89PNG seed 0")
    d = work / "instances" / NEW
    instance(d, "aaaaaaaaaaaaaaaa", "seed 0")
    births = {s: birth for s, birth in ((1, "bbbbbbbbbbbbbbbb"), (2, "cccccccccccccccc")) if s in seeds}
    for s, birth in births.items():          # made only with --appearance-seeds
        instance(d / "seeds" / str(s), birth, f"seed {s}")
    ref = td / "privileged.npz"
    ref.write_bytes(reference("aaaaaaaaaaaaaaaa"))
    a = types.SimpleNamespace(out=out, repo=repo, work=work, bench="roboarc", version="9.9.9", host_label="test",
                              no_oracle=set(), references=None, ref_paths={NEW: ref}, step_s={}, hf_revision=None,
                              appearance_seeds=seeds, goal_pictures=None, gpu="0",
                              premade={})   # RoboARC copy: its write_release names premade instances
    b = dict(family="blanket_fold", tier="easy", old=OLD, new=NEW, variation="blanket_fold_easy", released_simulator_keys=[],
             simulator_overrides={}, render_host=False, frozen_assets=False, released_task_parameters=False,
             grasp_weld_links=None, package_text=None, task="blanket_fold", template=None, design=None, scene_seed=0,
             gripper_space="native")
    return a, b, births


def write(a, b):
    rr.write_release(a, [b], "0" * 40, {"source": {"sha256": "5" * 64}}, {}, {"reference": "noinst"},
                     {NEW: {"gpu_physics": False}})


def files(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


class Restamp(unittest.TestCase):
    def test_only_the_birth_changes(self):
        ref = reference("aaaaaaaaaaaaaaaa")
        out = rr.restamp(ref, "0123456789abcdef")
        zi, zo = zipfile.ZipFile(io.BytesIO(ref)), zipfile.ZipFile(io.BytesIO(out))
        self.assertEqual(zi.namelist(), zo.namelist())
        for n in zi.namelist():
            if n != "instance_sha.npy":
                self.assertEqual(zi.read(n), zo.read(n), n)
        self.assertEqual(zo.read("instance_sha.npy"), npy_text("0123456789abcdef"))
        self.assertEqual(rr.npz_text(io.BytesIO(out), "instance_sha.npy"), "0123456789abcdef")
        self.assertEqual(zipfile.ZipFile(io.BytesIO(rr.restamp(ref, "aaaaaaaaaaaaaaaa"))).read("instance_sha.npy"),
                         zi.read("instance_sha.npy"))
        self.assertEqual(rr.npz_text(io.BytesIO(rr.restamp(ref, "abc")), "instance_sha.npy"), "abc")
        with self.assertRaises(SystemExit):
            rr.restamp(ref, "0" * 17)
        try:
            import numpy as np
        except ImportError:
            return
        with np.load(io.BytesIO(out)) as z:
            self.assertEqual(str(z["instance_sha"]), "0123456789abcdef")


class WriteRelease(unittest.TestCase):
    def test_seed_files(self):
        with tempfile.TemporaryDirectory() as td:
            a, b, births = tree(Path(td), [1, 2])
            rb = a.out / "scripts/roboarc"
            (rb / "oracles" / NEW).mkdir(parents=True)                 # a previous release's seed 3
            (rb / "oracles" / NEW / "privileged.s3.npz").write_bytes(b"old")
            write(a, b)
            pkg = rb / "packages" / f"{NEW}-privileged"
            lock0 = json.loads((pkg / "runtime_binding.upstream.json").read_text())
            ref = (rb / "oracles" / NEW / "privileged.npz").read_bytes()
            assets = json.loads((rb / "assets.json").read_text())
            for s, birth in births.items():
                sd = a.work / "instances" / NEW / "seeds" / str(s)
                si = json.loads((sd / "instance.json").read_text())
                lock = json.loads((pkg / f"runtime_binding.s{s}.upstream.json").read_text())
                self.assertEqual(lock, dict(lock0, instance_sha256=rr.canonical(si), asset_files={
                    "instance_assets/cloth.png": hashlib.sha256(f"texture seed {s}".encode()).hexdigest()}))
                self.assertNotEqual(lock["instance_sha256"], lock0["instance_sha256"])
                self.assertEqual((rb / "oracles" / NEW / f"privileged.s{s}.npz").read_bytes(), rr.restamp(ref, birth))
                self.assertEqual(assets["instances"][NEW]["seeds"][str(s)], {"instance_sha256": rr.canonical(si)})
                self.assertIn(f"instances/{NEW}/seeds/{s}/instance_assets/cloth.png", assets["files"])
            self.assertFalse((rb / "oracles" / NEW / "privileged.s3.npz").exists())
            self.assertFalse((rb / "oracles" / NEW / "standard.s1.npz").exists())

    def test_no_seeds_writes_what_it_wrote_before(self):
        with tempfile.TemporaryDirectory() as td:
            got = {}
            for seeds in ([], [1, 2]):           # the same paths (release.json names the work dir)
                a, b, _ = tree(Path(td) / "t", seeds)
                write(a, b)
                got[len(seeds)] = files(a.out)
                shutil.rmtree(Path(td) / "t")
            seed_file = lambda k: "/runtime_binding.s" in k or ".s1.npz" in k or ".s2.npz" in k  # noqa: E731
            self.assertEqual(set(got[0]), {k for k in got[2] if not seed_file(k)})
            for k, v in got[0].items():
                if k != "scripts/roboarc/assets.json":
                    self.assertEqual(v, got[2][k], k)
            a0, a2 = (json.loads(got[n]["scripts/roboarc/assets.json"]) for n in (0, 2))
            self.assertTrue(all("seeds" not in r for r in a0["instances"].values()))
            del a2["instances"][NEW]["seeds"]
            a2["files"] = {k: v for k, v in a2["files"].items() if "/seeds/" not in k}
            self.assertEqual(a0, a2)


class GoalPictures(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.a, self.b, self.births = tree(Path(self.td.name), [1, 2])
        write(self.a, self.b)
        self.rb = self.a.out / "scripts/roboarc"

    def tearDown(self):
        self.td.cleanup()

    def pictures(self, root: Path, birth_of=None):
        for s, birth in self.births.items():
            d = root / NEW / "seeds" / str(s)
            d.mkdir(parents=True, exist_ok=True)
            (d / "target_cam_high.png").write_bytes(f"\x89PNG seed {s}".encode())
            (d / "start_cam_high.png").write_bytes(b"review only")
            (d / "target_images.json").write_text(json.dumps({"state_sha": (birth_of or {}).get(s, birth)}))

    def placed(self):
        assets = json.loads((self.rb / "assets.json").read_text())
        for s in self.births:
            want = f"\x89PNG seed {s}".encode()
            for m in ("privileged", "standard"):
                goal = self.rb / "packages" / f"{NEW}-{m}" / "tests/seeds" / str(s) / "goal"
                self.assertEqual(files(goal), {"target_cam_high.png": want}, f"{m} seed {s}")
            self.assertEqual(assets["instances"][NEW]["seeds"][str(s)]["goal"],
                             {"target_cam_high.png": hashlib.sha256(want).hexdigest()})

    def test_from_goal_pictures(self):
        self.a.goal_pictures = Path(self.td.name) / "rendered"
        self.pictures(self.a.goal_pictures)
        rr.seed_goal_pictures(self.a, [self.b], "img:1", ())
        self.placed()

    def test_missing_picture_or_other_birth_stops(self):
        self.a.goal_pictures = Path(self.td.name) / "rendered"
        self.pictures(self.a.goal_pictures, birth_of={2: "dddddddddddddddd"})
        with self.assertRaisesRegex(SystemExit, "seed 2: .*names dddddddddddddddd.*render them with: docker run"):
            rr.seed_goal_pictures(self.a, [self.b], "img:1", ())
        self.pictures(self.a.goal_pictures)
        (self.a.goal_pictures / NEW / "seeds/1/target_cam_high.png").unlink()
        with self.assertRaisesRegex(SystemExit, r"seed 1: .*missing \['target_cam_high.png'\]"):
            rr.seed_goal_pictures(self.a, [self.b], "img:1", ())

    def test_rendered_in_the_release_image(self):
        (self.rb / "goal_images").mkdir()
        (self.rb / "goal_images/fold_target_images.py").write_text("# renderer\n")
        cmds, run = [], rr.run

        def fake(cmd, log=None, cwd=None, check=True):   # the render: the pictures of the instance it was given
            cmds.append(cmd)
            s = int(cmd[-3].rsplit("/", 1)[1])
            self.pictures(Path(self.td.name) / "fake")
            shutil.copytree(Path(self.td.name) / "fake" / NEW / "seeds" / str(s),
                            self.a.work / cmd[-1].removeprefix("/work/"), dirs_exist_ok=True)
            return ""
        rr.run = fake
        try:
            rr.seed_goal_pictures(self.a, [self.b], "img:1", (("OPENBLAS_CORETYPE", "Haswell"),))
        finally:
            rr.run = run
        self.placed()
        self.assertEqual(len(cmds), 2)
        c = cmds[0]
        self.assertEqual(c[:2], ["docker", "run"])
        self.assertIn("device=0", c)
        self.assertIn("OPENBLAS_CORETYPE=Haswell", c)
        self.assertIn(f"{self.a.work / 'instances'}:/instances:ro", c)
        self.assertEqual(c[c.index("img:1") + 1:c.index("img:1") + 3], ["sh", "-c"])
        self.assertEqual(c[-3:], [f"/instances/{NEW}/seeds/1", "fold_target_images.py", f"/work/goal/{NEW}/seeds/1"])
        rr.run = lambda *x, **k: self.fail("rendered again")   # pictures already rendered in this work dir: used again
        try:
            rr.seed_goal_pictures(self.a, [self.b], "img:1", ())
        finally:
            rr.run = run

    def test_no_renderer_stops(self):
        with self.assertRaisesRegex(SystemExit, "need a renderer"):
            rr.seed_goal_pictures(self.a, [self.b], "img:1", ())
        self.b["task"] = "rope_threading"
        with self.assertRaisesRegex(SystemExit, "no GOAL_RENDERERS entry for rope_threading"):
            rr.seed_goal_pictures(self.a, [self.b], "img:1", ())

    def test_bases_without_pictures_untouched(self):
        for m in ("privileged", "standard"):
            shutil.rmtree(self.rb / "packages" / f"{NEW}-{m}" / "environment/goal")
        before = files(self.a.out)
        rr.seed_goal_pictures(self.a, [self.b], "img:1", ())
        self.assertEqual(files(self.a.out), before)


class AssetsPin(unittest.TestCase):
    def test_pin_keeps_seeds_and_their_goal_pins(self):
        assets = load("assets_pin", ROOT / "scripts/robotangle/assets.py")
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "dataset"
            (d / "image").mkdir(parents=True)
            (d / "image/release-abc.tar.gz").write_bytes(b"tar")
            instance(d / "instances" / NEW, "aaaaaaaaaaaaaaaa", "seed 0")
            for s in (1, 2):
                instance(d / "instances" / NEW / "seeds" / str(s), "bbbbbbbbbbbbbbb" + str(s), f"seed {s}")
            canon = lambda p: assets.canonical_sha(p)  # noqa: E731
            s1, s2 = (canon(d / "instances" / NEW / "seeds" / str(s) / "instance.json") for s in (1, 2))
            assets.PIN = Path(td) / "assets.json"
            assets.PIN.write_text(json.dumps({"instances": {NEW: {"instance_sha256": "x", "seeds": {
                "1": {"instance_sha256": s1, "goal": {"target_cam_high.png": "g1"}},
                "2": {"instance_sha256": "an earlier seed 2", "goal": {"target_cam_high.png": "g2"}}}}}}))
            pin = assets.pin_from(d, "rev")
            self.assertEqual(pin["instances"][NEW]["seeds"], {"1": {"instance_sha256": s1, "goal": {"target_cam_high.png": "g1"}},
                                                              "2": {"instance_sha256": s2}})
            self.assertIn(f"instances/{NEW}/seeds/2/instance.json", pin["files"])
            self.assertEqual(assets.check(d, pin), [])
            (d / "instances" / NEW / "seeds/1/instance.json").write_text("{}")
            self.assertTrue(any(f"instances/{NEW}/seeds/1/instance.json: canonical" in x for x in assets.check(d, pin)))


class StageSeedFile(unittest.TestCase):
    def test_seed_file(self):
        si = load("stage_instance", ROOT / "images/robotangle/stage_instance.py")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            si.SRC = td / "instances"
            task = td / "task"
            task.mkdir()
            for s, name in ((0, "runtime_binding.json"), (1, "runtime_binding.s1.json")):
                inst = instance(si.SRC / NEW / ("seeds/1" if s else ""), "b" * 16, f"seed {s}")
                (task / name).write_text(json.dumps({"instance_sha256": rr.canonical(inst), "asset_files": {
                    "instance_assets/cloth.png": hashlib.sha256(f"texture seed {s}".encode()).hexdigest()}}))
            for seed, look in ((None, "seed 0"), ("0\n", "seed 0"), ("1\n", "seed 1")):
                (task / "seed").unlink(missing_ok=True)
                if seed is not None:
                    (task / "seed").write_text(seed)
                dest = td / f"dest{seed!r}"
                self.assertEqual(si.main([NEW, str(dest), "--binding", str(task / "runtime_binding.json"), "--seed-file",
                                          str(task / "seed")]), 0)
                self.assertEqual(json.loads((dest / "instance.json").read_text())["look"], look)
            (task / "seed").write_text("2\n")
            self.assertEqual(si.main([NEW, str(td / "dest2"), "--binding", str(task / "runtime_binding.json"), "--seed-file",
                                      str(task / "seed")]), 1)


if __name__ == "__main__":
    unittest.main()
