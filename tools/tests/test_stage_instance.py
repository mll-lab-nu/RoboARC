"""Tests for source/images/roboarc/stage_instance.py (the image's instance stager) and its --seed-file, on a fake
/opt/robotangle/instances tree: no image, standard library only.

    python3 tools/tests/test_stage_instance.py

- no --seed-file, and a seed file of 0: the base's own instance and runtime_binding.json, as before seeds;
- a seed S > 0: the instance from seeds/<S>/ and runtime_binding.s<S>.json, which becomes runtime_binding.json; no other
  seed's binding is left in the destination;
- fail closed: a seed without its instance or its binding, a seed file that is not a non-negative integer, an asset that
  is not the bound one.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import shutil
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("stage_instance", REPO / "source/images/roboarc/stage_instance.py")
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def instance(d: pathlib.Path, colour: str) -> dict:
    """A frozen instance in d (instance.json + instance_assets/) and its binding."""
    (d / "instance_assets" / "mujoco").mkdir(parents=True)
    inst = {"simulator": {"physics_engine": "mujoco"}, "appearance": colour}
    (d / "instance.json").write_text(json.dumps(inst))
    (d / "instance_assets" / "mujoco" / "scene.xml").write_text("<mujoco/>")
    (d / "instance_assets" / "visual.txt").write_text(colour)
    canon = json.dumps(inst, sort_keys=True, separators=(",", ":"), allow_nan=False)
    files = {"instance_assets/mujoco/scene.xml": hashlib.sha256(b"<mujoco/>").hexdigest(),
             "instance_assets/visual.txt": hashlib.sha256(colour.encode()).hexdigest()}
    return {"schema_version": 1, "instance_sha256": hashlib.sha256(canon.encode()).hexdigest(), "asset_files": files}


class StageInstance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.tmp.name)
        self.src = root / "opt" / "instances"
        b = self.src / "claws-03-i01"
        self.locks = {0: instance(b, "brown")}
        for s, colour in ((1, "copper"), (2, "steel")):
            self.locks[s] = instance(b / "seeds" / str(s), colour)
        stage.SRC = self.src
        self.task = root / "task"            # what a task dir copies into the image: bindings + seed file
        self.task.mkdir()
        (self.task / "runtime_binding.json").write_text(json.dumps(self.locks[0]))
        for s in (1, 2):
            (self.task / f"runtime_binding.s{s}.json").write_text(json.dumps(self.locks[s]))
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def run_stage(self, *args):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = stage.main(list(args))
        return rc, err.getvalue()

    def seed_file(self, text):
        f = self.root / "seed"
        f.write_text(text)
        return str(f)

    def staged(self, dest):
        return (json.loads((dest / "runtime_binding.json").read_text()), (dest / "instance_assets/visual.txt").read_text(),
                sorted(p.name for p in dest.glob("runtime_binding.s*.json")))

    def test_no_seed_file_and_seed_0(self):
        for extra in ([], ["--seed-file", self.seed_file("0\n")]):
            dest = self.root / f"dest{len(extra)}"
            rc, err = self.run_stage("claws-03-i01", str(dest), "--binding", str(self.task / "runtime_binding.json"), *extra)
            self.assertEqual(rc, 0, err)
            self.assertEqual(self.staged(dest), (self.locks[0], "brown", []))

    def test_seed_2_in_place(self):
        """A privileged task: the bindings and the seed file are copied into <dest> first (no --binding)."""
        dest = self.root / "app"
        shutil.copytree(self.task, dest)
        rc, err = self.run_stage("claws-03-i01", str(dest), "--seed-file", self.seed_file("2\n"), "--purge")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.staged(dest), (self.locks[2], "steel", []))
        self.assertFalse(self.src.exists())                    # --purge

    def test_seed_1_with_binding_dir(self):
        """A standard sim service: --binding names /srv/runtime_binding.json; the seed's binding is beside it."""
        dest = self.root / "inst"
        rc, err = self.run_stage("claws-03-i01", str(dest), "--binding", str(self.task / "runtime_binding.json"),
                                 "--seed-file", self.seed_file("1"))
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.staged(dest), (self.locks[1], "copper", []))

    def test_fail_closed(self):
        b = str(self.task / "runtime_binding.json")
        rc, err = self.run_stage("claws-03-i01", str(self.root / "d1"), "--binding", b, "--seed-file", self.seed_file("3"))
        self.assertEqual(rc, 1)
        self.assertIn("seed 3", err)                                       # no instance of seed 3
        (self.task / "runtime_binding.s2.json").unlink()
        rc, err = self.run_stage("claws-03-i01", str(self.root / "d2"), "--binding", b, "--seed-file", self.seed_file("2"))
        self.assertEqual(rc, 1)
        self.assertIn("has no binding", err)                               # no binding of seed 2
        for bad in ("", "x", "-1", "1.0"):
            rc, err = self.run_stage("claws-03-i01", str(self.root / "d3"), "--binding", b, "--seed-file", self.seed_file(bad))
            self.assertEqual(rc, 1, bad)
        (self.src / "claws-03-i01/seeds/1/instance_assets/visual.txt").write_text("brown")   # not the bound asset
        rc, err = self.run_stage("claws-03-i01", str(self.root / "d4"), "--binding", b, "--seed-file", self.seed_file("1"))
        self.assertEqual(rc, 1)
        self.assertIn("instance_assets differ", err)
        self.assertEqual(self.run_stage("claws-03-i01", str(self.root / "d5"), "--seed-file", self.seed_file("1"))[0], 1)


if __name__ == "__main__":
    unittest.main()
