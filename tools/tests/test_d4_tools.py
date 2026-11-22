"""Tests of the appearance-seed evidence tools that run without the image (standard library only):

    python3 tools/tests/test_d4_tools.py

- tools/release/state_parts.py compare: equal but `task` passes, any other part differing (or a part missing) fails;
- tools/release/d4_table.py: one row per `D4 {...}` line, the pass count, bases without a result named;
- tools/release/check_seed_physics.py run (the replays faked): a seed passes when it ends as seed 0 but for the task's
  instance id, and fails on any other part, on a task part differing beyond its instance id, or on a seed derived from
  another seed 0; harbor-list: every changed base, one base per other family (the first with a reference), nop only for
  a base without a reference.
"""
import hashlib, importlib.util, json, pathlib, shutil, subprocess, sys, tempfile, unittest, zipfile

REL = pathlib.Path(__file__).resolve().parents[2] / "tools" / "release"


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


class Tools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def parts(self, name, **parts):
        base = dict(world="w0", robot="r0", targets="t0", state="s0", task="k0")
        base.update(parts)
        p = self.d / name
        p.write_text(json.dumps({"kind": "end", "replays": [{"parts": base, "state_sha": "x", "success": True}]}))
        return p

    def test_compare(self):
        a = self.parts("a.json")
        self.assertEqual(run(REL / "state_parts.py", "compare", a, self.parts("b.json", task="k1")).returncode, 0)
        p = run(REL / "state_parts.py", "compare", a, self.parts("c.json", world="w1"))
        self.assertEqual(p.returncode, 1)
        self.assertIn("world", json.loads(p.stdout)["parts_differing"])
        c = self.d / "birth.json"
        c.write_text(json.dumps({"kind": "birth", "parts": dict(world="w0", robot="r0", targets="t0", state="s0")}))
        self.assertEqual(run(REL / "state_parts.py", "compare", a, c).returncode, 1)     # a part missing

    def test_table(self):
        out = self.d / "out"
        out.mkdir()
        row = dict(base="b-01-i01", seed=1, birth="ab", a_files=True, a_fields=True, a_model=True, b_strict=True,
                   c_birth=True, c_birth_parts_differing=["task"], d_restamp=True, ok=True)
        (out / "b-01-i01.log").write_text("noise\nD4 " + json.dumps(row) + "\nD4 " + json.dumps(dict(row, seed=2, ok=False)) + "\nrc=0\n")
        (out / "c-01-i01.log").write_text("Traceback\nrc=1\n")
        p = run(REL / "d4_table.py", out)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("1 / 2 seed instances pass", p.stdout)
        self.assertIn("c-01-i01", p.stdout)
        self.assertEqual(len(json.loads((out / "d4.json").read_text())), 2)


def npy16(s: str) -> bytes:
    """A 0-d '<U16' .npy (a reference's instance_sha.npy)."""
    head = "{'descr': '<U16', 'fortran_order': False, 'shape': (), }"
    head += " " * (64 - (10 + len(head) + 1) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + len(head).to_bytes(2, "little") + head.encode() + s.encode("utf-32-le").ljust(64, b"\0")


def canonical(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class SeedPhysics(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("check_seed_physics", REL / "check_seed_physics.py")
        self.csp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.csp)
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def base(self, derived_from=None):
        b = self.d / "inst" / "x-01-i01"
        shutil.rmtree(b, ignore_errors=True)
        births = {0: "a" * 16, 1: "b" * 16, 2: "c" * 16}
        for s, birth in births.items():
            d = b if s == 0 else b / "seeds" / str(s)
            (d / "instance_assets").mkdir(parents=True)
            inst = dict(state_sha=birth)
            if s:
                inst["derived_from"] = dict(instance_sha256=derived_from or canonical(dict(state_sha=births[0])))
            (d / "instance.json").write_text(json.dumps(inst))
        ref = self.d / "privileged.npz"
        with zipfile.ZipFile(ref, "w") as z:
            z.writestr("actions.npy", b"rows")
            z.writestr("instance_sha.npy", npy16(births[0]))
        return b, ref, births

    def check(self, ends, derived_from=None):
        """run() with replays faked: ends[seed] = (parts, task_but_instance_id) of its end state."""
        b, ref, births = self.base(derived_from)
        seen = {}

        def replay(task, instance, traj, fresh, tooling):
            inst = json.loads(instance.read_text())
            s = next(k for k, v in births.items() if v == inst["state_sha"])
            with zipfile.ZipFile(traj) as z:          # the seed's own birth stamped on its reference
                seen[s] = z.read("instance_sha.npy") == npy16(inst["state_sha"])
            parts, task = ends[s]
            one = dict(success=True, rows=4, birth=inst["state_sha"], instance_exact=True, state_sha=f"end{s}",
                       parts=parts, task_but_instance_id=task)
            return dict(replays=[one, dict(one)], all_success=True, deterministic=True)
        self.csp.replay = replay
        args = self.csp.argparse.Namespace(base=b, task="x", reference=ref, seeds="1,2", fresh=2, tooling="/t",
                                           out=self.d / "out.json")
        rc = self.csp.run(args)
        self.assertEqual(seen, {0: True, 1: True, 2: True})
        return rc, json.loads((self.d / "out.json").read_text())

    def test_identity(self):
        p = dict(world="w", robot="r", targets="t", state="s", coverage="c", task="k0")
        rc, out = self.check({0: (p, "t0"), 1: (dict(p, task="k1"), "t0"), 2: (dict(p, task="k2"), "t0")})
        self.assertEqual((rc, out["ok"]), (0, True))
        rc, out = self.check({0: (p, "t0"), 1: (dict(p, task="k1"), "t0"), 2: (dict(p, world="w2", task="k2"), "t0")})
        self.assertEqual((rc, out["checks"]["identity"]), (1, False))
        self.assertEqual(out["seeds_detail"]["2"]["parts_differing"], ["task", "world"])
        rc, out = self.check({0: (p, "t0"), 1: (dict(p, task="k1"), "t1"), 2: (dict(p, task="k2"), "t0")})
        self.assertEqual((rc, out["checks"]["identity"]), (1, False))      # the task part beyond its instance id
        rc, out = self.check({s: (dict(p, task=f"k{s}"), "t0") for s in (0, 1, 2)}, derived_from="0" * 64)
        self.assertEqual((rc, out["checks"]["derived"], out["checks"]["identity"]), (1, False, True))

    def test_harbor_list(self):
        tree = self.d / "tree"
        (tree / "scripts/roboarc").mkdir(parents=True)
        (tree / "images/roboarc").mkdir(parents=True)
        (tree / "images/roboarc/VERSION").write_text("9.9.9\n")
        rel = {"a-01-i01": dict(family="fa", oracle="none"), "a-02-i01": dict(family="fa"),
               "b-01-i01": dict(family="fb", gpu_physics=True), "b-02-i01": dict(family="fb"),
               "c-01-i01": dict(family="fc", oracle="none")}
        (tree / "scripts/roboarc/release.json").write_text(json.dumps(dict(bases=rel)))
        p = run(REL / "check_seed_physics.py", "harbor-list", "--tree", tree, "--changed", "fb", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        got = json.loads(p.stdout)
        self.assertEqual(got["bases"], ["a-02-i01", "b-01-i01", "b-02-i01", "c-01-i01"])
        self.assertEqual((got["nop_only"], got["gpu_bases"]), (["c-01-i01"], ["b-01-i01"]))
        self.assertEqual(len(got["trials"]), 3 * 2 * (3 * 2 + 1))
        p = run(REL / "check_seed_physics.py", "harbor-list", "--tree", tree, "--changed", "b-02-i01", "--out", "/o")
        self.assertIn("# bases: a-02-i01 b-02-i01 c-01-i01\n", p.stdout)    # fb is covered by its changed base
        self.assertNotIn("--gpu-bases", p.stdout)
        p = run(REL / "check_seed_physics.py", "harbor-list", "--tree", tree, "--changed", "fb", "--out", "/o")
        self.assertIn("--agents oracle,nop --gpu-bases b-01-i01 a-02-i01 b-01-i01 b-02-i01\n", p.stdout)
        self.assertIn("--run /o/s2/harbor_nop --agents nop c-01-i01", p.stdout)
        self.assertNotEqual(run(REL / "check_seed_physics.py", "harbor-list", "--tree", tree, "--changed", "zz").returncode, 0)


if __name__ == "__main__":
    unittest.main()
