"""Tests for scripts/ablations/cond_info.py, the information conditions C00 / C10 / C0G / C1G (E1, E2, and E3's C00), on
one committed Standard task of every family E1, E2 and E3 use: each edit applies, the service compiles, the text says
what the agent gets and nothing it does not, no world coordinates or simulator world frame reach the agent, and every
variant's DIGEST is the locked one (scripts/ablations/digests/). Standard library only:

    python3 scripts/ablations/tests/test_cond_info.py
"""
import pathlib
import re
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts" / "ablations"), str(REPO / "scripts")]
import experiments  # noqa: E402
import make_variants as mv  # noqa: E402

BASES = sorted({experiments.full(b) for b in experiments.A15 + experiments.E1_BASES}
               | {experiments.full(b) for b in experiments.E3_BASES})


def one_per_family(bases):
    seen, out = set(), []
    for b in bases:
        fam = re.sub(r"-\d+-i01$", "", b)
        if fam not in seen:
            seen.add(fam); out.append(b)
    return out


class Conditions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = pathlib.Path(cls.tmp.name)
        cls.built = {}
        rigid = one_per_family(sorted({experiments.full(b) for b in experiments.A15 + experiments.E1_BASES}))
        deform = one_per_family(sorted(experiments.full(b) for b in experiments.E3_BASES))
        for b in rigid:
            for t in ("q1c00", "q1c10", "q1c0g", "q1c1g"):
                cls.built[(b, t)] = mv.make_one(mv.REGISTRY[t], b, "standard", REPO / "tasks", cls.out)
        for b in deform:                          # E3's control
            cls.built[(b, "q1c00")] = mv.make_one(mv.REGISTRY["q1c00"], b, "standard", REPO / "tasks", cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def files(self, b, t):
        d = self.built[(b, t)][0]
        return ((d / "environment/sim/server.py").read_text(), (d / "instruction.md").read_text(),
                (d / "environment/rcb_client.py").read_text())

    def base_files(self, b):
        d = REPO / "tasks" / f"{b}-standard"
        return ((d / "environment/sim/server.py").read_text(), (d / "instruction.md").read_text(),
                (d / "environment/rcb_client.py").read_text())

    def test_services_compile(self):
        for (b, t), (d, _) in self.built.items():
            compile((d / "environment/sim/server.py").read_text(), f"{d.name}/server.py", "exec")
            compile((d / "environment/rcb_client.py").read_text(), f"{d.name}/rcb_client.py", "exec")

    def test_c00_agent_sees_nothing_new(self):
        for (b, t) in self.built:
            if t != "q1c00":
                continue
            srv, ins, cli = self.files(b, t)
            bsrv, bins, bcli = self.base_files(b)
            self.assertEqual(ins, bins, b)
            self.assertEqual(cli, bcli, b)
            self.assertEqual(srv.count("# probing v2 log patch"), 3, b)   # the log only

    def test_c10_poses_robot_frame(self):
        for (b, t) in self.built:
            if t != "q1c10":
                continue
            srv, ins, _ = self.files(b, t)
            self.assertIn('"objects": objects_robot()}', srv)
            self.assertIn("p, q2 = to_robot(", srv)          # never the simulator's world frame
            self.assertIn('meta["objects"]', ins)
            self.assertNotRegex(ins, r"(?i)no object poses|nothing reports their poses")
            self.assertNotIn("GET /geometry", srv)
            self.assertNotIn('"state": state_robot()', srv)

    def test_c0g_shapes_without_poses(self):
        for (b, t) in self.built:
            if t != "q1c0g":
                continue
            srv, ins, cli = self.files(b, t)
            self.assertIn('if self.path == "/geometry"', srv)
            self.assertIn("def geometry(self) -> dict:", cli)
            body = srv.split("def geometry_parts", 1)[1].split("GEOMETRY_BODY", 1)[0]
            self.assertNotIn("to_robot(", body)              # each part in its own frame: no position of the scene
            self.assertNotIn("objects_robot()", srv)
            self.assertNotIn("state_robot()", srv)
            self.assertNotIn('meta["objects"]', ins)
            self.assertRegex(ins, r"Nothing tells you where the\s+objects are")

    def test_c1g_poses_and_shapes_no_state(self):
        for (b, t) in self.built:
            if t != "q1c1g":
                continue
            srv, ins, cli = self.files(b, t)
            self.assertIn('"objects": objects_robot()}', srv)
            self.assertIn('if self.path == "/geometry"', srv)
            self.assertNotIn('"state": state_robot()', srv)
            self.assertNotIn('meta["state"]', ins)
            self.assertIn("**Object poses and part shapes are given (this variant).**", ins)

    def test_no_world_coordinates_in_texts(self):
        """No table height, no robot base position and no coordinate tuple in what the agent reads."""
        for (b, t), (d, _) in self.built.items():
            ins = (d / "instruction.md").read_text()
            self.assertNotRegex(ins, r"(?i)table (top )?height (is|of|=)\s*-?\d|world\s+[xyz]\s*=", d.name)
            self.assertNotRegex(ins, r"\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*\)", d.name)

    def test_locked_digests(self):
        for (b, t), (d, digest) in self.built.items():
            lock = mv.load_lock(t).get(d.name)
            if lock is not None:
                self.assertEqual(digest, lock, d.name)


if __name__ == "__main__":
    unittest.main()
