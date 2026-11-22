"""Tests for scripts/ablations/cond_state_deform.py, the CS (`cstate`, "+state") condition of E3, on every E3 base: each
edit applies, the service and the client compile, the rigid parts' geometry code is C1G's, the text says what the agent
gets and nothing it does not, the leak check is clean and catches what it should, the build is deterministic and every
DIGEST is the locked one (scripts/ablations/digests/cstate.tsv). The service's deformable-state code itself runs against
small hand-built models when numpy is there (a chain authored with its joints at the far end, as the cord's second
half; a cloth grid); against the real scenes it is checked by scripts/ablations/check_cstate.py. Standard library
(numpy optional):

    python3 scripts/ablations/tests/test_cond_state_deform.py
"""
import pathlib
import re
import shutil
import sys
import tempfile
import types
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts" / "ablations"), str(REPO / "scripts")]
import cond_info as ci  # noqa: E402
import cond_state_deform as cs  # noqa: E402
import experiments  # noqa: E402
import make_variants as mv  # noqa: E402

try:
    import numpy as np
except ImportError:            # the laptop's plain python: the model tests are skipped
    np = None

BASES = sorted(experiments.full(b) for b in experiments.E3_BASES)


class Build(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = pathlib.Path(cls.tmp.name)
        cls.built = {b: mv.make_one(mv.REGISTRY["cstate"], b, "standard", REPO / "tasks", cls.out) for b in BASES}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_e3_base(self):
        self.assertEqual(len(BASES), 18)
        fams = {cs.family_of(d) for d, _ in self.built.values()}
        self.assertEqual(fams, set(cs.FAMILIES))

    def test_leak_check_clean(self):
        for b, (d, _) in self.built.items():
            self.assertEqual(cs.leak_check(d), [], b)

    def test_digests_locked(self):
        lock = mv.load_lock("cstate")
        for b, (d, digest) in self.built.items():
            self.assertEqual(lock.get(d.name), digest, f"{d.name}: DIGEST not the locked one")

    def test_deterministic(self):
        b = BASES[0]
        with tempfile.TemporaryDirectory() as t:
            d2, digest2 = mv.make_one(mv.REGISTRY["cstate"], b, "standard", REPO / "tasks", pathlib.Path(t))
            d1, digest1 = self.built[b]
            self.assertEqual(digest1, digest2)
            for f in sorted(p.relative_to(d1) for p in d1.rglob("*") if p.is_file()):
                self.assertEqual((d1 / f).read_bytes(), (d2 / f).read_bytes(), str(f))

    def test_only_service_client_text_and_toml_change(self):
        for b, (d, _) in self.built.items():
            src = REPO / "tasks" / f"{b}-standard"
            changed = sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()
                             and (not (src / p.relative_to(d)).is_file() or (src / p.relative_to(d)).read_bytes() != p.read_bytes()))
            self.assertEqual(changed, ["DIGEST", "environment/rcb_client.py", "environment/sim/server.py", "instruction.md",
                                       "task.toml"], b)

    def test_service(self):
        for b, (d, _) in self.built.items():
            fam = cs.family_of(d)
            t = (d / "environment/sim/server.py").read_text()
            compile(t, "server.py", "exec")
            compile((d / "environment/rcb_client.py").read_text(), "rcb_client.py", "exec")
            for piece in (ci.LOG_SUBMIT_NEW, ci.LOG_OBS_NEW, ci.OBJ_FN, ci.FULL_GET_NEW, cs.STATE_META_NEW, cs.STATE_SPEC_NEW,
                          cs.STATE_FRAME_NEW, cs.server_block(fam)):
                self.assertEqual(t.count(piece), 1, f"{b}: {piece[:60]!r}")
            self.assertNotIn("state_robot()}", t)
            # the order: objects_robot, the C1G geometry with the cstate block, then meta_common
            self.assertLess(t.index("\ndef objects_robot()"), t.index("\ndef geometry_robot()"))
            self.assertLess(t.index("\ndef deformables_robot()"), t.index("\nGEOMETRY_BODY = "))
            self.assertLess(t.index("\nGEOMETRY_BODY = "), t.index("\ndef meta_common("))

    def test_geometry_is_c1g_for_rigid_parts(self):
        """Undo the three cstate edits of the C1G block: what is left is C1G's block, character for character."""
        for fam in cs.FAMILIES:
            t = cs.server_block(fam)
            deform = repr(dict(sorted(cs.FAMILIES[fam][0].items())))
            t = t.replace(cs.GEO_LOOP_NEW, cs.GEO_LOOP_OLD).replace(cs.GEO_REGION_NEW, cs.GEO_REGION_OLD)
            t = t.replace(cs.STATE_FN.replace("@DEFORMABLE@", deform).lstrip("\n") + "\n" + cs.GEO_BODY_NEW, cs.GEO_BODY_OLD)
            self.assertEqual(t, ci._geo_only_fn(), fam)

    def test_instruction(self):
        for b, (d, _) in self.built.items():
            fam = cs.family_of(d)
            (name, kind), = cs.FAMILIES[fam][0].items()
            noun = cs.FAMILIES[fam][1]
            ins = (d / "instruction.md").read_text()
            self.assertEqual(ins.count("### Part shapes and deformable state (this variant)"), 1, b)
            self.assertIn(f"**Object poses, part shapes and the {noun}'s state are given (this variant).**", ins)
            self.assertIn(f'`meta["deformables"]["{name}"]`', ins)
            self.assertIn("grid_shape" if kind == "cloth" else "capsule axis runs from point k to point k + 1", ins)
            self.assertEqual(re.findall(cs.LEFT, ins), [], b)
            self.assertNotIn('meta["state"]', ins)
            for placeholder in ("@NOUN@", "@NAME@", "@FAMILY@", "@DEFORMABLE@"):
                self.assertNotIn(placeholder, ins, b)
            base_ins = (REPO / "tasks" / f"{b}-standard" / "instruction.md").read_text()
            # everything before the Rules and after the section is the base task's
            self.assertEqual(ins[:ins.index("## Rules")], base_ins[:base_ins.index("## Rules")], b)
            self.assertEqual(ins[ins.index("## External material"):], base_ins[base_ins.index("## External material"):], b)

    def test_toml(self):
        for b, (d, _) in self.built.items():
            toml = (d / "task.toml").read_text()
            self.assertIn('ablation = "cstate"', toml)
            self.assertIn("object poses, part shapes and deformable state given", toml)
            self.assertIn(f'name = "roboarc/{d.name}"', toml)

    def test_leak_check_catches(self):
        b = "roboarc-rope-threading-06-i01"
        d0, _ = self.built[b]
        cases = {
            "reads the reward": ("environment/sim/server.py", "def deformables_robot() -> dict:\n",
                                 "def deformables_robot() -> dict:\n    _ = S._reward\n"),
            "reads the task state": ("environment/sim/server.py", "    w = S.env._world; d = w.data; out = {}\n",
                                     "    w = S.env._world; d = w.data; out = {}; _ = S.env._task.task_state\n"),
            "world frame": ("environment/sim/server.py", "@T_RW", ""),
            "wrong deformable": ("environment/sim/server.py", "DEFORMABLE = {'rope': 'chains'}", "DEFORMABLE = {'cord': 'chains'}"),
            "no-poses rule back": ("instruction.md", "### Part shapes and deformable state (this variant)\n",
                                   "### Part shapes and deformable state (this variant)\nNothing tells you where the objects are.\n"),
            "a coordinate": ("instruction.md", "**Deformable state.**", "**Deformable state.** The ring is at (0.41, -0.12, 0.05)."),
            "meta state": ("instruction.md", "**Deformable state.**", '**Deformable state.** See meta["state"].'),
            "toml": ("task.toml", "object poses, part shapes and deformable state given", "no privileged state"),
        }
        for what, (f, old, new) in cases.items():
            with tempfile.TemporaryDirectory() as t:
                d = pathlib.Path(t) / d0.name
                shutil.copytree(d0, d)
                p = d / f
                text = p.read_text()
                if old == "@T_RW":   # every robot-frame transform of the block gone: world coordinates
                    a = text.index("# ---------------------------------------------------------------- cstate")
                    z = text.index("\nGEOMETRY_BODY = ")
                    block = text[a:z].replace("@ T_RW[:3, :3].T + T_RW[:3, 3]", "").replace("to_robot(", "(lambda p, q: (p, q))(")
                    text = text[:a] + block + text[z:]
                else:
                    self.assertEqual(text.count(old), 1, what)
                    text = text.replace(old, new)
                p.write_text(text)
                self.assertNotEqual(cs.leak_check(d), [], what)

    def test_refuses_other_tasks(self):
        with tempfile.TemporaryDirectory() as t:
            for src in (REPO / "tasks" / "roboarc-bed-bolt-01-i01-standard", REPO / "tasks" / "roboarc-rope-threading-06-i01-privileged"):
                dst = pathlib.Path(t) / src.name
                shutil.copytree(src, dst)
                with self.assertRaises(SystemExit):
                    cs.make(src, dst)


def _fake_world(np):
    """Two hand-built entities on a fake MuJoCo model: `cord` (a handle with a free joint; chain A authored joints-at-origin,
    chain B authored as the cord's second half: each link's ball joint at the far end of its capsule) and `blanket` (a 3 x 2
    grid cloth). World poses are set directly (the code reads xpos / xmat / xquat only)."""
    def quat_to_mat(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    qy = np.array([np.cos(-np.pi / 4), 0.0, np.sin(-np.pi / 4), 0.0])   # geom z -> body -x (the cord's capsules)
    # bodies: 0 world, 1 handle, 2-3 chain A, 4-5 chain B (4 child of handle, 5 child of 4)
    L = 0.02
    xpos = np.array([[0, 0, 0], [1.0, 0, 0.5], [1.0, 0, 0.5], [1.0 + L, 0, 0.5], [0, 0, 0], [0, 0, 0]], dtype=float)
    model = types.SimpleNamespace(
        body_geomadr=np.array([0, 0, 1, 2, 3, 4]), body_geomnum=np.array([0, 1, 1, 1, 1, 1]),
        body_parentid=np.array([0, 0, 1, 2, 1, 4]), body_weldid=np.array([0, 1, 1, 1, 1, 1]),
        body_jntadr=np.array([-1, 0, 1, 2, 3, 4]), body_jntnum=np.array([0, 1, 1, 1, 1, 1]),
        jnt_type=np.array([0, 1, 1, 1, 1]), jnt_pos=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [L, 0, 0], [L, 0, 0]], dtype=float),
        geom_contype=np.ones(5, int), geom_conaffinity=np.ones(5, int), geom_type=np.array([6, 3, 3, 3, 3]),
        geom_size=np.array([[0.02, 0.01, 0.005]] + [[0.002, L / 2, 0]] * 4),
        geom_pos=np.array([[0, 0, 0]] + [[L / 2, 0, 0]] * 4, dtype=float), geom_quat=np.array([[1.0, 0, 0, 0]] + [qy] * 4),
        geom_dataid=np.array([-1] * 5))
    model.geom = lambda g: types.SimpleNamespace(name=["cord::handle_box", "cord::c0", "cord::c1", "cord::c3", "cord::c2"][g])
    data = types.SimpleNamespace(xpos=xpos, xmat=np.tile(np.eye(3).reshape(1, 9), (6, 1)),
                                 xquat=np.tile(np.array([1.0, 0, 0, 0]), (6, 1)))
    # chain B: every capsule runs along +x from its body's origin and its joint is at the capsule's far end, so the chain
    # runs from the handle along -x
    data.xpos[4] = [1.0 - L, 0.1, 0.5]           # its far end (the joint with the handle) at x = 1.0
    data.xpos[5] = [1.0 - 2 * L, 0.1, 0.5]       # its far end (the joint with b0) at b0's origin
    rest = np.array([[a * 0.01, b * 0.01, 0.0] for a in range(3) for b in range(2)])
    faces = np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3]])
    cloth = types.SimpleNamespace(rest=rest, faces=faces, contact_radius_m=0.003,
                                  vertices=lambda d=None: rest + np.array([0.0, 0.0, 0.25]))
    links = [types.SimpleNamespace(idx=i, name=n) for i, n in ((1, "handle"), (2, "a0"), (3, "a1"), (4, "b0"), (5, "b1"))]
    world = types.SimpleNamespace(model=model, data=data, _ropes={}, _cloths={"blanket": cloth},
                                  read_state=lambda: {"grasp": {"config": {"targets": [
                                      {"entity": "cord", "link": "a1", "center_m": [L / 2, 0, 0], "radius_m": 0.008},
                                      {"entity": "blanket", "material_min_m": []}]}}})
    ents = {"cord": types.SimpleNamespace(links=links), "blanket": types.SimpleNamespace(links=[])}
    S = types.SimpleNamespace(env=types.SimpleNamespace(_world=world), _entities=lambda: ents)
    mj = types.ModuleType("mujoco")
    mj.mjtGeom = types.SimpleNamespace(mjGEOM_SPHERE=2, mjGEOM_CAPSULE=3, mjGEOM_ELLIPSOID=4, mjGEOM_CYLINDER=5, mjGEOM_BOX=6,
                                       mjGEOM_MESH=7)
    mj.mjtJoint = types.SimpleNamespace(mjJNT_FREE=0, mjJNT_BALL=1)
    return S, types.SimpleNamespace(quat_to_mat=quat_to_mat), mj


@unittest.skipIf(np is None, "numpy is not installed")
class Block(unittest.TestCase):
    def test_chains_and_cloth(self):
        S, M, mj = _fake_world(np)
        T_RW = np.eye(4); T_RW[:3, 3] = [0.1, 0.0, -0.4]          # robot frame = world shifted

        def to_robot(p, q):
            return np.asarray(p, dtype=float) + T_RW[:3, 3], np.asarray(q, dtype=float)
        ns = {"np": np, "S": S, "M": M, "T_RW": T_RW, "to_robot": to_robot, "_qmul": lambda a, b: np.asarray(b),
              "_rr": lambda a, nd=6: [round(float(x), nd) for x in np.asarray(a, dtype=float).reshape(-1)]}
        code = cs.STATE_FN.replace("@DEFORMABLE@", repr({"blanket": "cloth", "cord": "chains"}))
        saved = sys.modules.get("mujoco"); sys.modules["mujoco"] = mj
        try:
            exec(compile(code, "cstate block", "exec"), ns)
            static = ns["deformables_static"]()
            now = ns["deformables_robot"]()
        finally:
            if saved is None:
                sys.modules.pop("mujoco", None)
            else:
                sys.modules["mujoco"] = saved
        L = 0.02
        cord = static["cord"]
        self.assertEqual([c["links"] for c in cord["chains"]], [["a0", "a1"], ["b0", "b1"]])
        self.assertEqual([c["parent"] for c in cord["chains"]], ["handle", "handle"])
        self.assertEqual(sorted(cord["rigid_links"]), ["handle"])
        self.assertEqual(cord["grasp_regions"], [{"link": "a1", "chain": 0, "segment": 1, "t": 0.5, "off_axis_m": 0.0, "radius": 0.008}])
        pts = [np.array(c["points"]) - T_RW[:3, 3] for c in now["cord"]["chains"]]
        # chain A: from the handle outward along +x
        np.testing.assert_allclose(pts[0], [[1.0, 0, 0.5], [1.0 + L, 0, 0.5], [1.0 + 2 * L, 0, 0.5]], atol=1e-9)
        # chain B (joints at the far end): from the handle (x = 1.0) outward along -x, its last point the last origin
        np.testing.assert_allclose(pts[1], [[1.0, 0.1, 0.5], [1.0 - L, 0.1, 0.5], [1.0 - 2 * L, 0.1, 0.5]], atol=1e-9)
        self.assertEqual(now["cord"]["links"]["handle"]["position"], [1.1, 0.0, 0.1])
        self.assertEqual(static["blanket"]["grid_shape"], [3, 2])
        self.assertEqual(static["blanket"]["spacing_m"], [0.01, 0.01])
        self.assertEqual(now["blanket"]["grid_shape"], [3, 2])
        np.testing.assert_allclose(np.array(now["blanket"]["vertices"]) - T_RW[:3, 3],
                                   S.env._world._cloths["blanket"].vertices(), atol=1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
