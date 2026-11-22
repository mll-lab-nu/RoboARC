"""Tests for scripts/ablations/cond_obs.py, E5's observation conditions V1 and V2, on committed Standard tasks of every
family E5 uses (the 15 A15 bases):
- V1: no `meta["safety"]` reaches the agent (both response paths are stripped; the text promises none), while the
  service still computes and logs it as the Standard task does;
- V2: the generated sensor code, run on a stub camera rig (numpy): the depth noise has the stated standard deviation,
  depth unit, Min-Z and fill rate, is reproducible and fresh per frame; the calibration error has exactly the stated
  size; every stated calibration (spec mount, meta cameras, per-frame extrinsic_cv) describes the same erroneous camera.
Standard library (numpy for the V2 maths, skipped without it):

    python3 scripts/ablations/tests/test_cond_obs.py
"""
import math
import pathlib
import re
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "scripts" / "ablations"), str(REPO / "scripts")]
import cond_obs  # noqa: E402
import experiments  # noqa: E402
import make_variants as mv  # noqa: E402

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def one_per_family(bases):
    seen, out = set(), []
    for b in bases:
        fam = re.sub(r"-\d+-i01$", "", b)
        if fam not in seen:
            seen.add(fam); out.append(b)
    return out


class Variants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = pathlib.Path(cls.tmp.name)
        cls.built = {(b, t): mv.make_one(mv.REGISTRY[t], b, "standard", REPO / "tasks", out)
                     for b in one_per_family([experiments.full(x) for x in experiments.A15]) for t in ("obsv1", "obsv2")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_v1_no_safety_for_the_agent(self):
        for (b, t), (d, digest) in self.built.items():
            srv = (d / "environment/sim/server.py").read_text()
            ins = (d / "instruction.md").read_text()
            compile(srv, d.name, "exec")
            self.assertEqual(srv.count("unfelt(on_main("), 2, d.name)
            self.assertIn('"safety": robot_felt(S.safety())', srv)   # still measured (the monitor's window), then removed
            self.assertIn('"safety": out.get("safety")', srv)        # and logged (C00's log patch)
            self.assertNotIn('meta["safety"]', ins)
            self.assertNotIn("meta safety", srv.split("SPEC = {", 1)[1].split("L.check_spec(SPEC)", 1)[0])
            self.assertEqual(cond_obs.leak_check(d), [])
            lock = mv.load_lock(t).get(d.name)
            if lock is not None:
                self.assertEqual(digest, lock, d.name)

    def test_v1_unfelt(self):
        src = cond_obs.V1_FN
        ns = {}
        exec(src, ns)
        out, parts = ns["unfelt"](({"safety": {"forcing": 1}, "done": False}, ["p"]))
        self.assertEqual(out, {"done": False})
        self.assertEqual(parts, ["p"])

    def test_v2_text_and_hooks(self):
        for (b, t), (d, _) in self.built.items():
            if t != "obsv2":
                continue
            srv = (d / "environment/sim/server.py").read_text()
            ins = (d / "instruction.md").read_text()
            for hook in ("v2_extrinsic(c, (np.asarray(", "return {c: v2_info(c, camera_to_robot(", "v2_depth(c, f.depth[c])",
                         '"intrinsic_cv": v2_K(c, CAM_INFO[c]["intrinsic_cv"])', 'v2_info(c, camera_to_robot(CAM_INFO[c]["mount"]))'):
                self.assertEqual(srv.count(hook), 1, (d.name, hook))
            self.assertIn("**Camera model (this variant).**", ins)
            self.assertLess(ins.index("**Camera model (this variant).**"), ins.index("### Action types"))


@unittest.skipIf(np is None, "numpy is not installed")
class V2Maths(unittest.TestCase):
    def rig(self):
        """The V2 block run on a stub rig: one static camera, one on a robot link."""
        K = [[400.0, 0.0, 424.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
        ns = {"np": np, "math": math, "CAMS": ["cam_high", "cam_wrist"],
              "CAM_INFO": {"cam_high": {"mount": {"type": "static"}, "intrinsic_cv": K},
                           "cam_wrist": {"mount": {"type": "link", "parent_link": "l"}, "intrinsic_cv": K}},
              "M": type("M", (), {"_r": staticmethod(lambda a, nd=7: np.round(np.asarray(a, dtype=np.float64), nd).tolist())})}
        exec(cond_obs.v2_block(), ns)
        return ns

    def test_calibration_error_size_and_determinism(self):
        a, b = self.rig(), self.rig()
        for c in ("cam_high", "cam_wrist"):
            T = a["V2_ERR"][c]["T"]
            np.testing.assert_array_equal(T, b["V2_ERR"][c]["T"])
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(T[:3, :3]) - 1) / 2))))
            self.assertAlmostEqual(ang, cond_obs.CALIB["rot_deg"], places=9)
            self.assertAlmostEqual(np.linalg.norm(T[:3, 3]) * 1000, cond_obs.CALIB["trans_mm"], places=9)
            self.assertAlmostEqual(abs(a["V2_ERR"][c]["focal"] - 1), cond_obs.CALIB["focal_rel"], places=12)
            np.testing.assert_allclose(a["V2_ERR"][c]["T_inv"] @ T, np.eye(4), atol=1e-12)
        self.assertFalse(np.allclose(a["V2_ERR"]["cam_high"]["T"], a["V2_ERR"]["cam_wrist"]["T"]))

    def test_stated_calibration_is_one_camera(self):
        ns = self.rig()
        rng = np.random.default_rng(0)

        def pose():
            q = rng.standard_normal(4); q /= np.linalg.norm(q); w, x, y, z = q
            T = np.eye(4)
            T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
            T[:3, 3] = rng.standard_normal(3)
            return T
        GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
        # a static camera: world_T_cam; the stated extrinsic (cam_from_world) and cam2world_gl must be inverse
        W = pose(); E = np.linalg.inv(W)
        info = ns["v2_info"]("cam_high", {"extrinsic_cv": E[:3].tolist(), "cam2world_gl": (W @ GL2CV).tolist(),
                                          "mount": {"type": "static", "extrinsic_cv": E[:3].tolist(), "cam2world_gl": (W @ GL2CV).tolist()}})
        E_s = np.eye(4); E_s[:3] = np.asarray(info["extrinsic_cv"])
        np.testing.assert_allclose(np.asarray(info["cam2world_gl"]) @ GL2CV @ E_s, np.eye(4), atol=1e-6)
        np.testing.assert_allclose(info["mount"]["extrinsic_cv"], info["extrinsic_cv"], atol=1e-9)
        np.testing.assert_allclose(ns["v2_extrinsic"]("cam_high", E[:3]), np.asarray(info["extrinsic_cv"]), atol=1e-6)
        np.testing.assert_allclose(np.linalg.inv(E_s), W @ ns["V2_ERR"]["cam_high"]["T"], atol=1e-6)
        # a wrist camera: world_T_cam = FK(link) @ link_T_cam_cv; the agent's FK with the stated link_T_cam_cv gives the
        # camera whose extrinsic the per-frame extrinsic_cv states
        L, C = pose(), pose()
        E_true = np.linalg.inv(L @ C)
        stated = ns["v2_info"]("cam_wrist", {"mount": {"type": "link", "parent_link": "l", "link_T_cam_cv": C.tolist()}})
        C_s = np.asarray(stated["mount"]["link_T_cam_cv"])
        E_frame = np.eye(4); E_frame[:3] = ns["v2_extrinsic"]("cam_wrist", E_true[:3])
        np.testing.assert_allclose(np.linalg.inv(L @ C_s), E_frame, atol=1e-6)
        K_s = np.asarray(ns["v2_K"]("cam_wrist", [[400.0, 0.0, 424.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]))
        self.assertAlmostEqual(K_s[0, 0], 400.0 * ns["V2_ERR"]["cam_wrist"]["focal"], places=5)
        self.assertEqual((K_s[0, 2], K_s[1, 2]), (424.0, 240.0))

    def test_depth_noise(self):
        ns = self.rig()
        for c, sensor, z0 in (("cam_high", "D435i", 1.0), ("cam_wrist", "D405", 0.4)):
            z = np.full((480, 848), z0, dtype=np.float32)
            z[:10] = 0.0                                         # no reading stays none
            z[10:20] = cond_obs.SENSORS[sensor]["min_z_m"] * 0.9   # too close: none
            a = ns["v2_depth"](c, z)
            self.assertEqual(a.shape, z.shape)
            self.assertTrue(np.all(a[:20] == 0))
            body = a[20:].astype(np.float64)
            valid = body > 0
            self.assertAlmostEqual(1 - valid.mean(), 1 - cond_obs.SENSORS[sensor]["fill"], delta=0.002)
            sd = body[valid].std()
            self.assertAlmostEqual(sd / cond_obs.sigma_m(sensor, z0), 1.0, delta=0.05)
            unit = cond_obs.SENSORS[sensor]["unit_m"]
            self.assertLess(np.abs(body[valid] / unit - np.round(body[valid] / unit)).max(), 1e-2)
            b = ns["v2_depth"](c, z)                             # the next frame: fresh noise
            self.assertFalse(np.array_equal(a, b))
        again = self.rig()                                       # a new episode: the same frames again
        z = np.full((48, 84), 1.0, dtype=np.float32)
        np.testing.assert_array_equal(self.rig()["v2_depth"]("cam_high", z), again["v2_depth"]("cam_high", z))


if __name__ == "__main__":
    unittest.main()
