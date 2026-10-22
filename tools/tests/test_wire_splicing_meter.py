"""wire_splicing's splice meter (user decision 2026-10-19; runs/wire_splice_design_20261014, registration_draft.md
"Success display"): the reading model (criteria.meter_reading) and the meter's MJCF (geometry.wire_splicing.meter_xml).

The reading: no path (no resistance, the needle at its rest stop) unless A's eye is closed, B threads it once and B's coil
lies on A's bare core; otherwise R = 10 mOhm / turns round both of A's strands; the needle is in the red band (from
10 mOhm, one turn round both) exactly when every shape condition holds, and a coil of enough turns that fails another
condition (here: the tail not folded back, or fewer than 1.5 turns about the axis) stops the needle just below the
band's edge. The MJCF: the needle's centre of mass stays on its hinge axis at any angle (setting it moves no mass), and
its hinge's zero is the rest stop.

Round both strands (criteria.around_both, user decision 2026-10-22): a coil round both of A's strands passes, a loop round
one strand only or one passing between the strands fails though it winds 1.5 turns about the axis between them; the
-i01 reference's end state (tools/tests/fixtures/wire_splicing_states.json) passes every shape condition, and its coil
at the end of grip-turn 9 (0.94 turns about the axis) goes 0.91 turns round both.

  python tools/tests/test_wire_splicing_meter.py
"""
from pathlib import Path
import json
import os
import sys
import unittest

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2] / "source"
sys.path.insert(0, str(ROOT))

from topogym.assets.geometry import wire_splicing as geo
from topogym.tasks.wire_splicing import criteria as C
from topogym.tasks.wire_splicing.common import WireSplicingTask

D = geo.CORE_D
TERMINAL = WireSplicingTask.defaults["terminal"]
EDGE = geo.dial_position(geo.R_TURN_MOHM / geo.BAND_TURNS)
MIN_TURNS = TERMINAL["min_wrap_turns"]     # 1.5 (user, 2026-10-21; 3 from 2026-10-20, 6.5 before)
MIN_BOTH = TERMINAL["min_both_turns"]      # 1 (user, 2026-10-22)


def splice(turns=7., both=None, pierce=1, coil_r_D=1.4, coil_z=(-1., 25.), stem=32., tail_dir=-.9, tail_r_D=2.5):
    """A measurement as criteria.measure returns it for an eye with a coil on its doubled section (both: its turns
    round both strands, by default all of them)."""
    return dict(eye=True, pierce=pierce, wrap_turns=turns, both_turns=turns if both is None else both, coil_nodes=60,
                coil_mean_r_D=coil_r_D, coil_z_from_eye_mm=coil_z, stem_len_mm=stem, tail_dir=tail_dir,
                tail_r_D=tail_r_D)


def shape_checks(m):
    return C.shape_checks(m, D, min_turns=TERMINAL["min_wrap_turns"], min_both_turns=TERMINAL["min_both_turns"],
                          max_coil_r_D=TERMINAL["max_coil_radius_d"], tail_dot=TERMINAL["tail_fold_dot"],
                          min_tail_r_D=TERMINAL["min_tail_radius_d"])


def reading(m):
    checks = shape_checks(m)
    return C.meter_reading(m, checks), checks


class MeterReading(unittest.TestCase):
    def assert_open(self, r):
        self.assertFalse(r["conducting"])
        self.assertIsNone(r["resistance_mohm"])
        self.assertFalse(r["in_band"])
        self.assertEqual(r["dial_position"], geo.METER["rest_u"])
        self.assertEqual(r["needle_rad"], 0.)          # the rest stop is the hinge's zero (the scene's qpos0)

    def test_no_path(self):
        """No eye, an eye B does not thread (or threads twice), or no coil on the bare core: no reading."""
        self.assert_open(reading(dict(eye=False))[0])
        self.assert_open(reading(splice(pierce=0))[0])
        self.assert_open(reading(splice(pierce=2))[0])
        self.assert_open(reading(splice(coil_z=(-10., 25.)))[0])           # the coil runs past the neck
        self.assert_open(reading(dict(splice(), coil_nodes=0, wrap_turns=0., both_turns=0.))[0])

    def test_complete_splice_in_band(self):
        r, checks = reading(splice(turns=7.))
        self.assertTrue(all(checks.values()))
        self.assertTrue(r["conducting"] and r["in_band"])
        self.assertAlmostEqual(r["resistance_mohm"], 10. / 7.)
        self.assertGreaterEqual(r["dial_position"], EDGE)
        self.assertAlmostEqual(r["dial_position"], np.log10(100. / (10. / 7.)) / 2.)

    def test_band_edge_when_another_condition_fails(self):
        """Seven tight turns on the bare core (more than the band needs), but the tail not folded back (or the coil loose): the resistance is the
        coil's, and the needle stops EDGE_GAP below the band's edge."""
        for m in (splice(turns=7., tail_dir=.2), splice(turns=7., coil_r_D=2.3)):
            r, checks = reading(m)
            self.assertTrue(checks["wrap"])
            self.assertFalse(all(checks.values()))
            self.assertTrue(r["conducting"])
            self.assertFalse(r["in_band"])
            self.assertAlmostEqual(r["resistance_mohm"], 10. / 7.)
            self.assertAlmostEqual(r["dial_position"], EDGE - C.EDGE_GAP)
            self.assertGreater(r["needle_rad"], geo.needle_q(EDGE))          # left of the band (angles fall rightward)

    def test_too_few_turns_below_band(self):
        self.assertEqual((MIN_TURNS, MIN_BOTH), (1.5, 1.))
        # under one turn round both strands, though 1.6 turns about the axis: below the band, R = 10 / 0.8 mOhm
        r, checks = reading(splice(turns=1.6, both=.8))
        self.assertTrue(checks["wrap"])
        self.assertFalse(checks["round_both"] or r["in_band"])
        self.assertAlmostEqual(r["resistance_mohm"], 10. / .8)
        self.assertAlmostEqual(r["dial_position"], np.log10(100. / 12.5) / 2.)
        # a full turn round both strands but 1.2 turns about the axis: 10 mOhm, the band's edge, yet out of the band
        r, checks = reading(splice(turns=1.2, both=1.))
        self.assertTrue(checks["round_both"])
        self.assertFalse(checks["wrap"] or r["in_band"])
        self.assertAlmostEqual(r["resistance_mohm"], 10.)
        self.assertAlmostEqual(r["dial_position"], EDGE - C.EDGE_GAP)
        # 0.98 turns round both would read within EDGE_GAP of the band: the needle stays EDGE_GAP below its edge
        r, _ = reading(splice(turns=1.6, both=.98))
        self.assertFalse(r["in_band"])
        self.assertAlmostEqual(r["dial_position"], EDGE - C.EDGE_GAP)
        # exactly the thresholds: in the band, at its edge
        r, _ = reading(splice(turns=MIN_TURNS, both=MIN_BOTH))
        self.assertTrue(r["in_band"])
        self.assertAlmostEqual(r["dial_position"], EDGE)

    def test_scale_clamped(self):
        """A path above 100 mOhm (under 0.1 turn) reads at the 100 mOhm end, not the rest stop; under 1 mOhm at 1."""
        self.assertEqual(reading(splice(turns=.02, tail_dir=.2))[0]["dial_position"], 0.)
        self.assertEqual(reading(splice(turns=12.))[0]["dial_position"], 1.)

    def test_band_edge_is_the_round_both_threshold(self):
        self.assertAlmostEqual(geo.R_TURN_MOHM / geo.BAND_TURNS, 10. / TERMINAL["min_both_turns"])
        self.assertAlmostEqual(EDGE, .5)                  # 10 mOhm: the dial's middle


class MeterModel(unittest.TestCase):
    def test_needle_moves_no_mass(self):
        m = mujoco.MjModel.from_xml_string(geo.meter_xml())
        d = mujoco.MjData(m)
        needle = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "needle")
        joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "needle")
        self.assertEqual(m.nq, 1)
        self.assertEqual(float(m.qpos0[m.jnt_qposadr[joint]]), geo.needle_q(geo.METER["rest_u"]))
        com = []
        for u in (geo.METER["rest_u"], 0., .5, EDGE, 1.):
            d.qpos[m.jnt_qposadr[joint]] = geo.needle_q(u)
            mujoco.mj_kinematics(m, d)
            com.append(np.array(d.xipos[needle]))
        self.assertTrue(all(np.array_equal(c, com[0]) for c in com))
        # every geom is visual: none collides (the MuJoCo scene keeps only the case, on the static panel)
        self.assertFalse(np.any(m.geom_contype) or np.any(m.geom_conaffinity))
        self.assertEqual(int(m.body_jntnum[m.geom_bodyid[m.geom_type == mujoco.mjtGeom.mjGEOM_MESH][0]]), 0)


def doubled_end(a):
    """A's nodes for an eye at the bottom (z = 0) of two straight strands along +z, the stem at x = -a (strip first,
    from z = 50 mm down) and the return leg at x = +a (up to z = 45 mm), joined by a half circle; with the strip node 0,
    the neck nodes k1, k2 and the axis (midway, toward the strip)."""
    stem = np.c_[np.full(17, -a), np.zeros(17), np.linspace(.050, .010, 17)]
    phi = np.linspace(np.pi, 2 * np.pi, 9)[1:-1]
    eye = np.c_[a * np.cos(phi), np.zeros(7), .010 + a * np.sin(phi)]
    back = np.c_[np.full(15, a), np.zeros(15), np.linspace(.010, .045, 15)]
    PA = np.vstack([stem, eye, back])
    return PA, 16, 16 + 8, np.array([0., 0., .030]), np.array([0., 0., 1.])


def helix(centre_x, r, turns, z0=.020, z1=.030, n=40):
    t = np.linspace(0., 2 * np.pi * turns, n)
    return np.c_[centre_x + r * np.cos(t), r * np.sin(t), np.linspace(z0, z1, n)]


class RoundBoth(unittest.TestCase):
    """criteria.around_both: the coil's turns round both of A's strands together, about each strand's own centreline."""

    def test_round_both_strands(self):
        PA, k1, k2, c0, ahat = doubled_end(.0018)            # strands touching (centres one core diameter apart)
        both, run = C.around_both(PA, 0, k1, k2, c0, ahat, helix(0., .0045, 1.5))
        self.assertAlmostEqual(both, 1.5, places=2)
        self.assertEqual(run, (0, 39))

    def test_loop_round_one_strand_only(self):
        """1.5 turns round the stem alone, wide enough to enclose the axis: 1.5 turns about the axis, none round both."""
        PA, k1, k2, c0, ahat = doubled_end(.003)
        Q = helix(-.003, .004, 1.5)
        self.assertGreater(abs(C.winding(Q, c0, ahat)), 1.4)
        self.assertLess(C.around_both(PA, 0, k1, k2, c0, ahat, Q)[0], .6)

    def test_loop_passing_between_the_strands(self):
        """1.5 turns round the return leg and the axis, passing between the two strands: none round both."""
        PA, k1, k2, c0, ahat = doubled_end(.003)
        Q = helix(.001, .003, 1.5)
        self.assertGreater(abs(C.winding(Q, c0, ahat)), 1.4)
        self.assertLess(C.around_both(PA, 0, k1, k2, c0, ahat, Q)[0], .6)

    def test_reference_states(self):
        states = json.loads((Path(__file__).resolve().parent / "fixtures" / "wire_splicing_states.json").read_text())
        iA, iB = geo.WIRES["A"]["n_ins"], geo.WIRES["B"]["n_ins"]
        end = states["reference_end"]
        m = C.measure(np.array(end["PA"]), np.array(end["PB"]), iA, iB, D)
        checks = shape_checks(m)
        self.assertTrue(all(checks.values()), checks)
        self.assertGreater(m["both_turns"], 1.5)            # 1.56 round both (1.96 about the axis)
        self.assertLess(m["both_turns"], m["wrap_turns"])
        r = C.meter_reading(m, checks)
        self.assertTrue(r["in_band"])
        self.assertAlmostEqual(r["resistance_mohm"], 10. / m["both_turns"])
        nine = states["grip_turn_9"]                         # a ring that looks closed: 0.91 turns round both
        m = C.measure(np.array(nine["PA"]), np.array(nine["PB"]), iA, iB, D)
        self.assertGreater(m["wrap_turns"], .9)
        self.assertLess(m["both_turns"], 1.)
        self.assertFalse(shape_checks(m)["round_both"])


class CoreLine(unittest.TestCase):
    """The splice's axis (criteria.core_line) from the stem and only the tail nodes that lie beside it: A's end folded
    back past a short coil tilts it under 5 deg (the return leg beside only part of the stem moves the midline a little);
    with every tail node fitted, as before v4, it tilted 29 deg."""

    def test_folded_tail_keeps_the_axis(self):
        z = np.arange(0.050, 0.0099, -0.0025)                      # the stem, strip first, toward the eye (-z)
        stem = np.c_[np.zeros_like(z), np.zeros_like(z), z]
        bend = np.array([[0.002, 0., 0.006], [0.004, 0., 0.007]])   # the eye's bend
        beside = np.c_[np.full(5, 0.0036), np.zeros(5), np.linspace(0.010, 0.020, 5)]   # the return leg by the stem
        folded = np.array([[0.012, 0., 0.024], [0.016, 0., 0.024], [0.022, 0., 0.018], [0.026, 0., 0.010],
                           [0.028, 0., 0.002], [0.029, 0., -0.006], [0.030, 0., -0.014], [0.030, 0., -0.022]])
        PA = np.vstack([stem, bend, beside, folded])
        k1, k2 = len(stem) - 1, len(stem) + len(bend)
        c0, a = C.core_line(PA, 0, k1, k2)
        self.assertLess(np.degrees(np.arccos(abs(a[2]))), 5.)
        self.assertGreater(a[2], 0.)                                # from the eye toward the strip
        n = k1                                                      # the pre-v4 fit: as many tail nodes as stem nodes
        pts = np.vstack([PA[:k1 + 1], PA[k2:k2 + n + 1]])
        _, _, vt = np.linalg.svd(pts - pts.mean(axis=0))
        self.assertGreater(np.degrees(np.arccos(abs(vt[0][2]))), 20.)


if __name__ == "__main__":
    unittest.main()
