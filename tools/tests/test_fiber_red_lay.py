"""fiber_bundling's redesigned tiers (fiber-bundling 07-10): the `lay` rule names the strand by its colour, so either red
fiber may meet it (numpy + scipy, the simulator source; no MuJoCo, no image):

    python3 tools/tests/test_fiber_red_lay.py

- the strands drawn in the marked strand's colour are those with palette entry i % len(COLORS) equal to its own
  (16 strands: 0 and 8 are red, and strand_palette_shift never moves entry 0);
- the marked strand (0) in place: the rule holds and reports strand 0, exactly as before;
- strand 8 in place, strand 0 not: the rule holds, reporting strand 8;
- neither red strand in place (another colour is): the rule fails, reporting strand 0 as before.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from topogym.assets.geometry.fiber_bundling import COLORS  # noqa: E402
from topogym.tasks.fiber_bundling.redesign import tasks as T  # noqa: E402

ANGLE, TOL = -130.0, 20.0


def task_with(roots):
    """A redesigned task stand-in: 16 strands whose roots sit at the given angles (deg, from the ring's top about the
    feed direction +x), the ring axis through the origin."""
    t = object.__new__(T.RedesignFiberTask)
    t.config = {"redesign": {"lay": {"strand": 0, "angle_deg": ANGLE, "tolerance_deg": TOL}}}
    pts = []
    for i in range(16):
        a = np.radians(roots.get(i, 90.0))          # every other strand far from the target angle
        # normal +x, up +z, left = n x up = -y: angle = atan2(v.left, v.up)
        pts.append(np.array([[0.0, 0.0, 0.0], [0.0, -np.sin(a) * 0.01, np.cos(a) * 0.01]]))
    t._strand_points = lambda state: pts
    t._front_point = lambda state: np.zeros(3)
    return t


def lay(roots):
    return task_with(roots).lay_term(None, np.zeros(3), np.array([1.0, 0.0, 0.0]))


def main() -> int:
    assert T.same_colour_strands(0, 16) == [8], T.same_colour_strands(0, 16)
    assert all(COLORS[i % len(COLORS)] == COLORS[0] for i in (0, 8))
    a = lay({0: ANGLE})
    assert a["within"] and a["strand"] == 0, a
    b = lay({8: ANGLE + 5.0})
    assert b["within"] and b["strand"] == 8, b
    c = lay({3: ANGLE})                              # a blue strand in place: not a red fiber
    assert not c["within"] and c["strand"] == 0, c
    d = lay({0: ANGLE + 50.0, 8: ANGLE - 50.0})     # both red fibers out of tolerance
    assert not d["within"] and d["strand"] == 0, d
    print("ok fiber lay rule: either red fiber (strands 0, 8) meets it; another colour does not")
    return 0


if __name__ == "__main__":
    sys.exit(main())
