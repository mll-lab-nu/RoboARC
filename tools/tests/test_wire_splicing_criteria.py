#!/usr/bin/env python
"""topogym/tasks/wire_splicing/criteria.py around_both when one of A's strands is a single node.

A's return strand runs from the eye's apex to A's node farthest back toward the strip. When A's tail, past the eye,
never comes back toward the strip (the stress round v2.13b of wire-splicing-01-i01, 2026-10-25: prefix3_e8 and
prefix5_e8, the tail pulled straight out), that node is the apex itself and the strand has no segment. around_both then
took argmin over an empty array, the task's step raised, and the simulator faulted (a confirmed BREAKDOWN). Such a
strand reaches no station along the axis, so the coil gains no turn round both strands.

A here: 10 nodes on the axis (z from +50 to -50 mm; the strip at node 0, the eye between nodes 3 and 6), B's coil a
helix of 3 mm radius round it.
  - the tail straight on past the eye (z falling to the end): the return strand is node 6 alone; 0 turns, no error;
  - the tail folded back toward the strip (nodes 7-9 rising to +30 mm): the coil winds round both strands, about
    0.53 turns, as before the fix.

  python tools/tests/test_wire_splicing_criteria.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))


def main():
    from topogym.tasks.wire_splicing import criteria as C
    ahat, c0 = np.array([0., 0., 1.]), np.zeros(3)
    straight = np.c_[np.zeros(10), np.zeros(10), np.linspace(.05, -.05, 10)]
    s = np.linspace(0, 6, 30)
    coil = np.c_[.003 * np.cos(s), .003 * np.sin(s), np.linspace(.04, -.04, 30)]
    turns, run = C.around_both(straight, 0, 3, 6, c0, ahat, coil)
    assert (turns, run) == (0., None), (turns, run)
    folded = straight.copy()
    folded[7:, 2] = np.linspace(-.01, .03, 3)
    turns, run = C.around_both(folded, 0, 3, 6, c0, ahat, coil)
    assert abs(turns - .5269) < 1e-3 and run == (0, 20), (turns, run)
    print(f"PASS (straight tail: 0 turns; folded tail: {turns:.4f} turns over coil nodes {run})")


if __name__ == "__main__":
    main()
