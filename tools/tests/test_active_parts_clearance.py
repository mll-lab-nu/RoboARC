"""The release clearance policy active_parts_clearance_v1 (evaluation/terminal/settling.py, the blanket families since
RoboARC 2.1.0): the boxes of robot links fixed to the world are left out, every other box keeps the clearance (numpy +
scipy, the simulator source; no MuJoCo, no image):

    python3 tools/tests/test_active_parts_clearance.py

A cloth (particle target, 8 mm padding) lies flat, released and still; the robot is two boxes: a fixed base and a moving
gripper.
- the base box 2 mm from the cloth's padded box, the gripper far: all_parts_clearance fails (robot_clear False), the
  active policy passes and reports one exempt box;
- the gripper 2 mm away, the base far: both policies fail;
- both far: both pass (the active policy measures to the gripper only);
- the active policy without the geometry's robot_fixed flags: evaluation unavailable, never a pass;
- all_parts_clearance's metrics are the same as before (no policy keys added);
- the policy is accepted for particle and rigid targets, and fixed_contact_active_clearance_v1 still only for FEM ones;
- blanket_flatten (and blanket_fold, its subclass) use it, with terminal revision release_settling_v2_active_parts.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source"))
from topogym.evaluation.terminal.settling import SettlingSpec, evaluate_settling  # noqa: E402

PTS = np.array([[x, y, 0.001] for x in np.linspace(-.3, .3, 7) for y in np.linspace(-.24, .24, 5)])
PAD = .008
CLOTH = np.array([PTS.min(0) - PAD, PTS.max(0) + PAD])


def box(x0, x1, y0=-.1, y1=.1, z0=0., z1=.08):
    return [[x0, y0, z0], [x1, y1, z1]]


def state(robot, fixed=None):
    g = dict(schema_version=2, physics_step=7, complete=True, robot_present=True, robot_bounds_m=np.array(robot, float),
             target_bounds_m={}, robot_convex_planes_m=[np.zeros((0, 4)) for _ in robot], fem_elements={})
    if fixed is not None:
        g["robot_fixed"] = list(fixed)
    return dict(physics_step=7, assist_active=False, release_geometry=g,
                particles={"blanket": dict(position_m=PTS, velocity_m_s=np.zeros_like(PTS))})


def run(policy, robot, fixed=None):
    spec = SettlingSpec(particle_targets=("blanket",), particle_radius_m=PAD, robot_contact_policy=policy)
    reached, metrics, _ = evaluate_settling(state(robot, fixed), spec, None)
    return metrics


edge = CLOTH[1, 0]
near_base = box(edge + .002, edge + .30)                 # 2 mm outside the padded cloth box
far = box(edge + .20, edge + .40)
far_gripper = box(-.05, .05, z0=.50, z1=.60)
near_gripper = box(-.05, .05, z0=CLOTH[1, 2] + .002, z1=CLOTH[1, 2] + .1)

# a base 2 mm from the cloth: all parts fails, the active policy passes (the base exempt)
m = run("all_parts_clearance", [near_base, far_gripper])
assert not m["robot_clear"] and abs(m["clearance_lower_bound_m"] - .002) < 1e-9, m
assert "robot_contact_policy" not in m and "fixed_robot_boxes_exempt" not in m, m
m = run("active_parts_clearance_v1", [near_base, far_gripper], fixed=[True, False])
assert m["robot_clear"] and m["fixed_robot_boxes_exempt"] == 1 and m["robot_contact_policy"] == "active_parts_clearance_v1", m
assert m["clearance_lower_bound_m"] > .4, m

# a moving gripper 2 mm from the cloth: both fail
for policy, fixed in (("all_parts_clearance", None), ("active_parts_clearance_v1", [True, False])):
    m = run(policy, [far, near_gripper], fixed)
    assert not m["robot_clear"] and abs(m["clearance_lower_bound_m"] - .002) < 1e-9, (policy, m)

# both far: both pass; the active policy measures to the gripper only
a, b = run("all_parts_clearance", [far, far_gripper]), run("active_parts_clearance_v1", [far, far_gripper], [True, False])
assert a["robot_clear"] and b["robot_clear"], (a, b)
assert abs(a["clearance_lower_bound_m"] - .2) < 1e-9, a
assert abs(b["clearance_lower_bound_m"] - (.50 - CLOTH[1, 2])) < 1e-9, b

# no flags: unavailable, not clear
m = run("active_parts_clearance_v1", [near_base, far_gripper])
assert not m["robot_clear"] and "robot_fixed_flags" in m["missing_evidence"] and m["evaluation_status"] == "unavailable", m

# the specs the policies accept
SettlingSpec(rigid_targets=("piece",), robot_contact_policy="active_parts_clearance_v1")
try:
    SettlingSpec(particle_targets=("blanket",), robot_contact_policy="fixed_contact_active_clearance_v1")
    raise AssertionError("fixed_contact_active_clearance_v1 must stay FEM-only")
except ValueError:
    pass

# the blanket families use it (read from the source: the task modules need trimesh, which this tier does not install)
SRC = Path(__file__).resolve().parents[2] / "source/topogym/tasks"
flatten = (SRC / "blanket_flatten/common.py").read_text()
fold = (SRC / "blanket_fold/common.py").read_text()
assert 'terminal_revision = "release_settling_v2_active_parts"' in flatten, "blanket_flatten's terminal revision"
assert 'robot_contact_policy="active_parts_clearance_v1"' in flatten, "blanket_flatten's settling policy"
assert "class BlanketFoldTask(BlanketFlattenTask)" in fold and "terminal_revision" not in fold \
    and "def final_settling_spec" not in fold, "blanket_fold inherits the revision and the spec"
print("ok active_parts_clearance_v1: fixed boxes exempt, moving boxes kept, flags required, blanket families on it")
