"""clock_assembling: assemble a watch movement's gear train enlarged 8 times with two ALOHA arms (design:
runs/watch_assembly_design_20261014, README_v3.md; registration proposal REGISTRATION_PROPOSAL.md there).

The scene is one MJCF entity in the ALOHA mount frame (topogym.assets.geometry.clock_gear_train): the movement, the stands,
the four wheels, the bridge, two screws and the probe. Seating, meshing, the gates, the screws and the bridge's fixing are
the task's mechanism (mechanism.GearTrainRules), stepped by the MuJoCo world every tick: the task runs on MuJoCo only.

Goal: the four wheels seated on their lower pivots, every upper pivot (the barrel: its arbor) in its bridge hole, the
bridge seated, both screws tight and the bridge fixed, the three meshes coupled, and the run test: the barrel wound by at
least wind_min_rad through its ratchet (the click holds it), then the train running on the mainspring alone for the rules'
5 s window: the fourth wheel at least run_turns_min turns, stalled at most run_stall_max_s, at the train's ratio (-450)
within ratio_tolerance; all together for stable_duration_s. Holding is allowed."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.assets.geometry import clock_gear_train as geo
from topogym.tasks.base import EntitySpec, RigidTaskDefinition, TaskSpec

# The ALOHA's mount in the world (configs/env_configs/world_tabletop_aloha_v1.yaml robot_mount.pos x, y): the entity's
# origin, so the design's mount-frame coordinates hold as they are (as bed_bolt).
MOUNT_XY_M = (-.1356786, 0.)


def recommended_grasp_assist(variation_id, seed=0, *, task_parameters=None):
    from topogym.config.registry import task_definition
    return task_definition("clock_assembling", variation_id, seed)(task_parameters, seed=seed).grasp_assist()


class ClockAssemblingTask(RigidTaskDefinition):
    task_id = "clock_assembling"
    physics_model = "rigid_articulated"
    terminal_revision = "clock_gear_train_v2"
    defaults = dict(stable_duration_s=.5, terminal=dict(
        seat_height_m=.001,                     # a wheel's origin above its seat (the rule seats at 0.8 mm)
        pivot_depth_m=.001,                     # an upper pivot's tip inside its bridge hole
        bridge_height_m=.0005, bridge_lateral_m=.0005, bridge_tilt_rad=float(np.radians(.5)),
        wind_min_rad=float(np.radians(120.)),   # the arbor's wind held by the click when the run test starts
        run_turns_min=5.,                       # the fourth wheel's turns in the rules' 5 s window
        run_stall_max_s=.2,                     # time in that window with the fourth wheel under 10 % of its free-run speed
        ratio_tolerance=.02))                   # the fourth wheel's turn per drum turn in that window, relative to -450
    # the bridge is 21 cm wide and the probe 10 cm long: a hand holding either far from its middle (bed_bolt's setting)
    grasp_max_tcp_distance_m = .15

    def grasp_assist(self):
        """contact_weld_v1 on every part (the wheels, the bridge, the screws, the probe), each as a whole part."""
        from topogym.utils.grasp import GraspAssistConfig, rigid_grasp_region
        return GraspAssistConfig("contact_weld_v1", tuple(rigid_grasp_region(geo.ENTITY, part, geo.part_points(part))
                                                          for part in geo.GRASP_LINKS),
                                 max_tcp_distance_m=self.grasp_max_tcp_distance_m)

    def origin(self):
        return np.r_[MOUNT_XY_M, self.table_height_m]

    def movement_origin(self):
        """W, the movement frame's origin, in the world."""
        return self.origin() + geo.P_W

    def spins(self):
        """Each wheel's turn about its axis in its cup at birth: the design's turn plus the seed's U(-pi, pi)."""
        rng = np.random.default_rng(self.seed)
        return {n: float(np.angle(np.exp(1j * (geo.BASE_SPIN[n] + rng.uniform(-np.pi, np.pi))))) for n in geo.WHEELS}

    def declare(self, asset_dir, table_height_m):
        self.table_height_m = float(table_height_m)
        folder = Path(asset_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{geo.ENTITY}.xml"
        path.write_text(geo.entity_xml(self.spins()))
        self.local_vertices = {geo.ENTITY: geo.part_points("barrel")}     # the entity's base link
        return TaskSpec((EntitySpec(geo.ENTITY, "mjcf", path, tuple(self.origin()), fixed=False),), {}, dict(
            task_id=self.task_id, instance_id=self.instance_id, physics_engine="mujoco",
            movement_origin_m=self.movement_origin().tolist(), wheels=list(geo.WHEELS), total_ratio=geo.TOTAL_RATIO,
            screw_turns=geo.SCREW_TURNS, goal_requires_release=False))

    def finalize_spec(self, spec, *, asset_dir, scene_config):
        mount = np.asarray(scene_config.env.robot_mount.pos, dtype=float)
        if np.max(np.abs(mount[:2] - MOUNT_XY_M)) > 1e-9 or abs(mount[2] - self.table_height_m) > 1e-9 \
                or np.any(np.asarray(scene_config.env.robot_mount.euler, dtype=float)):
            raise ValueError("clock_assembling is laid out in the ALOHA tabletop's mount frame")
        return spec

    def mujoco_mechanism(self, world):
        from .mechanism import GearTrainRules
        return GearTrainRules(world, self.movement_origin())

    def _part_pose(self, links, name):
        link = links[name]
        q = np.asarray(link["quaternion_wxyz"], dtype=float)
        return np.asarray(link["position_m"], dtype=float), Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()

    def evaluate_state(self, state):
        rules = state.get("mechanism")
        if rules is None:
            raise RuntimeError("clock_assembling runs on MuJoCo: the world reads no mechanism")
        _, _, _, failure = self.rigid_measurements(state)
        if failure:
            return False, failure, {}
        links = state["articulations"][geo.ENTITY]["links"]
        poses = {n: self._part_pose(links, n) for n in geo.PARTS}
        for n, (p, R) in poses.items():
            if not np.isfinite(np.r_[p, R.ravel()]).all():
                return False, "Non-finite physical state", {}
            if (p + geo.part_points(n) @ R.T)[:, 2].min() < self.table_height_m - .1:
                return False, "A task body fell below the table", {}
        t = self.config["terminal"]
        o = self.movement_origin()
        bp, BR = poses["bridge"]
        bp = bp - o
        metrics = dict(evaluation_status="available", holding_allowed=True)
        seated = in_bridge = True
        for n in geo.WHEELS:
            p, R = poses[n]
            p = p - o
            seated_n = bool(rules[f"{n}_seated"] and p[2] <= t["seat_height_m"])
            z = geo.SHOULDER_Z + geo.UP_PIVOT_L if n != "barrel" else geo.H_BRIDGE + geo.T_BRIDGE
            v = p + R @ np.array([0., 0., z]) - (bp + BR @ np.array([geo.POS[n][0], geo.POS[n][1], 0.]))
            h = float(v @ BR[:, 2])
            lateral = float(np.linalg.norm(v - h * BR[:, 2]))
            radius = geo.PIVOT_HOLE_R if n != "barrel" else geo.BARREL_ARBOR_R + .0005
            in_n = bool(lateral <= radius and h >= t["pivot_depth_m"])
            metrics |= {f"{n}_seated": seated_n, f"{n}_in_bridge": in_n, f"{n}_pivot_lateral_m": lateral,
                        f"{n}_pivot_depth_m": h}
            seated, in_bridge = seated and seated_n, in_bridge and in_n
        tilt = float(np.arccos(np.clip(BR[2, 2], -1., 1.)))
        bridge_seated = bool(abs(bp[2] - geo.H_BRIDGE) <= t["bridge_height_m"] and np.hypot(bp[0], bp[1]) <= t["bridge_lateral_m"]
                             and tilt < t["bridge_tilt_rad"])
        fixed = bool(rules["fixed"] and rules["screw1_tight"] and rules["screw2_tight"])
        meshed = all(rules[f"{n}_meshed"] for n in geo.PARENT)
        wound = float(rules["run_wound_rad"])
        turns, stall, ratio = float(rules["run_fourth_turns"]), float(rules["run_stall_s"]), float(rules["run_ratio"])
        ran = bool(rules["run_done"] and rules["run_ratio_measured"] and wound >= t["wind_min_rad"]
                   and turns >= t["run_turns_min"] and stall <= t["run_stall_max_s"]
                   and abs(ratio / geo.TOTAL_RATIO - 1.) <= t["ratio_tolerance"])
        metrics |= dict(bridge_seated=bridge_seated, bridge_height_error_m=float(bp[2] - geo.H_BRIDGE),
                        bridge_tilt_rad=tilt, bridge_fixed=fixed, meshed=meshed, wound_rad=wound, run_done=bool(rules["run_done"]),
                        run_fourth_turns=turns, run_stall_s=stall, run_ratio=ratio, run_test=ran,
                        screw1_turns=float(rules["screw1_turns"]), screw2_turns=float(rules["screw2_turns"]))
        return bool(seated and in_bridge and bridge_seated and fixed and meshed and ran), None, metrics

    def goal_words(self):
        """(summary, success conditions) of the public goal (goal_description.describe_goal)."""
        t = self.config["terminal"]
        deg = lambda r: f"{np.degrees(r):g} degrees"
        return ("Assemble the watch movement's gear train: the four wheels on their pivots in the main plate, the train "
                "bridge over them on its pillars with every upper pivot in its hole, the two bridge screws tight; then wind "
                "the barrel through its ratchet, let go and watch the fourth wheel run.", [
                    f"Seat the barrel, the center wheel, the third wheel and the fourth wheel on their lower pivots in the "
                    f"main plate (each wheel within {t['seat_height_m'] * 1000:g} mm of its seat); a wheel seats once its "
                    f"lower pivot is down in its hole, within {.6:g} mm of the hole's axis and tilted less than 3 degrees. "
                    "Each wheel's pinion must mesh with the previous wheel's teeth (barrel, center, third, fourth); a wheel "
                    "lowered with its pinion's teeth tip to tip on the other wheel's teeth stops until it is turned a little.",
                    f"Every upper pivot (the barrel: its arbor) inside its hole in the bridge, at least "
                    f"{t['pivot_depth_m'] * 1000:g} mm deep and within the hole's radius of its axis.",
                    f"The bridge seated on its pillars: within {t['bridge_height_m'] * 1000:g} mm of its seat height, "
                    f"{t['bridge_lateral_m'] * 1000:g} mm of its seat position and tilted less than {deg(t['bridge_tilt_rad'])}.",
                    f"Both screws in the bridge's screw holes and turned clockwise by {geo.SCREW_TURNS:g} full turns with "
                    "their heads on the bridge (no thread is simulated: the turns are counted while a head sits on the "
                    "bridge); with both tight and the bridge seated, the bridge is fixed.",
                    f"After the bridge is fixed, wind the barrel: turn its square (the ratchet under it turns with it) "
                    f"counterclockwise seen from above; the click lets it turn that way only and holds it one ratchet tooth "
                    f"({np.degrees(geo.CLICK_PITCH):g} degrees) behind the furthest point reached. Wind it at least "
                    f"{deg(t['wind_min_rad'])} (as held by the click), then stop winding: the mainspring drives the barrel "
                    f"and the train. Half a second after the last winding the rules time a "
                    f"{5:g} s run: the fourth wheel must turn at least {t['run_turns_min']:g} times in it, spend at most "
                    f"{t['run_stall_max_s']:g} s slower than a tenth of its free-run speed (the speed the mainspring's "
                    "present windup gives the train against its own damping and friction), and turn "
                    f"{abs(geo.TOTAL_RATIO):g} times as much as the barrel in the opposite direction, within "
                    f"{t['ratio_tolerance'] * 100:g} %. Winding again restarts the run.",
                    "Holding parts is allowed; the probe and the hands may stay anywhere.",
                ])

    def terminal_metadata(self):
        return super().terminal_metadata() | dict(
            goal_kind="gear_train_assembled", total_ratio=geo.TOTAL_RATIO, screw_turns=geo.SCREW_TURNS,
            rules="seat, gate, mesh, screw and fixing rules of the task's mechanism (carrier welds, joint equalities for "
                  "the meshes, gate contacts, screw welds after two turns, the bridge welded once both are tight); the run "
                  "test (the mainspring a hinge stiffness released at the fixing, the click a joint limit the rules move, "
                  "a 5 s run window)")
