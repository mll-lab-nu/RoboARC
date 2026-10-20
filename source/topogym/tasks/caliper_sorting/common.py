"""Caliper sorting: four sleeves that look alike stand in a rack; they are of two classes, which differ in one diameter by
three vernier divisions. The robot measures them with a vernier caliper fixed face up on a stand, its scale in its own
unit (it brings each sleeve to the jaws, moves the slider and turns the slider's locking screw, reads the vernier) and
puts each into the bin whose label gives its two readings on this caliper.

Scene and parts: topogym.assets.geometry.caliper_sorting. Design record: runs/caliper_sorting_design_20261014
(README_v3.md, REGISTRATION_PROPOSAL.md); family notes: docs/rcb/families/caliper_sorting.md."""
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.tasks.base import EntitySpec, RigidTaskDefinition, TaskSpec


class CaliperSortingTask(RigidTaskDefinition):
    task_id = "caliper_sorting"
    variation_id = "caliper_sorting_vernier"
    family_id = "vernier_caliper_sleeves"
    model_revision = "caliper_sorting_v3"
    # v2 (2026-10-19, the user, the metric review runs/robotangle_mujoco/metric_review_20261019/caliper_zh.md): every
    # sleeve released (no robot contact), and a sleeve left still in another class's bin ends the episode in failure
    terminal_revision = "sleeves_in_labelled_bins_v2"
    difficulty = None
    defaults = dict(stable_duration_s=.5,
                    terminal=dict(bin_half_extent_m=.04, min_height_m=.003, max_height_m=.04,
                                  max_linear_speed_m_s=.001, max_angular_speed_rad_s=.02))
    # Every sleeve and the slider are grasp targets of both hands: a sleeve as a whole link, the slider within 20 mm of its
    # grip (slider x 44 mm), so that a pinch on the locking screw's knob never takes hold of it. The caliper's beam is fixed
    # on its stand (2026-10-18) and no target; the knob is no target either: it turns by friction only.
    grasp_max_tcp_distance_m = .15

    def declare(self, asset_dir, table_height_m):
        from topogym.assets.geometry.caliper_sorting import build_caliper_sorting
        self.table_height_m = float(table_height_m)
        scene = build_caliper_sorting(Path(asset_dir), self.seed)
        self.scenario = scene.scenario
        self.sleeves = tuple(f"sleeve_{k}" for k in range(len(scene.scenario["sleeves"])))
        self.local_vertices = dict(scene.local_vertices)
        self.bin_centres_mount = [np.r_[b, 0.] for b in scene.bins]
        entities = tuple(EntitySpec(e.name, "urdf", e.path, e.pos, e.quat, fixed=e.fixed, material={"friction": .3},
                                    morph_options={"merge_fixed_links": False, **e.options},
                                    initial_joint_positions=dict(e.joints))
                         for e in scene.entities)
        self.weld_targets = tuple(dict(entity=entity, link=link, center_m=np.asarray(centre).tolist(), radius_m=radius)
                                  for (entity, link), (centre, radius) in scene.grasp_regions.items())
        return TaskSpec(entities, {"rigid": {"iterations": 100, "max_collision_pairs": 8192, "constraint_timeconst": .001}},
                        dict(task_id=self.task_id, instance_id=self.instance_id, scenario=scene.scenario,
                             initial_state="caliper closed and locked on its stand, sleeves in the rack; robot at home",
                             holding_allowed=False, recommended_grasp_assist=dict(
                                 model="contact_weld_v1", initial_intent="open", targets=list(self.weld_targets))))

    def finalize_spec(self, spec, *, asset_dir, scene_config):
        """The scene is declared in the ALOHA mount frame (the robot's base frame: env.robot_mount plus its riser; table
        top z = 0 without a riser): moved into the world as interlocked_claws moves its source frame."""
        mount = scene_config.env.robot_mount
        rotation = Rotation.from_euler('xyz', mount.euler, degrees=True)
        offset = np.asarray(mount.pos, dtype=float) + rotation.apply([0., 0.,
            float(scene_config.robot_control.robot.get('base_riser_height_m', 0.))])
        entities = tuple(replace(e, position_m=tuple(offset + rotation.apply(e.position_m)),
            quaternion_wxyz=tuple((rotation * Rotation.from_quat(np.asarray(e.quaternion_wxyz)[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]))
            for e in spec.entities)
        self.mount_rotation, self.mount_offset = rotation, offset
        self.bin_centres = [offset + rotation.apply(c) for c in self.bin_centres_mount]
        return super().finalize_spec(replace(spec, entities=entities), asset_dir=asset_dir, scene_config=scene_config)

    def mujoco_mechanism(self, world):
        """The locking screw: locks the slider where it stands from 20 deg off the loose stop, frees it at 10 deg."""
        from .lock import CaliperLock
        return CaliperLock(world)

    # -- goal
    def _in_bin(self, position_m, kb):
        """A centre (world, m) inside bin kb's box: within the half extent of its centre in the mount's x and y, between
        the two heights above the mount plane."""
        t = self.config["terminal"]
        rel = self.mount_rotation.inv().apply(np.asarray(position_m, dtype=float) - self.bin_centres[kb])
        return bool(abs(rel[0]) <= t["bin_half_extent_m"] and abs(rel[1]) <= t["bin_half_extent_m"]
                    and t["min_height_m"] <= rel[2] <= t["max_height_m"])

    def evaluate_state(self, state):
        """Success now: every sleeve in the bin labelled with its class, every sleeve still, none held by an assisted
        grasp and none touched by the robot (released). Also reports the sleeves left in another class's bin (still, not
        held, not touched): evaluate_transition times them."""
        _, _, _, failure = self.rigid_measurements(state)
        if failure:
            return False, failure, {}
        t = self.config["terminal"]
        held = sorted({h["binding"]["target"][0] for h in ((state.get("grasp") or {}).get("hands") or {}).values()
                       if h.get("binding") is not None} & set(self.sleeves))
        placed, speeds, bins, wrong = [], [], [], []
        for k, name in enumerate(self.sleeves):
            body = state["bodies"][name]
            cls = self.scenario["sleeves"][k]["cls"]
            kb = self.scenario["bins"].index(cls)
            v = float(np.linalg.norm(body["linear_velocity_m_s"]))
            w = float(np.linalg.norm(body["angular_velocity_rad_s"]))
            still = v <= t["max_linear_speed_m_s"] and w <= t["max_angular_speed_rad_s"]
            placed.append(self._in_bin(body["position_m"], kb))
            wrong.append(bool(still and name not in held and self.released(state, [name])
                              and any(self._in_bin(body["position_m"], j) for j in range(len(self.bin_centres)) if j != kb)))
            speeds.append((v, w))
            bins.append(kb)
        quiet = all(v <= t["max_linear_speed_m_s"] and w <= t["max_angular_speed_rad_s"] for v, w in speeds)
        released = self.released(state, self.sleeves)
        reached = bool(all(placed) and quiet and not held and released)
        return reached, None, dict(sleeves_in_their_bins=int(sum(placed)), sleeve_count=len(self.sleeves),
                                   sleeve_in_its_bin=placed, sleeve_target_bin=bins, held_sleeves=held,
                                   sleeves_released=bool(released), sleeve_left_in_wrong_bin=wrong,
                                   maximum_sleeve_speed_m_s=max(v for v, _ in speeds),
                                   maximum_sleeve_angular_speed_rad_s=max(w for _, w in speeds), sleeves_quiet=bool(quiet))

    def initial_task_state(self, world_state):
        """For each sleeve, the simulated time (s) since which it has stood still in another class's bin, -1 if not."""
        return dict(wrong_bin_since_s=[-1.] * len(self.sleeves))

    def validate_task_state(self, state):
        super().validate_task_state(state)
        if state and (set(state) != {"wrong_bin_since_s"} or len(state["wrong_bin_since_s"]) != len(self.sleeves)):
            raise ValueError("Invalid caliper_sorting history")

    def evaluate_transition(self, state, task_state):
        """A sleeve left still in another class's bin (no assisted grasp on it, no robot touching it) for the goal's
        confirmation time (stable_duration_s) ends the episode in failure: the episode ends at the first success, so
        without this a sorter could learn the right bins by trying placements one after the other."""
        reached, failure, metrics = self.evaluate_state(state)
        history = task_state or self.initial_task_state(state)
        if failure:
            return reached, failure, metrics, history
        now = float(state["sim_time_s"])
        since = [(s if s >= 0 else now) if w else -1. for s, w in zip(history["wrong_bin_since_s"],
                                                                         metrics["sleeve_left_in_wrong_bin"])]
        limit = float(self.config.get("stable_duration_s", .5))
        late = [k for k, s in enumerate(since) if s >= 0 and now - s + 1e-9 >= limit]
        metrics = {**metrics, "wrong_bin_time_s": [round(now - s, 6) if s >= 0 else 0. for s in since]}
        if late:
            return False, "A sleeve was left in a bin that is not its own", metrics, dict(wrong_bin_since_s=since)
        return reached, None, metrics, dict(wrong_bin_since_s=since)

    def terminal_metadata(self):
        t = self.config["terminal"]
        return super().terminal_metadata() | dict(
            goal_kind="every_sleeve_in_the_bin_labelled_with_its_class",
            rule=(f"each sleeve's centre within {t['bin_half_extent_m'] * 1000:g} mm of its class's bin centre in x and y "
                  f"and {t['min_height_m'] * 1000:g}-{t['max_height_m'] * 1000:g} mm above the table; every sleeve at most "
                  f"{t['max_linear_speed_m_s'] * 1000:g} mm/s and {t['max_angular_speed_rad_s']:g} rad/s; no assisted "
                  "grasp on a sleeve and no robot contact with one (released); the caliper is not judged; a sleeve "
                  "left still, released, in another class's bin for the confirmation time is a failure"),
            sleeve_classes=[s["cls"] for s in self.scenario["sleeves"]], bin_labels=list(self.scenario["bins"]),
            class_readings=dict(self.scenario["classes"]))


def recommended_grasp_assist(variation_id, seed=0, *, task_parameters=None):
    """contact_weld_v1 on the slider (about its grip) and every sleeve (a whole link), for both hands."""
    import tempfile
    from topogym.config.registry import task_definition
    from topogym.utils.grasp import GraspAssistConfig, GraspTarget
    task = task_definition("caliper_sorting", variation_id, seed)(task_parameters, seed=seed)
    with tempfile.TemporaryDirectory() as folder:
        task.declare(Path(folder), 0.)
    return GraspAssistConfig("contact_weld_v1", tuple(GraspTarget(t["entity"], t["link"], tuple(t["center_m"]), t["radius_m"])
                                                      for t in task.weld_targets),
                             max_tcp_distance_m=task.grasp_max_tcp_distance_m)
