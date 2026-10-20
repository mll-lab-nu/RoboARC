"""bed_bolt: drive a long hex-socket bolt into a blind hole in a bed frame's corner block with an L-shaped hex key, two
ALOHA arms (design: runs/bed_bolt_design_20261012, README sections 7-9).

The scene is three MJCF entities in the ALOHA mount frame (topogym.assets.geometry.bed_bolt): bed_frame (static), bolt
(the bolt and the thread carriage) and key. The thread is the task's mechanism (thread.BoltThread), stepped by the MuJoCo
world every tick: the task runs on MuJoCo only.

Goal: the bolt seated (the thread caught and the head within seat_tolerance_m of the block face), its axis within
axis_tolerance_rad of the bore's, the bolt and the key still, for stable_duration_s."""
from pathlib import Path

import numpy as np

from topogym.assets.geometry import bed_bolt as geo
from topogym.tasks.base import EntitySpec, RigidTaskDefinition, TaskSpec

# The ALOHA's mount in the world (configs/env_configs/world_tabletop_aloha_v1.yaml robot_mount.pos x, y): the entities'
# origin, so the design's mount-frame coordinates hold as they are.
MOUNT_XY_M = (-.1356786, 0.)


def recommended_grasp_assist(variation_id, seed=0, *, task_parameters=None):
    from topogym.config.registry import task_definition
    return task_definition("bed_bolt", variation_id, seed)(task_parameters, seed=seed).grasp_assist()


class BedBoltTask(RigidTaskDefinition):
    task_id = "bed_bolt"
    physics_model = "rigid_articulated"
    terminal_revision = "bed_bolt_seated_v1"
    defaults = dict(stable_duration_s=.5, terminal=dict(seat_tolerance_m=.0015, axis_tolerance_rad=float(np.radians(2.))))
    # The key is 20 cm long: a hand holding it near the bend has its TCP far from the key's centre (Rotor's setting).
    grasp_max_tcp_distance_m = .15
    # The bolt's recess and the tool's tips (topogym.assets.geometry.bed_bolt DRIVES); the public words for the tool
    drive = "hex"
    tool_words = dict(hex="L-shaped hex key", torx="L-shaped Torx key (a six-lobe star tip)",
                      square="ratcheting L-shaped square key (a square tip on a ratchet)",
                      phillips="ratcheting L-shaped Phillips key (a cross tip on a ratchet)",
                      slotted="ratcheting L-shaped slotted key (a flat blade on a ratchet)")

    def grasp_assist(self):
        """contact_weld_v1 on the bolt and on the key, each as a whole part."""
        from topogym.utils.grasp import GraspAssistConfig, rigid_grasp_region
        return GraspAssistConfig("contact_weld_v1", (rigid_grasp_region("bolt", "bolt", geo.bolt_points()),
                                                     rigid_grasp_region("key", "key", geo.key_points())),
                                 max_tcp_distance_m=self.grasp_max_tcp_distance_m)

    def origin(self):
        return np.r_[MOUNT_XY_M, self.table_height_m]

    def declare(self, asset_dir, table_height_m):
        self.table_height_m = float(table_height_m)
        folder = Path(asset_dir)
        folder.mkdir(parents=True, exist_ok=True)
        entities = []
        for name in geo.ENTITIES:
            path = folder / f"{name}.xml"
            path.write_text(geo.entity_xml(name, drive=self.drive))
            entities.append(EntitySpec(name, "mjcf", path, tuple(self.origin()), fixed=name == "bed_frame"))
        self.local_vertices = {"bolt": geo.bolt_points(), "key": geo.key_points()}
        hole = self.hole_T()
        return TaskSpec(tuple(entities), {}, dict(
            task_id=self.task_id, instance_id=self.instance_id, physics_engine="mujoco",
            hole_mouth_m=hole[:3, 3].tolist(), hole_axis=hole[:3, 0].tolist(), bolt_length_m=geo.BOLT["L"],
            lead_m=geo.BOLT["lead"], thread=dict(geo.THREAD), goal_requires_release=False,
            **({} if self.drive == "hex" else dict(drive=self.drive))))

    def finalize_spec(self, spec, *, asset_dir, scene_config):
        mount = np.asarray(scene_config.env.robot_mount.pos, dtype=float)
        if np.max(np.abs(mount[:2] - MOUNT_XY_M)) > 1e-9 or abs(mount[2] - self.table_height_m) > 1e-9 \
                or np.any(np.asarray(scene_config.env.robot_mount.euler, dtype=float)):
            raise ValueError("bed_bolt is laid out in the ALOHA tabletop's mount frame")
        return spec

    def hole_T(self):
        """The bore mouth frame in the world (x into the bore)."""
        T = geo.hole_T()
        T[:3, 3] += self.origin()
        return T

    def mujoco_mechanism(self, world):
        from .thread import BoltThread
        fingers = {i for links in self._finger_links_by_hand.values() for i in links}
        return BoltThread(world, self.hole_T(), fingers, drive=self.drive)

    def evaluate_state(self, state):
        from scipy.spatial.transform import Rotation
        thread = state.get("mechanism")
        if thread is None:
            raise RuntimeError("bed_bolt runs on MuJoCo: the world reads no thread")
        poses, _, quiet, failure = self.rigid_measurements(state)
        if failure:
            return False, failure, {}
        position, rotation = poses["bolt"]
        hole = self.hole_T()
        depth = float((position - hole[:3, 3]) @ hole[:3, 0])           # the bolt frame's origin is its tip
        axis = float(np.arccos(np.clip(-rotation.apply([1., 0., 0.]) @ hole[:3, 0], -1., 1.)))
        slide = float(state["articulations"]["bolt"]["joints"]["thr_s"]["position"][0])
        terminal = self.config["terminal"]
        seated = bool(thread["engaged"] and slide >= geo.BOLT["L"] - terminal["seat_tolerance_m"])
        aligned = axis <= terminal["axis_tolerance_rad"]
        return seated and aligned and quiet, None, dict(
            evaluation_status="available", seated=seated, aligned=aligned, quiet=bool(quiet), engaged=bool(thread["engaged"]),
            engaged_turns=float(thread["turns"]), tip_depth_m=depth, axis_angle_rad=axis, key_slipping=bool(thread["slipped"]),
            seat_depth_m=geo.BOLT["L"], holding_allowed=True)

    def goal_words(self):
        """(summary, success conditions) of the public goal (goal_description.describe_goal)."""
        t, c = self.config["terminal"], geo.THREAD
        tool = self.tool_words[self.drive]
        return (f"Drive the long bolt into the blind hole in the bed frame's corner block with the {tool} until its "
                f"head is within {t['seat_tolerance_m'] * 1000:g} mm of the block face.", [
                    f"Drive the bolt in until its head is within {t['seat_tolerance_m'] * 1000:g} mm of the block face, "
                    f"its tip at least {(geo.BOLT['L'] - t['seat_tolerance_m']) * 1000:g} mm deep in the hole, with the thread "
                    f"caught, and its axis within {np.degrees(t['axis_tolerance_rad']):g} degrees of the hole's axis.",
                    f"The hole is threaded. The thread catches when the bolt's tip is {c['catch_depth_m'] * 1000:g} to "
                    f"{c['catch_depth_max_m'] * 1000:g} mm inside the hole, within {c['catch_radial_m'] * 1000:g} mm of the hole's "
                    f"axis and {c['catch_angle_deg']:g} degrees of its direction, with the bolt's thread lined up with the "
                    "hole's (a straight push lines up somewhere in that range; turning the bolt about its axis moves where); "
                    f"each turn then drives the bolt {geo.BOLT['lead'] * 1000:g} mm in. A bolt the thread does not hold cannot "
                    f"be pushed in past the thread: not past {c['catch_depth_max_m'] * 1000:g} mm before it has ever caught, and "
                    "afterwards not deeper than it has already been turned in; it catches again only there, in the same "
                    f"thread. With fewer than {c['hold_turns']:g} turns "
                    "engaged the thread cannot carry the bolt: when no finger and no key end holds it, it droops, and past "
                    f"{c['fall_tilt_deg']:g} degrees it lets go and falls out.",
                    "Once the thread has engaged, pushing or pulling the bolt along its axis does not advance it. Fingers "
                    "cannot turn the bolt (they slip); use the key.",
                    *self.ratchet_words(),
                    "Keep the bolt's and the key's linear speed at most 1 mm/s and angular speed at most 0.02 rad/s. Holding "
                    "is allowed; the key may stay in the socket or anywhere else.",
                ])

    def ratchet_words(self):
        """The ratcheting key's one plain sentence (DRIVES ratchet), none for the plain keys."""
        c = geo.DRIVES[self.drive]
        if not c.get("ratchet"):
            return []
        words = ["The key has a ratchet at its bend: swung in the tightening direction it turns its tip, and with it the bolt; "
                 "swung back, the handle turns freely (with a slight drag) while the tip stays still in the socket, so the tip "
                 "need not come out between swings."]
        if "across" in c["in_socket"]:
            words.append("The slot runs across the whole head and is open at both ends: the blade can slide out along it.")
        return words

    def terminal_metadata(self):
        return super().terminal_metadata() | dict(
            goal_kind="bolt_seated", seat_depth_m=geo.BOLT["L"], lead_m=geo.BOLT["lead"],
            thread="a carriage at the bore mouth, the lead a joint equality, a weld to the bolt switched by the task's "
                   "mechanism (catch lined up with the hole's thread within its first 8.2 mm or where it has been turned in, "
                   "a stop there for a loose bolt, hold from 2 turns, let go below them)")
