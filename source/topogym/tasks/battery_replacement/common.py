"""battery_replacement: open a game controller's battery cover, pry the two old AA cells out with the pry stick, put
them in the recycling box, seat two new cells with the right polarity and latch the cover, two ALOHA arms (design:
runs/controller_battery_design_20261014, README.md v2 section 5 and README_v3.md v3b).

The scene is seven MJCF entities in the ALOHA mount frame (topogym.assets.geometry.battery_replacement): controller (the
static shell with its cradle, the two spring contacts and the detent plunger on slide joints, and the props: the stick
holder, the tray, the recycling box, the cover's pad), cover, old_A, old_B, new_1, new_2 and stick (free bodies). Every
mechanism is physical (contacts, joint springs, hard limits); the task runs on MuJoCo.

Grasping: contact_weld_v1 on the cover, the stick's handle and the four cells (rule C of the design's GRASP_PROPOSAL.md,
the user's choice of 2026-10-18). The old cells are WeakHoldGraspTarget (mu 1, firm at 0.268, 60 mm): two fingers
touching a seated cell's top cannot take it; a cell lifted by the stick can be plucked out by a fingertip pinch (central
angle between the two contacts at least 90 deg), but a hold below 150 deg is lost once the cell has moved 60 mm, so a
cell is carried only held from both sides. The new cells are ForceClosureGraspTarget (mu 0.268): taken only by a pinch
whose contacts are at least 150 deg apart around the cell.

Goal (all held for stable_duration_s): each slot holds a new cell, seated with the right polarity (axis within
axis_tolerance_rad of the slot's, centre within lateral_tolerance_m sideways (1.2 mm: the slot's free play is 1 mm
each side, and a cell resting against a slot wall measures 1.03 mm off centre, its contact sinking slightly; the user's
choice of 2026-10-17),
vertical_tolerance_m vertically and axial_tolerance_m along the axis of the seated pose, the slot's spring contact
compressed at least spring_min_compression_m); the cover closed
and latched (within cover_tolerance_m in x and y, between -cover_sink_m and cover_lift_m in z, cover_tilt_rad, the
plunger in its groove); both old cells inside the recycling box; the cells and the cover at most quiet_speed_m_s and
quiet_spin_rad_s. Holding is allowed; where the stick ends does not matter."""
from pathlib import Path

import numpy as np

from topogym.assets.geometry import battery_replacement as geo
from topogym.tasks.base import EntitySpec, RigidTaskDefinition, TaskSpec

# The ALOHA's mount in the world (configs/env_configs/world_tabletop_aloha_v1.yaml robot_mount.pos x, y): the entities'
# origin, so the design's mount-frame coordinates hold as they are.
MOUNT_XY_M = (-.1356786, 0.)
# A firm hold of a cell: its two finger contacts at least 150 deg apart around the cell (force closure at mu = tan 15 deg);
# a weak (fingertip) hold of an old cell is lost once the cell has moved this far from where it was taken.
FIRM_MU, WEAK_TRAVEL_M = .268, .06
# The new cells' polarity in the tray per scene seed (seed % 4): '+' = that cell's + end toward +x.
POLARITIES = ("+-", "-+", "++", "--")


def recommended_grasp_assist(variation_id, seed=0, *, task_parameters=None):
    from topogym.config.registry import task_definition
    return task_definition("battery_replacement", variation_id, seed)(task_parameters, seed=seed).grasp_assist()


class BatteryReplacementTask(RigidTaskDefinition):
    task_id = "battery_replacement"
    physics_model = "rigid_articulated"
    terminal_revision = "battery_replacement_swap_v1"
    defaults = dict(stable_duration_s=.5, terminal=dict(
        axis_tolerance_rad=float(np.radians(5.)), lateral_tolerance_m=.0012, vertical_tolerance_m=.001, axial_tolerance_m=.0015,
        spring_min_compression_m=.001, cover_tolerance_m=.0005, cover_sink_m=.0002, cover_lift_m=.0006,
        cover_tilt_rad=float(np.radians(1.)), plunger_min_m=.00025, quiet_speed_m_s=.005, quiet_spin_rad_s=.1))

    @property
    def polarity(self):
        return POLARITIES[self.seed % len(POLARITIES)]

    def grasp_assist(self):
        """contact_weld_v1 on the cover, the stick's handle and the four cells, each as a whole part; the old cells on a
        force-closure pinch, carried only by a firm one (WeakHoldGraspTarget); the new cells only on a firm pinch."""
        from topogym.utils.grasp import ForceClosureGraspTarget, GraspAssistConfig, WeakHoldGraspTarget, rigid_grasp_region
        targets = [rigid_grasp_region("cover", "cover", geo.cover_points()),
                   rigid_grasp_region("stick", "stick", geo.handle_points())]
        for name in geo.CELLS:
            region = rigid_grasp_region(name, "cell", geo.cell_points())
            if name.startswith("old"):
                region = WeakHoldGraspTarget(region.entity, region.link, region.center_m, region.radius_m,
                                             friction_coefficient=1., firm_friction_coefficient=FIRM_MU,
                                             weak_hold_travel_m=WEAK_TRAVEL_M)
            else:
                region = ForceClosureGraspTarget(region.entity, region.link, region.center_m, region.radius_m,
                                                 friction_coefficient=FIRM_MU)
            targets.append(region)
        return GraspAssistConfig("contact_weld_v1", tuple(targets))

    def origin(self):
        return np.r_[MOUNT_XY_M, self.table_height_m]

    def world_T(self, T_mount):
        """A mount-frame pose in the world."""
        T = np.array(T_mount, dtype=float)
        T[:3, 3] += self.origin()
        return T

    def declare(self, asset_dir, table_height_m):
        self.table_height_m = float(table_height_m)
        folder = Path(asset_dir)
        folder.mkdir(parents=True, exist_ok=True)
        entities = []
        for name in geo.ENTITIES:
            path = folder / f"{name}.xml"
            path.write_text(geo.entity_xml(name, self.polarity))
            # The controller's shell body has no joint: MuJoCo welds it to the world (as fiber_bundling's terminal); the
            # entity is not declared fixed, so nyx_direct draws its visuals at the body (it draws no fixed MJCF entity).
            entities.append(EntitySpec(name, "mjcf", path, tuple(self.origin())))
        self.local_vertices = {name: geo.cell_points() for name in geo.CELLS}
        self.local_vertices |= {"cover": geo.cover_points(), "stick": geo.stick_points()}
        return TaskSpec(tuple(entities), {}, dict(
            task_id=self.task_id, instance_id=self.instance_id, physics_engine="mujoco", polarity=self.polarity,
            compartment_T=self.world_T(geo.compartment_T()).tolist(),
            seated_T={s: self.world_T(geo.seated_T(s)).tolist() for s in "AB"},
            cover_closed_T=self.world_T(geo.cover_closed_T()).tolist(),
            recycling_box_xy_m=(np.asarray(geo.BOX_XY_M) + MOUNT_XY_M).tolist(), goal_requires_release=False))

    def finalize_spec(self, spec, *, asset_dir, scene_config):
        mount = np.asarray(scene_config.env.robot_mount.pos, dtype=float)
        if np.max(np.abs(mount[:2] - MOUNT_XY_M)) > 1e-9 or abs(mount[2] - self.table_height_m) > 1e-9 \
                or np.any(np.asarray(scene_config.env.robot_mount.euler, dtype=float)):
            raise ValueError("battery_replacement is laid out in the ALOHA tabletop's mount frame")
        return spec

    def _slot(self, slot, poses, joints):
        """The cell seated in slot (name, polarity right) or None, and the slot's measurements."""
        from scipy.spatial.transform import Rotation
        t = self.config["terminal"]
        seat = self.world_T(geo.seated_T(slot))
        plus = seat[:3, 0]                                      # the slot's + direction
        compression = geo.SPRING_PRE * geo.MM - float(joints[f"spring_{slot}"]["position"][0])
        found, measured = None, {}
        for name in geo.CELLS:
            position, rotation = poses[name]
            axis = rotation.apply([1., 0., 0.])
            offset = seat[:3, :3].T @ (position - seat[:3, 3])      # in the seated cell's frame: x along, y across, z up
            angle = float(np.arccos(np.clip(abs(axis @ plus), -1., 1.)))
            if (angle <= t["axis_tolerance_rad"] and abs(offset[1]) <= t["lateral_tolerance_m"]
                    and abs(offset[2]) <= t["vertical_tolerance_m"] and abs(offset[0]) <= t["axial_tolerance_m"]
                    and compression >= t["spring_min_compression_m"]):
                found = (name, bool(axis @ plus > 0))
                measured = dict(cell=name, polarity_right=found[1], axis_angle_rad=angle, offset_m=offset.tolist())
        return found, dict(measured, spring_compression_m=compression)

    def evaluate_state(self, state):
        from scipy.spatial.transform import Rotation
        poses, _, _, failure = self.rigid_measurements(state)
        if failure:
            return False, failure, {}
        t = self.config["terminal"]
        joints = state["articulations"]["controller"]["joints"]
        slots, metrics = {}, {}
        for slot in "AB":
            slots[slot], metrics[f"slot_{slot}"] = self._slot(slot, poses, joints)
        swapped = all(slots[s] is not None and slots[s][0].startswith("new") and slots[s][1] for s in "AB")
        closed = self.world_T(geo.cover_closed_T())
        position, rotation = poses["cover"]
        d = closed[:3, :3].T @ (position - closed[:3, 3])
        tilt = float((Rotation.from_matrix(closed[:3, :3]).inv() * rotation).magnitude())
        plunger = float(joints["plunger"]["position"][0])
        latched = bool(abs(d[0]) <= t["cover_tolerance_m"] and abs(d[1]) <= t["cover_tolerance_m"]
                       and -t["cover_sink_m"] <= d[2] <= t["cover_lift_m"] and tilt <= t["cover_tilt_rad"]
                       and plunger >= -t["plunger_min_m"])
        box = np.asarray(geo.BOX_XY_M) + MOUNT_XY_M
        rim = self.table_height_m + (geo.BOX_IN[2] + 3.) * geo.MM
        inner = np.asarray(geo.BOX_IN[:2]) * geo.MM / 2
        recycled = all(bool(np.all(np.abs(poses[n][0][:2] - box) <= inner) and poses[n][0][2] <= rim) for n in ("old_A", "old_B"))
        quiet = True
        for name in (*geo.CELLS, "cover"):
            body = state["bodies"][name]
            quiet = quiet and np.linalg.norm(body["linear_velocity_m_s"]) <= t["quiet_speed_m_s"] \
                and np.linalg.norm(body["angular_velocity_rad_s"]) <= t["quiet_spin_rad_s"]
        return swapped and latched and recycled and bool(quiet), None, dict(
            evaluation_status="available", swapped=swapped, cover_latched=latched, recycled=recycled, quiet=bool(quiet),
            cover_offset_m=d.tolist(), cover_tilt_rad=tilt, plunger_m=plunger, holding_allowed=True, **metrics)

    def goal_words(self):
        """(summary, success conditions) of the public goal (goal_description.describe_goal)."""
        t = self.config["terminal"]
        mm = lambda v: f"{v * 1000:g} mm"
        tilt = round(float(np.degrees(t["cover_tilt_rad"])), 6)
        return ("Replace the two old AA cells in the game controller's battery compartment with the two new ones, with the "
                "right polarity, put the old cells in the recycling box and close the battery cover.", [
                    f"Each of the two slots holds a new cell (black body, copper band), fully seated: its axis within "
                    f"{np.degrees(t['axis_tolerance_rad']):g} degrees of the slot's, its centre within {mm(t['lateral_tolerance_m'])} "
                    f"sideways, {mm(t['vertical_tolerance_m'])} vertically and {mm(t['axial_tolerance_m'])} along the axis of the "
                    f"seated position, and the slot's spring contact compressed at least {mm(t['spring_min_compression_m'])}.",
                    "Polarity: a cell's + end (the small nub, and the '+' printed on its coloured band) faces that slot's + "
                    "contact plate, marked with a red '+' on the compartment floor; its - end presses the spring contact.",
                    f"The cover is closed and latched: within {mm(t['cover_tolerance_m'])} of its closed position along and "
                    f"across the controller, between {mm(t['cover_sink_m'])} below and {mm(t['cover_lift_m'])} above it, "
                    f"tilted at most {tilt:g} degree{'s' * (tilt != 1.)}, its detent plunger in the groove.",
                    "Both old cells (grey body, tan band) lie inside the recycling box (their centres within its inner walls, "
                    "below its rim).",
                    f"The four cells and the cover move at most {mm(t['quiet_speed_m_s'])}/s and {t['quiet_spin_rad_s']:g} "
                    "rad/s. Holding parts is allowed; the pry stick may end anywhere on the table.",
                ])

    def terminal_metadata(self):
        return super().terminal_metadata() | dict(
            goal_kind="battery_swap", polarity=self.polarity,
            grasp_rules="contact_weld_v1; the old cells on a force-closure pinch (mu 1), a hold below 150 deg (mu 0.268) "
                        "lost after 60 mm (WeakHoldGraspTarget); the new cells only at 150 deg or more (ForceClosureGraspTarget, mu 0.268)")
