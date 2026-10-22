"""wire_splicing: splice two stiff solid-core wires, two ALOHA arms, MuJoCo only (design:
runs/wire_splice_design_20261014, README_v5 sections 1-5, README_v6, registration_draft.md).

The scene is four MJCF entities in the ALOHA mount frame (topogym.assets.geometry.wire_splicing): wire_a and wire_b
(each its terminal block and its elastoplastic chain, laid on the table), jig (the bending post) and meter (the splice
meter by the terminals, visual only). The wires are laid 1.5 mm above their resting height and settle under gravity
before the birth (the family's reset_physics_steps).

Goal (the design's v5_criteria.py, README_v5 5.2), from the two wires' centreline nodes: A's bare end bent into an eye
(turned back past 150 degrees, closed at the neck), B's bare wire through the eye once (linking number +-1), B's bare
tail wound at least 1.5 turns round A's doubled end, at least one full turn of the coil round both of A's strands
together (criteria.around_both: about each strand's own centreline, not only about the line between them; user decision
2026-10-22), tight (the coil's nodes within 2 core diameters of the axis on average), on the bare core (between the neck and A's strip, each widened by one diameter), A's tail end folded back
(its last three links pointing back toward the eye, its tip at least 1.2 diameters from the axis), no assisted grasp
active and the splice still (its bare nodes' largest speed relative to their mean at most 5 mm/s). The design judged the
state at release and 2 s later; here every condition must hold for stable_duration_s (2 s) on end, with no more than
max_turns_lost turns lost and the same linking number as at the window's first tick.

The splice meter (user decision 2026-10-19; meter.SpliceMeter, the task's mechanism) shows the shape conditions as a
modelled resistance, R = 10 mOhm / turns round both strands (the red band from 10 mOhm, one turn), no reading without an
eye, its threading and a coil on the bare core; its
needle is in the red band exactly when every shape condition holds (criteria.meter_reading). It only displays: the
predicate above is computed from the nodes as before (judge), and the meter's reading goes into the metrics."""
from pathlib import Path

import numpy as np

from topogym.assets.geometry import wire_splicing as geo
from topogym.tasks.base import EntitySpec, TaskDefinition, TaskSpec
from . import criteria as C

# The ALOHA's mount in the world (configs/env_configs/world_tabletop_aloha_v1.yaml robot_mount.pos x, y): the entities'
# origin, so the design's mount-frame coordinates hold as they are.
MOUNT_XY_M = (-.1356786, 0.)
WIRE_ENTITIES = ("wire_a", "wire_b")


def recommended_grasp_assist(variation_id, seed=0, *, task_parameters=None):
    from topogym.config.registry import task_definition
    return task_definition("wire_splicing", variation_id, seed)(task_parameters, seed=seed).grasp_assist()


class WireSplicingTask(TaskDefinition):
    task_id = "wire_splicing"
    physics_model = "rigid_articulated"
    terminal_revision = "wire_splice_eye_wrap_v5"       # v2 (user, 2026-10-20): 3 turns; v3 (2026-10-21): 1.5; v4: the core line from the stem and the tail beside it; v5 (2026-10-22): a full turn round both strands
    instruction = ("Splice the red and the blue wire. Bend the red wire's bare end into a small eye round the rod of the "
                   "bending post, thread the blue wire's bare end through the eye, turn the wires so that their insulated "
                   "parts point away from each other, wind the blue wire's bare tail tightly round the red wire's doubled "
                   "end one and a half turns, at least one full turn of it round both strands of the doubled end "
                   "together, fold the red wire's leftover end back over the coil, then let go. The "
                   "needle meter on the panel by the terminals reads this splice: its needle stays in the red band "
                   "only when the splice is complete.")
    defaults = dict(stable_duration_s=2., terminal=dict(
        min_wrap_turns=1.5, min_both_turns=1., max_coil_radius_d=2., tail_fold_dot=.5, min_tail_radius_d=1.2,
        max_turns_lost=.25, max_splice_speed_m_s=.005))

    def grasp_assist(self):
        """contact_weld_v1 on every link of both wires; a hand pinching both welds, of each wire, its touched link
        nearest the TCP and every other link whose centre lies between the two pads (MultiRopeGraspAssistConfig
        weld_each_rope, weld_links_between_pads; user decision 2026-10-18), a hand on one wire its bilaterally touched
        link nearest the TCP and that wire's other links between the pads. With hold_between_pads (user decision
        2026-10-19) a hand also holds, the same way, when both fingers touch the wires and every such contact lies
        between the pads: the red wire's two legs side by side, each pad on one, and the eye's two sides with the blue
        wire's lead beside them."""
        from topogym.utils.grasp import GraspTarget, MultiRopeGraspAssistConfig
        targets = []
        for entity in WIRE_ENTITIES:
            wire = geo.WIRE_OF[entity]
            for k, length in enumerate(geo.segments(wire)):
                r = geo.R_INS if k < geo.WIRES[wire]["n_ins"] else geo.R_CU
                targets.append(GraspTarget(entity, geo.link_name(entity, k), (0., 0., float(length / 2)),
                                           float(length / 2 + r + .002)))
        return MultiRopeGraspAssistConfig("contact_weld_v1", tuple(targets), weld_each_rope=WIRE_ENTITIES,
                                          weld_links_between_pads=True, hold_between_pads=True)

    def origin(self):
        return np.r_[MOUNT_XY_M, self.table_height_m]

    def declare(self, asset_dir, table_height_m):
        self.table_height_m = float(table_height_m)
        folder = Path(asset_dir)
        folder.mkdir(parents=True, exist_ok=True)
        entities = []
        for name in geo.ENTITIES:
            path = folder / f"{name}.xml"
            path.write_text(geo.entity_xml(name))
            entities.append(EntitySpec(name, "mjcf", path, tuple(self.origin()), fixed=name == "jig"))
        return TaskSpec(tuple(entities), {}, dict(
            task_id=self.task_id, instance_id=self.instance_id, physics_engine="mujoco",
            wires={e: dict(links=geo.n_links(geo.WIRE_OF[e]), strip_node=geo.WIRES[geo.WIRE_OF[e]]["n_ins"],
                           core_diameter_m=geo.CORE_D, insulation_diameter_m=geo.INS_D) for e in WIRE_ENTITIES},
            goal_requires_release=True))

    def finalize_spec(self, spec, *, asset_dir, scene_config):
        mount = np.asarray(scene_config.env.robot_mount.pos, dtype=float)
        if np.max(np.abs(mount[:2] - MOUNT_XY_M)) > 1e-9 or abs(mount[2] - self.table_height_m) > 1e-9 \
                or np.any(np.asarray(scene_config.env.robot_mount.euler, dtype=float)):
            raise ValueError("wire_splicing is laid out in the ALOHA tabletop's mount frame")
        return spec

    # ---------------------------------------------------------------- measurements
    @staticmethod
    def nodes(state, entity):
        """The wire's centreline nodes (world frame): every link origin, then the end of the last link."""
        from scipy.spatial.transform import Rotation
        wire = geo.WIRE_OF[entity]
        links = state["articulations"][entity]["links"]
        n = geo.n_links(wire)
        P = np.array([links[geo.link_name(entity, k)]["position_m"] for k in range(n)], dtype=float)
        q = np.asarray(links[geo.link_name(entity, n - 1)]["quaternion_wxyz"], dtype=float)
        tip = P[-1] + Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0., 0., geo.segments(wire)[-1]])
        return np.vstack([P, tip])

    @staticmethod
    def splice_speed(state):
        """The bare links' largest speed relative to their mean motion (m/s): the splice's own deformation."""
        V = []
        for entity in WIRE_ENTITIES:
            wire = geo.WIRE_OF[entity]
            links = state["articulations"][entity]["links"]
            V += [links[geo.link_name(entity, k)]["linear_velocity_m_s"]
                  for k in range(geo.WIRES[wire]["n_ins"], geo.n_links(wire))]
        V = np.asarray(V, dtype=float)
        return float(np.linalg.norm(V - V.mean(axis=0), axis=1).max())

    def goal_words(self):
        """(summary, success conditions) of the public goal (goal_description.describe_goal)."""
        t, d = self.config["terminal"], geo.CORE_D * 1000
        return ("Splice the red and the blue wire: bend the red wire's bare end into an eye round the rod of the bending "
                "post, thread the blue wire's bare end through the eye, wind the blue wire's bare tail round the red wire's "
                f"doubled end at least {t['min_wrap_turns']:g} turns, at least {t['min_both_turns']:g} full turn of them "
                "round both strands of the doubled end together, fold the red wire's leftover end back over the coil, "
                "then let go.", [
                    f"The red wire's bare end is turned back on itself by more than 150 degrees into an eye that is closed "
                    "at its neck: the two legs touch where they first meet below the bend.",
                    "The blue wire passes through the closed eye exactly once (its linking number with the eye is +1 or -1).",
                    f"The blue wire's bare tail winds at least {t['min_wrap_turns']:g} turns round the red wire's doubled "
                    "section. The turns are counted about the doubled section's axis: the line fitted to the red wire's stem "
                    "(its bare part from the strip to the neck) and the return-leg points that lie beside it, within "
                    f"{2.5 * d:.1f} mm ({2.5:g} core diameters; the core is {d:.2f} mm thick) of the stem's own line. The "
                    f"counted coil is the longest run of the blue wire's bare points within {2.5 * d:.1f} mm of that axis "
                    "and between the neck and the red wire's strip, each widened by one core diameter.",
                    f"At least {t['min_both_turns']:g} full turn of the coil goes round both strands of the red wire's "
                    "doubled end together, not only round the line between them. The strands: the stem side, from the "
                    "strip to the eye's apex (its point nearest the eye end of the axis), and the return side, from the apex "
                    "to the return leg's point farthest toward the strip (a leftover end folded back from there is not one "
                    "of them). For each point of the counted coil, each strand's point at the same station along the axis "
                    "is taken, where the strand crosses the plane through the coil point perpendicular to the axis, and the "
                    "coil point's angle about it in that plane; along the coil these angles accumulate into turns about "
                    "each strand, a step counting only where both strands have their points. Some contiguous part of the "
                    f"coil must wind at least {t['min_both_turns']:g} turn about each strand, both the same way as the "
                    "coil winds about the axis: a loop round one strand only, or one passing between the strands, does "
                    "not count.",
                    f"The coil is tight: its points lie on average at most {t['max_coil_radius_d'] * d:.1f} mm "
                    f"({t['max_coil_radius_d']:g} core diameters) from the axis.",
                    f"The red wire's leftover end is folded back: the direction from its fourth-last to its last point "
                    f"points back toward the eye (its dot product with the axis, which runs from the eye toward the strip, "
                    f"is at most -{t['tail_fold_dot']:g}), and its tip is at least {t['min_tail_radius_d'] * d:.1f} mm "
                    f"({t['min_tail_radius_d']:g} core diameters) from the axis.",
                    f"Let go: no hand holds a wire. The blue wire's turns may drop by at most {t['max_turns_lost']:g} and "
                    "the threading may not change while the conditions are held, and the splice must be still: every bare "
                    "point's speed relative to the bare points' mean motion at most "
                    f"{t['max_splice_speed_m_s'] * 1000:g} mm/s.",
                    "The needle meter on the panel by the terminals displays this splice: no reading unless the eye is "
                    "closed, the blue wire threads it once and the coil lies on the bare core; otherwise the equivalent "
                    "resistance 10 mOhm divided by the coil's turns round both strands, on a logarithmic dial from 100 to "
                    f"1 mOhm. Its needle enters the red band ({10. / t['min_both_turns']:g} mOhm or less) only when every "
                    "condition above but the stillness and the release holds. It is a display of these conditions, not a "
                    "further one.",
                ])

    def judge(self, PA, PB):
        """The measurement of the splice from the two wires' nodes and its shape conditions (the success conditions less
        the hold, quiet and hands-off ones); the task's predicate and the splice meter (meter.SpliceMeter) use it."""
        t = self.config["terminal"]
        m = C.measure(PA, PB, geo.WIRES["A"]["n_ins"], geo.WIRES["B"]["n_ins"], geo.CORE_D)
        return m, C.shape_checks(m, geo.CORE_D, min_turns=t["min_wrap_turns"], min_both_turns=t["min_both_turns"],
                                 max_coil_r_D=t["max_coil_radius_d"], tail_dot=t["tail_fold_dot"],
                                 min_tail_r_D=t["min_tail_radius_d"])

    def mujoco_mechanism(self, world):
        from .meter import SpliceMeter
        return SpliceMeter(world, self.judge)

    def initial_task_state(self, world_state):
        return dict(hold_reference=None)

    def validate_task_state(self, state):
        super().validate_task_state(state)
        if state and set(state) != {"hold_reference"}:
            raise ValueError("Invalid wire_splicing history")
        reference = (state or {}).get("hold_reference")
        if reference is not None and set(reference) != {"wrap_turns", "pierce"}:
            raise ValueError("Invalid wire_splicing hold reference")

    def evaluate_state(self, state):
        return self.evaluate_transition(state, self.task_state)[0:3]

    def evaluate_transition(self, state, task_state):
        for entity in WIRE_ENTITIES:
            points = np.array([x["position_m"] for x in state["articulations"][entity]["links"].values()])
            if not np.isfinite(points).all():
                return False, "Non-finite wire state", {}, task_state
        t = self.config["terminal"]
        m, checks = self.judge(self.nodes(state, "wire_a"), self.nodes(state, "wire_b"))
        speed = self.splice_speed(state)
        checks.update(quiet=speed <= t["max_splice_speed_m_s"], hands_off=not state.get("assist_active", False))
        now = all(checks.values())
        reference = (task_state or {}).get("hold_reference")
        if now:
            if reference is None:       # the window's first tick: the turns and the linking number to hold
                reference = dict(wrap_turns=float(m["wrap_turns"]), pierce=int(m["pierce"]))
            held = (reference["wrap_turns"] - m["wrap_turns"] <= t["max_turns_lost"]
                    and m["pierce"] == reference["pierce"])
        else:
            held = False
        if not held:
            reference = None
        checks["held"] = held
        metrics = {k: v for k, v in m.items()}
        if state.get("mechanism") is not None:      # the splice meter's reading of this tick (display only)
            metrics["meter"] = dict(state["mechanism"])
        metrics.update(checks=checks, splice_speed_m_s=speed, evaluation_status="available",
                       goal_evaluation_complete=self.goal_complete)
        return bool(now and held), None, metrics, dict(hold_reference=reference)
