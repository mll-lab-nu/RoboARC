"""Orbits easy / medium / hard with the cord as a rigid capsule chain: proposed settings, the FEM settings are unchanged.

The fixture, the handle and the chain are one MJCF entity (topogym.assets.geometry.orbits_chain_tiers) for Genesis'
rigid solver (the simulator settings go with it: CPU, float64, one 2.5 ms substep, CG, one global constraint solve).
The geometry is the FEM setting's with the changes approved for the chain (the user, 2026-10-08):
- the fixture 1.4 times larger (every length but the cord's radius) and the stopper ball 0.6 times that, room for the
  robot's hands round the rings and the posts;
- the base 15 mm thinner (the rings, the posts' tops and the ball where they were) and ring 1's radius 1 cm larger,
  its centre 1 cm further -x (its +x rim stays on P1): the hands' reach under the rings and round ring 1;
- ring 2 and every post and ring after it (P2 .. P_n, rings 2 .. n) 20 mm further +x, only the P1-P2 spacing grows
  (the coordinator, 2026-10-09; development setting L0_bt15_r1b1_r2on20): room for the hands threading ring 2 from
  below, the words unchanged; until then chain231_l0bt15r1b1_v1 (ring2_dx_m 0, K0 chain_k0_v2);
- hard only: every ring (the posts' tops and the ball with them) 40 mm lower and the rings' stack 50 mm shorter, rings
  1 .. 4 at z 303 / 238 / 174 / 110 mm (the user, 2026-10-12, candidate A2 of
  runs/oracle_dev/orbits_chain/hard_reach_r2on20/summary.md): the hands' reach through ring 1; easy and medium unchanged;
- the cord a chain of 23.1 mm capsule links of 2.618 mm radius on ball joints with a bending stiffness of 1 MPa
  (k = E pi r^4 / 4 / L), 2.14 / 2.6 / 3.4 m long (93 / 113 / 147 links), closed through the handle's eyes;
- the table 5 cm longer on the goal side (topogym/config/profiles/task_configs/orbits.yaml), room for the goal region.

The start is the fixed state K0 (the loop round the last post on the base, both legs along it, the handle pulled
straight on the table; a packaged resource). It is not authored in the MJCF: Genesis drops the collisions of an
entity's parts that touch in its authored pose, so the chain is authored straight high above the fixture and K0 is
written into its joints when the entities are bound, before the episode's first state is recorded: a reset and a
replay both start from it.

The goal is the FEM settings' goal: the whole cord and the handle inside the goal region, the fixture on the system
side, slow, and the loop's tie at eye 1 intact. The goal region is chosen from the start state.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from topogym.assets.geometry.orbits_chain_tiers import (CORD_RGBA, RING_SEGMENTS, chain_layout, chain_points,
                                                        chain_qpos, eye_offsets, grasp_targets, handle_outline,
                                                        links_for, ring_radius, scene_xml)
from topogym.assets.scene_visuals import surface
from topogym.evaluation.task_metrics import body_points, quiet_body
from topogym.utils.geometry.regions import placement_measurements
from ..base import EntitySpec, TaskSpec
from ..scene_presets import record_preset
from .common import CORD_PROFILES, profile_value

ENTITY = "orbits_chain"
LOAD_TOLERANCE_M = .001      # K0 written: every cord point and the handle this near its target (forward kinematics)
# the approved chain and fixture changes (module docstring); cord_length_m per tier
CHAIN = dict(fixture_scale=1.4, ball_scale=.6, cord_radius_m=.002618, link_m=.0231, cord_e_pa=1e6, base_drop_m=.015,
             ring1_grow_m=.01, ring2_dx_m=.02, ring_drop_m=0., ring_stack_cut_m=0.)


def cord_contact(xml, solref):
    """The chain MJCF with the cord links' geoms (cord_*) at their own contact time constant."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    for g in root.iter("geom"):
        if g.get("name", "").startswith("cord_"):
            g.set("solref", solref)
    return ET.tostring(root, encoding="unicode")


class OrbitsChainTask:
    """The rigid-chain settings' behaviour, first in the MRO before the FEM tier's class (module docstring)."""
    family_id = "rigid_chain"
    difficulty = None
    physics_model = "rigid_articulated"
    model_revision = "chain231_l0bt15r1b1r2on20_v1"
    placement_attachment_names = ()
    # The rigid families' CPU float64 solve with one substep and one global constraint solve instead of the FEM
    # settings' IPC (rcb_topogym simulator_config); no reset settle: K0 is the start as written.
    simulator_settings = dict(compute_backend="cpu", precision="64", solver_substeps=1, reset_physics_steps=0,
                              rigid_options={"use_contact_island": False})
    # On MuJoCo (the RoboTangle core): our own environment's physics (topogym/planning/orbits/mj_env.py Config, the
    # user 10-12: kept as the setting's own profile, profiles.py orbits_chain_own_v1): one 2.5 ms MuJoCo step per tick,
    # ten ticks per 25 ms control step, no reset settle (K0 is the start as written, as our env starts).
    mujoco_settings = dict(solver_substeps=1, reset_physics_steps=0)
    # The cord links' contact time constant (Config.cord_solref; the user 10-10: Place ends that kept jittering at the
    # authored 5 ms came to rest within 0.5 s at 10 ms). The authored MJCF keeps SOLREF for the fixture and the handle.
    CORD_SOLREF = ".01 1"
    LINEAR_LIMIT_M_S = .01
    ANGULAR_LIMIT_RAD_S = .02
    TIE_LIMIT_M = .002
    tier = None              # 'easy' / 'medium' / 'hard'

    @classmethod
    def chain_defaults(cls, base, cord_length_m, **fixture):
        from ..base import merge_parameters
        return merge_parameters(base.defaults, {"chain": dict(CHAIN, cord_length_m=cord_length_m, **fixture)})

    def cord_rgba(self):
        """The chain links' colour (appearance only): CORD_RGBA at the default cord_finish, else the finish's."""
        if self.config["cord_finish"] == self.defaults["cord_finish"]:
            return CORD_RGBA
        return " ".join(f"{v:.4g}" for v in surface(self.config["cord_finish"])["color"])

    def make_layout(self):
        """The FEM setting's geometry with the approved changes (orbits_chain_tiers.chain_layout)."""
        c = self.config["chain"]
        profile = CORD_PROFILES[self.config["cord_profile"]]
        rings = int(self.config["geometry"].get("ring_count", 4))
        return chain_layout(self.config["geometry"], self.config["linear_scale"], fixture_scale=c["fixture_scale"],
                            ball_scale=c["ball_scale"], cord_radius_m=c["cord_radius_m"], base_drop_m=c["base_drop_m"],
                            ring1_grow_m=c["ring1_grow_m"], handle_x_m=profile_value(profile, "handle_x_m", rings),
                            ring2_dx_m=c["ring2_dx_m"], ring_drop_m=c["ring_drop_m"],
                            ring_stack_cut_m=c["ring_stack_cut_m"])

    def chain_size(self):
        """(links, link length m)."""
        c = self.config["chain"]
        n = links_for(c["cord_length_m"], c["link_m"])
        return n, c["cord_length_m"]/n

    def start_state(self):
        """K0 (the packaged resource asset_resource): dict(points, handle_pos, handle_quat), task-local."""
        from topogym.assets.library import resource_path
        with np.load(resource_path(self.asset_resource)) as z:
            return {k: np.asarray(z[k], float) for k in ("points", "handle_pos", "handle_quat")}

    def grasp_assist(self):
        """contact_weld_v1 on every link (an 8 mm sphere round its middle) and the handle (45 mm round its centre),
        the default thresholds: a pinch holds the chain where it welds."""
        from topogym.utils.grasp import GraspAssistConfig, GraspTarget
        n, link_m = self.chain_size()
        return GraspAssistConfig("contact_weld_v1", tuple(GraspTarget(ENTITY, name, centre, radius)
                                                          for name, centre, radius in grasp_targets(n, link_m)))

    def declare(self, asset_dir, table_height_m):
        self.table_height_m = float(table_height_m)
        preset = record_preset(self, asset_dir)
        self.layout = self.make_layout()
        self.n, self.link_m = self.chain_size()
        self.yaw_deg = float(self.config["yaw_deg"])
        if not np.isfinite(self.yaw_deg):
            raise ValueError("Invalid Orbits yaw_deg")
        if self.config["placement_anchor"] != "base_center":
            raise ValueError("the rigid-chain settings place the fixture by its base centre")
        self.turn = R.from_euler("z", self.yaw_deg, degrees=True)
        anchor = self.turn.apply([0., 0., self.layout.base_height_m/2])
        self.origin = np.r_[np.asarray(self.config["base_center_xy_m"], float)-anchor[:2], table_height_m]
        self.chain_state0 = self.start_state()
        if len(self.chain_state0["points"])-1 != self.n:
            raise ValueError(f"K0 has {len(self.chain_state0['points'])-1} links, the chain {self.n}")
        path = Path(asset_dir)/f"{ENTITY}.xml"
        path.write_text(cord_contact(scene_xml(self.layout, self.n, self.link_m, self.yaw_deg,
                                               self.config["chain"]["cord_e_pa"], self.cord_rgba()), self.CORD_SOLREF))
        self.handle_points = handle_outline(self.layout)
        self.fixture_points = self.fixture_cloud()
        self.placement_targets, self.placement_remainder = ("cord",), ("fixture",)
        return TaskSpec((EntitySpec(ENTITY, "mjcf", path, tuple(self.origin)),),
                        {"rigid": dict(enable_self_collision=True, constraint_solver="CG")},
                        dict(**preset, origin_m=self.origin.tolist(), yaw_deg=self.yaw_deg, task_id=self.task_id,
                             chain_links=self.n, chain_link_m=self.link_m,
                             goal_evaluation_complete=bool(self.goal_complete)))

    def world(self, local):
        return self.turn.apply(np.asarray(local, float))+self.origin

    def fixture_cloud(self):
        """World points of the base (its corners), the posts (their axes), the rings' tubes and the ball."""
        lay = self.layout
        corners = np.array([[x*lay.base_length_m/2, y*lay.base_width_m/2, z*lay.base_height_m]
                            for x in (-1, 1) for y in (-1, 1) for z in (0, 1)])
        tops = [lay.ball_z, *lay.ring_z]
        posts = [np.column_stack([np.full(8, x), np.zeros(8), np.linspace(lay.base_height_m, z, 8)])
                 for x, z in zip(lay.post_x, tops)]
        a = np.arange(RING_SEGMENTS)*2*np.pi/RING_SEGMENTS
        rings = [np.column_stack([x+ring_radius(lay, k)*np.cos(a), ring_radius(lay, k)*np.sin(a), np.full(len(a), z)])
                 for k, (x, z) in enumerate(zip(lay.ring_x, lay.ring_z))]
        ball = np.array([lay.post_x[0], 0., lay.ball_z])+lay.ball_radius_m*np.vstack([np.eye(3), -np.eye(3)])
        return self.world(np.vstack([corners, *posts, *rings, ball]))

    def placement_clouds(self, spec):
        """The start state's cord and handle, and the fixture (the goal region does not depend on later states)."""
        st = self.chain_state0
        r = self.layout.cord_radius_m
        cord = (st["points"][:, None, :]+r*np.vstack([np.eye(3), -np.eye(3)])[None]).reshape(-1, 3)
        hq = st["handle_quat"]
        handle = R.from_quat([hq[1], hq[2], hq[3], hq[0]]).apply(self.handle_points)+st["handle_pos"]
        return dict(cord=self.world(np.vstack([cord, handle])), fixture=self.fixture_points)

    def bind(self, entities):
        """K0 written into the chain's joints (zero velocities) before the episode's first state; checked by forward
        kinematics (Slotted found Genesis placing a written chain state up to 66 mm off after 135-174 deg bends)."""
        super().bind(entities)
        entity = entities[ENTITY]
        state = self.chain_state0
        mujoco_view = hasattr(entity, "world")        # topogym.world.mujoco.views.EntityView (no set_qpos there)
        for name, q in chain_qpos(self.layout, state, self.yaw_deg).items():
            if name == "root":
                q = np.r_[q[:3]+self.origin, q[3:]]
            joint = entity.get_joint(name)
            if mujoco_view:
                entity.world.data.qpos[entity._qadr[joint.qs_idx_local]] = q
                entity.world.data.qvel[entity._dadr[joint.dofs_idx_local]] = 0.
            else:
                entity.set_qpos(np.asarray(q, float), qs_idx_local=joint.qs_idx_local, zero_velocity=True)
        if mujoco_view:
            entity.world._state_changed()
            links = [entity.get_link(f"link_{i:03d}") for i in range(self.n)] + [entity.get_link("handle")]
            pos = np.array([link.get_pos() for link in links])
            quat = np.array([link.get_quat() for link in links])
            idx, hidx = list(range(self.n)), self.n
        else:
            pos = np.asarray(entity.get_links_pos().cpu(), float)
            quat = np.asarray(entity.get_links_quat().cpu(), float)
            idx = [entity.get_link(f"link_{i:03d}").idx_local for i in range(self.n)]
            hidx = entity.get_link("handle").idx_local
        mats = R.from_quat(quat[idx][:, [1, 2, 3, 0]]).as_matrix()
        points = self.turn.inv().apply(chain_points(pos[idx], mats, self.link_m, self.n//2)-self.origin)
        handle = self.turn.inv().apply(pos[hidx]-self.origin)
        error = max(float(np.linalg.norm(points-state["points"], axis=1).max()),
                    float(np.linalg.norm(handle-state["handle_pos"])))
        if error > LOAD_TOLERANCE_M:
            raise ValueError(f"K0 written {error*1e3:.2f} mm off its target (forward kinematics, "
                             f"tolerance {LOAD_TOLERANCE_M*1e3:.1f} mm)")

    def link_cover(self):
        """A link's capsule axis ends and their radial extremes, in the link frame."""
        r = self.layout.cord_radius_m
        return np.array([[x, *o] for x in (0., self.link_m) for o in ((0., 0.), (r, 0.), (-r, 0.), (0., r), (0., -r))])

    def tie_residual(self, state):
        """The tie residual at eye 1 (m): the last link's end against the handle's eye 1."""
        links = state["articulations"][ENTITY]["links"]
        end = body_points(links[f"link_{self.n-1:03d}"], [[self.link_m, 0., 0.]])[0]
        eye = body_points(links["handle"], [eye_offsets(self.layout)[1]])[0]
        return float(np.linalg.norm(end-eye))

    def cord_cloud(self, state):
        """World points of every link's cover and the handle, and the tie residual at eye 1 (m). The links' covers in
        one batched rotation (every tick on MuJoCo: per-link rotations cost more than the physics step)."""
        links = state["articulations"][ENTITY]["links"]
        cover = self.link_cover()
        bodies = [links[f"link_{i:03d}"] for i in range(self.n)]
        pos = np.array([b["position_m"] for b in bodies], float)
        quat = np.array([b["quaternion_wxyz"] for b in bodies], float)
        if pos.shape != (self.n, 3) or quat.shape != (self.n, 4) or not np.isfinite(np.c_[pos, quat]).all():
            raise ValueError("Invalid body pose")
        mats = R.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()
        covers = np.einsum("nij,kj->nki", mats, cover)+pos[:, None, :]
        return np.vstack([body_points(links["handle"], self.handle_points), covers.reshape(-1, 3)]), self.tie_residual(state)

    def evaluate_goal_region(self, state, region):
        """The whole chain and the handle inside the goal, the fixture on the system side, slow and tied. With the
        handle's centre outside the (convex) region some handle point is too: not reached, and only the tie is
        measured (the evaluation runs every physics tick)."""
        links = state["articulations"][ENTITY]["links"]
        if not np.all(region.contains(np.asarray(links["handle"]["position_m"], float)[None])):
            tie = self.tie_residual(state)
            intact = tie <= self.TIE_LIMIT_M
            return False, None if intact else "connection_invalid", dict(
                all_target_inside=False, handle_centre_inside=False, connections_intact=intact, tie_residual_m=tie,
                evaluation_status="available", holding_allowed=True)
        points, tie = self.cord_cloud(state)
        metrics = placement_measurements(region, {"cord": points}, {"fixture": self.fixture_points})
        speeds = [float(np.linalg.norm(links[f"link_{i:03d}"]["linear_velocity_m_s"])) for i in range(self.n)]
        quiet = (max(speeds) <= self.LINEAR_LIMIT_M_S
                 and quiet_body(links["handle"], self.LINEAR_LIMIT_M_S, self.ANGULAR_LIMIT_RAD_S))
        intact = tie <= self.TIE_LIMIT_M
        metrics.update(quiet=bool(quiet), maximum_link_speed_m_s=max(speeds), connections_intact=intact,
                       tie_residual_m=tie, evaluation_status="available", holding_allowed=True)
        reached = metrics["all_target_inside"] and metrics["remainder_on_system_side"] and quiet and intact
        return bool(reached), None if intact else "connection_invalid", metrics
