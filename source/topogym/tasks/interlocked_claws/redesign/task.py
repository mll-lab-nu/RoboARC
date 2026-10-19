"""Task classes of the redesigned interlocked_claws tiers (step 3, agent "claws"; not registered in the repo's
registry: tools register them in-process, see register()).

What stays the family's: the robot (original ALOHA), the pregrasp birth convention (both hands already closed on their
grip blocks, contact_weld_v1 with initial intent "close"), the goal (the placement contract of the low released tiers:
the entire brown piece inside the green dashed goal area, the other piece outside on the system side, both quiet for
0.5 s, holding allowed: the task text of TASKS["interlocked_claws"]), the materials (density 600, friction .3) and the
MuJoCo profile (claws_v6).
What is new: the pieces (compound boxes, tiers.py), their start pose and one grasp
region per grip block (a piece may have several grips; every grasp lands on the weld the same way).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from topogym.tasks.interlocked_claws.common import InterlockedClawsTask
from .shapes import DENSITY, Piece, union_mesh

VARIATIONS = {
    "very_easy": "interlocked_claws_rd_very_easy",
    "easier": "interlocked_claws_rd_easier",
    "easy": "interlocked_claws_rd_easy",
    "medium": "interlocked_claws_rd_medium",
    "hard": "interlocked_claws_rd_hard",
}
GRIP_RADIUS_M = .03
FINGER_ON_GRIP_M = .0222        # the family's finger position closed on a 28 mm grip block


def _scalar(v):
    return np.float64(v["np"]) if isinstance(v, dict) else v


def _piece(rec):
    body = [dict(size=[_scalar(v) for v in b["size"]],
                 pose=np.asarray(b["pose"], float) if b["pose_kind"] == "array" else b["pose"]) for b in rec["body"]]
    grips = {g: (np.asarray(T, float) if rec["grip_kind"][g] == "array" else T) for g, T in rec["grips"].items()}
    return Piece(rec["name"], body, grips, tuple(_scalar(v) for v in rec["color"]))


def load_design(tier):
    """A tier's runtime design record (pieces A / B, A_in_assembly, engaged, start, hands, birth_flip, birth_q,
    clearance) from its frozen data file."""
    from topogym.assets.library import resource_path
    rec = json.loads(Path(resource_path(f"variations/interlocked_claws/rd_{tier}.json")).read_text())
    if rec.get("schema") != "claws_rd_runtime/1" or rec.get("tier") != tier:
        raise ValueError(f"claws rd {tier}: not a runtime design record")
    d = {k: rec[k] for k in ("name", "start", "hands", "birth_flip", "birth_q")}
    d.update(clearance=_scalar(rec["clearance"]), A=_piece(rec["A"]), B=_piece(rec["B"]),
             A_in_assembly=np.asarray(rec["A_in_assembly"], float), engaged=np.asarray(rec["engaged"], float))
    return d


class RedesignClawsTask(InterlockedClawsTask):
    tier = None
    frozen_identity_exempt = ("geometry_report.json",)    # written by the authoring build only (it held the route)
    family_id = "redesign_keyhole_gates"
    model_revision = "rd_v1"
    pregrasp_weld = True
    separation_only = False
    terminal_revision = "placement_region_v3_grasp_contact_2mm"

    def design(self):
        """The tier's pieces, start pose and birth hands, as frozen data
        (topogym/assets/resources/variations/interlocked_claws/rd_<tier>.json, exported from the authoring factories by
        scripts/robotangle/redesign/interlocked_claws/export_runtime.py): the same values, bit for bit."""
        return load_design(self.tier)

    def build_assets(self, folder):
        folder = Path(folder)
        d = self.design()
        A, B = d["A"], d["B"]
        T_A = np.asarray(d["A_in_assembly"], float)
        T_B = T_A @ np.asarray(d["engaged"], float)
        self.table_start = d.get("start") == "table"
        if self.table_start:
            T_A, T_B, settle = settle_on_table(A, B, T_A, T_B)
            d["settle"] = settle
        root = folder / "family"
        # each piece's frame (its free joint) at its composite centre of mass (validator, 10-11: a frame off the COM
        # couples translation and rotation in M; an impact that jumps the spin then breaks the energy books); the
        # design keeps its own frames: design pose = body pose @ trans(-com)
        self.piece_com = {p.name: p.com() for p in (A, B)}
        framed = {p.name: p.shifted(self.piece_com[p.name]) for p in (A, B)}
        reports = {n: framed[n].save(root, DENSITY) for n in (B.name, A.name)}
        bodies = []
        for piece, T in ((B, T_B), (A, T_A)):          # target (brown) first: placement_targets = bodies[0]
            fp = framed[piece.name]
            mesh = union_mesh(fp.boxes)
            Tb = np.asarray(T, float).copy()
            Tb[:3, 3] = Tb[:3, 3] + Tb[:3, :3] @ self.piece_com[piece.name]
            bodies.append(SimpleNamespace(name=piece.name, mesh=mesh, pose=Tb, color=piece.color,
                                          asset_path=root / f"{piece.name}.urdf", piece=fp))
        # birth: each hand closed on its grip (left on A's, right on B's), the arms by the reference ALOHA FK
        hand_piece = {"left": (A, T_A, d["hands"]["left"]), "right": (B, T_B, d["hands"]["right"])}
        flip = d.get("birth_flip") or {}
        targets = [T @ np.asarray(p.grips[g], float) @ (np.diag([1., -1., -1., 1.]) if flip.get(side) else np.eye(4))
                   for side, (p, T, g) in hand_piece.items()]
        if self.table_start:          # both hands empty at the family's upstream neutral pose, grippers open
            self.initial_robot_q = np.tile([0., -.96, 1.16, 0., -.3, 0., OPEN_M, OPEN_M], 2)
        else:
            self.initial_robot_q = birth_qpos(targets, first_seeds=d.get("birth_q"))
        self.design_record = d
        self.birth_hands = {side: (p.name, g) for side, (p, _, g) in hand_piece.items()}
        # no geometry_report.json: the frozen instances' identity entry for it is not compared (frozen_identity_exempt)
        report = dict(family=self.family_id, tier=self.tier, design=d["name"],
                      pieces=reports, clearance_design_m=d["clearance"],
                      initial_robot_qpos=self.initial_robot_q.tolist(), birth_hands=self.birth_hands,
                      collision_representation="compound exact boxes",
                      body_frame="composite centre of mass (design frame + com)",
                      piece_com={k: v.tolist() for k, v in self.piece_com.items()},
                      robot_geometry_modified=False, initialization="pregrasp birth, both hands closed on grips")
        return SimpleNamespace(bodies=bodies, reference={"density_kg_m3": DENSITY}), report

    def finalize_spec(self, spec, *, asset_dir, scene_config):
        from dataclasses import replace
        spec = super().finalize_spec(spec, asset_dir=asset_dir, scene_config=scene_config)
        if getattr(self, "table_start", False):
            g = scene_config.robot_control.gripper
            targets = {side: {"gripper_position_m": float(g.open_qpos)} for side in ("left", "right")}
            spec = replace(spec, initial_robot_targets=targets,
                           metadata={**spec.metadata, "initial_robot_targets": targets,
                                     "initial_state": "assembly resting on the table; both hands empty at home"})
        return spec

    def pregrasp_targets(self, assembly, entities, transformed):
        """Every grip block of both pieces is a grasp region: centre = its TCP point (the grip link's frame is the
        piece frame), radius 3 cm."""
        targets = []
        for body in assembly.bodies:
            for g, T in body.piece.grips.items():
                targets.append(dict(entity=body.name, link=g, center_m=np.asarray(T, float)[:3, 3].tolist(),
                                    radius_m=GRIP_RADIUS_M))
        return tuple(targets)


def birth_qpos(targets, wrist=(-1.6, 1.85), first_seeds=None):
    """Both arms' joints placing each TCP (reference ALOHA FK, the frame the pieces are authored in) on its grip frame:
    per arm, bounded least squares from several seeds; among the exact solutions (residual < 1e-9) the one with the
    least bent wrist and forearm roll. Fingers closed on the grips."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from topogym.assets.geometry.interlocked_claws import ARM_LIMITS, load_reference, tcp_frames
    reference = load_reference()
    q = InterlockedClawsTask.opposed_q(.13, .30528805, FINGER_ON_GRIP_M).copy()
    lo, hi = np.array(ARM_LIMITS).T
    lo, hi = lo.copy(), hi.copy()
    lo[4], hi[4] = wrist
    for k, (arm, target) in enumerate(zip((0, 8), targets)):
        best = None
        rng = np.random.default_rng(7 + k)
        fixed = [[0., -.67, 1.05, 0., -.38, 0.], [0., -.3, .6, 0., .3, 0.], [.5, -.4, .8, 0., -.3, 0.],
                 [-.5, -.4, .8, 0., -.3, 0.], [0., -.6, .9, 1.2, .6, -1.2], [0., -.6, .9, -1.2, .6, 1.2],
                 [.8, -.2, .4, .8, .5, -.8], [-.8, -.2, .4, -.8, .5, .8], [0., -.1, .1, 0., 1.4, 0.]]
        if first_seeds and first_seeds.get(("left", "right")[k]) is not None:
            # the layout's birth branch (the placement search's): its exact solution, if it has one, is taken
            x0 = np.clip(np.asarray(first_seeds[("left", "right")[k]], float), lo + 1e-6, hi - 1e-6)
            def res0(x, arm=arm):
                qq = q.copy(); qq[arm:arm + 6] = x
                tcp = tcp_frames(reference, qq)[k]
                return np.r_[tcp[:3, 3] - target[:3, 3],
                             .1 * Rotation.from_matrix(tcp[:3, :3] @ target[:3, :3].T).as_rotvec()]
            r = least_squares(res0, x0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=2000)
            if np.abs(r.fun).max() <= 1e-8 and abs(r.x[5]) <= 2.8:
                q[arm:arm + 6] = r.x
                continue
        for seed in fixed + rng.uniform(lo, hi, size=(40, 6)).tolist():
            def res(x, arm=arm):
                qq = q.copy(); qq[arm:arm + 6] = x
                tcp = tcp_frames(reference, qq)[k]
                return np.r_[tcp[:3, 3] - target[:3, 3],
                             .1 * Rotation.from_matrix(tcp[:3, :3] @ target[:3, :3].T).as_rotvec()]
            r = least_squares(res, np.clip(seed, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), xtol=1e-14, ftol=1e-14,
                              gtol=1e-14, max_nfev=2000)
            if np.abs(r.fun).max() > 1e-8:
                continue
            if abs(r.x[5]) > 2.8:          # core: never wrist_rotate near 180 deg
                continue
            cost = abs(r.x[4]) + .3 * abs(r.x[3]) + .1 * abs(r.x[5])
            if best is None or cost < best[0]:
                best = (cost, r.x)
        if best is None:
            raise ValueError(f"redesign birth: no exact grasp for arm {k}")
        q[arm:arm + 6] = best[1]
    q[[6, 7, 14, 15]] = FINGER_ON_GRIP_M
    return q


OPEN_M = .037


def settle_on_table(A, B, T_A, T_B, seconds=5.):
    """The assembly dropped 1 mm onto the table and settled (MuJoCo 3.14, the claws contact class, 0.625 ms steps):
    -> (A's pose, B's pose, report). Deterministic: the same bytes give the same poses."""
    import mujoco
    from scipy.spatial.transform import Rotation

    def body(name, piece, T):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
        geoms = []
        for b in piece.boxes:
            P = np.asarray(b["pose"], float)
            gq = Rotation.from_matrix(P[:3, :3]).as_quat()[[3, 0, 1, 2]]
            geoms.append(f'<geom type="box" size="{" ".join(map(repr, (np.asarray(b["size"]) / 2).tolist()))}" '
                         f'pos="{" ".join(map(repr, P[:3, 3].tolist()))}" quat="{" ".join(map(repr, gq.tolist()))}" '
                         f'density="{DENSITY}" friction=".3 .005 .0001"/>')
        return (f'<body name="{name}" pos="{" ".join(map(repr, T[:3, 3].tolist()))}" '
                f'quat="{" ".join(map(repr, q.tolist()))}"><freejoint/>' + "".join(geoms) + "</body>")
    lo = min((A.corners() @ T_A[:3, :3].T + T_A[:3, 3])[:, 2].min(), (B.corners() @ T_B[:3, :3].T + T_B[:3, 3])[:, 2].min())
    dz = (.02 + .001) - lo
    T_A, T_B = T_A.copy(), T_B.copy()
    T_A[2, 3] += dz; T_B[2, 3] += dz
    xml = ('<mujoco><option timestep="0.000625" integrator="implicitfast" solver="Newton" iterations="100" '
           'cone="elliptic" impratio="10"/><default><geom solref="0.00125 2" solimp="0.9 0.99 0.001 0.5 2" '
           'condim="3"/></default><worldbody><geom type="box" size="1 1 0.01" pos="0 0 0.01" friction="1 .005 .0001"/>'
           + body("A", A, T_A) + body("B", B, T_B) + "</worldbody></mujoco>")
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_DIAGEXACT
    d = mujoco.MjData(m)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    out = []
    for name in ("A", "B"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        T = np.eye(4)
        T[:3, :3] = d.xmat[b].reshape(3, 3)
        T[:3, 3] = d.xpos[b]
        out.append(T)
    speed = float(np.abs(d.qvel).max())
    return out[0], out[1], dict(seconds=seconds, final_max_qvel=speed, drop_m=1e-3,
                                A_moved_m=float(np.linalg.norm(out[0][:3, 3] - T_A[:3, 3])),
                                B_moved_m=float(np.linalg.norm(out[1][:3, 3] - T_B[:3, 3])))


def _make(tier):
    name = "".join(w.capitalize() for w in tier.split("_"))
    return type(f"RedesignClaws{name}", (RedesignClawsTask,), dict(
        tier=tier, variation_id=VARIATIONS[tier], difficulty=tier))


RedesignClawsVeryEasy = _make("very_easy")
RedesignClawsEasier = _make("easier")
RedesignClawsEasy = _make("easy")
RedesignClawsMedium = _make("medium")
RedesignClawsHard = _make("hard")
CLASSES = {"very_easy": RedesignClawsVeryEasy, "easier": RedesignClawsEasier, "easy": RedesignClawsEasy,
           "medium": RedesignClawsMedium, "hard": RedesignClawsHard}


def register(tier):
    """Append the tier's variation record in-process (as session_kit.register_variation does); idempotent."""
    from topogym.config import variation_records
    vid = VARIATIONS[tier]
    for row in variation_records.RECORDS:
        if row["task_id"] == "interlocked_claws" and row["variation_id"] == vid:
            return vid
    cls = CLASSES[tier]
    rec = dict(task_id="interlocked_claws", variation_id=vid, difficulty=tier, family_id=cls.family_id,
               model_revision=cls.model_revision,
               implementation=f"topogym.tasks.interlocked_claws.redesign.task:{cls.__name__}",
               aliases=[], status="integrated", reason=None, validation="redesign_dev", pool="redesign")
    variation_records.RECORDS = (*variation_records.RECORDS, rec)
    return vid


def make_session(tier, seed=0, **kw):
    """A fresh MuJoCo Session of a redesigned tier (native gripper space)."""
    vid = register(tier)
    from rcb_topogym import Session
    return Session("interlocked_claws", variation_id=vid, seed=seed, gripper_space="native", strict=False, **kw)
