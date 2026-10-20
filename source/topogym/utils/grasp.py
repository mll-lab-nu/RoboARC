"""Pure grasp declarations and geometry; no scene, controller or physics writes.

Contact filtering and material intersection adapted from historical Fiber and
Blanket helpers at cb60f01b32786cc9c26f754f91b7be4bef3f9ca6. No legacy imports.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation


TASK_CONTACT_PENETRATION_M = .001
GRASP_CONTACT_PENETRATION_M = .002


@dataclass(frozen=True)
class GraspTarget:
    """Owner-declared spherical grasp region in an authored rigid link frame."""
    entity: str
    link: str
    center_m: tuple[float, float, float] = (0., 0., 0.)
    radius_m: float = .03

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.entity, self.link)):
            raise ValueError("Grasp entity and link must be nonempty names")
        center = np.asarray(self.center_m, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Grasp center must contain three finite coordinates")
        if isinstance(self.radius_m, bool) or not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("Grasp radius must be finite and positive")
        object.__setattr__(self, "center_m", tuple(map(float, center)))

    @property
    def key(self):
        return (self.entity, self.link)


def rigid_grasp_region(entity, link, vertices, *, margin_m=.005):
    """Enclose authored link-local geometry for tasks allowing surface regrasp.

    This only declares a region. The rigid backend still requires fresh closing
    intent, simultaneous contacts from both fingers, and the usual health gates.
    """
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Grasp geometry must contain finite link-local vertices")
    if not math.isfinite(margin_m) or margin_m <= 0:
        raise ValueError("Grasp region margin must be finite and positive")
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = float(np.linalg.norm(points-center, axis=1).max()) + margin_m
    return GraspTarget(entity, link, tuple(center), radius)


@dataclass(frozen=True)
class SoftGraspTarget:
    """Allowed material region; dynamic indices are chosen only near the hand.

    Bounds/radii are in rest-material metres (rope arc length or cloth UV).
    Empty bounds allow the whole entity except permanent/fixed constraints.
    """
    entity: str
    material_min_m: tuple[float, ...] = ()
    material_max_m: tuple[float, ...] = ()
    spatial_radius_m: float = .014
    material_radius_m: float = .018
    contact_distance_m: float = .006
    max_strain_ratio: float = 1.3
    strength: float = 100.
    require_exposed_patch: bool = True

    def __post_init__(self):
        if not isinstance(self.entity, str) or not self.entity:
            raise ValueError("Soft grasp entity must be named")
        if not isinstance(self.require_exposed_patch, bool):
            raise ValueError("require_exposed_patch must be boolean")
        lo, hi = np.asarray(self.material_min_m), np.asarray(self.material_max_m)
        if lo.ndim != 1 or hi.shape != lo.shape or len(lo) not in (0, 1, 2):
            raise ValueError("Material bounds must be matching arc-length or UV coordinates")
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
            raise ValueError("Invalid material bounds")
        object.__setattr__(self, "material_min_m", tuple(map(float, lo)))
        object.__setattr__(self, "material_max_m", tuple(map(float, hi)))
        for name in ("spatial_radius_m", "material_radius_m", "contact_distance_m", "strength", "max_strain_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name}")
        if self.max_strain_ratio <= 1:
            raise ValueError("max_strain_ratio must exceed one")

    @property
    def key(self):
        return (self.entity,)


@dataclass(frozen=True)
class GraspAssistConfig:
    """Episode configuration, never a policy command. Disabled by default."""
    model: str = "none"
    targets: tuple[GraspTarget | SoftGraspTarget, ...] = ()
    stable_time_s: float = .02
    max_tcp_distance_m: float = .06
    max_penetration_m: float = GRASP_CONTACT_PENETRATION_M
    max_translation_error_m: float = .01
    max_rotation_error_rad: float = .15
    max_finger_opening_fraction: float = .15
    intent_deadband: float = .002
    # "close": a hand whose gripper is already commanded closed at episode birth
    # (a pregrasp segment0) may acquire without a later closing command.
    initial_intent: str = "open"

    def __post_init__(self):
        if self.initial_intent not in {"open", "close"}:
            raise ValueError("initial_intent must be open or close")
        if self.model not in {"none", "contact_weld_v1", "ipc_soft_band_v1", "pbd_local_patch_v1"}:
            raise ValueError("Unsupported grasp assistance model")
        targets = tuple(self.targets)
        target_type = GraspTarget if self.model in {"none", "contact_weld_v1"} else SoftGraspTarget
        if not all(isinstance(t, target_type) for t in targets):
            raise ValueError(f"Grasp targets must be {target_type.__name__} declarations")
        if target_type is SoftGraspTarget:
            dimension = 1 if self.model == "ipc_soft_band_v1" else 2
            if any(t.material_min_m and len(t.material_min_m) != dimension for t in targets):
                raise ValueError("Material bounds dimension does not match grasp model")
        if len({t.key for t in targets}) != len(targets):
            raise ValueError("Declare each grasp link once")
        if (self.model == "none") != (not targets):
            raise ValueError("Enabled assist requires targets; disabled assist accepts none")
        object.__setattr__(self, "targets", targets)
        for name in ("stable_time_s", "max_tcp_distance_m", "max_penetration_m",
                     "max_translation_error_m", "max_rotation_error_rad", "max_finger_opening_fraction", "intent_deadband"):
            v = getattr(self, name)
            if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.intent_deadband >= 1:
            raise ValueError("intent_deadband must be a fraction below one")
        if self.max_finger_opening_fraction > 1:
            raise ValueError("max_finger_opening_fraction must be at most one")


# -- Opt-in contact_weld_v1 rules (2026-10-14), MuJoCo backend only (topogym/world/mujoco/grasp.py). They are subclasses
# with their own fields, not new fields of GraspTarget / GraspAssistConfig: every existing configuration keeps its
# dataclass fields, so its asdict (the grasp state in every tick record, hence each frozen instance's state_sha) and its
# grasp decisions stay exactly as before. SimulatorConfig reads either back from a mapping by its own field name.
@dataclass(frozen=True)
class ForceClosureGraspTarget(GraspTarget):
    """A GraspTarget that contact_weld_v1 welds only on a force-closure pinch (controller battery design D2,
    runs/controller_battery_design_20261014: two fingers touching the top of a cell seated in its slot must not pull it
    out). Besides the usual bilateral contact, some contact of one finger and some contact of the other must lie on a
    line inside both friction cones: the line between the two contact points (in the plane of the two normals) makes
    at most atan(friction_coefficient) with each contact's normal into the target (two-finger force closure, Nguyen
    1988, force_closure_pair). Across a cylinder that is a central angle between the two contacts of at least
    180 - 2 atan(mu) degrees (90 at mu 1). Checked at acquisition only; a weld once made is not re-checked.
    friction_coefficient: the finger-target contacts' mu (MuJoCo takes the larger of the two geoms'; the ALOHA's
    fingers: 1)."""
    friction_coefficient: float = 1.

    def __post_init__(self):
        super().__post_init__()
        mu = self.friction_coefficient
        if isinstance(mu, bool) or not isinstance(mu, (int, float)) or not math.isfinite(mu) or mu <= 0:
            raise ValueError("Grasp friction coefficient must be finite and positive")
        object.__setattr__(self, "friction_coefficient", float(mu))


@dataclass(frozen=True)
class WeakHoldGraspTarget(ForceClosureGraspTarget):
    """A ForceClosureGraspTarget whose weak holds may only lift it a short way (controller battery, rule C of
    runs/controller_battery_design_20261014/GRASP_PROPOSAL.md, the user's choice of 2026-10-18: an old cell may be
    plucked out of its slot by the fingertips but must be held from both sides to be carried). A hold is firm when its
    acquisition pinch also closes force at firm_friction_coefficient (across a cylinder: a central angle between the two
    contacts of at least 180 - 2 atan(firm mu) degrees, 150 at 0.268), else weak; a weak hold is lost (reason
    "weak_hold_travel", as any lost grasp: the hand opens before it takes hold again) once the target link's origin is
    more than weak_hold_travel_m from where it was taken. A firm hold has no such limit. Both decided at acquisition."""
    firm_friction_coefficient: float = .268
    weak_hold_travel_m: float = .06

    def __post_init__(self):
        super().__post_init__()
        firm, travel = self.firm_friction_coefficient, self.weak_hold_travel_m
        if isinstance(firm, bool) or not isinstance(firm, (int, float)) or not math.isfinite(firm) or firm <= 0:
            raise ValueError("Grasp firm friction coefficient must be finite and positive")
        if firm >= self.friction_coefficient:
            raise ValueError("Grasp firm friction coefficient must be below the friction coefficient")
        if isinstance(travel, bool) or not isinstance(travel, (int, float)) or not math.isfinite(travel) or travel <= 0:
            raise ValueError("Grasp weak-hold travel must be finite and positive")
        object.__setattr__(self, "firm_friction_coefficient", float(firm))
        object.__setattr__(self, "weak_hold_travel_m", float(travel))


@dataclass(frozen=True)
class MultiRopeGraspAssistConfig(GraspAssistConfig):
    """contact_weld_v1 where one hand may weld several ropes at once (wire splice design, question 5,
    runs/wire_splice_design_20261014: each hand pinches two wires one behind the other, so each finger touches only one
    of them, and the right hand twists the two tips together).

    weld_each_rope names entities (rope chains or MJCF link chains) whose grasp targets are their links. A finger
    touches a named rope when a contact row joins it to one of the rope's target links inside that target's region,
    with the plain filters (penetration 0..max_penetration_m, target centre within max_tcp_distance_m of the TCP).
    A rope pinch: each finger touches a named rope, the two fingers' ropes include two different ones, and every
    touched rope is touched by both fingers (side by side) or touches another touched rope (a contact row between their
    links inside the region of a touched link: the wires pressed together between the fingers). Then the hand welds,
    for every touched rope, its touched link nearest the TCP, all to that hand, as one grasp (one binding, acquired
    after the usual stable_time_s, lost and released together). Without a rope pinch a named rope is a candidate as a
    rope chain is: its bilaterally touched link nearest the TCP. Candidates of other entities follow the plain rule.

    rope_max_rotation_error_rad: the relative rotation a rope pinch's welds may drift before the grasp is lost
    (relative_pose_error); None keeps max_rotation_error_rad. Twisting the wires turns each welded tip link up to
    0.41 rad off its weld (the design's v4 measurement), past the plain 0.15 rad."""
    weld_each_rope: tuple[str, ...] = ()
    rope_max_rotation_error_rad: float | None = None

    def __post_init__(self):
        super().__post_init__()
        ropes = tuple(self.weld_each_rope)
        if self.model != "contact_weld_v1":
            raise ValueError("weld_each_rope is a contact_weld_v1 rule")
        if not ropes or not all(isinstance(r, str) and r for r in ropes) or len(set(ropes)) != len(ropes):
            raise ValueError("weld_each_rope must name distinct entities")
        if not set(ropes) <= {t.entity for t in self.targets}:
            raise ValueError("weld_each_rope names an entity without grasp targets")
        object.__setattr__(self, "weld_each_rope", ropes)
        limit = self.rope_max_rotation_error_rad
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))
                                  or not math.isfinite(limit) or limit <= 0):
            raise ValueError("rope_max_rotation_error_rad must be finite and positive")
        if limit is not None:
            object.__setattr__(self, "rope_max_rotation_error_rad", float(limit))


def force_closure_pair(position_a, normal_a, position_b, normal_b, friction_coefficient):
    """Whether two frictional contacts close force (Nguyen 1988): the line between them lies inside both friction
    cones, i.e. the direction from each contact to the other makes at most atan(mu) with that contact's unit normal
    (pointing into the held body).

    The line is taken in the plane of the two normals (its component along normal_a x normal_b dropped; along the
    normal when the normals are parallel): a fingertip pressed on a cylinder touches it along a line in that direction,
    and MuJoCo reports one point anywhere on that line (two fingers on the lifted AA cell: points 9.8 mm apart along
    its axis), so the best pair of points of the two lines is the one across. For a circular section this is the
    central-angle rule: the two contacts subtend at least 180 - 2 atan(mu) degrees. Coincident points never close."""
    line = np.asarray(position_b, dtype=float) - np.asarray(position_a, dtype=float)
    normal_a, normal_b = np.asarray(normal_a, dtype=float), np.asarray(normal_b, dtype=float)
    axis = np.cross(normal_a, normal_b)
    if np.linalg.norm(axis) > 1e-9:
        axis /= np.linalg.norm(axis)
        line -= np.dot(line, axis) * axis
    else:
        line = np.dot(line, normal_a) * normal_a
    length = float(np.linalg.norm(line))
    if not length > 0:
        return False
    line /= length
    cos_cone = 1. / math.sqrt(1. + float(friction_coefficient) ** 2)
    return bool(np.dot(line, normal_a) >= cos_cone and np.dot(-line, normal_b) >= cos_cone)


def closed_fraction(value, open_value, closed_value):
    values = np.asarray([value, open_value, closed_value], dtype=float)
    if not np.isfinite(values).all() or open_value == closed_value:
        raise ValueError("Invalid gripper coordinate semantics")
    return float((value - open_value) / (closed_value - open_value))


def rotation(quaternion_wxyz):
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1.) > 1.e-5:
        raise ValueError("Expected unit wxyz orientation")
    return Rotation.from_quat(q[[1, 2, 3, 0]])


def relative_pose(parent_position, parent_quaternion, position, quaternion):
    r = rotation(parent_quaternion)
    return dict(position_m=r.inv().apply(np.asarray(position)-parent_position).tolist(),
                quaternion_wxyz=(r.inv()*rotation(quaternion)).as_quat()[[3, 0, 1, 2]].tolist())


def relative_error(reference, actual):
    return (float(np.linalg.norm(np.asarray(reference["position_m"])-actual["position_m"])),
            float((rotation(reference["quaternion_wxyz"]).inv()*rotation(actual["quaternion_wxyz"])).magnitude()))


def bilateral_contact_rows(contacts, fingers, target_link, *, center_m, radius_m, max_penetration_m):
    """One native contact snapshot, one target link, both distinct fingers.

    Never merge contacts from different substeps or different candidate links.
    Negative penetration is speculative separation, not acquisition evidence.
    """
    if len(set(fingers)) != 2:
        raise ValueError("Exactly two distinct finger links are required")
    a, b = (np.asarray(contacts[k]).reshape(-1) for k in ("link_a", "link_b"))
    depth = np.asarray(contacts["penetration"]).reshape(-1)
    position = np.asarray(contacts["position"]).reshape(-1, 3)
    if len({len(a), len(b), len(depth), len(position)}) != 1:
        raise ValueError("Mismatched contact arrays")
    valid = (np.isfinite(depth) & (depth >= 0) & (depth <= max_penetration_m)
             & np.isfinite(position).all(axis=1)
             & (np.linalg.norm(position-np.asarray(center_m), axis=1) <= radius_m))
    rows = []
    for finger in fingers:
        indices = np.flatnonzero(valid & (((a == target_link) & (b == finger)) |
                                          ((b == target_link) & (a == finger))))
        if not len(indices):
            return ()
        rows.extend(indices.tolist())
    return tuple(sorted(set(rows)))


def held_finger_contact_mask(contacts, fingers_by_hand, movable_links, grasp=None):
    """Identify grasp contacts in one snapshot, with or without assistance.

    Physical holding requires both declared fingers of the same hand touching
    the same movable task link. A live rigid-assist binding also identifies that
    hand's held link when its weld carries the load and one finger loses contact.
    No force threshold: zero-force contacts count, speculative separation does not.
    This does not grant the held object's other contacts a larger tolerance.
    """
    a, b = (np.asarray(contacts[k]) for k in ("link_a", "link_b"))
    depth = np.asarray(contacts["penetration"])
    touching = np.isfinite(depth) & (depth >= 0)
    held = np.zeros(depth.shape, dtype=bool)
    hands = (grasp or {}).get("hands", {})
    for hand, fingers in fingers_by_hand.items():
        targets = []
        for finger in fingers:
            targets.append(set(b[touching & (a == finger)]) | set(a[touching & (b == finger)]))
        selected = targets[0] & targets[1] & movable_links
        state = hands.get(hand, {})
        binding = state.get("binding")
        if state.get("phase") in {"registered", "holding"} and binding is not None:
            links = binding.get("links", ())
            if binding.get("hand") == hand and len(links) == 2 and links[1] in movable_links:
                selected.add(links[1])
        held |= touching & ((np.isin(a, fingers) & np.isin(b, list(selected))) |
                            (np.isin(b, fingers) & np.isin(a, list(selected))))
    return held


def material_patch(positions, material_coordinates, center_index, *, spatial_radius_m, material_radius_m):
    """Spatial intersection with material neighbourhood; empty stays empty."""
    x, uv = np.asarray(positions), np.asarray(material_coordinates)
    if x.ndim != 2 or x.shape[1] != 3 or uv.ndim != 2 or len(x) != len(uv):
        raise ValueError("Invalid material patch arrays")
    if not np.isfinite(x).all() or not np.isfinite(uv).all():
        raise ValueError("Nonfinite material patch")
    if type(center_index) is not int or not 0 <= center_index < len(x):
        raise ValueError("Invalid material index")
    if not all(np.isfinite(v) and v > 0 for v in (spatial_radius_m, material_radius_m)):
        raise ValueError("Patch radii must be finite and positive")
    return np.flatnonzero((np.linalg.norm(x-x[center_index], axis=1) <= spatial_radius_m) &
                         (np.linalg.norm(uv-uv[center_index], axis=1) <= material_radius_m))


def surface_distances(points, vertices, faces):
    """Unsigned distances to actual triangles, not to a TCP or vertex cloud."""
    import trimesh
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    return trimesh.proximity.closest_point_naive(mesh, np.asarray(points))[1]


def material_components(indices, coordinates, radius):
    """Stable connected components in material space; separated layers stay apart."""
    from scipy.spatial import cKDTree
    indices = np.asarray(indices, dtype=int)
    tree = cKDTree(np.asarray(coordinates)[indices])
    remaining, result = set(range(len(indices))), []
    while remaining:
        todo, found = [min(remaining)], []
        while todo:
            k = todo.pop()
            if k not in remaining:
                continue
            remaining.remove(k)
            found.append(int(indices[k]))
            todo.extend(i for i in tree.query_ball_point(tree.data[k], radius) if i in remaining)
        result.append(np.array(sorted(found), dtype=int))
    return result


def exposed_patch(positions, material, indices, *, radius, clearance, remote_from_patch=False):
    """Reject a patch under a materially distant layer (gravity-up convention).

    remote_from_patch=False (the flex's rule): a covering particle is more than 2 radius in material from the patch
    particle it covers. remote_from_patch=True (the VBD cloth's rule, 2026-10-11): it is more than 2 radius in material
    from EVERY patch particle. A patch centred on a fold's bend reaches down the bottom layer, under the top layer a few
    cm past the bend; that top layer continues the patch's own surface (within 2 radius of its top part), so it is not a
    covering layer. A layer folded over the patch is at least twice its distance to the crease away in material, and
    still covers. With the flex's 3 mm layer spacing (below the 6 mm clearance) the two rules agree on bends; with VBD's
    7 mm offset only the second admits a bend grip."""
    x, uv, indices = np.asarray(positions), np.asarray(material), np.asarray(indices)
    if not len(indices):
        return True
    # Only particles within `radius` (horizontally) of a patch particle can cover it, all inside the patch's horizontal
    # box grown by the radius (and 1 nm, far above rounding): the same per-particle test, on those particles only.
    low = x[indices, :2].min(axis=0)-radius-1e-9
    high = x[indices, :2].max(axis=0)+radius+1e-9
    near = np.flatnonzero(np.all((x[:, :2] >= low) & (x[:, :2] <= high), axis=1))
    x_near, uv_near = x[near], uv[near]
    if remote_from_patch:
        from scipy.spatial import cKDTree
        remote_all = cKDTree(uv[indices]).query(uv_near)[0] > 2*radius
    for i in indices:
        local = np.linalg.norm(x_near[:, :2]-x[i, :2], axis=1) <= radius
        remote = remote_all if remote_from_patch else np.linalg.norm(uv_near-uv[i], axis=1) > 2*radius
        if np.any(local & remote & (x_near[:, 2] > x[i, 2]+clearance)):
            return False
    return True
