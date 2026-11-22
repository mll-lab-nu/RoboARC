"""pbd_local_patch_v1 on MuJoCo: the soft grasp's selection and health rules (SoftGraspBackend, unchanged) on a MuJoCo
cloth (cloth.py), its native constraint a pool of inactive connect equalities per hand (cloth.add_cloth).

Acquisition pins every flex vertex that carries the admitted patch (the corners of each patch particle's triangle; an
18 mm patch on the coarse blanket flex: at most 14 vertices) to the hand link at their current offsets, so each patch
particle moves with the hand as Genesis attaches the patch's particles themselves: each pool equality is re-targeted
(model.eq_obj2id, model.eq_data anchors) and activated. Release deactivates them. The world's checkpoints carry
model.eq_data and model.eq_obj2id, and restore re-checks every binding against the active pins.

The grasp slips (clamp, never fail), as the assisted rigid grasp does (grasp.py WELD_SLIP_FORCE_N, WELD_SLIP_WINDOW_S):
after every substep the vector sum of a hand's pin forces (the force the hand puts on the cloth) enters a 20 ms window;
once its mean exceeds the robot's slip force the patch releases at the tick's end, reason "grasp_slipped".
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from topogym.world.physics.grasp_soft import SoftGraspBackend
from topogym.utils.grasp import rotation
from .grasp import WELD_SLIP_FORCE_N, WELD_SLIP_WINDOW_S


class MujocoClothGraspBackend(SoftGraspBackend):
    evidence_source = "mujoco_flex_bilateral_finger_surface_proximity"

    def __init__(self, world, robot, config):
        import mujoco
        super().__init__(world, robot, config)
        m = world.model
        self.cloths = {}
        for name, entity in self.entities.items():
            cloth = world._cloths.get(name)
            uv = getattr(entity, "grasp_material_coordinates", None)
            if cloth is None or uv is None:
                raise ValueError(f"pbd_local_patch_v1 on MuJoCo needs a cloth with its birth material map: {name}")
            self.initialize_material(name, uv, uv)
            self.cloths[name] = cloth
        self.pool = {}
        for hand in self.hands:
            names = [pin for c in self.cloths.values() for pin in c.record["cloth"]["pins"].get(hand, [])]
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n) for n in names]
            if not ids or min(ids) < 0:
                raise ValueError(f"scene lacks the cloth pins of hand {hand}")
            self.pool[hand] = ids
        self.slip_force = WELD_SLIP_FORCE_N.get(robot.description.robot_id)
        self.slip_samples = max(1, round(WELD_SLIP_WINDOW_S / float(m.opt.timestep)))
        self._window = {}       # hand -> its slip window (the last slip_samples substeps' pin force sums, N)
        self._slipped = set()   # hands whose window mean exceeded the slip force

    # -- the cloth
    def positions(self, name):
        return self.entities[name].get_particles_pos()

    def _pins_for(self, name, indices):
        cloth = self.cloths[name]
        return sorted({int(v) for i in indices for v, w in zip(cloth.corners[int(i)], cloth.weights[int(i)]) if w > 1e-12})

    def attach_native(self, binding):
        name, hand = binding["target"][0], binding["hand"]
        cloth, w = self.cloths[name], self.world
        m, d = w.model, w.data
        verts = self._pins_for(name, binding["indices"])
        free = [e for e in self.pool[hand] if not d.eq_active[e]]
        if len(verts) > len(free):
            raise ValueError(f"{hand}: a patch of {len(verts)} cloth vertices exceeds the pin pool ({len(free)} free)")
        link = self.hands[hand]
        p, R = np.asarray(link.get_pos()), rotation(link.get_quat()).as_matrix()
        x = cloth.vertices(d)
        pins = []
        for e, v in zip(free, verts):
            m.eq_obj2id[e] = int(cloth.bodies[v])
            m.eq_data[e, 0:3] = R.T @ (x[v] - p)
            m.eq_data[e, 3:6] = 0.
            d.eq_active[e] = 1
            pins.append([int(e), int(v)])
        binding["pins"] = pins
        self._window[hand] = deque([0.] * self.slip_samples, maxlen=self.slip_samples)
        self._slipped.discard(hand)

    def detach_native(self, binding):
        d = self.world.data
        for e, _ in binding["pins"]:
            d.eq_active[e] = 0
        self._window.pop(binding["hand"], None)
        self._slipped.discard(binding["hand"])

    def exists(self, binding):
        m, d = self.world.model, self.world.data
        cloth = self.cloths[binding["target"][0]]
        return bool(binding.get("pins")) and all(
            d.eq_active[e] and int(m.eq_obj2id[e]) == int(cloth.bodies[v]) for e, v in binding["pins"])

    def verify_native_ownership(self, bindings):
        d = self.world.data
        held = {int(e) for b in bindings for e, _ in b["pins"]}
        active = {int(e) for ids in self.pool.values() for e in ids if d.eq_active[e]}
        if held != active:
            raise ValueError("Unexpected active cloth pins")

    def penetration_reason(self, name, indices):
        """SoftGraspBackend's convex-obstacle guard on MuJoCo's collision geoms (every MuJoCo collision geom is convex:
        a primitive or a mesh's convex hull)."""
        from scipy.spatial import ConvexHull
        x = self.positions(name)[indices]
        for entity_name, entity in self.world.entities.items():
            if entity_name == "robot" or not getattr(entity, "links", None):
                continue
            for link in entity.links:
                for geom in link.geoms:
                    v = np.asarray(geom.get_verts()).reshape(-1, 3)
                    if len(v) < 4 or np.linalg.matrix_rank(v - v.mean(axis=0)) < 3:
                        continue
                    if np.any(x.max(axis=0) < v.min(axis=0)) or np.any(x.min(axis=0) > v.max(axis=0)):
                        continue
                    planes = ConvexHull(v).equations
                    distance = (x @ planes[:, :3].T + planes[:, 3]).max(axis=1)
                    if np.any(distance < -self.config.max_penetration_m):
                        return "material_obstacle_penetration"
        return None

    # -- the slip rule
    def sample(self):
        """After every substep: each holding hand's pin force sum enters its window; a mean above the slip force marks
        the hand slipped. MuJoCo puts the equality rows first (d.ne of them, by equality id), a connect's 3 rows."""
        if not self._window or self.slip_force is None:
            return
        d = self.world.data
        ids, force = np.asarray(d.efc_id[:d.ne]), np.asarray(d.efc_force)
        for hand, window in self._window.items():
            total = np.zeros(3)
            for e in self.pool[hand]:
                if d.eq_active[e]:
                    rows = np.flatnonzero(ids == e)
                    if len(rows):
                        total += force[rows[0]:rows[0] + 3]
            window.append(math.hypot(*total))
            if sum(window) / self.slip_samples > self.slip_force:
                self._slipped.add(hand)

    def check_health(self, binding):
        reason = super().check_health(binding)
        if reason is None and binding["hand"] in self._slipped:
            reason = "grasp_slipped"
        return reason

    def capture_slip(self):
        return dict(windows={hand: tuple(w) for hand, w in self._window.items()}, slipped=sorted(self._slipped))

    def restore_slip(self, state):
        self._window = {hand: deque(w, maxlen=self.slip_samples) for hand, w in state["windows"].items()}
        self._slipped = set(state["slipped"])

    def read_model(self):
        return dict(super().read_model(), native_constraint="connect equalities hand -> patch vertices (MuJoCo flex)",
                    slip_force_n=self.slip_force, slip_window_s=WELD_SLIP_WINDOW_S,
                    attachment_scope="soft pins; the robot's own contacts still act on the cloth")


# The pin compression rule (VBD cloth, 2026-10-12): a pinned patch is kinematic, so the hand can drive it into other
# layers of the sheet, where the self-contact barrier (layer offset self_margin_m, 7 mm) then swings by joules per step
#. A pinch cannot hold a layer jammed into
# another: when a pinned vertex comes closer than PIN_COMPRESSION_FRACTION of the layer offset to a triangle of material
# farther than PIN_COMPRESSION_MATERIAL_M from every pinned vertex (another layer, not the patch's own neighbourhood),
# the assist lets go, as for strain and follow error. Host-side float64 geometry on the cloth state: the same on the
# CPU and GPU steps, no state of its own. Threshold 0.85 x 7 mm = 5.95 mm: normal holds stay >= 6.7 mm (six routes'
# phase ends, min 6.73 mm; per tick, easier4's route 6.95 mm and a stress stream's 9.96 mm), k1's pinned patch fell
# 6.9 -> 4.1 mm with the barrier swings from about 5.4 mm.
PIN_COMPRESSION_FRACTION, PIN_COMPRESSION_MATERIAL_M = .85, .03


def _point_triangle_distance(p, a, b, c):
    """Distances from points p (n, 3) to triangles (a, b, c) (n, 3 each), row by row (Ericson's closest point)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    q = np.empty_like(p)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v, w = vb / denom, vc / denom
        q[:] = a + ab * v[:, None] + ac * w[:, None]                     # inside the face
        m = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)                   # edge bc
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q[m] = (b + (c - b) * t[:, None])[m]
        m2 = (vb <= 0) & (d2 >= 0) & (d6 <= 0)                            # edge ac
        t = d2 / (d2 - d6)
        q[m2] = (a + ac * t[:, None])[m2]
        m3 = (vc <= 0) & (d1 >= 0) & (d3 <= 0)                            # edge ab
        t = d1 / (d1 - d3)
        q[m3] = (a + ab * t[:, None])[m3]
    for mask, corner in (((d1 <= 0) & (d2 <= 0), a), ((d3 >= 0) & (d4 <= d3), b), ((d6 >= 0) & (d5 <= d6), c)):
        q[mask] = corner[mask]
    return np.linalg.norm(p - q, axis=1)


def pin_compression(cloth, idx):
    """The least distance from the pinned vertices idx to a triangle of other material (every corner farther than
    PIN_COMPRESSION_MATERIAL_M in the rest sheet from every pinned vertex); inf when there is none near."""
    from scipy.spatial import cKDTree
    idx = np.asarray(idx, dtype=np.int64)
    X, rest, tri = cloth.vertices(), np.asarray(cloth.rest, dtype=np.float64), cloth._tri
    far = cKDTree(rest[idx]).query(rest)[0] > PIN_COMPRESSION_MATERIAL_M
    other = tri[far[tri].all(axis=1)]
    if not len(other):
        return float("inf")
    reach = float(cloth.settings["self_margin_m"]) + .012            # a vertex within the margin of a 12 mm triangle
    centroid = X[other].mean(axis=1)
    pairs = cKDTree(X[idx]).query_ball_tree(cKDTree(centroid), reach)
    pi = np.array([i for i, near in enumerate(pairs) for _ in near], dtype=np.int64)
    ti = np.array([t for near in pairs for t in sorted(near)], dtype=np.int64)
    if not len(pi):
        return float("inf")
    a, b, c = (X[other[ti, k]] for k in range(3))
    return float(_point_triangle_distance(X[idx[pi]], a, b, c).min())


class MujocoVBDClothGraspBackend(MujocoClothGraspBackend):
    """pbd_local_patch_v1 on a VBD cloth (cloth_vbd.py): the same selection, health and slip rules; the native constraint
    makes every grid vertex that carries the admitted patch kinematic at its offset in the hand frame (ClothVBD.pin),
    so it follows the hand every step; release gives the vertices their masses back. The slip rule reads the force the
    hand puts on the cloth after every substep (ClothVBD.pin_force: minus the pinned vertices' membrane forces and
    weight)."""
    evidence_source = "vbd_cloth_bilateral_finger_surface_proximity"
    exposure_remote_from_patch = True        # VBD's 7 mm layers: a bend grip is not occluded by its own top layer

    def __init__(self, world, robot, config):
        SoftGraspBackend.__init__(self, world, robot, config)
        m = world.model
        self.cloths = {}
        for name, entity in self.entities.items():
            cloth = world._cloths.get(name)
            uv = getattr(entity, "grasp_material_coordinates", None)
            if cloth is None or uv is None or not getattr(cloth, "steps_outside", False):
                raise ValueError(f"pbd_local_patch_v1 on a VBD cloth needs the cloth with its birth material map: {name}")
            self.initialize_material(name, uv, uv)
            self.cloths[name] = cloth
        self.hand_body = {hand: int(link.idx) for hand, link in self.hands.items()}
        self.pool = {}
        self.slip_force = WELD_SLIP_FORCE_N.get(robot.description.robot_id)
        self.slip_samples = max(1, round(WELD_SLIP_WINDOW_S / float(m.opt.timestep)))
        self._window = {}
        self._slipped = set()

    def attach_native(self, binding):
        name, hand = binding["target"][0], binding["hand"]
        cloth = self.cloths[name]
        verts = self._pins_for(name, binding["indices"])
        cloth.pin(hand, self.hand_body[hand], verts)
        binding["pins"] = [[k, int(v)] for k, v in enumerate(verts)]
        self._window[hand] = deque([0.] * self.slip_samples, maxlen=self.slip_samples)
        self._slipped.discard(hand)

    def check_health(self, binding):
        reason = super().check_health(binding)
        if reason is None:
            cloth = self.cloths[binding["target"][0]]
            pin = cloth.pins.get(binding["hand"])
            if pin is not None and len(pin["idx"]):
                gap = pin_compression(cloth, pin["idx"])
                if np.isfinite(gap):     # health is observed state: finite numbers only (none: no other layer near)
                    self._last_health.setdefault(binding["hand"], {})["min_pin_layer_gap_m"] = gap
                if gap < PIN_COMPRESSION_FRACTION * float(cloth.settings["self_margin_m"]):
                    reason = "material_self_compression"
        return reason

    def detach_native(self, binding):
        cloth = self.cloths[binding["target"][0]]
        if binding["hand"] in cloth.pins:
            cloth.unpin(binding["hand"])
        self._window.pop(binding["hand"], None)
        self._slipped.discard(binding["hand"])

    def exists(self, binding):
        cloth = self.cloths[binding["target"][0]]
        pin = cloth.pins.get(binding["hand"])
        return bool(binding.get("pins")) and pin is not None and \
            [int(v) for v in pin["idx"]] == sorted(int(v) for _, v in binding["pins"])

    def verify_native_ownership(self, bindings):
        held = {b["hand"] for b in bindings}
        pinned = {hand for c in self.cloths.values() for hand in c.pins}
        if held != pinned:
            raise ValueError("Unexpected pinned cloth vertices")

    def sample(self):
        """After every substep: each holding hand's force on the cloth enters its window; a mean above the slip force
        marks the hand slipped."""
        if not self._window or self.slip_force is None:
            return
        for hand, window in self._window.items():
            f = next((c.pin_force[hand] for c in self.cloths.values() if hand in c.pin_force), None)
            window.append(0. if f is None else math.hypot(*f))
            if sum(window) / self.slip_samples > self.slip_force:
                self._slipped.add(hand)

    def read_model(self):
        return dict(SoftGraspBackend.read_model(self), native_constraint="kinematic patch vertices (VBD cloth)",
                    slip_force_n=self.slip_force, slip_window_s=WELD_SLIP_WINDOW_S,
                    attachment_scope="kinematic pins; the hand feels their reaction one step later (xfrc_applied)")
