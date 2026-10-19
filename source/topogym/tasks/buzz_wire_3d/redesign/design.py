"""A buzz_wire_3d redesign tier as data: the wire (dense toy-coordinate controls), the wand, the insulated rest
sleeves, the wooden obstacles and the deck placement. `layout()` turns it into the family's own BuzzWireLayout, so the
wire, the wand, the deck, the endpoint mounts, the start and the finish poses come from the family's generator
(topogym.assets.geometry.buzz_wire_3d) unchanged."""
from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np

from topogym.assets.geometry.buzz_wire_3d import BuzzWireLayout, SpatialWire, endpoint_pose

TABLE_Z = 0.91964            # the tabletop height of every buzz scene (frozen scenes: wire body z)
DECK_TOP = 0.125 + 0.0175     # base_elevation_m + base thickness (toy z)
WIRE_R = 0.00147
TUBE_R = 0.00126


@dataclass
class Design:
    name: str
    tier: str
    controls: np.ndarray                  # (N, 3) toy coordinates, absolute z
    sections: list                        # [(name, control index)], increasing
    loop_radius_m: float = 0.0154
    sleeves: list = field(default_factory=list)      # [(s0, s1)] rail arc-length ranges (m), insulated rest sleeves
    sleeve_radius_m: float = 0.003
    obstacles: list = field(default_factory=list)    # [dict(kind=box|cylinder, pos, size|radius+height, euler_deg)]
    center_xy_m: tuple = (-0.08, -0.27)               # world xy of the deck centre
    base_size_m: tuple = (0.5201, 0.3689, 0.0175)
    notes: str = ""
    robot_mount: tuple | None = None                  # (x, y, yaw_deg): the Panda's table-top mount; None = default

    def layout(self) -> BuzzWireLayout:
        c = np.asarray(self.controls, float)
        return BuzzWireLayout(
            template_id="authored_multistage_v1",
            authored_controls_m=tuple(map(tuple, c)),
            authored_section_ends=tuple((n, int(i)) for n, i in self.sections),
            base_size_m=tuple(self.base_size_m), horizontal_scale=.42, depth_scale=.42, height_scale=.315,
            wire_radius_m=WIRE_R, loop_radius_m=float(self.loop_radius_m), loop_tube_radius_m=TUBE_R, loop_segments=64,
            stem_length_m=.01596, grip_length_m=.03052, grip_radius_m=.0049, grip_shape="box",
            grip_half_thickness_m=.0035, grip_flat_roll_deg=0., end_stop_radius_m=.009, end_stop_shape="block_mount",
            end_stop_thickness_m=.0028, rail_segment_m=.0028, witness_step_m=.000175, end_reserve_m=.098,
            clearance_cover_m=.00035, finish="toy_wood_copper", post_radius_m=0., post_height_m=0.,
            mount_size_m=(.008, .008, .012), base_elevation_m=.125, route_elevation_m=0., supported_endpoints=True,
            endpoint_support_part="deck", rest_handle_pitch_deg=0., rest_finish_yaw_deg=0.,
            rest_finish_quaternion_wxyz=(), finish_cradle_mount_clearance_m=0., finish_cradle_neck_pair=False,
            spiral_exit_roll_deg=150, spiral_exit_roll_start=.965, spiral_exit_roll_end=.995, upright_handle_pitch_deg=45)

    def wire(self) -> SpatialWire:
        return SpatialWire(self.layout())

    def origin(self):
        return np.array([*self.center_xy_m, TABLE_Z])

    def summary(self) -> dict:
        w = self.wire()
        L = self.layout()
        rail = w.rail
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rail, axis=0), axis=1))]
        p0, r0 = endpoint_pose(w, 0)
        p1, r1 = endpoint_pose(w, 1)
        return dict(name=self.name, tier=self.tier, rail_length_m=round(float(arc[-1]), 4),
                    loop_radius_m=self.loop_radius_m,
                    radial_clearance_m=round(self.loop_radius_m - TUBE_R - WIRE_R, 5),
                    sleeves=[[round(a, 4), round(b, 4)] for a, b in self.sleeves], obstacles=len(self.obstacles),
                    rail_min=rail.min(0).round(4).tolist(), rail_max=rail.max(0).round(4).tolist(),
                    start_pose=dict(p=p0.round(4).tolist(), q_wxyz=r0.as_quat()[[3, 0, 1, 2]].round(5).tolist()),
                    finish_pose=dict(p=p1.round(4).tolist(), q_wxyz=r1.as_quat()[[3, 0, 1, 2]].round(5).tolist()),
                    min_curvature_radius_m=round(float(w.minimum_curvature_radius), 4),
                    center_xy_m=list(self.center_xy_m), sections=[n for n, _ in self.sections],
                    robot_mount=None if self.robot_mount is None else [round(float(v), 4) for v in self.robot_mount])


DEFAULT_MOUNT = (-0.57, 0.0, 0.0)                  # the scene config's Panda mount (world x, y; yaw deg)
# the scene config's link0 footprint (mount frame) and table; the mount plane is the table surface (scene config rule)
FOOTPRINT = ((-0.154079, -0.094614), (0.071567, 0.094671))
TABLE_XY = ((-0.8856786, -0.7), (0.6143214, 0.7))
EDGE_MARGIN = 0.01        # the scene config's robot_mount.edge_margin
DECK_CLEARANCE = 0.02     # footprint to the toy's deck (its feet stand inside it)


def footprint_corners(mount):
    x, y, yaw = mount
    (a, b), (c, d) = FOOTPRINT
    local = np.array([[a, b], [c, b], [c, d], [a, d]])
    t = np.deg2rad(yaw)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return local @ rot.T + [x, y]


def _separated(p, q):
    """Separating-axis test of two convex polygons (corner arrays)."""
    for poly in (p, q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            if (p @ n).max() < (q @ n).min() or (q @ n).max() < (p @ n).min():
                return True
    return False


def mount_problems(mount, center_xy, base_size):
    """Why a table-top Panda mount (x, y, yaw_deg) is not legal for a deck at center_xy (size base_size): its link0
    footprint must lie inside the table (scene edge margin) and keep DECK_CLEARANCE from the deck. [] = legal."""
    out = []
    f = footprint_corners(mount)
    lo, hi = np.asarray(TABLE_XY[0]) + EDGE_MARGIN, np.asarray(TABLE_XY[1]) - EDGE_MARGIN
    if (f < lo).any() or (f > hi).any():
        out.append("footprint leaves the table")
    hx, hy = base_size[0] / 2 + DECK_CLEARANCE, base_size[1] / 2 + DECK_CLEARANCE
    deck = np.asarray(center_xy) + np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    if not _separated(f, deck):
        out.append("footprint within %.0f mm of the deck" % (DECK_CLEARANCE * 1e3))
    return out


def rail_arc(rail):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(rail, axis=0), axis=1))]


def sleeve_segments(rail, sleeves):
    """Rail segments (a, b) covered by the rest sleeves: the sleeve is a chain of capsules along the same rail."""
    arc = rail_arc(rail)
    out = []
    for s0, s1 in sleeves:
        idx = np.flatnonzero((arc >= s0) & (arc <= s1))
        if len(idx) < 2:
            continue
        pts = rail[idx[0]:idx[-1] + 1]
        out += [(a, b) for a, b in zip(pts[:-1], pts[1:])]
    return out
