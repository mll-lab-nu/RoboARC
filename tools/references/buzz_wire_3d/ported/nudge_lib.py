"""Distance queries for the nudge planner without touching the Session's model: mj_geomDistance between the robot's
gripper / wrist geoms and the static geoms near the TCP, and between the wand metal and the wire metal."""
import numpy as np, mujoco
from topogym.planning.buzz_wire_3d.redesign.kin import ROBOT_BODIES

NEAR = ("link5", "link6", "link7", "hand", "tool_mount", "r85_base_mount", "r85_base", "r85_right_driver", "r85_right_coupler",
        "r85_right_spring_link", "r85_right_follower", "left_finger", "r85_right_silicone_pad", "r85_left_driver",
        "r85_left_coupler", "r85_left_spring_link", "r85_left_follower", "right_finger", "r85_left_silicone_pad")


class Dist:
    def __init__(self, kin, cls):
        m = kin.m
        self.m, self.kin = m, kin
        bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "" for g in range(m.ngeom)]
        coll = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
        self.robot = np.flatnonzero(np.array([b in NEAR for b in bn]) & coll)
        self.static = np.flatnonzero(kin.static_geom)
        gn = [kin.geom_name(g) for g in range(m.ngeom)]
        self.deck = np.array(["finish_support" in n or "wire_base" in n for n in gn])
        self.wire_metal = np.flatnonzero(cls["wire_metal"])
        self.wand_metal = np.flatnonzero(cls["wand_metal"])
        self.ft = np.zeros(6)

    def _rb(self, g):
        d = self.kin.d
        return d.geom_xpos[g], self.m.geom_rbound[g]

    def robot_static(self, dmax=0.01, skip_deck=True):
        """min distance robot(near) - static (deck excluded if skip_deck: returned separately) -> (dmin, pair, deck_min)"""
        m, d = self.m, self.kin.d
        tcp = d.xpos[m.body("hand").id]
        sx = d.geom_xpos[self.static]
        near_static = self.static[np.linalg.norm(sx - tcp, axis=1) < 0.35 + m.geom_rbound[self.static]]
        best, pair, deck = 1.0, None, 1.0
        for g1 in self.robot:
            p1, r1 = d.geom_xpos[g1], m.geom_rbound[g1]
            for g2 in near_static:
                if np.linalg.norm(d.geom_xpos[g2] - p1) - r1 - m.geom_rbound[g2] > dmax:
                    continue
                dist = mujoco.mj_geomDistance(m, d, int(g1), int(g2), dmax, self.ft)
                if self.deck[g2]:
                    deck = min(deck, dist)
                elif dist < best:
                    best, pair = dist, (g1, g2)
        return best, pair, deck

    def wand_wire(self, dmax=0.01):
        m, d = self.m, self.kin.d
        c = d.xpos[self.kin.wand_bid]
        wires = self.wire_metal[np.linalg.norm(d.geom_xpos[self.wire_metal] - c, axis=1) < 0.12]
        best = 1.0
        for g1 in self.wand_metal:
            p1, r1 = d.geom_xpos[g1], m.geom_rbound[g1]
            for g2 in wires:
                if np.linalg.norm(d.geom_xpos[g2] - p1) - r1 - m.geom_rbound[g2] > dmax:
                    continue
                best = min(best, mujoco.mj_geomDistance(m, d, int(g1), int(g2), dmax, self.ft))
        return best
