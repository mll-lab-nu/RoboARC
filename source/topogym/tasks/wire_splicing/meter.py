"""wire_splicing's splice meter: the task mechanism the MuJoCo world steps once per tick (MujocoWorld.set_mechanism).

User decision 2026-10-19 (runs/wire_splice_design_20261014, registration_draft.md "Success display"): a needle meter
on a test panel by the terminals (topogym.assets.geometry.wire_splicing.meter_xml) shows the splice's equivalent
resistance, a documented model computed from the configuration (criteria.meter_reading), and never decides success.
Every tick, after the MuJoCo steps: the tick's end kinematics (mj_kinematics, the same poses the tick's read_state
gives the task), the splice's measurement and shape conditions (the task's own judge), the reading, and the needle
hinge set to the reading's angle with zero speed. The needle collides with nothing and its centre of mass lies on its
hinge axis, so setting it moves no mass and changes no energy. Its reading is part of every checkpoint (capture /
restore); the needle's angle is in the world's own state (qpos)."""
from __future__ import annotations

import numpy as np

from topogym.assets.geometry import wire_splicing as geo

SEP = "::"


class SpliceMeter:
    def __init__(self, world, judge):
        """judge(PA, PB) -> (measurement, shape checks): the task's (WireSplicingTask.judge)."""
        import mujoco
        m = world.model
        joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"meter{SEP}needle")
        if joint < 0:
            raise ValueError("wire_splicing: the scene has no meter needle")
        self.qa, self.va = int(m.jnt_qposadr[joint]), int(m.jnt_dofadr[joint])
        self.bodies = {}
        for entity, wire in geo.WIRE_OF.items():
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{entity}{SEP}{geo.link_name(entity, k)}")
                   for k in range(geo.n_links(wire))]
            if min(ids) < 0:
                raise ValueError(f"wire_splicing: the scene lacks links of {entity}")
            self.bodies[entity] = np.asarray(ids, dtype=np.int64)
        self.tip = {entity: float(geo.segments(wire)[-1]) for entity, wire in geo.WIRE_OF.items()}
        self.world, self.m, self.judge = world, m, judge
        self.reading = self._measure(m, world.data)
        self._set_needle(world.data)

    def nodes(self, d, entity) -> np.ndarray:
        """The wire's centreline nodes from the data's kinematics (WireSplicingTask.nodes on read_state's poses)."""
        from scipy.spatial.transform import Rotation
        ids = self.bodies[entity]
        P = np.asarray(d.xpos)[ids]
        q = np.asarray(d.xquat[ids[-1]], dtype=float)
        tip = P[-1] + Rotation.from_quat(q[[1, 2, 3, 0]]).apply([0., 0., self.tip[entity]])
        return np.vstack([P, tip])

    def _measure(self, m, d) -> dict:
        import mujoco
        from . import criteria as C
        mujoco.mj_kinematics(m, d)          # the tick's end poses (the world's readout recomputes the same)
        measurement, checks = self.judge(self.nodes(d, "wire_a"), self.nodes(d, "wire_b"))
        return C.meter_reading(measurement, checks)

    def _set_needle(self, d) -> None:
        d.qpos[self.qa] = self.reading["needle_rad"]
        d.qvel[self.va] = 0.

    # ---- the world's calls
    def tick(self, m, d) -> None:
        self.reading = self._measure(m, d)
        self._set_needle(d)

    def capture(self) -> dict:
        return dict(self.reading)

    def restore(self, snap: dict) -> None:
        self.reading = dict(snap)

    def read(self) -> dict:
        """The reading: conducting, resistance_mohm (None: no path), turns, in_band, dial_position, needle_rad."""
        return dict(self.reading)
