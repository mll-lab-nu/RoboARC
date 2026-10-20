"""clock_assembling's rules: the task mechanism the MuJoCo world steps once per tick (MujocoWorld.set_mechanism).

The scene (topogym.assets.geometry.clock_gear_train) holds a carrier at each lower pivot hole (lean hinges tx, ty and a
spin hinge, no collision), a mesh gate at each hole mouth but the barrel's, and inactive equalities: seat welds (carrier
to wheel), mesh couplings (joint equalities between spin hinges), lean locks, the screws' welds to the bridge and the
bridge's weld to the world. Every tick, after the MuJoCo steps (design: runs/watch_assembly_design_20261014,
README_v3.md section 4; mv_rules.py there):
  gate     while a wheel's lower pivot tip is 0-4 mm above its hole mouth within 1.5 mm of the hole axis and its pinion
           meets the seated previous wheel's teeth tip on tip (|leaf phase - tooth phase| < 0.2 of a pitch), the gate at the
           hole mouth collides with the wheels (the arbor's bottom lands on it); it closes only while the pivot is above
           it and nothing overlaps it, and opens when the phases part (the wheel turned by about half a leaf);
  seat     a wheel whose lower pivot is down in its hole (axis within 0.6 mm of the hole's, origin at most 0.8 mm above the
           plate face, tilt < 3 deg) and not on a closed gate is welded to its carrier, the carrier's light joints first
           set to the wheel's orientation (as bed_bolt's thread carriage);
  mesh     two seated wheels whose pinion and teeth meet are coupled (joint equality, polycoef[1] = -teeth / leaves), the
           constant chosen so the leaves sit in the tooth gaps;
  screws   a screw whose head sits on the bridge (0-0.8 mm above the bridge top, axis within 0.6 mm of its hole) counts
           its clockwise turns relative to the bridge; at 2 turns it is tight, welded to the bridge (no thread simulated);
  fix      both screws tight and the bridge seated (underside within 0.5 mm of its seat, tilt < 0.5 deg): the bridge is
           welded to the world, every wheel's lean hinges locked where they are (joint equalities), the pivot-hole
           contacts switched off (the locks now hold the arbors; locked arbors rubbing in their holes would brake the
           train); the barrel's arbor welded to its click carrier and the mainspring's lock released;
  click    from then on the click hinge's lower limit is the largest angle the arbor reached less one ratchet tooth (a
           joint limit: the arbor winds one way only, the mainspring then drives the drum and the train);
  run test once the arbor has not advanced for RUN_SETTLE_S after a wind, a RUN_WINDOW_S window measures the train
           running on the mainspring alone: the fourth wheel's turns, the time it spends slower than STALL_FRACTION of
           its free-run speed (the speed the mainspring's present windup gives against the train's damping and friction,
           from the model: within 1 % of the measured speed on the -i01 reference) and its turn per drum turn. Winding
           again restarts the window.
Its state and the contact bits it writes are part of every checkpoint (capture / restore); the equalities' data and
switches are in the world's own state (model.eq_data, mjData.eq_active)."""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.assets.geometry import clock_gear_train as geo

SEP = "::"
GATE_LATERAL_M, GATE_PHASE = .0015, .2
SEAT_LATERAL_M, SEAT_HEIGHT_M, SEAT_TILT_DEG = .0006, .0008, 3.
SCREW_HEIGHT_M, SCREW_LATERAL_M = .0008, .0006
FIX_HEIGHT_M, FIX_TILT_DEG = .0005, .5
RUN_SETTLE_S, RUN_WINDOW_S = .5, 5.          # the run test: no winding for 0.5 s, then a 5 s window
RUN_MIN_WIND = math.radians(10.)             # a window opens only on a wound arbor
STALL_FRACTION = .1                          # slower than this fraction of the free-run speed counts as stalled
PIVOT_CONTACTS = ("hole_", "endstone_", "pad_", "padlow_", "cham_")


def wrap(x):
    return (x + .5) % 1. - .5


def yaw_of(R) -> float:
    return math.atan2(R[1, 0], R[0, 0])


class GearTrainRules:
    def __init__(self, world, origin):
        import mujoco
        m = world.model
        E, B, J, G = (mujoco.mjtObj.mjOBJ_EQUALITY, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT,
                      mujoco.mjtObj.mjOBJ_GEOM)

        def idx(kind, name):
            i = mujoco.mj_name2id(m, kind, f"{geo.ENTITY}{SEP}{name}")
            if i < 0:
                raise ValueError(f"clock_assembling: the scene has no {name}")
            return int(i)
        self.world, self.m = world, m
        self.o = np.asarray(origin, dtype=float)
        self.b = {n: idx(B, n) for n in geo.PARTS}
        self.bcar = {n: idx(B, f"carr_{n}_s") for n in geo.WHEELS}
        self.jq = {n: {a: int(m.jnt_qposadr[idx(J, f"{a}_{n}")]) for a in ("tx", "ty", "spin")} for n in geo.WHEELS}
        self.jv = {n: {a: int(m.jnt_dofadr[idx(J, f"{a}_{n}")]) for a in ("tx", "ty", "spin")} for n in geo.WHEELS}
        self.e_seat = {n: idx(E, f"seat_{n}") for n in geo.WHEELS}
        self.e_mesh = {n: idx(E, f"mesh_{n}") for n in geo.PARENT}
        self.e_lock = {(a, n): idx(E, f"lock_{a}_{n}") for n in geo.WHEELS for a in ("tx", "ty")}
        self.e_fix = idx(E, "bridge_fix")
        self.e_spring_lock, self.e_arbor = idx(E, "spring_lock"), idx(E, "arbor_fix")
        self.j_click = idx(J, "click_barrel")
        self.q_click = int(m.jnt_qposadr[self.j_click])
        self.v_click = int(m.jnt_dofadr[self.j_click])
        self.b_arbor, self.b_click_parent = idx(B, "barrel_arbor"), idx(B, "carr_barrel_y")
        self.click_range0 = float(m.jnt_range[self.j_click, 0])
        # the free run: the fourth wheel's speed for the mainspring's windup q against the train reflected to the drum
        # (spin hinges' damping and friction loss times the squared / plain ratios, the mainspring hinge's damping)
        j_spring = idx(J, "spring_barrel")
        self.q_spring = int(m.jnt_qposadr[j_spring])
        self.k_spring = float(m.jnt_stiffness[j_spring])
        ratio = {"barrel": 1., "center": geo.RATIO["center"]}
        ratio["third"] = ratio["center"] * geo.RATIO["third"]
        ratio["fourth"] = ratio["third"] * geo.RATIO["fourth"]
        self.drag = float(sum(m.dof_damping[self.jv[n]["spin"]] * ratio[n] ** 2 for n in geo.WHEELS)
                          + m.dof_damping[int(m.jnt_dofadr[j_spring])])
        self.friction = float(sum(m.dof_frictionloss[self.jv[n]["spin"]] * abs(ratio[n]) for n in geo.WHEELS))
        self.e_screw = {s: idx(E, f"{s}_fix") for s in ("screw1", "screw2")}
        self.gate = {n: idx(G, f"gate_{n}") for n in geo.PARENT}
        names = [mujoco.mj_id2name(m, G, g) or "" for g in range(m.ngeom)]
        prefixes = tuple(f"{geo.ENTITY}{SEP}{p}" for p in PIVOT_CONTACTS)
        self.pivot_contacts = np.array([g for g, nm in enumerate(names) if nm.startswith(prefixes)], dtype=np.int64)
        self.wheel_geoms = {n: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == self.b[n]
                                and (m.geom_contype[g] or m.geom_conaffinity[g])] for n in geo.WHEELS}
        # the bits the scene compiled (the gates collide at compile time: a geom compiled with contype = conaffinity = 0
        # never enters collision detection); the gates start open
        self.gate_bits = {n: int(m.geom_conaffinity[g]) for n, g in self.gate.items()}
        self.pivot_bits = (m.geom_contype[self.pivot_contacts].copy(), m.geom_conaffinity[self.pivot_contacts].copy())
        self._fromto = np.zeros(6)
        self.seated = {n: False for n in geo.WHEELS}
        self.gate_closed = {n: False for n in geo.PARENT}
        self.mesh_on = {n: False for n in geo.PARENT}
        self.turns = {"screw1": 0., "screw2": 0.}
        self.tight = {"screw1": False, "screw2": False}
        self.last_rel = {"screw1": None, "screw2": None}
        self.fixed = False
        self.arbor0 = self.max_arbor = self.last_advance_t = None
        self.run = self._run_blank()
        self.gate_blocks = {n: 0 for n in geo.PARENT}
        for n in geo.PARENT:
            self._set_gate(n, False)

    # ---- geometry readouts (W: the movement frame, axes as the world's)
    def _T(self, d, b):
        return np.asarray(d.xpos[b]), np.asarray(d.xmat[b]).reshape(3, 3)

    def wheel_local(self, d, n):
        p, R = self._T(d, self.b[n])
        return p - self.o, R

    def spin(self, d, n) -> float:
        return float(d.qpos[self.jq[n]["spin"]])

    def bridge_local(self, d):
        p, R = self._T(d, self.b["bridge"])
        return p - self.o, R

    def tip_offset(self, d, n):
        """The upper pivot tip (the barrel: its arbor's top) relative to its bridge hole: (horizontal vector in W, height
        above the bridge underside along the bridge's axis)."""
        p, R = self.wheel_local(d, n)
        z = geo.SHOULDER_Z + geo.UP_PIVOT_L if n != "barrel" else geo.H_BRIDGE + geo.T_BRIDGE
        tip = p + R @ np.array([0., 0., z])
        bp, BR = self.bridge_local(d)
        hole = bp + BR @ np.array([geo.POS[n][0], geo.POS[n][1], 0.])
        axis = BR[:, 2]
        v = tip - hole
        h = float(v @ axis)
        return v - h * axis, h

    def hole_radius(self, n) -> float:
        return geo.PIVOT_HOLE_R if n != "barrel" else geo.BARREL_ARBOR_R + .0005

    def entered(self, d, n, depth_m=.0003) -> bool:
        """The pivot tip is inside its bridge hole by at least depth_m."""
        lat, h = self.tip_offset(d, n)
        return bool(np.linalg.norm(lat) <= self.hole_radius(n) and h >= depth_m)

    def bridge_seat_error(self, d):
        bp, BR = self.bridge_local(d)
        tilt = math.degrees(math.acos(float(np.clip(BR[2, 2], -1., 1.))))
        return float(bp[2] - geo.H_BRIDGE), float(np.hypot(bp[0], bp[1])), tilt

    def phase_error(self, d, n) -> float:
        """Leaf phase minus tooth phase at the line of centres, in pitches (0: tip on tip; +-0.5: a leaf in a gap)."""
        par = geo.PARENT[n]
        pw, Rw = self.wheel_local(d, n)
        pp, Rp = self.wheel_local(d, par)
        a = math.atan2(pp[1] - pw[1], pp[0] - pw[0])
        u = (a - yaw_of(Rw)) * geo.GEARS[n]["pin"] / (2 * math.pi)
        v = (a + math.pi - yaw_of(Rp)) * geo.GEARS[par]["teeth"] / (2 * math.pi)
        return wrap(u - v)

    def _clear(self, d, geoms_a, geoms_b) -> bool:
        """No geom of a overlaps a geom of b (MuJoCo's convex distance, whatever their contact bits)."""
        import mujoco
        for a in geoms_a:
            for b in geoms_b:
                if mujoco.mj_geomDistance(self.m, d, int(a), int(b), .002, self._fromto) < 0:
                    return False
        return True

    # ---- switches
    def _set_gate(self, n, closed):
        self.gate_closed[n] = bool(closed)
        self.m.geom_conaffinity[self.gate[n]] = self.gate_bits[n] if closed else 0

    def _set_pivot_contacts(self, on):
        self.m.geom_contype[self.pivot_contacts] = self.pivot_bits[0] if on else 0
        self.m.geom_conaffinity[self.pivot_contacts] = self.pivot_bits[1] if on else 0

    def _weld_here(self, d, eid):
        """Switch weld eid on at its two bodies' present relative pose (body2 0: the world)."""
        b1, b2 = int(self.m.eq_obj1id[eid]), int(self.m.eq_obj2id[eid])
        p1, R1 = self._T(d, b1)
        p2, R2 = self._T(d, b2)
        q = Rotation.from_matrix(R1.T @ R2).as_quat()
        self.m.eq_data[eid, 0:3] = 0.
        self.m.eq_data[eid, 3:6] = R1.T @ (p2 - p1)
        self.m.eq_data[eid, 6:10] = (q[3], q[0], q[1], q[2])
        self.m.eq_data[eid, 10] = 1.
        d.eq_active[eid] = 1

    # ---- the rules
    def _gate(self, d, n, p, lat):
        low = p[2] - geo.LOW_PIVOT_L              # the lower pivot tip's height above the plate face
        near = lat <= GATE_LATERAL_M and (-.0005 <= low <= geo.LOW_PIVOT_L or self.gate_closed[n])
        tip_on_tip = bool(near and self.seated[geo.PARENT[n]] and abs(self.phase_error(d, n)) < GATE_PHASE)
        if tip_on_tip and low >= 0. and not self.gate_closed[n] and self._clear(d, [self.gate[n]], self.wheel_geoms[n]):
            self._set_gate(n, True)
            self.gate_blocks[n] += 1
        elif self.gate_closed[n] and not tip_on_tip:
            self._set_gate(n, False)

    def _seat(self, d, n, R):
        import mujoco
        a, b, c = Rotation.from_matrix(R).as_euler("XYZ")
        d.qpos[self.jq[n]["tx"]] = float(np.clip(a, -geo.TILT_MAX, geo.TILT_MAX))
        d.qpos[self.jq[n]["ty"]] = float(np.clip(b, -geo.TILT_MAX, geo.TILT_MAX))
        d.qpos[self.jq[n]["spin"]] = float(c)
        for k in ("tx", "ty", "spin"):
            d.qvel[self.jv[n][k]] = 0.
        mujoco.mj_kinematics(self.m, d)
        self._weld_here(d, self.e_seat[n])
        self.seated[n] = True

    def _mesh(self, d, n, par):
        """Couple n to par with the leaves in the gaps: the free parent turned by the phase error."""
        err = self.phase_error(d, n)                       # want +-0.5
        shift = (.5 if err >= 0 else -.5) - err            # pitches the leaf must move relative to the tooth
        dpar = shift * 2 * math.pi / geo.GEARS[par]["teeth"]
        e = self.e_mesh[n]
        r = geo.RATIO[n]
        self.m.eq_data[e, :5] = 0.
        self.m.eq_data[e, 0] = self.spin(d, n) - r * (self.spin(d, par) + dpar)
        self.m.eq_data[e, 1] = r
        d.eq_active[e] = 1
        self.mesh_on[n] = True

    def _screws(self, d):
        bp, BR = self.bridge_local(d)
        for s, k in (("screw1", "P1"), ("screw2", "P2")):
            if self.tight[s]:
                continue
            p, R = self._T(d, self.b[s])
            hole = bp + BR @ np.array([geo.PILLARS[k][0] + geo.SCREW_OFF[0], geo.PILLARS[k][1] + geo.SCREW_OFF[1],
                                       geo.T_BRIDGE])
            axis = BR[:, 2]
            v = p - self.o - hole
            h = float(v @ axis)
            lat = float(np.linalg.norm(v - h * axis))
            rel = yaw_of(BR.T @ R)
            if 0. <= h <= SCREW_HEIGHT_M and lat <= SCREW_LATERAL_M:
                if self.last_rel[s] is not None:
                    dy = (rel - self.last_rel[s] + math.pi) % (2 * math.pi) - math.pi
                    self.turns[s] += -dy / (2 * math.pi)       # clockwise seen from above tightens
                self.last_rel[s] = rel
                if self.turns[s] >= geo.SCREW_TURNS:
                    self._weld_here(d, self.e_screw[s])
                    self.tight[s] = True
            else:
                self.last_rel[s] = None

    def _fix(self, d):
        self._weld_here(d, self.e_fix)
        for n in geo.WHEELS:
            for a in ("tx", "ty"):
                e = self.e_lock[(a, n)]
                self.m.eq_data[e, :5] = 0.
                self.m.eq_data[e, 0] = d.qpos[self.jq[n][a]]
                d.eq_active[e] = 1
        self._set_pivot_contacts(False)
        # the arbor onto its click carrier (the carrier's light hinge first set to the arbor's turn, as a wheel's seat),
        # then the mainspring free
        import mujoco
        Ry = np.asarray(d.xmat[self.b_click_parent]).reshape(3, 3)
        Ra = np.asarray(d.xmat[self.b_arbor]).reshape(3, 3)
        d.qpos[self.q_click] = yaw_of(Ry.T @ Ra)
        d.qvel[self.v_click] = 0.
        mujoco.mj_kinematics(self.m, d)
        self._weld_here(d, self.e_arbor)
        d.eq_active[self.e_spring_lock] = 0
        self.fixed = True
        self.arbor0 = self.max_arbor = float(d.qpos[self.q_click])
        self.last_advance_t = float(d.time)
        self._set_click(self.max_arbor - geo.CLICK_PITCH)

    # ---- the run test
    @staticmethod
    def _run_blank():
        return dict(active=False, t0=None, t_prev=None, fourth0=None, drum0=None, wound=None, stall_s=0., done=False,
                    turns=0., ratio=None, drum_turn=0., speed_sum=0., free_sum=0.)

    def free_speed(self, d) -> float:
        """The fourth wheel's free-run speed (rad/s) for the mainspring's present windup."""
        torque = max(self.k_spring * float(d.qpos[self.q_spring]) - self.friction, 0.)
        return abs(geo.TOTAL_RATIO) * torque / self.drag

    def _set_click(self, lower):
        self.m.jnt_range[self.j_click, 0] = float(lower)

    def _click_and_run(self, d):
        a = float(d.qpos[self.q_click])
        t = float(d.time)
        if a > self.max_arbor + 1e-9:                     # wound further: the click's limit follows, one tooth behind
            self.max_arbor = a
            self.last_advance_t = t
            self._set_click(a - geo.CLICK_PITCH)
            if self.run["active"] or self.run["done"]:    # winding again restarts the run test
                self.run = self._run_blank()
        r = self.run
        if not r["active"] and not r["done"]:
            if t - self.last_advance_t >= RUN_SETTLE_S and self.max_arbor - self.arbor0 >= RUN_MIN_WIND:
                r.update(active=True, t0=t, t_prev=t, fourth0=self.spin(d, "fourth"), drum0=self.spin(d, "barrel"),
                         wound=a - self.arbor0, stall_s=0., speed_sum=0., free_sum=0.)
            return
        if r["active"]:
            w = float(d.qvel[self.jv["fourth"]["spin"]])
            free = self.free_speed(d)
            if abs(w) < STALL_FRACTION * free:
                r["stall_s"] += t - r["t_prev"]
            r["speed_sum"] += abs(w) * (t - r["t_prev"])
            r["free_sum"] += free * (t - r["t_prev"])
            r["t_prev"] = t
            if t - r["t0"] >= RUN_WINDOW_S:
                dfourth = self.spin(d, "fourth") - r["fourth0"]
                ddrum = self.spin(d, "barrel") - r["drum0"]
                r.update(active=False, done=True, turns=abs(dfourth) / (2 * math.pi), drum_turn=ddrum,
                         ratio=dfourth / ddrum if abs(ddrum) > 1e-9 else None)

    # ---- the world's calls
    def tick(self, m, d):
        import mujoco
        changed = False
        for n in geo.WHEELS:
            if self.seated[n]:
                continue
            p, R = self.wheel_local(d, n)
            lat = float(np.hypot(p[0] - geo.POS[n][0], p[1] - geo.POS[n][1]))
            tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1., 1.))))
            if n in geo.PARENT:
                self._gate(d, n, p, lat)
            if lat <= SEAT_LATERAL_M and p[2] <= SEAT_HEIGHT_M and tilt < SEAT_TILT_DEG and not self.gate_closed.get(n, False):
                self._seat(d, n, R)
                changed = True
        for n, par in geo.PARENT.items():
            if not self.mesh_on[n] and self.seated[n] and self.seated[par]:
                self._mesh(d, n, par)
                changed = True
        if changed:
            mujoco.mj_kinematics(m, d)
        self._screws(d)
        if self.fixed:
            self._click_and_run(d)
        elif all(self.tight.values()):
            dz, _, tilt = self.bridge_seat_error(d)
            if abs(dz) <= FIX_HEIGHT_M and tilt < FIX_TILT_DEG:
                self._fix(d)

    def capture(self) -> dict:
        return dict(seated=dict(self.seated), gate_closed=dict(self.gate_closed), mesh_on=dict(self.mesh_on),
                    turns=dict(self.turns), tight=dict(self.tight), last_rel=dict(self.last_rel), fixed=self.fixed,
                    arbor0=self.arbor0, max_arbor=self.max_arbor, last_advance_t=self.last_advance_t, run=dict(self.run),
                    gate_blocks=dict(self.gate_blocks))

    def restore(self, snap: dict) -> None:
        self.seated = {n: bool(v) for n, v in snap["seated"].items()}
        self.mesh_on = {n: bool(v) for n, v in snap["mesh_on"].items()}
        self.turns = {s: float(v) for s, v in snap["turns"].items()}
        self.tight = {s: bool(v) for s, v in snap["tight"].items()}
        self.last_rel = {s: None if v is None else float(v) for s, v in snap["last_rel"].items()}
        self.fixed = bool(snap["fixed"])
        f = lambda v: None if v is None else float(v)     # noqa: E731
        self.arbor0, self.max_arbor, self.last_advance_t = f(snap["arbor0"]), f(snap["max_arbor"]), f(snap["last_advance_t"])
        self.run = dict(snap["run"])
        self._set_click(self.max_arbor - geo.CLICK_PITCH if self.fixed else self.click_range0)
        self.gate_blocks = {n: int(v) for n, v in snap["gate_blocks"].items()}
        for n, closed in snap["gate_closed"].items():
            self._set_gate(n, bool(closed))
        self._set_pivot_contacts(not self.fixed)

    def read(self) -> dict:
        r = self.run
        out = dict(fixed=self.fixed, wound_rad=float(self.max_arbor - self.arbor0) if self.fixed else 0.,
                   run_active=bool(r["active"]), run_done=bool(r["done"]),
                   run_wound_rad=float(r["wound"]) if r["wound"] is not None else 0.,
                   run_fourth_turns=float(r["turns"]), run_stall_s=float(r["stall_s"]),
                   run_ratio=float(r["ratio"]) if r["ratio"] is not None else 0., run_ratio_measured=r["ratio"] is not None,
                   run_speed_of_free=float(r["speed_sum"] / r["free_sum"]) if r["free_sum"] else 0.)
        for n in geo.WHEELS:
            out[f"{n}_seated"] = self.seated[n]
        for n in geo.PARENT:
            out[f"{n}_meshed"] = self.mesh_on[n]
            out[f"{n}_gate_closed"] = self.gate_closed[n]
        for s in ("screw1", "screw2"):
            out[f"{s}_turns"] = float(self.turns[s])
            out[f"{s}_tight"] = self.tight[s]
        return out
