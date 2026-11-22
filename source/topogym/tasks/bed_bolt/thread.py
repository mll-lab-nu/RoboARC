"""bed_bolt's thread: the task mechanism the MuJoCo world steps once per tick (MujocoWorld.set_mechanism).

The scene (topogym.assets.geometry.bed_bolt) holds a thread carriage at the bore mouth (tilt hinges thr_ty, thr_tz ->
slide thr_s -> spin thr_h; the lead is a joint equality, the seat the slide's range) and an inactive weld from the
carriage's spin body to the bolt. Every tick, after the MuJoCo steps (design: runs/bed_bolt_design_20261012, README 8):
  catch    not caught, the bolt's tip at least 2.5 mm inside the bore and no deeper than the stop allows (below), within
           1 mm of its axis, its axis within 2 deg of the bore's, lined up with the hole's thread (its roll about the bore
           axis within phase_tol_deg of phase_rad + depth / lead x 2 pi: the hole is threaded), and 0.5 s since the
           thread last let go: the carriage is set onto the bolt (tilts to the bolt's axis, slide = the tip's depth,
           spin = depth / lead x 2 pi), the weld anchored at the bodies' present relative pose and switched on, and the
           bore's contacts and the stop switched off (the thread guides the bolt);
  stop     a loose bolt cannot be pushed past the thread (user 2026-10-17): a collision-only plate across the bore whose
           face the tip meets, at the first thread (8.2 mm, catch_depth_max_m) until the thread has caught and from then
           at the deepest it has been turned in (s_max) if that is deeper, so a bolt that let go catches again only where
           the thread has been. It is moved only while off and switched on only where nothing overlaps it;
  hold     caught with at least hold_turns engaged turns (slide / lead): both tilt hinges held by joint equalities
           (thread_tilt_y / _z), set at their angles on that tick, each target eased to 0 by tilt_ease_deg_s while the
           actual tilt is within tilt_gap_deg of it (the bolt re-centres; something holding it tilted holds the target),
           released on the tick the thread falls below hold_turns or lets go; the tilt hinges' only stiffness is k_weak;
  let go   caught, below hold_turns, drooping more than fall_tilt_deg (the two tilt angles combined) with nothing holding
           the bolt (no robot finger touching it and no key end in its socket), or the slide below backed_out_m: the weld
           off, the bore's contacts on again once no bore wall overlaps the bolt;
  lock     caught: the spin held at its angle by a joint equality (thread_lock, set at that tick's angle), so a push or
           pull along the axis or a finger's twist does not turn the bolt (the user, 2026-10-16: a push must not advance
           it, fingers must slip). No key end in the socket: locked, always. A key end in it: static friction, the lock
           breaks away on the tick the spin carries more than SPIN_FRICTION (the lock row and the spin hinge's friction
           row together), the friction row acts from then, and the spin locks again once a whole rate_window_s since the
           break-away has passed and its rate over that window is below relock_rad_s. Released when the thread lets go.
While the thread is engaged the bolt is no assisted-grasp target (grasp_allowed; the user, 2026-10-16: fingers turning
the bolt directly must slip); the fingers' contacts stay, so a light pinch still supports it by friction.
Contacts switched off come back only where nothing overlaps them: switched on into an overlap they would push the parts
apart with the energy of that overlap (the stress observer's first check: a key drawn out of the socket after a slip,
the socket's contacts back on at once, stored 0.7 J against it).
Its state (and the model fields it writes) is part of every checkpoint: capture() / restore(). The user removed the key's
slip (a key end in the socket while the bolt turned faster than 3 rad/s lost its grip until drawn out) from the family on
2026-10-18. read() still reports its two flags, slipped (always false) and socket_on (always true), and the rate window
still follows the key's turn about the bore axis while the thread is not caught (the lock reads the rate only while
caught), so every state hash, the -i01 references' included, stays as it was.
"The socket" is the recess of the scene's drive (topogym.assets.geometry.bed_bolt DRIVES: the hex socket of
bed_bolt_crossbar, or another drive's recess with the matching tips on the tool).
  ratchet  the ratcheting driver's drives (DRIVES ratchet; RATCHET): the hinge key_ratchet between the L and the bit turns
           one way only: every tick its lower limit is raised to its angle when the angle has risen (the bit turned
           forward against the L: the L swung back while the bit stays in the recess), so the bit can never turn back
           against the L (the L swung forward carries the bit); the free-wheel's drag is the hinge's friction. The limit
           (a model field) is part of capture() / restore() and read()."""
from __future__ import annotations

from collections import deque

import numpy as np

from topogym.assets.geometry.bed_bolt import (BOLT, DRIVES, SPIN_FRICTION, THREAD, key_ends, key_geom_names,
                                               key_end_points, stop_x)

SEP = "::"
PRESS_M = .0005             # a catch is allowed with the tip this far into the stop's face


class BoltThread:
    def __init__(self, world, Th, finger_bodies, drive="hex"):
        import mujoco
        m = world.model
        name = lambda kind, n: mujoco.mj_name2id(m, kind, f"{'key' if n == 'key' else 'bolt'}{SEP}{n}")
        J, B, G = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
        self.j = {n: name(J, n) for n in ("thr_ty", "thr_tz", "thr_s", "thr_h")}
        if min(self.j.values()) < 0:
            raise ValueError("bed_bolt: the scene has no thread carriage")
        self.qa = {n: int(m.jnt_qposadr[i]) for n, i in self.j.items()}
        self.va = {n: int(m.jnt_dofadr[i]) for n, i in self.j.items()}
        self.weld = name(mujoco.mjtObj.mjOBJ_EQUALITY, "thread_weld")
        self.lock = name(mujoco.mjtObj.mjOBJ_EQUALITY, "thread_lock")
        self.tilt_eq = {j: name(mujoco.mjtObj.mjOBJ_EQUALITY, f"thread_tilt_{j[-1]}") for j in ("thr_ty", "thr_tz")}
        if min(self.weld, self.lock, *self.tilt_eq.values()) < 0:
            raise ValueError("bed_bolt: the scene has no thread weld, spin lock or tilt holds")
        self.spin_dof = int(m.jnt_dofadr[self.j["thr_h"]])
        self._ctype = (int(mujoco.mjtConstraint.mjCNSTR_EQUALITY), int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
        self.bolt, self.key, self.spin_body = name(B, "bolt"), name(B, "key"), name(B, "thr_spin")
        self.bore = np.array([mujoco.mj_name2id(m, G, f"bed_frame{SEP}bore_{i}") for i in range(16)])
        self.drive = drive                                   # the recess and the tool's tips (DRIVES)
        self.bolt_geoms = [name(G, n) for n in ("bolt_tip", "bolt_shank", "bolt_head")]      # what can reach the bore
        self.stop, self.stop_body = name(G, "thr_stop_plate"), name(B, "thr_stop")
        if min(self.stop, self.stop_body) < 0:
            raise ValueError("bed_bolt: the scene has no thread stop")
        self.key_geoms = [mujoco.mj_name2id(m, G, f"key{SEP}{n}") for n in key_geom_names(drive)]
        self.ratchet = None                                  # the ratchet's hinge (joint id) and the bit's body
        if DRIVES[drive].get("ratchet"):
            self.ratchet = mujoco.mj_name2id(m, J, f"key{SEP}key_ratchet")
            self.bit = mujoco.mj_name2id(m, B, f"key{SEP}key_bit")
            if min(self.ratchet, self.bit) < 0:
                raise ValueError("bed_bolt: the ratcheting driver has no hinge or no bit")
            self.ratchet_q = int(m.jnt_qposadr[self.ratchet])
            self.ratchet_lock = float(m.jnt_range[self.ratchet, 0])
        if min(self.bolt_geoms + self.key_geoms) < 0:
            raise ValueError("bed_bolt: the bolt's or the key's geoms are missing")
        self._fromto = np.zeros(6)
        self.bore_bits = (m.geom_contype[self.bore].copy(), m.geom_conaffinity[self.bore].copy())
        self.stop_bits = (int(m.geom_contype[self.stop]), int(m.geom_conaffinity[self.stop]))
        self.fingers = np.asarray(sorted(finger_bodies), dtype=np.int64)
        self.Th = np.asarray(Th, dtype=float)
        self.lead = BOLT["lead"]
        self.c = dict(THREAD)
        self.tick_s = float(world.physics_dt_s)
        self.window = max(1, int(round(self.c["rate_window_s"] / self.tick_s)))
        self.world, self.m = world, m
        self.engaged = False
        self.bore_on = True                                  # the bore's contacts
        self.stop_on = True                                  # the stop plate's contacts
        self.s_max = 0.                                      # the deepest the thread has carried the bolt (slide, m)
        self.t_let_go = -1e9
        self.t_break = -1e9                                  # the spin lock's last break-away (key in the socket)
        self.angles: deque = deque(maxlen=self.window + 1)   # the turn angle each tick (spin hinge or key about the bore)
        self.rate = 0.
        self.events = 0                                      # catches and let-gos so far (a count for read())

    # ---- geometry
    def _bolt_axis(self, d):
        R = np.asarray(d.xmat[self.bolt]).reshape(3, 3)
        x = self.Th[:3, 0]
        rel = np.asarray(d.xpos[self.bolt]) - self.Th[:3, 3]
        depth = float(rel @ x)
        radial = float(np.linalg.norm(rel - depth * x))
        angle = float(np.arccos(np.clip(-R[:, 0] @ x, -1., 1.)))
        return depth, radial, angle, R

    def roll(self, R) -> float:
        """The bolt's roll about the bore axis: its y axis from the mouth frame's y toward z."""
        y = R[:, 1]
        return float(np.arctan2(y @ self.Th[:3, 2], y @ self.Th[:3, 1]))

    def phase_error(self, depth, R) -> float:
        """How far (rad, +-pi) the bolt's thread is turned from lining up with the hole's at this depth."""
        e = self.roll(R) - self.c["phase_rad"] - 2 * np.pi * depth / self.lead
        return float((e + np.pi) % (2 * np.pi) - np.pi)

    def stop_limit(self) -> float:
        """The deepest a loose bolt may catch: the first thread, or the deepest the thread has carried it."""
        return max(self.c["catch_depth_max_m"], self.s_max)

    def turns(self, d) -> float:
        return float(d.qpos[self.qa["thr_s"]]) / self.lead

    def tilt(self, d) -> float:
        return float(np.hypot(d.qpos[self.qa["thr_ty"]], d.qpos[self.qa["thr_tz"]]))

    def key_end_in_socket(self, d):
        """The key end in the bolt's socket ('short', 'long' or None) and its depth below the head face (m)."""
        Rb, pb = np.asarray(d.xmat[self.bolt]).reshape(3, 3), np.asarray(d.xpos[self.bolt])
        Rk, pk = np.asarray(d.xmat[self.key]).reshape(3, 3), np.asarray(d.xpos[self.key])
        face = BOLT["L"] + BOLT["head_h"]
        window, ends = DRIVES[self.drive]["in_socket"], key_ends(self.drive)
        for end, local in key_end_points().items():
            if end not in ends:
                continue
            lb = Rb.T @ (pk + Rk @ local - pb)
            if self.drive == "hex":                         # bed_bolt_crossbar's check as it was
                inside = face - BOLT["hex_t"] - .0015 <= lb[0] <= face + .0005 and np.hypot(lb[1], lb[2]) < .0025
            elif "across" in window:                        # a slot: across its mid-plane and along it
                a = np.radians(DRIVES[self.drive]["rotate_deg"])
                along, across = lb[1] * np.cos(a) + lb[2] * np.sin(a), -lb[1] * np.sin(a) + lb[2] * np.cos(a)
                inside = (window["depth"][0] <= face - lb[0] <= window["depth"][1] and abs(across) < window["across"]
                          and abs(along) < window["along"])
            else:
                inside = window["depth"][0] <= face - lb[0] <= window["depth"][1] and np.hypot(lb[1], lb[2]) < window["radial"]
            if inside:
                return end, float(face - lb[0])
        return None, 0.

    def _lever_angle(self, d) -> float:
        """The key's long arm about the bore axis (the ratcheting driver: its bit's frame x, the long arm's with the
        ratchet at 0)."""
        Rk = np.asarray(d.xmat[self.key if self.ratchet is None else self.bit]).reshape(3, 3)
        lever = self.Th[:3, :3].T @ (-Rk[:, 0])
        return float(np.arctan2(-lever[1], lever[2]))

    def _supported(self, d) -> bool:
        """A robot finger touches the bolt, or a key end sits in its socket."""
        n = int(d.ncon)
        if n:
            b1 = self.m.geom_bodyid[np.asarray(d.contact.geom1[:n])]
            b2 = self.m.geom_bodyid[np.asarray(d.contact.geom2[:n])]
            if np.any(((b1 == self.bolt) & np.isin(b2, self.fingers)) | ((b2 == self.bolt) & np.isin(b1, self.fingers))):
                return True
        return self.key_end_in_socket(d)[0] is not None

    def _clear(self, d, geoms_a, geoms_b) -> bool:
        """No geom of a overlaps a geom of b (MuJoCo's convex distance, whatever their contact bits)."""
        import mujoco
        for a in geoms_a:
            for b in geoms_b:
                if mujoco.mj_geomDistance(self.m, d, int(a), int(b), .002, self._fromto) < 0:
                    return False
        return True

    # ---- switches
    def _bore(self, on):
        self.bore_on = bool(on)
        self.m.geom_contype[self.bore] = self.bore_bits[0] if on else 0
        self.m.geom_conaffinity[self.bore] = self.bore_bits[1] if on else 0

    def _stop(self, on):
        self.stop_on = bool(on)
        self.m.geom_contype[self.stop] = self.stop_bits[0] if on else 0
        self.m.geom_conaffinity[self.stop] = self.stop_bits[1] if on else 0

    def _place_stop(self, d):
        """The plate at stop_x(s_max) along its body's x (the bore axis); its world pose written for the clear check (a
        static body: the next kinematics gives the same)."""
        self.m.geom_pos[self.stop, 0] = stop_x(self.s_max, self.c)
        R = np.asarray(d.xmat[self.stop_body]).reshape(3, 3)
        d.geom_xpos[self.stop] = np.asarray(d.xpos[self.stop_body]) + R @ self.m.geom_pos[self.stop]

    def _catch(self, m, d, depth, R):
        import mujoco
        bl = self.Th[:3, :3].T @ (-R[:, 0])                  # the bolt's tip direction in the mouth frame
        d.qpos[self.qa["thr_ty"]] = -np.arctan2(bl[2], bl[0])
        d.qpos[self.qa["thr_tz"]] = np.arctan2(bl[1], np.hypot(bl[0], bl[2]))
        d.qpos[self.qa["thr_s"]] = depth
        d.qpos[self.qa["thr_h"]] = depth / self.lead * 2 * np.pi
        for n in self.va:
            d.qvel[self.va[n]] = 0.
        mujoco.mj_kinematics(m, d)
        R1, p1 = np.asarray(d.xmat[self.spin_body]).reshape(3, 3), np.asarray(d.xpos[self.spin_body])
        R2, p2 = np.asarray(d.xmat[self.bolt]).reshape(3, 3), np.asarray(d.xpos[self.bolt])
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R1.T @ R2).as_quat()
        m.eq_data[self.weld, 0:3] = 0.
        m.eq_data[self.weld, 3:6] = R1.T @ (p2 - p1)
        m.eq_data[self.weld, 6:10] = (q[3], q[0], q[1], q[2])
        m.eq_data[self.weld, 10] = 1.
        d.eq_active[self.weld] = 1
        self._bore(False)
        self._stop(False)
        self.s_max = max(self.s_max, float(depth))
        self.engaged = True
        self.angles.clear()
        self.events += 1

    def _let_go(self, d):
        d.eq_active[self.weld] = 0
        self.engaged = False
        self.t_let_go = float(d.time)
        self.angles.clear()
        self.events += 1

    def _ratchet(self, m, d):
        q = float(d.qpos[self.ratchet_q])
        if q > self.ratchet_lock:
            self.ratchet_lock = q
            m.jnt_range[self.ratchet, 0] = q

    # ---- the world's calls
    def tick(self, m, d):
        c = self.c
        if self.ratchet is not None:
            self._ratchet(m, d)
        angle = float(d.qpos[self.qa["thr_h"]]) if self.engaged else self._lever_angle(d)
        if self.angles:
            prev = self.angles[-1]
            if not self.engaged:                                    # the key's angle: continuous across +-pi
                angle = prev + float(np.arctan2(np.sin(angle - prev), np.cos(angle - prev)))
        self.angles.append(angle)
        self.rate = self._rate()
        if not self.engaged:
            depth, radial, angle, R = self._bolt_axis(d)
            # the tip may press into the stop's face (stop_margin_m past the limit) by a fraction of a millimetre
            if (c["catch_depth_m"] <= depth <= self.stop_limit() + c["stop_margin_m"] + PRESS_M
                    and radial < c["catch_radial_m"] and angle < np.radians(c["catch_angle_deg"])
                    and d.time - self.t_let_go > c["recatch_wait_s"]
                    and abs(self.phase_error(depth, R)) < np.radians(c["phase_tol_deg"])):
                self._catch(m, d, depth, R)
        else:
            self.s_max = max(self.s_max, float(d.qpos[self.qa["thr_s"]]))
            n = self.turns(d)
            if n < c["hold_turns"] and self.tilt(d) > np.radians(c["fall_tilt_deg"]) and not self._supported(d):
                self._let_go(d)
            elif d.qpos[self.qa["thr_s"]] < c["backed_out_m"]:
                self._let_go(d)
        if not self.engaged and not self.bore_on and self._clear(d, self.bolt_geoms, self.bore):
            self._bore(True)
        if not self.engaged and not self.stop_on:
            self._place_stop(d)
            if self._clear(d, self.bolt_geoms + self.key_geoms, [self.stop]):
                self._stop(True)
        end, _ = self.key_end_in_socket(d)
        self._tilt_hold(m, d)
        self._spin_lock(m, d, end)

    def _tilt_hold(self, m, d):
        c = self.c
        hold = self.engaged and self.turns(d) >= c["hold_turns"]
        ease, gap = np.radians(c["tilt_ease_deg_s"]) * self.tick_s, np.radians(c["tilt_gap_deg"])
        for j, e in self.tilt_eq.items():
            q = float(d.qpos[self.qa[j]])
            if not hold:
                d.eq_active[e] = 0
            elif not d.eq_active[e]:
                m.eq_data[e, 0] = q
                d.eq_active[e] = 1
            else:
                t = float(m.eq_data[e, 0])
                if t != 0. and abs(t - q) < gap:
                    m.eq_data[e, 0] = t - np.sign(t) * min(abs(t), ease)

    def spin_torque(self, d) -> float:
        """The torque the spin's constraints carry (the lock row and the friction row, N m), from the last solve."""
        n = int(d.nefc)
        t, i, f = np.asarray(d.efc_type[:n]), np.asarray(d.efc_id[:n]), np.asarray(d.efc_force[:n])
        rows = ((t == self._ctype[0]) & (i == self.lock)) | ((t == self._ctype[1]) & (i == self.spin_dof))
        return float(f[rows].sum())

    def _spin_lock(self, m, d, end):
        c, locked = self.c, self.locked(d)
        if not self.engaged:
            lock = False
        elif end is None:
            lock = True
        elif locked:
            lock = abs(self.spin_torque(d)) <= SPIN_FRICTION
            if not lock:
                self.t_break = float(d.time)
        else:
            lock = d.time - self.t_break >= c["rate_window_s"] and self.rate < c["relock_rad_s"]
        if lock != locked:
            if lock:
                m.eq_data[self.lock, 0] = d.qpos[self.qa["thr_h"]]
            d.eq_active[self.lock] = int(lock)

    def locked(self, d) -> bool:
        """The spin lock is on (its switch and angle are the world's equality state, which every checkpoint holds)."""
        return bool(d.eq_active[self.lock])

    def _rate(self):
        """The turn over the last rate_window_s (fewer ticks right after a catch or a let-go), rad / s."""
        if len(self.angles) < 2:
            return 0.
        return abs(self.angles[-1] - self.angles[0]) / ((len(self.angles) - 1) * self.tick_s)

    def capture(self) -> dict:
        snap = dict(engaged=self.engaged, bore_on=self.bore_on,
                    t_let_go=self.t_let_go, t_break=self.t_break, angles=list(self.angles),
                    rate=self.rate, events=self.events, stop_on=self.stop_on, s_max=self.s_max)
        if self.ratchet is not None:
            snap["ratchet_lock"] = self.ratchet_lock
        return snap

    def restore(self, snap: dict) -> None:
        self.engaged = bool(snap["engaged"])
        self.t_let_go, self.rate, self.events = float(snap["t_let_go"]), float(snap["rate"]), int(snap["events"])
        self.t_break = float(snap["t_break"])
        self.angles = deque((float(a) for a in snap["angles"]), maxlen=self.window + 1)
        self._bore(bool(snap["bore_on"]))
        self.s_max = float(snap["s_max"])
        self.m.geom_pos[self.stop, 0] = stop_x(self.s_max, self.c)     # placed from s_max while on (moved only while off)
        self._stop(bool(snap["stop_on"]))
        if self.ratchet is not None:
            self.ratchet_lock = float(snap["ratchet_lock"])
            self.m.jnt_range[self.ratchet, 0] = self.ratchet_lock

    def grasp_allowed(self, entity, link) -> bool:
        """The assisted grasp asks (topogym/world/mujoco/grasp.py): the bolt is no target while the thread is engaged."""
        return not (self.engaged and entity == "bolt")

    def read(self) -> dict:
        """The thread's readings and every discrete state capture() holds (the instance's state_sha hashes them), with the
        switches the world's equality state holds (the spin lock, the tilt holds) and their targets."""
        d, m = self.world.data, self.m
        tilt_eq = list(self.tilt_eq.values())
        return dict(engaged=self.engaged, slipped=False, bore_on=self.bore_on, socket_on=True,
                    t_let_go=self.t_let_go, t_break=self.t_break, angles=list(self.angles), rate_rad_s=self.rate,
                    events=self.events, stop_on=self.stop_on, s_max_m=self.s_max, stop_limit_m=self.stop_limit(),
                    locked=self.locked(d), lock_angle_rad=float(m.eq_data[self.lock, 0]),
                    tilt_held=[bool(d.eq_active[e]) for e in tilt_eq],
                    tilt_target_rad=[float(m.eq_data[e, 0]) for e in tilt_eq],
                    turns=self.turns(d), tilt_rad=self.tilt(d),
                    **({} if self.ratchet is None else dict(ratchet_lock_rad=self.ratchet_lock,
                                                            ratchet_rad=float(d.qpos[self.ratchet_q]))))
