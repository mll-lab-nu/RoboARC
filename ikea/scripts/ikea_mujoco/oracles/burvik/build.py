"""BURVIK closed-loop reference, phase by phase (privileged, true state). The base rows are the standard agent's retimed
demo (p2_cl, within the speed limits); the controller keeps them where they still work and replaces the arms' targets
where the prism physics moves the parts:

  P0  rows      0 - 64000: base rows (legs 0/1 threaded into the top, the top lifted and turned on its side)
  P1  rows  64000 - 70000: base rows, both TCPs 15 mm lower through the rack placement and the release (the top stands
                           in the rack instead of tipping over it), the offset ramped out on the retreat
  P2  rows  70000 - 78500: base rows (the right hand takes the handle)
  P3  the handle (right arm, servo in the top's frame): pre-pose 30 mm out along the axis, aligned, then in at 2 mm/s to
      0.3 mm past the seat; held there (servo) until the handle weld (after both handle legs)
  P4  leg_2 (left arm, base rows to the grasp, then a servo in the handle's frame): pre-pose 20 mm out, axis aligned,
      in at 2 mm/s while turning, 0.3 mm past the seat, until its weld; release, retreat, blend back to the base rows
  P5  leg_3: the same
  P6  base rows to the end (the upright turn), the right arm's targets carried with the handle hold's offset
State snapshots after every phase (snap_<phase>.pkl) make each phase's search cheap.
"""
import os, sys, pickle, json, argparse
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orc import Orc, setup
from servo import PoseServo, rotvec


def axis_servo_err(d, b, axis_part_local, axis_world_star):
    """rotation vector that brings the body's local axis onto the world axis (minimal)."""
    a = d.xmat[b].reshape(3, 3) @ axis_part_local
    c = np.cross(a, axis_world_star); s = np.linalg.norm(c); cth = float(a @ axis_world_star)
    return np.zeros(3) if s < 1e-12 else c / s * np.arctan2(s, cth)


class Builder:
    def __init__(self, code, scenes, base, out):
        self.R = setup(code, scenes)
        self.o = Orc(self.R)
        self.base = np.load(base)["ctrl"]
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.mech = self.o.mech
        self.ops = {op["key"]: op for op in self.mech.ops}
        hk = float(os.environ.get("BURVIK_HKP", "0.08"))
        self.hsv = PoseServo(self.o, "right", "handle", kp=hk, kr=hk)
        hla = float(os.environ.get("BURVIK_HOLD_LEAD_ALL", "0"))       # >0: the hold's lead bounded in every phase
        self.hsv.lead_max = hla if hla > 0 else None
        self.log = []; self.tilt = None
        self.debug_contacts = float(os.environ.get("BURVIK_DEBUG_CONTACTS_MM", "0")) / 1000

    # ---- snapshots
    def snap(self, name, k):
        pickle.dump(dict(o=self.o.save(), k=k, j7_off=getattr(self, "j7_off", 0.)), open(os.path.join(self.out, f"snap_{name}.pkl"), "wb"), protocol=4)

    def load(self, name):
        s = pickle.load(open(os.path.join(self.out, f"snap_{name}.pkl"), "rb"))
        self.o.restore(s["o"]); self.j7_off = s.get("j7_off", 0.); return s["k"]

    def note(self, **kw):
        kw["row"] = len(self.o.rows); kw["active"] = self.o.active()
        self.log.append(kw); print(json.dumps(kw, default=str), flush=True)

    # ---- the handle target and hold
    def handle_star(self, g):
        if getattr(self, "tilt", None):
            th = self.tilt_theta(); c, s_ = np.cos(th), np.sin(th)
            Ry = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]); pv = self.tilt["pivot"]
            return pv + Ry @ (self.tilt["p0"] - pv) + np.array([0., 0., self.tilt_lift()]), Ry @ self.tilt["R0"]
        op = self.ops["s2-handle-two-holes"]; d = self.o.d; bt = self.o.m.body("top").id
        Rt = np.zeros(9); mujoco.mju_quat2Mat(Rt, np.asarray(op["target_quat"], float)); Rt = Rt.reshape(3, 3)
        Rh = d.xmat[bt].reshape(3, 3)
        return d.xpos[bt] + Rh @ (np.asarray(op["target_pos"]) + g * np.asarray(op["axis"], float)), Rh @ Rt

    def tilt_start(self, theta, n, lift=None):
        """start rotating the top+handle assembly (the right arm holds the handle; after leg_2 the handle is locked to the
        top) by theta (rad) about world y through the top's support point (the mean top-world contact), ramped over n
        rows: the hold's target becomes that world pose. theta=None: back to the top-relative hold."""
        m, d = self.o.m, self.o.d; top = m.body("top").id
        pts = [np.array(d.contact[i].pos) for i in range(d.ncon)
               if {int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])} == {top, 0} and d.contact[i].efc_address >= 0]
        if theta is None:
            self.tilt = None; return
        prev = self.tilt
        p0, R0 = (prev["p0"], prev["R0"]) if prev else self.handle_star(-0.0003)
        th0 = self.tilt_theta() if prev else 0.
        z0 = self.tilt_lift() if prev else 0.; z1 = z0 if lift is None else lift
        # the pivot: the top's lowest support contacts (the jig's V-guards under its bottom edge; it leans on posts higher up)
        if pts:
            zmin = min(p_[2] for p_ in pts); low = [p_ for p_ in pts if p_[2] < zmin + 0.02]
        pivot = prev["pivot"] if prev else (np.mean(low, axis=0) if pts else d.xpos[top].copy())
        self.tilt = dict(p0=p0, R0=R0, pivot=pivot, th0=th0, th1=theta, z0=z0, z1=z1, r0=len(self.o.rows), n=n)
        self.note(phase="tilt", pivot=np.round(pivot * 1000, 1).tolist(), th0_deg=round(np.degrees(th0), 2), th1_deg=round(np.degrees(theta), 2), n=n, ncontacts=len(pts))

    def tilt_a(self):
        t = self.tilt; a = min(1., max(0., (len(self.o.rows) - t["r0"]) / t["n"]))
        return 0.5 - 0.5 * np.cos(np.pi * a)

    def tilt_theta(self):
        t = self.tilt; return t["th0"] + (t["th1"] - t["th0"]) * self.tilt_a()

    def tilt_lift(self):
        t = self.tilt; return t["z0"] + (t["z1"] - t["z0"]) * self.tilt_a()

    def hold_tick(self, row, g=-0.0003, extra_left=None):
        """the right arm holds the handle at g (servo); the row's left arm part as given."""
        p, Rm = self.handle_star(g)
        return self.hsv.tick(row, p, Rm)

    # ---- phases
    def p0(self):
        self.o.run(self.base[:64000]); self.snap("p0", 64000)

    def p1(self, d_mm=15.):
        k0 = len(self.o.rows); d = d_mm / 1000
        def off(k):
            k1, k2, k3, k4 = 400, 2800, 3600, 4800
            z = 0. if k < k1 else -d * (k - k1) / (k2 - k1) if k < k2 else -d if k < k3 else -d * (1 - (k - k3) / (k4 - k3)) if k < k4 else 0.
            return {"left": np.array([0, 0, z]), "right": np.array([0, 0, z])}
        self.o.run(self.base[64000:70000], offset=off)
        p, Rm = self.o.body("top"); self.note(phase="p1", top=np.round(p, 4).tolist(), top_z=np.round(Rm[:, 2], 3).tolist())
        self.snap("p1", 70000)

    def p2(self):
        self.o.run(self.base[70000:78500]); self.snap("p2", 78500)

    def p3(self, above=0.030, speed=0.002, depth=-0.0003, k_end=85500):
        """handle in; returns at base index k_end with the handle held."""
        k = 78500; g = above
        while k < k_end:
            p, Rm = self.handle_star(g)
            ep, ew = self.hsv.tick(self.base[k], p, Rm)
            if (ep < 0.001 and ew < 0.01) or g < above:
                g = max(depth, g - speed * self.o.s.control_dt)
            k += 1
        op = self.ops["s2-handle-two-holes"]
        gap, lat, ang, Rr = self.mech.measure_lean(op); rel, _ = self.mech._local(op)
        self.note(phase="p3", grasp=self.grasp(), tau_r=np.round(self.o.d.actuator_force[self.o.arm["right"]], 1).tolist(), gap_mm=round(gap * 1000, 3), lat_mm=round(lat * 1000, 3), ang=round(ang, 3),
                  holes_mm=round(self.mech._two_holes(op, Rr, rel) * 1000, 3))
        self.snap("p3", k)
        return k

    def leg_star(self, key, g):
        op = self.ops[key]; d = self.o.d; bh = self.o.m.body(op["host"]).id
        Rh = d.xmat[bh].reshape(3, 3)
        return d.xpos[bh] + Rh @ (np.asarray(op["target_pos"]) + g * np.asarray(op["axis"], float)), Rh @ np.asarray(op["axis"], float)

    def leg(self, key, k0, k_release_base, k_rejoin, above=0.020, speed=0.002, depth=-0.0003, spin=float(os.environ.get("BURVIK_SPIN", "-2.2")), lead=0.001, thread_start=0.0105, kp_ax=0.03,
            kp=0.08, vmax=0.0004, max_ticks=int(os.environ.get("BURVIK_LEG_MAX_TICKS", "24000"))):
        """the left arm takes over the leg at base index k0 (already grasped by the base rows): servo the leg (axis on the
        op axis, its point on the target at gap g) and spin it about the axis at `spin` rad/s once in the near zone, until
        the weld; then open, retreat 40 mm along the axis and blend back to the base rows at k_rejoin."""
        o, d, m = self.o, self.o.d, self.o.m
        op = self.ops[key]; b = m.body(op["part"]).id
        pa = np.asarray(op["part_axis"], float)
        lim = o.s.step_limit[o.arm["left"]] * 0.9
        g = above; k = k0; ticks = 0; welded_at = None; base_ix = k0; started = False; paused = False
        grip_closed = self.base[k0][o.grip["left"]]
        while ticks < max_ticks:
            row = np.array(self.base[min(base_ix, len(self.base) - 1)], dtype=np.float64)
            row[o.grip["left"]] = grip_closed                        # keep the hand closed (the base rows may open)
            # the right arm holds the handle
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            ph, Rh = self.handle_star(-0.0003)
            eph, ewh = self.hsv.errors(ph, Rh)
            vh = np.clip(self.hsv.kp * eph, -self.hsv.vmax, self.hsv.vmax); wh = np.clip(self.hsv.kr * ewh, -self.hsv.wmax, self.hsv.wmax)
            tcp_r = d.site_xpos[o.site["right"]]
            dqr = np.clip(o.dls("right", vh + np.cross(wh, tcp_r - d.xpos[self.hsv.b]), wh), -self.hsv.lim, self.hsv.lim)
            # the leg
            p_star, ax_star = self.leg_star(key, g)
            # the leg's target point is its frame origin (the mechanism measures the part frame)
            e_p = p_star - d.xpos[b]
            e_w = axis_servo_err(d, b, pa, ax_star)
            mt = self.mech.metrics.get(key, {})
            near = mt.get("near", False)
            gap_now = mt.get("gap_m", g)
            # in the thread (near the seat): the axial target follows the leg just ahead of where the thread has pulled
            # it (lead 1 mm / turn; the spin drives the advance), the spin stops at the seat
            # own geometry (the mechanism's near zone drops out at 2 mm lateral; the controller must not)
            Rhh = d.xmat[m.body(op["host"]).id].reshape(3, 3)
            rel_h = Rhh.T @ (d.xpos[b] - d.xpos[m.body(op["host"]).id]) - np.asarray(op["target_pos"])
            axh = np.asarray(op["axis"], float)
            gap_now = float(rel_h @ axh); lat_now = float(np.linalg.norm(rel_h - gap_now * axh))
            if not started and gap_now < thread_start and lat_now < 0.001:
                started = True
            seated = started and gap_now < 0.0002
            hp_, hR_ = self.handle_star(-0.0003); he_ = float(np.linalg.norm(self.hsv.errors(hp_, hR_)[0]))
            if started and not seated:
                paused = (lat_now > 0.0012 or he_ > 0.0015) if not paused else (lat_now > 0.0008 or he_ > 0.001)
            threading = started and not seated and not paused
            if threading:                     # the axial target advances at the thread's lead x the spin rate
                g = min(g, gap_now) - float(os.environ.get("BURVIK_LEAD", "0.001")) * abs(spin) / (2 * np.pi) * o.s.control_dt
                if abs(mt.get("turns", 0)) > 0 and False: pass
            elif seated or paused:
                g = gap_now
            w_spin = (spin * o.s.control_dt) * ax_star if threading else np.zeros(3)
            e_ax = float(e_p @ ax_star) * ax_star; e_lat = e_p - e_ax
            v = kp * e_lat + (float(os.environ.get("BURVIK_KPAX", kp_ax)) if threading else kp) * e_ax; nv = np.linalg.norm(v)
            if nv > vmax: v *= vmax / nv
            w = np.clip(0.08 * e_w, -0.002, 0.002) + w_spin
            tcp = d.site_xpos[o.site["left"]]
            dql = np.clip(o.dls("left", v + np.cross(w, tcp - d.xpos[b]), w), -lim, lim)
            prev = o.rows[-1]
            row[o.arm["right"]] = prev[o.arm["right"]] + dqr
            row[o.arm["left"]] = prev[o.arm["left"]] + dql
            o.step(row)
            ticks += 1
            if not started and (np.linalg.norm(e_p) < 0.001 and np.linalg.norm(e_w) < 0.01 or g < above):
                g = max(thread_start - 0.0005, g - speed * o.s.control_dt)
            if self.debug_contacts and ticks % 100 == 0 and g < self.debug_contacts:
                hop = self.ops["s2-handle-two-holes"]; hg, hl, ha, _ = self.mech.measure_lean(hop)
                Rhh = d.xmat[m.body(op["host"]).id].reshape(3, 3)
                self.note(phase="dbg", g_mm=round(g * 1000, 2), gap=round(gap_now * 1000, 2), e_lat_host=np.round(1000 * (Rhh.T @ e_lat), 2).tolist(),
                          e_ax_mm=round(1000 * float(e_p @ ax_star), 2), tilt=round(np.degrees(np.linalg.norm(e_w)), 2),
                          turns=round(mt.get("turns", 0), 3), handle=dict(gap=round(hg * 1000, 2), lat=round(hl * 1000, 2), ang=round(ha, 2)),
                          threading=bool(threading), near=bool(near), paused=bool(paused), he_mm=round(he_ * 1000, 2))
                cs = []
                for c in d.contact[:d.ncon]:
                    b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                    if b in (b1, b2) and c.efc_address >= 0 and "finger" not in m.body(b2 if b1 == b else b1).name:
                        o2 = c.geom2 if b1 == b else c.geom1
                        own = c.geom1 if b1 == b else c.geom2
                        cs.append((m.geom(int(own)).name, m.geom(int(o2)).name or f"g{int(o2)}:{m.body(int(m.geom_bodyid[o2])).name}:t{int(m.geom_type[o2])}",
                                   round(-1000 * float(c.dist), 2), round(abs(float(d.efc_force[c.efc_address])), 1)))
                self.note(phase="contacts", g_mm=round(g * 1000, 2), gap=round(1000 * self.mech.metrics.get(key, {}).get("gap_m", 0), 2),
                          lat=round(1000 * self.mech.metrics.get(key, {}).get("lateral_m", 0), 2), contacts=sorted(cs, key=lambda x: -x[3])[:12], ncon_leg=len(cs))
            if ticks % 1000 == 0:
                mt = self.mech.metrics.get(key, {})
                self.note(phase=key, g_mm=round(g * 1000, 2), e_p_mm=round(np.linalg.norm(e_p) * 1000, 2),
                          e_w_deg=round(np.degrees(np.linalg.norm(e_w)), 2), **{x: (round(v_, 4) if isinstance(v_, float) else v_) for x, v_ in mt.items()})
            if self.mech.active(key):
                welded_at = len(o.rows); break
        if welded_at is None:
            self.note(phase=key, result="no weld"); self.snap(key + "_fail", base_ix); return None
        self.note(phase=key, result="welded", ticks=ticks)
        # release: open the hand (base open value), hold the handle, retreat 40 mm out along the axis, then blend
        open_val = self.base[k_release_base][o.grip["left"]]
        for i in range(160):
            row = np.array(o.rows[-1]); row[o.grip["left"]] = open_val
            ph, Rh = self.handle_star(-0.0003); self.hsv.tick(row, ph, Rh)
        _, ax_star = self.leg_star(key, 0.)
        for i in range(400):
            row = np.array(o.rows[-1]); mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            dql = np.clip(o.dls("left", -ax_star * 0.0001, np.zeros(3)), -lim, lim)
            row[o.arm["left"]] += dql
            ph, Rh = self.handle_star(-0.0003); self.hsv.tick(row, ph, Rh)
        start = np.array(o.rows[-1][o.arm["left"]]); n = 1200
        for i in range(n):
            tgt = self.base[k_rejoin - n + i]
            a = (i + 1) / n
            row = np.array(tgt, dtype=np.float64)
            row[o.arm["left"]] = (1 - a) * start + a * tgt[o.arm["left"]]
            ph, Rh = self.handle_star(-0.0003); self.hsv.tick(row, ph, Rh)
        self.snap(key, k_rejoin)
        return k_rejoin

    def grasp(self, side="right", part="handle"):
        """(contacts, N) between the side's fingers and the part."""
        m, d = self.o.m, self.o.d; n = 0; f = 0.
        for c in d.contact[:d.ncon]:
            s_ = {m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name}
            if part in s_ and s_ & {side + "_left_finger", side + "_right_finger"} and c.efc_address >= 0:
                n += 1; f += abs(float(d.efc_force[c.efc_address]))
        return n, round(f, 1)

    def axial_force(self, b, host, ax):
        """sum over the leg-host contacts of the force on the leg along ax (+ = pushed out of the hole), N."""
        m, d = self.o.m, self.o.d; f6 = np.zeros(6); tot = 0.
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if {b1, b2} != {b, host} or c.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fw = c.frame.reshape(3, 3).T @ f6[:3]                 # force on geom2, world
            tot += float(fw @ ax) * (1. if b2 == b else -1.)
        return tot

    def leg_contacts(self, b):
        m, d = self.o.m, self.o.d; cs = []
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if b in (b1, b2) and c.efc_address >= 0 and "finger" not in m.body(b2 if b1 == b else b1).name:
                own, oth = (c.geom1, c.geom2) if b1 == b else (c.geom2, c.geom1)
                cs.append((m.geom(int(own)).name, m.geom(int(oth)).name or f"g{int(oth)}:{m.body(int(m.geom_bodyid[oth])).name}",
                           round(-1000 * float(c.dist), 3), round(abs(float(d.efc_force[c.efc_address])), 1)))
        return sorted(cs, key=lambda x: -x[3])

    def arm_contacts(self, side):
        """contacts of the side's arm links/hand with anything but its own fingers' grasped part: {pair: N}."""
        m, d = self.o.m, self.o.d; out = {}
        for c in d.contact[:d.ncon]:
            n1, n2 = m.body(int(m.geom_bodyid[c.geom1])).name, m.body(int(m.geom_bodyid[c.geom2])).name
            if c.efc_address < 0 or not (n1.startswith(side + "_") or n2.startswith(side + "_")):
                continue
            if ("finger" in n1 and n2.startswith("leg")) or ("finger" in n2 and n1.startswith("leg")):
                continue
            k = n1 + "|" + n2; out[k] = round(out.get(k, 0) + abs(float(d.efc_force[c.efc_address])), 1)
        return out

    def free_table(self, side, j7q, step_deg=5, clear=float(os.environ.get("BURVIK_RATCHET_CLEAR", "0.006"))):
        """collision-free j7 angles for the side's hand at its current pose: {k: free} for q7 + k*step (k = 0..359/step),
        free = every hand/finger/link7 geom clear of every world (fixture) geom by `clear` (kinematics on a copy)."""
        m = self.o.m; d2 = mujoco.MjData(m); d2.qpos[:] = self.o.d.qpos; fr = np.zeros(6)
        hand = [g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name in (side + "_hand", side + "_left_finger", side + "_right_finger", side + "_link7", side + "_link6")
                and (m.geom_contype[g] | m.geom_conaffinity[g])]
        world = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == 0 and (m.geom_contype[g] | m.geom_conaffinity[g]) and m.geom(g).name != "floor"]
        q0 = float(d2.qpos[j7q]); out = {}
        for k in range(360 // step_deg):
            d2.qpos[j7q] = q0 + np.radians(k * step_deg); mujoco.mj_kinematics(m, d2)
            dmin = min(mujoco.mj_geomDistance(m, d2, g, w, clear + 0.01, fr) for g in hand for w in world)
            out[k] = dmin > clear
        return q0, step_deg, out

    def leg_force(self, key, k0, k_release_base, k_rejoin, above=0.020, speed=0.002, thread_start=0.0105,
                  spin=float(os.environ.get("BURVIK_SPIN", "-2.2")), fmin=float(os.environ.get("BURVIK_FMIN", "3")),
                  fmax=float(os.environ.get("BURVIK_FMAX", "8")), ax_step=float(os.environ.get("BURVIK_AXSTEP", "2e-6")),
                  kp=0.08, vmax=0.0004, max_ticks=int(os.environ.get("BURVIK_LEG_MAX_TICKS", "24000"))):
        """the left arm takes over the leg at base index k0 (grasped by the base rows). Approach: servo the leg's frame
        to `above` out on the op axis, axis aligned, then in at `speed` to thread_start. Threading: the lateral offset
        and the tilt servoed by DLS (no spin, no axial term), the axial target stepped by ax_step per tick to keep the
        leg's axial contact force with the host in [fmin, fmax] N, and the wrist j7 target advanced at `spin` rad/s
        (signed about the op axis). At the seat (gap < 0.25 mm) the spin stops (j7 target = j7), the press is held, until
        the weld; then open, retreat 40 mm along the axis and blend back to the base rows at k_rejoin."""
        o, d, m = self.o, self.o.d, self.o.m
        op = self.ops[key]; b = m.body(op["part"]).id; host = m.body(op["host"]).id
        pa = np.asarray(op["part_axis"], float); axh = np.asarray(op["axis"], float)
        lim = o.s.step_limit[o.arm["left"]] * 0.9
        j7a = o.arm["left"][6]; j7j = int(m.actuator_trnid[j7a, 0]); j7q = int(m.jnt_qposadr[j7j])
        grip_closed = self.base[k0][o.grip["left"]]
        g = above; ticks = 0; welded_at = None; mode = "approach"; seat_ticks = 0; turned = 0.; lead_ax = 0.; t_thread = 0
        ratchet = os.environ.get("BURVIK_RATCHET", "0") == "1" and key == os.environ.get("BURVIK_RATCHET_KEY", "s2-leg-handle-right")
        sub = "spin"; t_sub = 0; strokes = 0; table = None; q_back = None
        open_val = float(os.environ.get("BURVIK_RATCHET_OPEN", "182.142857")); look = np.radians(float(os.environ.get("BURVIK_RATCHET_LOOK", "8")))
        back_rate = float(os.environ.get("BURVIK_RATCHET_BACK", "2.0"))
        def is_free(q):
            q0, st, tb = table; kk = int(np.floor(((q - q0) % (2 * np.pi)) / np.radians(st) + 1e-9)) % len(tb)
            return tb[kk] and tb[(kk + 1) % len(tb)]
        mujoco.mj_kinematics(m, d)
        T0 = None
        while ticks < max_ticks:
            row = np.array(self.base[k0], dtype=np.float64)
            row[o.grip["left"]] = grip_closed
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            # the right arm holds the handle
            ph, Rh = self.handle_star(-0.0003)
            eph, ewh = self.hsv.errors(ph, Rh); he_ = float(np.linalg.norm(eph))
            vh = np.clip(self.hsv.kp * eph, -self.hsv.vmax, self.hsv.vmax); wh = np.clip(self.hsv.kr * ewh, -self.hsv.wmax, self.hsv.wmax)
            tcp_r = d.site_xpos[o.site["right"]]
            dqr = np.clip(o.dls("right", vh + np.cross(wh, tcp_r - d.xpos[self.hsv.b]), wh), -self.hsv.lim, self.hsv.lim)
            # the leg in the host frame
            Rhh = d.xmat[host].reshape(3, 3); ax_star = Rhh @ axh
            rel_h = Rhh.T @ (d.xpos[b] - d.xpos[host]) - np.asarray(op["target_pos"])
            gap_now = float(rel_h @ axh); lat_v = rel_h - gap_now * axh; lat_now = float(np.linalg.norm(lat_v))
            e_w = axis_servo_err(d, b, pa, ax_star)
            fax = self.axial_force(b, host, ax_star)
            if mode == "approach":
                p_star = d.xpos[host] + Rhh @ (np.asarray(op["target_pos"]) + g * axh)
                e_p = p_star - d.xpos[b]
                v = kp * e_p
                if (np.linalg.norm(e_p) < 0.001 and np.linalg.norm(e_w) < 0.01) or g < above:
                    g = max(thread_start - 0.0005, g - speed * o.s.control_dt)
                if gap_now < thread_start and lat_now < 0.001:
                    # the hand's pose in the host frame at the thread start: from here on the HAND is servoed (TCP on
                    # its line, the j7 axis on its direction), the leg turns with j7 and the TCP follows its advance
                    mode = "thread"; gap0 = gap_now; t_thread = ticks
                    a0 = Rhh.T @ (d.site_xpos[o.site["left"]] - d.xpos[host]); u0 = Rhh.T @ d.xaxis[j7j]
                spin_now = 0.
                w = np.clip(0.08 * e_w, -0.002, 0.002)
                nv = np.linalg.norm(v)
                if nv > vmax: v *= vmax / nv
                tcp = d.site_xpos[o.site["left"]]
                v = v + np.cross(w, tcp - d.xpos[b])
            if mode != "approach":
                press = float(os.environ.get("BURVIK_PRESS", "0.0002")) if mode == "thread" else 0.
                tcp = d.site_xpos[o.site["left"]]
                tcp_star = d.xpos[host] + Rhh @ a0
                e_t = tcp_star - tcp; e_t = e_t - float(e_t @ ax_star) * ax_star     # lateral: the actual TCP on the line
                # axial: the commanded target leads the actual TCP by `press` into the hole (bounded force, no windup)
                jp = np.zeros((3, m.nv)); mujoco.mj_jacSite(m, d, jp, None, o.site["left"])
                eq = np.asarray(o.rows[-1])[o.arm["left"]] - d.qpos[o.qadr["left"]]
                lead_ax = float((jp[:, o.vadr["left"]] @ eq) @ ax_star)
                v = float(os.environ.get("BURVIK_HANDKP", "0.1")) * e_t + 0.2 * (-press - lead_ax) * ax_star
                v = np.clip(v, -0.0004, 0.0004)
                w = np.clip(0.08 * axis_servo_err(d, m.jnt_bodyid[j7j], np.asarray(m.jnt_axis[j7j], float), Rhh @ u0), -0.002, 0.002)
                if mode == "thread":
                    spin_now = spin
                    if ratchet:
                        sgn_ = 1. if float(d.xaxis[j7j] @ ax_star) >= 0 else -1.; s7 = np.sign(sgn_ * spin)
                        q7 = float(d.qpos[j7q]); t_sub += 1
                        if table is None:
                            table = self.free_table("left", j7q)
                            self.note(phase="ratchet_table", free_deg=[k * table[1] for k, f in table[2].items() if f])
                        if sub == "spin" and not is_free(q7 + s7 * look):
                            sub = "open"; t_sub = 0; strokes += 1
                        if sub != "spin":
                            spin_now = 0.; press = 0.
                        if sub == "open" and t_sub >= int(os.environ.get("BURVIK_RATCHET_NOPEN", "120")):
                            # back (against the spin) to the start of this free window, a few degrees inside
                            qb = q7
                            for i_ in range(72):
                                if not is_free(qb - s7 * np.radians(5)): break
                                qb -= s7 * np.radians(5)
                            q_back = qb + s7 * np.radians(3); sub = "back"; t_sub = 0
                        elif sub == "back" and (q7 - q_back) * s7 <= 0.002:
                            sub = "close"; t_sub = 0
                        elif sub == "close" and t_sub >= int(os.environ.get("BURVIK_RATCHET_NCLOSE", "160")):
                            sub = "spin"; t_sub = 0; table = self.free_table("left", j7q)
                    if gap_now < float(os.environ.get("BURVIK_SEAT", "0.00025")) and (not ratchet or sub == "spin"):
                        mode = "seat"
                if mode == "seat":
                    spin_now = 0.; seat_ticks += 1
            if ratchet and mode == "thread" and sub != "spin":             # no press while the hand is open / turning back
                v = v - 0.2 * (-float(os.environ.get("BURVIK_PRESS", "0.0002"))) * ax_star
            dql = np.clip(o.dls("left", v, w), -lim, lim)
            prev = o.rows[-1]
            row[o.arm["right"]] = self.hsv.clamp_lead(prev[o.arm["right"]] + dqr)
            row[o.arm["left"]] = prev[o.arm["left"]] + dql
            if spin_now:
                sgn = 1. if float(d.xaxis[j7j] @ ax_star) >= 0 else -1.
                row[j7a] = prev[j7a] + dql[6] + sgn * spin_now * o.s.control_dt
                turned += spin_now * o.s.control_dt
            elif mode == "seat":
                row[j7a] = d.qpos[j7q]                                    # the spin's torque released
            elif ratchet and mode == "thread":
                if sub == "back":
                    row[j7a] = prev[j7a] + dql[6] - s7 * min(back_rate * o.s.control_dt, abs(float(d.qpos[j7q]) - q_back) + 1e-4)
                else:
                    row[j7a] = d.qpos[j7q] + dql[6]                      # hold the wrist where it is (no torque)
                if sub in ("open", "back"):
                    row[o.grip["left"]] = open_val
            o.step(row)
            ticks += 1
            if os.environ.get("BURVIK_DUMP_THREAD") and mode != "approach" and ticks - t_thread <= int(os.environ["BURVIK_DUMP_THREAD"]) and ticks % 5 == 0:
                hp_, hR_ = self.handle_star(-0.0003)
                self.note(phase="tdump", t=ticks - t_thread, gap=round(gap_now * 1000, 3), lat=round(lat_now * 1000, 3), he=round(1000 * float(np.linalg.norm(self.hsv.errors(hp_, hR_)[0])), 2),
                          q7=round(float(d.qpos[j7q]), 3), lead_ax=round(1e6 * lead_ax), v_um=np.round(1e6 * np.asarray(v), 1).tolist(),
                          ax=np.round(ax_star, 3).tolist(), e_tcp_mm=np.round(1000 * (tcp_star - d.site_xpos[o.site["left"]]), 2).tolist() if mode != "approach" else None,
                          j7_leg_deg=round(float(np.degrees(np.arccos(min(1., abs(float((d.xmat[b].reshape(3, 3) @ pa) @ d.xaxis[j7j])))))), 2),
                          contacts=self.leg_contacts(b)[:3], arm=self.arm_contacts("left"))
            if self.debug_contacts and ticks % 10 == 0 and (self.mech.metrics.get(key, {}).get("contact_N") or 0) > self.debug_contacts * 1000:
                self.note(phase="spike", gap=round(gap_now * 1000, 3), contacts=self.leg_contacts(b)[:10])
            if ticks % 200 == 0:
                # slip of the leg in the hand: the leg's pose in the TCP frame
                Rt_ = d.site_xmat[o.site["left"]].reshape(3, 3); pl = Rt_.T @ (d.xpos[b] - d.site_xpos[o.site["left"]])
                T0 = pl if T0 is None else T0
                mt = self.mech.metrics.get(key, {})
                self.note(phase="lf", mode=mode, sub=sub if ratchet else None, strokes=strokes, lgrasp=self.grasp("left", op["part"]), grasp=self.grasp(), hold_lead_um=None if self.hsv.last_lead is None else round(1e6 * self.hsv.last_lead), lead_um=round(1e6 * lead_ax, 1) if mode != "approach" else None, gap=round(gap_now * 1000, 3), lat=round(lat_now * 1000, 3),
                          tilt=round(np.degrees(np.linalg.norm(e_w)), 3), fax=round(fax, 2), turned=round(turned / (2 * np.pi), 2),
                          mturns=round(mt.get("turns", 0), 3), slip_mm=np.round(1000 * (pl - T0), 2).tolist(), he_mm=round(he_ * 1000, 2),
                          near=mt.get("near"), cN=mt.get("contact_N"), win=mt.get("contact_window_ok"))
            if self.mech.active(key):
                welded_at = len(o.rows); break
        if welded_at is None:
            self.note(phase=key, result="no weld"); self.snap(key + "_fail", k0); return None
        self.note(phase=key, result="welded", ticks=ticks)
        open_val = self.base[k_release_base][o.grip["left"]]
        for i in range(160):
            row = np.array(o.rows[-1]); row[o.grip["left"]] = open_val
            ph, Rh = self.handle_star(-0.0003); self.hsv.tick(row, ph, Rh)
        _, ax_star = self.leg_star(key, 0.)
        for i in range(400):
            row = np.array(o.rows[-1]); mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            dql = np.clip(o.dls("left", -ax_star * 0.0001, np.zeros(3)), -lim, lim)
            row[o.arm["left"]] += dql
            ph, Rh = self.handle_star(-0.0003); self.hsv.tick(row, ph, Rh)
        start = np.array(o.rows[-1][o.arm["left"]]); n = 1200
        tgt_end = np.array(self.base[k_rejoin][o.arm["left"]])
        # j7: blend to the base row's angle modulo whole turns (no unwinding)
        tgt_end_j7 = tgt_end[6] + 2 * np.pi * np.round((start[6] - tgt_end[6]) / (2 * np.pi))
        for i in range(n):
            tgt = self.base[k_rejoin - n + i]
            a = (i + 1) / n
            row = np.array(tgt, dtype=np.float64)
            t7 = tgt[o.arm["left"]].copy(); t7[6] = tgt_end_j7 + (tgt[o.arm["left"]][6] - tgt_end[6])
            row[o.arm["left"]] = (1 - a) * start + a * t7
            ph, Rh = self.handle_star(-0.0003); self.hsv.tick(row, ph, Rh)
        self.j7_off = tgt_end_j7 - tgt_end[6]
        self.snap(key, k_rejoin)
        return k_rejoin

    def leg_rel(self, key):
        """(gap, lateral) of the leg's frame against its op target in the host frame (m)."""
        op = self.ops[key]; m, d = self.o.m, self.o.d
        b = m.body(op["part"]).id; host = m.body(op["host"]).id; axh = np.asarray(op["axis"], float)
        Rhh = d.xmat[host].reshape(3, 3)
        rel = Rhh.T @ (d.xpos[b] - d.xpos[host]) - np.asarray(op["target_pos"])
        g = float(rel @ axh); return g, float(np.linalg.norm(rel - g * axh))

    def follow_until(self, key, k0, kmax, lat_max=float(os.environ.get("BURVIK_ENTRY_LAT", "0.12")),
                     gap_min=0.015, gap_max=float(os.environ.get("BURVIK_ENTRY_GAP", "0.06")), off_fn=None, k_check=0, rate_fn=None):
        """base rows (the right arm holding the handle) until the leg is in the pre-pose zone of its op (the base rows
        bring it there), then hand over to the leg controller; returns the base index."""
        o = self.o; m, d = o.m, o.d; j7 = o.arm["left"][6]; j7_off = getattr(self, "j7_off", 0.)
        k = k0; kf = float(k0); acc = np.zeros(7); prev = np.zeros(3)
        while k < kmax:
            # time dilation: rate_fn(k) base rows per tick (<1 slows the base rows down; rows interpolated)
            i0 = int(kf); fr = kf - i0
            row = (1 - fr) * np.asarray(self.base[i0], np.float64) + fr * np.asarray(self.base[min(i0 + 1, len(self.base) - 1)], np.float64)
            row[j7] += j7_off
            if off_fn is not None:                       # a world offset of the left TCP (integrated through DLS)
                z = np.asarray(off_fn(kf), float)
                if np.any(z != prev):
                    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                    acc += o.dls("left", z - prev, np.zeros(3)); prev = z
                row[o.arm["left"]] += acc
            ph, Rh = self.handle_star(-0.0003); self.hsv.tick(row, ph, Rh)
            kf = kf + (rate_fn(kf) if rate_fn is not None else 1.)
            if abs(kf - round(kf)) < 1e-9: kf = float(round(kf))
            if int(kf) == k: continue
            k = int(kf)
            if k % 20 == 0 and k >= k_check:
                g, l = self.leg_rel(key)
                if l < lat_max and gap_min < g < gap_max:
                    break
        g, l = self.leg_rel(key); tp, tR = self.o.body("top")
        self.note(phase="entry", key=key, k=k, gap_mm=round(g * 1000, 2), lat_mm=round(l * 1000, 2), top=np.round(tp * 1000, 1).tolist(),
                  top_z=np.round(tR[:, 2], 3).tolist())
        return k

    def follow(self, k0, k1, hold=True):
        """base rows k0..k1, the right arm holding the handle (servo) if hold; else the right arm's targets carried with
        their offset from the base row at k0 (the hand stays where the hold left it)."""
        o = self.o
        off = np.array(o.rows[-1][o.arm["right"]]) - self.base[k0][o.arm["right"]]
        j7 = o.arm["left"][6]; j7_off = getattr(self, "j7_off", 0.)
        for k in range(k0, k1):
            row = np.array(self.base[k], dtype=np.float64); row[j7] += j7_off
            if hold:
                ph, Rh = self.handle_star(-0.0003); self.hsv.tick(row, ph, Rh)
            else:
                row[o.arm["right"]] += off; o.step(row)
        return k1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code"); ap.add_argument("--scenes"); ap.add_argument("--base"); ap.add_argument("--out")
    ap.add_argument("--from", dest="start", default=None); ap.add_argument("--to", default="p6")
    a = ap.parse_args()
    B = Builder(a.code, a.scenes, a.base, a.out)
    order = ["p0", "p1", "p2", "p3", "leg2", "leg3", "p6"]
    LEG = B.leg_force if os.environ.get("BURVIK_LEGMODE", "force") == "force" else B.leg
    k = 0
    resume_l3 = a.start == "pre_leg3"                                    # resume at leg_3's entry (after the transit)
    if a.start:
        k = B.load(a.start)
        order = order[order.index("leg2" if resume_l3 else a.start) + 1:]
    for ph in order:
        if ph == "p0": B.p0(); k = 64000
        elif ph == "p1": B.p1(); k = 70000
        elif ph == "p2": B.p2(); k = 78500
        elif ph == "p3": k = B.p3()
        elif ph == "leg2":
            k = LEG("s2-leg-handle-left", k, 97027, 98500)
            if k is None: break
        elif ph == "leg3":
            # from here on the right arm's handle hold is lead-bounded: an unbounded hold winds up against the seated
            # handle (its target runs ahead of the blocked arm) and eventually pushes the top out of the rack
            hl = float(os.environ.get("BURVIK_HOLD_LEAD", "0.001")); B.hsv.lead_max = hl if hl > 0 else None
            # the base rows carry leg_3 (held horizontally at its far end) across the top's face to its hole. Two
            # changes, arm motion only: slowly (at the base speed the leg swings in the grasp), and with the TCP dx
            # further out from the face (the bolt end scrapes the leaning top and pushes it out of the rack); the
            # standoff is kept, the leg controller's approach closes it. Optional dz lower under leg_2.
            if not resume_l3:
                dz = float(os.environ.get("BURVIK_UNDER_DZ", "0.0")); dx = float(os.environ.get("BURVIK_OUT_DX", "0.05"))
                def under(kk, a=103850, b=104000, c=104800, e=105300):
                    fx = 0. if kk < a else min(1., (kk - a) / (b - a))
                    fz = 0. if kk < a else min(1., (kk - a) / (b - a)) if kk < c else max(0., 1 - (kk - c) / (e - c))
                    return np.array([dx * fx, 0., -dz * fz])
                slow = float(os.environ.get("BURVIK_TRANSIT_RATE", "0.25"))
                def rate(kk, a=103800, e=105400):
                    return slow if a <= kk < e else 1.
                k = B.follow_until("s2-leg-handle-right", k, 115000, off_fn=under, k_check=105300, rate_fn=rate,
                                   gap_max=float(os.environ.get("BURVIK_ENTRY_GAP", "0.09"))); B.snap("pre_leg3", k)
            hl3 = float(os.environ.get("BURVIK_HOLD_LEAD3", str(hl)))           # the bound while threading leg_3
            B.hsv.lead_max = hl3 if hl3 > 0 else None
            # the top leans ~6 deg in the rack: the legs then slope down away from it and the spinning left hand (at
            # leg_3's free end, low above the handle rack) sweeps into handle_rack_b. The right arm, holding the handle
            # (locked to the top by leg_2), turns the assembly upright about its stand support while leg_3 threads.
            # (turning it upright is blocked by the jig; lifting it out of the jig's V-guards is the assist used)
            tdeg = os.environ.get("BURVIK_TILT_DEG", "none"); ntilt = int(os.environ.get("BURVIK_TILT_ROWS", "1200"))
            lift = float(os.environ.get("BURVIK_LIFT", "0.04"))
            if tdeg != "none" or lift > 0:
                _, Rt_ = B.o.body("top"); tz = Rt_[:, 2]
                th = 0. if tdeg == "none" else -np.arctan2(tz[2], -tz[0]) if tdeg == "auto" else np.radians(float(tdeg))
                B.tilt_start(th, ntilt, lift=lift)
                for i in range(ntilt + 200):                                     # hold the left arm, turn the top
                    row = np.array(B.o.rows[-1]); ph, Rh = B.handle_star(-0.0003); B.hsv.tick(row, ph, Rh)
                tp, tR = B.o.body("top"); B.note(phase="tilted", grasp=B.grasp(), top=np.round(tp * 1000, 1).tolist(), top_z=np.round(tR[:, 2], 3).tolist(),
                                                 he_mm=round(1000 * float(np.linalg.norm(B.hsv.errors(*B.handle_star(-0.0003))[0])), 2))
            k = LEG("s2-leg-handle-right", k, 116360, 117800)
            if k is None: break
            if getattr(B, "tilt", None):                                         # the top back into the jig
                B.tilt_start(0., ntilt, lift=0.)
                for i in range(ntilt + 200):
                    row = np.array(B.o.rows[-1]); ph, Rh = B.handle_star(-0.0003); B.hsv.tick(row, ph, Rh)
                B.tilt_start(None, 0)
                tp, tR = B.o.body("top"); B.note(phase="untilted", top=np.round(tp * 1000, 1).tolist(), top_z=np.round(tR[:, 2], 3).tolist())
                B.snap("leg3", k)
        elif ph == "p6":
            B.follow(k, len(B.base), hold=False)
            fc = B.o.s.final_check()
            B.note(phase="end", success=fc["success"], report=fc["end"]["report"].get("failures"))
            B.snap("end", len(B.base))
            B.o.write(os.path.join(a.out, "ref.npz"))
        if ph == a.to:
            break
    json.dump(B.log, open(os.path.join(a.out, "log.json"), "w"), default=str, indent=1)


if __name__ == "__main__":
    main()
