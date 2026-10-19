"""Oracle of the Cast Cage Hedgehog (variation hedgehog_in_cage_cast_cage) on a frozen Session.

The star starts at rest inside the cage (the opening rest D25: E just out of W0, nothing graspable). In order:
  1. the rod: the hand over its handle (pointing down, its x axis along the rod), down, closed until the assisted weld
     holds, up;
  2. the push: to the stroke's start outside W90 (the rod along the stroke direction u, turned ROD["phi"] about it), the
     tip straight in by STROKE length, held still 1 s, straight back out. The star tips over onto the grasp rest T
     (U out of W0);
  3. the rod back to its place: over it, down, open, up;
  4. the star: held until still; it must lie at T (within REST_TOLERANCE of the planned T, U at least U_OUT_M out of
     the cage); U's pregrasp (the planned grasp in the star frame, on the measured star), the straight approach along
     the hand's z, closed until the weld holds;
  5. the held route: straight from where the star lies to the route state of HANDOVER_CHOICES whose move clears the cage
     most, then the route to its end, every pose through IK with the hand-in-star pose the weld actually took (as
     window_change), slowly;
  6. the carry: out along the exit direction, over the goal centre, down; held GOAL_HOLD_STEPS.
Free motions (home -> rod, rod -> stroke, stroke -> rod place, rod place -> U pregrasp) are HedgehogIK.plan_free_tcp
joint paths: the cage, mount and table are obstacles, the star at its measured pose too, the rod lying on the table
before it is taken and after it is put back, and while it is held it counts as part of the hand."""
import json
from pathlib import Path

import numpy as np
import fcl
from scipy.spatial.transform import Rotation, Slerp

ROUTE = Path(__file__).with_name("cast_cage_route.json")
GRASP = dict(psi=60., sign=-1, ell=.074)        # U grasp (placement study cc_ik_route2, yaw 120 / (.05, .05) / mount .24):
                                                # approach turned psi about U, x along sign*U, TCP ell from the centre
ROD = dict(axis="+x", phi=240., tip_m=.23)      # the rod along hand +x (across the pads), turned phi about the stroke
                                                # direction, its tip tip_m from the grip
STROKE = dict(entry_mm=(1.48, 19.694, -10.847), u=(.25634, -.87308, .41476), length_mm=27.338, pre_mm=15., speed_m_s=.04,
              dwell_steps=40)                 # cage frame, real mm (x scale); search_d25 node 1 without its sweep
HANDOVER_STEPS = 24
FREE_STEP_RAD = .02
HELD_STEP_RAD = .006
SETTLE_STEPS = 100
GOAL_HOLD_STEPS = 120
REST_TOLERANCE = (.008, np.radians(10.))   # the grasp rest T as the push leaves it: within this of the planned T
U_OUT_M = .02                              # and spike U at least this far past the cage's outer wall
HANDOVER_CHOICES = np.arange(54., 73., 2.)  # route states the star can be moved to straight from T (clear of the cage)
ARM_TOLERANCE_RAD = .001
PRE_M, LIFT_M, EXIT_M, LOW_M = .06, .12, .08, .02
GRIP_ABOVE_M = .0135     # the TCP over the handle's centre: closed, the Robotiq fingertips reach ~19 mm past the TCP, so
                         # they end ~3 mm above the table while the pads still cover 13 mm of the handle's 16 mm sides


def _pose(entity):
    p = np.asarray(entity.get_pos(), float).reshape(3)
    q = np.asarray(entity.get_quat(), float).reshape(4)
    return p, Rotation.from_quat(q[[1, 2, 3, 0]])


class CastCageOracle:
    def __init__(self, session, log=print):
        from topogym.planning.hedgehog_in_cage.kinematics import HedgehogIK
        self.s, self.log = session, log
        robot = session.env._robot
        self.entity, self.group = robot.entity, robot.groups["arm"]
        world = session.env._world
        self.world, self.ball, self.rod = world, world.entities["hedgehog_ball"], world.entities["push_rod"]
        self.cage = _pose(world.entities["hedgehog_cage"])
        self.kin = HedgehogIK(session.env)
        self.width = lambda native: session.robot.native_to_width("arm", native)
        self.w_closed = self.width(.8)
        self.q_cmd = np.asarray(session.joint_positions(), float)[:7]
        self.w_cmd = float(np.asarray(session.joint_positions(), float)[7])
        self.route = json.loads(ROUTE.read_text())
        self.scale = float(self.route["scale"])
        self.report = dict(stages=[])

    # -- low level (as window_change)
    def step_to(self, q, width, max_step):
        q0, w0 = self.q_cmd.copy(), self.w_cmd
        n = max(1, int(np.ceil(np.max(np.abs(np.asarray(q) - q0)) / max_step)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            self.s.step(np.r_[q0 + t * (np.asarray(q) - q0), w0 + t * (width - w0)])
        self.q_cmd, self.w_cmd = np.asarray(q, float).copy(), float(width)

    def hold(self, steps):
        for _ in range(steps):
            self.s.step(np.r_[self.q_cmd, self.w_cmd])

    def held(self):
        return bool(self.world.read_state().get("assist_active"))

    def settle(self, max_steps=80):
        for _ in range(max_steps):
            if np.max(np.abs(np.asarray(self.s.joint_positions(), float)[:7] - self.q_cmd)) < ARM_TOLERANCE_RAD:
                return True
            self.s.step(np.r_[self.q_cmd, self.w_cmd])
        return False

    def wait_width(self, max_steps=200):
        last = [float(self.s.gripper_width("arm"))]
        for _ in range(max_steps):
            self.s.step(np.r_[self.q_cmd, self.w_cmd])
            last.append(float(self.s.gripper_width("arm")))
            if len(last) > 4 and max(last[-5:]) - min(last[-5:]) < 5e-5:
                return True
        return False

    def close(self, width_from, max_steps=200):
        for k in range(max_steps):
            self.w_cmd = width_from + (self.w_closed - width_from) * min(1., (k + 1) / 10)
            self.s.step(np.r_[self.q_cmd, self.w_cmd])
            if self.held():
                self.hold(8)
                return True
        return self.held()

    def seed(self):
        q = self.kin.seed.copy()
        q[self.kin.arm_qindices] = self.q_cmd
        return q

    def solve(self, tcp):
        """IK of a TCP pose (position, Rotation) from the current command."""
        return np.asarray(self.kin.solve(tcp, self.seed()), float)[self.kin.arm_qindices]

    def line(self, a, b, step_m, width, max_step=FREE_STEP_RAD):
        """The TCP straight from pose a to pose b (positions linear, rotations slerped), IK per point."""
        n = max(1, int(np.ceil(np.linalg.norm(b[0] - a[0]) / step_m)))
        sl = Slerp([0, 1], Rotation.concatenate([a[1], b[1]]))
        for t in np.linspace(0, 1, n + 1)[1:]:
            self.step_to(self.solve((a[0] + t * (b[0] - a[0]), sl(t))), width, max_step)

    def free(self, tcp, width, label):
        rows = self.kin.plan_free_tcp(tcp, seed=self.seed(), random_seed=0, iterations=8000, minimum_clearance=.004)
        for r in rows:
            self.step_to(np.asarray(r["qpos"])[self.kin.arm_qindices], width, FREE_STEP_RAD)
        self.log(dict(free=label, rows=len(rows)))

    def tcp_now(self):
        q = self.kin.seed.copy()
        q[self.kin.arm_qindices] = np.asarray(self.s.joint_positions(), float)[:7]
        return self.kin.tcp_pose(q)

    # -- the rod and the star as planning obstacles / as part of the hand
    def _objects(self, entity):
        """FCL hulls of an entity's geoms in its own frame, and its pose."""
        from topogym.planning.buzz_wire_3d.kinematics import mesh_object
        p, r = _pose(entity)
        objs = [mesh_object(r.inv().apply(np.asarray(g.get_verts(), float).reshape(-1, 3) - p), g.init_faces)
                for link in entity.links for g in link.geoms]
        return objs, (p, r)

    def rod_obstacle(self, on):
        """The rod lying on the table as an obstacle of the free motions (on) or not."""
        from topogym.planning.buzz_wire_3d.kinematics import manager
        fixed = [o for o in self.kin.fixed.getObjects() if o not in getattr(self, "_rod_fixed", [])]
        self._rod_fixed = []
        if on:
            objs, (p, r) = self._objects(self.rod)
            for o in objs:
                o.setTransform(fcl.Transform(r.as_matrix(), p))
            self._rod_fixed = objs
        self.kin.fixed = manager(fixed + self._rod_fixed)

    def rod_in_hand(self, on):
        """While held, the rod's hulls ride on the hand link (its pose in the hand frame as the weld took it)."""
        from topogym.planning.buzz_wire_3d.kinematics import manager
        kin = self.kin
        hand = self.entity.get_link(self.group.hand_link).idx_local
        kin.robot_shapes = [(i, o) for i, o in kin.robot_shapes if o not in getattr(self, "_rod_held", [])]
        self._rod_held = []
        if on:
            from topogym.planning.buzz_wire_3d.kinematics import mesh_object
            hp, hr = _pose(self.entity.get_link(self.group.hand_link))
            objs = []
            for link in self.rod.links:
                for g in link.geoms:
                    v = np.asarray(g.get_verts(), float).reshape(-1, 3)
                    objs.append(mesh_object(hr.inv().apply(v - hp), g.init_faces))
            self._rod_held = objs
            kin.robot_shapes = kin.robot_shapes + [(hand, o) for o in objs]
        base = next(i for i, n in kin.names.items() if n == "link0")
        fingers = set(self.group.grasp.finger_links)
        kin.robot_manager = manager([o for i, o in kin.robot_shapes if i != base])
        kin.nonfinger_manager = manager([o for i, o in kin.robot_shapes if kin.names[i] not in fingers])

    def star_obstacle(self):
        """The star where it lies now (HedgehogIK checks the robot against it at kin.ball_pose)."""
        self.kin.ball_pose = _pose(self.ball)

    # -- frames
    def star_in_cage(self):
        p, r = _pose(self.ball)
        return self.cage[1].inv().apply(p - self.cage[0]), self.cage[1].inv() * r

    def cage_to_world(self, p, r):
        return self.cage[0] + self.cage[1].apply(p), self.cage[1] * r

    def grasp_in_star(self):
        """Hand TCP pose in the star frame: x along sign*U, approach z perpendicular to U turned by psi, TCP on U's axis."""
        du = np.asarray(self.route["u_dir_star"], float)
        a = np.cross(du, [0., 0., 1.]) if abs(du[2]) < .9 else np.cross(du, [1., 0., 0.])
        a /= np.linalg.norm(a)
        b = np.cross(du, a)
        psi = np.radians(GRASP["psi"])
        z = np.cos(psi) * a + np.sin(psi) * b
        x = GRASP["sign"] * du
        return du * GRASP["ell"], Rotation.from_matrix(np.c_[x, np.cross(z, x), z])

    def hand_in_star(self):
        """The TCP pose in the star frame as the weld actually took it."""
        tp, tr = self.tcp_now()
        bp, br = _pose(self.ball)
        return br.inv().apply(tp - bp), br.inv() * tr

    def rod_in_tcp(self):
        """The rod's pose in the TCP frame as the weld took it (its origin at the grip point, +x toward the tip)."""
        tp, tr = self.tcp_now()
        p, r = _pose(self.rod)
        return tr.inv().apply(p - tp), tr.inv() * r

    def tcp_for_rod(self, p, r):
        """The TCP pose that puts the held rod's body at (p, r)."""
        rp, rr = self.rod_tcp
        tr = r * rr.inv()
        return p - tr.apply(rp), tr

    def rod_grip(self):
        """The rod's grip point (the handle's centre: the rod body's origin) and its axis toward the tip (the body's +x)."""
        p, r = _pose(self.rod)
        return p, r.apply([1., 0., 0.])

    # -- stages
    def pick_rod(self, entry):
        """Stage 1; False if the weld does not take the rod."""
        p, d = self.rod_grip()
        a = 1. if ROD["axis"] == "+x" else -1.
        x = a * np.r_[d[:2], 0.] / np.linalg.norm(d[:2])     # the rod lies tipped 0.9 deg: the hand stays level
        z = np.array([0., 0., -1.])
        r = Rotation.from_matrix(np.c_[x, np.cross(z, x), z])
        grip = (p + [0., 0., GRIP_ABOVE_M], r)
        pre = (grip[0] + [0., 0., LIFT_M], r)
        w_open = self.width(0.)
        self.rod_obstacle(True)
        self.star_obstacle()
        self.free(pre, w_open, "to the rod")
        self.wait_width()
        self.rod_obstacle(False)
        self.line(pre, grip, .004, w_open)
        self.settle()
        if not self.close(w_open):
            tp, _ = self.tcp_now()
            entry.update(result="rod weld not engaged", tcp_minus_grip_mm=np.round((tp - grip[0]) * 1e3, 2).tolist(),
                         width_mm=round(float(self.s.gripper_width("arm")) * 1e3, 2))
            return False
        self.rod_in_hand(True)
        self.rod_tcp = self.rod_in_tcp()
        self.rod_place = _pose(self.rod)
        self.line(grip, pre, .004, self.w_closed)
        return True

    def stroke(self, entry):
        u = self.cage[1].apply(np.asarray(STROKE["u"], float))
        e = self.cage[0] + self.cage[1].apply(np.asarray(STROKE["entry_mm"], float) * self.scale / 1000.)
        rp, rr = self.rod_tcp
        a = rr.apply([1., 0., 0.])                          # the rod's axis in the TCP frame (about +x)
        tip = rp + rr.apply([ROD["tip_m"], 0., 0.])          # its tip in the TCP frame
        r = Rotation.from_rotvec(np.radians(ROD["phi"]) * u) * Rotation.align_vectors([u], [a])[0]
        tip0 = e - STROKE["pre_mm"] * self.scale / 1000. * u
        tip1 = e + STROKE["length_mm"] * self.scale / 1000. * u
        h0, h1 = (tip0 - r.apply(tip), r), (tip1 - r.apply(tip), r)
        self.free(h0, self.w_closed, "to the stroke")
        self.settle()
        step = STROKE["speed_m_s"] * self.s.control_dt if hasattr(self.s, "control_dt") else STROKE["speed_m_s"] * .025
        self.line(h0, h1, step, self.w_closed, max_step=1.)
        self.hold(STROKE["dwell_steps"])
        self.line(h1, h0, step, self.w_closed, max_step=1.)
        entry["stroke"] = dict(entry_world=e.tolist(), u_world=u.tolist())

    def put_rod_back(self, entry):
        p, r = self.rod_place
        place = self.tcp_for_rod(p + [0., 0., .001], r)       # where it was taken, 1 mm up
        pre = (place[0] + [0., 0., LIFT_M], place[1])
        self.free(pre, self.w_closed, "back over the rod's place")
        self.line(pre, place, .004, self.w_closed)
        self.settle()
        w_open = self.width(0.)
        self.step_to(self.q_cmd, w_open, FREE_STEP_RAD)
        self.wait_width()
        self.rod_in_hand(False)
        self.line(place, pre, .004, w_open)
        self.hold(20)
        self.rod_obstacle(True)

    def grasp_u(self, entry):
        self.hold(SETTLE_STEPS)
        p, r = self.star_in_cage()
        T = self.route["T"]
        dp = float(np.linalg.norm(p - np.asarray(T[:3])))
        da = float((r * Rotation.from_quat(T[3:]).inv()).magnitude())
        du = np.asarray(self.route["u_dir_star"], float)
        tip = p + r.apply(du * self.route["u_length_m"])
        u_out = float(np.hypot(*tip[:2]) - self.route["outer_radius_m"])
        entry["T_offset"] = dict(position_mm=round(dp * 1e3, 2), rotation_deg=round(np.degrees(da), 2),
                                 u_out_mm=round(u_out * 1e3, 2))
        if dp > REST_TOLERANCE[0] or da > REST_TOLERANCE[1] or u_out < U_OUT_M:
            entry["result"] = "star not at the grasp rest"
            return False
        self.star_obstacle()
        gp, gr = self.grasp_in_star()
        bp, br = _pose(self.ball)
        tcp = (bp + br.apply(gp), br * gr)
        pre = (tcp[0] - tcp[1].apply([0., 0., PRE_M]), tcp[1])
        w_open = self.width(.3)
        self.free(pre, w_open, "to U's pregrasp")
        self.wait_width()
        self.line(pre, tcp, .003, w_open)
        self.settle()
        if not self.close(w_open):
            entry["result"] = "U weld not engaged"
            return False
        return True

    def handover(self, p0, r0):
        """The route state (of HANDOVER_CHOICES) the star moves to straight from where it lies, the one whose straight
        move stays farthest from the cage (the star's hulls against the cage's, every 1/HANDOVER_STEPS)."""
        from topogym.planning.buzz_wire_3d.kinematics import distance, manager
        cage_objs, (cp, cr) = self._objects(self.world.entities["hedgehog_cage"])
        for o in cage_objs:
            o.setTransform(fcl.Transform(np.eye(3), np.zeros(3)))        # the cage frame
        cage = manager(cage_objs)
        star = manager(self.kin.ball_objects)
        cum = np.asarray(self.route["route_deg"])
        best = None
        for deg in HANDOVER_CHOICES:
            k = int(np.searchsorted(cum, deg))
            pk, rk = np.asarray(self.route["route"][k][:3]), Rotation.from_quat(self.route["route"][k][3:])
            sl = Slerp([0, 1], Rotation.concatenate([r0, rk]))
            poses = [(p0 + t * (pk - p0), sl(t)) for t in np.linspace(0, 1, HANDOVER_STEPS + 1)[1:]]
            gap = np.inf
            for pc, rc in poses:
                for o in self.kin.ball_objects:
                    o.setTransform(fcl.Transform(rc.as_matrix(), pc))
                star.update()
                gap = min(gap, distance(star, cage))
            if best is None or gap > best[0]:
                best = (gap, k, poses)
        return best

    def held_route(self, entry):
        """T (as measured) straight to the route state handover() picks, then the route to its end, through the
        measured hand-in-star pose."""
        hp, hr = self.hand_in_star()
        p0, r0 = self.star_in_cage()
        gap, k0, poses = self.handover(p0, r0)
        entry["handover"] = dict(route_deg=round(float(self.route["route_deg"][k0]), 1), clearance_mm=round(gap * 1e3, 2))
        poses += [(np.asarray(s[:3]), Rotation.from_quat(s[3:])) for s in self.route["route"][k0 + 1::2]]
        worst = 0.
        for pc, rc in poses:
            bp, br = self.cage_to_world(pc, rc)
            self.step_to(self.solve((bp + br.apply(hp), br * hr)), self.w_closed, HELD_STEP_RAD)
            p, _ = self.star_in_cage()
            worst = max(worst, float(np.linalg.norm(p - pc)))
        entry["route_tracking_max_mm"] = round(worst * 1e3, 2)
        return hp, hr

    def carry(self, hib, entry):
        """Out along the exit direction, over the goal centre, down until the star's lowest point is LOW_M above the
        table; the star's orientation kept."""
        hp, hr = hib
        task = self.s.env._task
        region = task.goal_region
        bp, br = _pose(self.ball)
        out = bp[:2] - self.cage[0][:2]
        out = np.r_[out / np.linalg.norm(out), 0.]
        low = min(float(br.apply(np.asarray(g.get_verts(), float).reshape(-1, 3) - bp)[:, 2].min())
                  for link in self.ball.links for g in link.geoms)
        goal = np.r_[region.center_xy_m, region.table_height_m + LOW_M - low]
        for a, b in ((bp, bp + EXIT_M * out), (bp + EXIT_M * out, goal + [0, 0, .06]), (goal + [0, 0, .06], goal)):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / .004)))
            for t in np.linspace(0, 1, n + 1)[1:]:
                c = a + t * (b - a)
                self.step_to(self.solve((c + br.apply(hp), br * hr)), self.w_closed, HELD_STEP_RAD * 2)
        self.settle()
        entry["goal"] = goal.tolist()

    def run(self):
        stages = [dict(name=n) for n in ("rod", "push", "rod back", "grasp U", "route", "carry")]
        self.report["stages"] = stages
        ok = self.pick_rod(stages[0])
        self.log(stages[0])
        if not ok:
            return self.report
        self.stroke(stages[1]); self.log(stages[1])
        self.put_rod_back(stages[2]); self.log(stages[2])
        if not self.grasp_u(stages[3]):
            self.log(stages[3])
            return self.report
        self.log(stages[3])
        hib = self.held_route(stages[4]); self.log(stages[4])
        self.carry(hib, stages[5])
        self.hold(GOAL_HOLD_STEPS)
        stages[5]["status"] = {k: v for k, v in self.s.status().items() if k in ("success", "episode_end", "sim_steps")}
        self.log(stages[5])
        self.report["success"] = bool(self.s.status().get("success"))
        return self.report


def run_oracle(session, log=print):
    """Entry point for run-oracle code and the Harbor bridge."""
    return CastCageOracle(session, log=log).run()
