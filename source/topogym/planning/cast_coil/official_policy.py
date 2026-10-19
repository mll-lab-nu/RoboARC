"""v2 very_easy oracle (the official 8-step Cast Coil, MuJoCo): the explicitly granted world observation in, JOINT_ABS out.

The plan is planning/motions/cast_coil_official.py from the measured resting pieces: pinch both grasp cells, lift
8 cm, carry both pieces to the assembly start, run the eight steps with each hand taking its share of every motion,
lower the assembled block onto the table, open and back off. A design record may ask for a joint-space carry
("carry": "joint": the arms move from the lift straight to the first solution of the rest of their plans, the hand
poses in between free, through a raised via configuration wherever a piece would come near the table) and for the hands' midpoint held still during the assembly ("anchor": "hands") or for the
pair's placement found by both arms together ("anchor": "arms", official_arms.OracleArms.placement). Before moving,
each arm's joint path is planned (planning/cast_coil/official_arms.py) and the arm is driven onto it from home with a
joint-space move. Holding pieces, it follows piece poses with a measured-pose compensation and a path clock (the path
pauses while tracking is outside its budget); during the assembly silver's target is the measured gold pose composed
with the planned relative pose, so the clearance-critical relative pose follows the route even while gold lags. Every
commanded hand pose becomes joint targets through the planner's IK started from the previous solution, and each arm
joint target moves at most official_arms.STEP_LIMIT_RAD per control step: the standard service's joint speed limit,
so the reference replays unchanged there. Contact-triggered grasp assistance belongs to the simulator, not the policy.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from topogym.assets.variation_resources import read_json
from topogym.planning.motions import cast_coil_official as M
from .official_arms import STEP_LIMIT_RAD

ROUTES = Path(__file__).parent / "routes"


def solution_record(geometry_resource):
    """A v2 tier's whole solution record: its packaged pieces (topogym/assets/resources/<geometry_resource>) plus the
    route and execution settings kept beside this module (routes/<the same file name>), which the image does not ship.
    The route file names the sha256 of the pieces it was recorded on."""
    from topogym.assets.library import RESOURCE_ROOT
    route = json.loads((ROUTES / Path(geometry_resource).name).read_text())
    catalog = json.loads((RESOURCE_ROOT / "catalog.json").read_text())["resources"][geometry_resource]
    if route["geometry"] != geometry_resource or route["geometry_sha256"] != catalog["sha256"]:
        raise ValueError(f"{geometry_resource} is not the geometry its route was recorded on")
    return dict(read_json(geometry_resource), **{k: v for k, v in route.items()
                                                  if k not in ("schema_version", "geometry", "geometry_sha256",
                                                               "description", "source")})

PARTS = {"left": "coil_silver", "right": "coil_gold"}
PIECES = {"left": "silver", "right": "gold"}
LIFT_M = .08
ASSEMBLY_CLEARANCE_M, RAISE_MARGIN_M = .03, .03     # anchor "hands" or "arms" / carry "joint" plans only
# carry "joint": where a held piece would come within CARRY_CLEARANCE_M of the table, the joint move goes through a
# raised via configuration, the halves checked again down to CARRY_VIA_DEPTH levels. A 6 x 8 x 6 lattice design's gold
# piece swung 84 mm into the table on the plain move (forearm roll 2.4 rad, wrist angle 1.6 rad, 2026-10-11)
CARRY_CLEARANCE_M, CARRY_VIA_DEPTH = .03, 3
OPEN, CLOSED = .037, .002
STEP_SECONDS = dict(slide_per_cell=3., long_slide=8., turn=6.)
# Steps whose smallest gap is below TIGHT_GAP_M run TIGHT_SLOWDOWN times slower: the hands trail a moving target in
# proportion to its speed, and the wrist roll that turns a piece about the window axis is the softest joint (kp 10.4).
TIGHT_GAP_M, TIGHT_SLOWDOWN = .002, 3.
# The ALOHA position servos are soft (shoulder kp 265 N m/rad against 20 N m s/rad joint damping, about 75 ms of lag)
# and the shoulder and elbow have static friction, so a moving hand trails its target by millimetres. The command
# leads the path by LEAD_S; the measured-pose compensation integrates against the current, not the leading, target.
LEAD_S = .08
# Relative-pose error allowed while a step advances: half of the step's smallest gap between the pieces (the real
# collision geometry along the planned route), within [0.35, 1.5] mm. 2026-10-09: the 6-cell slide and the first
# turn keep >= 4 mm, the next turn 3.4 mm, the last five steps 1.5 mm (the design clearance). The error is the
# largest material displacement (position plus radius x angle), which overstates how much gap is used: a quarter of
# the gap stalled the tight steps with no contact between the pieces at all.
RELATIVE_BUDGET_SHARE, RELATIVE_BUDGET_RANGE_M = .5, (.00035, .0015)


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]])


def _pieces_mjcf(record):
    """Both pieces as free bodies of boxes and pins (the URDFs' collision primitives), for measuring their gaps."""
    from topogym.assets.geometry.cast_coil_official import piece_primitives

    def numbers(values):
        return " ".join(f"{float(v):.17g}" for v in values)
    bodies = []
    for name, piece in (("coil_gold", "gold"), ("coil_silver", "silver")):
        geoms = []
        for shape in piece_primitives(record, piece):
            if shape[0] == "box":
                geoms.append(f'<geom type="box" pos="{numbers(shape[1])}" size="{numbers(shape[2])}"/>')
            else:
                geoms.append(f'<geom type="cylinder" fromto="{numbers(np.r_[shape[1], shape[2]])}" '
                             f'size="{shape[3]:.17g}"/>')
        bodies.append(f'<body name="{name}"><freejoint/>{"".join(geoms)}</body>')
    return f'<mujoco><worldbody>{"".join(bodies)}</worldbody></mujoco>'


def route_clearance(record, *, degrees_per_sample=1., metres_per_sample=.001, reach_m=.004):
    """{route step index: smallest gap between the pieces (m, at most reach_m)} along the planned relative route,
    with the pieces' collision primitives placed by qpos."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(_pieces_mjcf(record))
    model.geom_margin[:] = reach_m/2
    data = mujoco.MjData(model)
    gold = model.jnt_qposadr[model.body("coil_gold").jntadr[0]]
    silver = model.jnt_qposadr[model.body("coil_silver").jntadr[0]]
    data.qpos[gold:gold+7] = [0., 0., 0., 1., 0., 0., 0.]
    gaps = {}
    for index, _, relative in M.disassembly(record, degrees_per_sample=degrees_per_sample,
                                            metres_per_sample=metres_per_sample):
        data.qpos[silver:silver+3] = relative[0]
        data.qpos[silver+3:silver+7] = relative[1].as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(model, data)
        gap = min((float(c.dist) for c in data.contact[:data.ncon]), default=reach_m)
        gaps[index] = min(gaps.get(index, reach_m), gap)
    return gaps


def along(poses, u, centre=(0., 0., 0.), radius=0.):
    """Pose at fraction u of a sampled path's length: the travel of the point `centre` (body frame) plus `radius` times
    the turn angle, so a path whose samples are unevenly spaced (the carry: turns bunched between translations) is
    followed at an even speed; a path of equal samples by its sample index."""
    if len(poses) == 1:
        return poses[0]
    centre = np.asarray(centre, float)
    lengths = np.array([np.linalg.norm(a[0]+a[1].apply(centre)-b[0]-b[1].apply(centre)) + radius*(a[1]*b[1].inv()).magnitude()
                        for a, b in zip(poses[:-1], poses[1:])])
    if lengths.sum() <= 0:
        lengths = np.ones(len(poses)-1)
    at = float(np.clip(u, 0., 1.))*lengths.sum()
    ends = np.cumsum(lengths)
    i = min(int(np.searchsorted(ends, at)), len(lengths)-1)
    start = ends[i]-lengths[i]
    return M.interpolate(poses[i], poses[i+1], float(np.clip((at-start)/lengths[i], 0., 1.)) if lengths[i] > 0 else 1.)


class OfficialCoilPolicy:
    def __init__(self, *, seed=0, task_parameters=None, record=None):
        """record: a solution record of the official schema (cells, pins, grasp, route); default the v2 very_easy
        tier's."""
        from topogym.assets.geometry.cast_coil_official import piece_points, piece_primitives
        from topogym.tasks.cast_coil.cast_coil_v2_very_easy import CastCoilV2VeryEasyTask
        self.phase = "created"
        self.record = solution_record(CastCoilV2VeryEasyTask.geometry_resource) if record is None else record
        cell = float(self.record["cell_m"])
        self.centre = M.block_centre_cells(self.record)*cell
        self.radius = max(float(np.linalg.norm(piece_points(piece_primitives(self.record, p))-self.centre, axis=1).max())
                          for p in PIECES.values())
        self.centroids = {side: (np.asarray(self.record["pieces"][p]["cells"], float)+.5).mean(0)*cell
                          for side, p in PIECES.items()}
        self.tcp_in_piece = {side: M.grasp_tcp_in_piece(self.record, p) for side, p in PIECES.items()}
        self.samples, self.tracking = [], {}
        gaps = route_clearance(self.record)
        low, high = RELATIVE_BUDGET_RANGE_M
        count = len(self.record["route"])
        # assembly step k reverses route step count-k; its budget is set by the smaller gap of the two route samples
        self.relative_budget = {count-index: float(np.clip(RELATIVE_BUDGET_SHARE*gap, low, high))
                                for index, gap in gaps.items() if index >= 0}
        self.slowdown = {count-index: TIGHT_SLOWDOWN if gap < TIGHT_GAP_M else 1.
                         for index, gap in gaps.items() if index >= 0}

    def initialize(self, context):
        if context["action_spec"]["robot_id"] != "aloha" or context["action_spec"]["mode"] != "JOINT_ABS":
            raise ValueError("The official Coil oracle requires ALOHA JOINT_ABS")
        self.dt = context["timing_spec"]["control_dt_s"]
        self.grip, self.program = OPEN, None
        self.bias = {s: np.zeros(3) for s in PARTS}
        self.angle_bias = {s: np.zeros(3) for s in PARTS}
        self.arms = self.solution = self.command = None
        self.limited_steps = 0          # control steps on which the joint speed limit held a target back

    def act(self, obs):
        """The program's next action as joint targets: a hand pose through the planner's IK started from the arm's
        previous solution; each arm joint target moves at most STEP_LIMIT_RAD per control step towards the solution."""
        self.obs = obs
        if self.program is None:
            self.program = self.actions()
        action = next(self.program, None)
        if action is None:
            return None
        if self.command is None:
            self.command = {s: np.asarray(obs["robot"][s]["joint_position_target_rad"], float) for s in PARTS}
            self.solution = dict(self.command)
        out, limited = {}, False
        for side in PARTS:
            command = action.get(side, {})
            if "joint_position_rad" in command:
                self.solution[side] = np.asarray(command["joint_position_rad"], float)
            elif "position_world_m" in command:
                pose = (np.asarray(command["position_world_m"], float), rotation(command["orientation_wxyz"]))
                self.solution[side] = self.arms.solve(side, pose, self.solution)
            change = self.solution[side]-self.command[side]
            step = np.clip(change, -STEP_LIMIT_RAD, STEP_LIMIT_RAD)
            limited |= bool(np.any(step != change))
            self.command[side] = self.command[side]+step
            out[side] = dict(joint_position_rad=self.command[side].tolist())
            if "gripper_position_m" in command:
                out[side]["gripper_position_m"] = command["gripper_position_m"]
        self.limited_steps += limited
        return out

    def time(self):
        return self.obs["state"]["sim_time_s"]

    def bodies(self):
        """{side: (position, Rotation)} of the measured pieces (their body frames are the piece cell frames)."""
        out = {}
        for side, name in PARTS.items():
            body = self.obs["privileged"]["world"]["bodies"][name]
            out[side] = (np.asarray(body["position_m"], float), rotation(body["quaternion_wxyz"]))
        return out

    def tcp(self):
        return {s: (np.asarray(v["tcp_position_world_m"]), rotation(v["tcp_orientation_wxyz"]))
                for s, v in self.obs["robot"].items()}

    def hold(self, seconds):
        start = self.time()
        while self.time()-start < seconds-1e-8:
            yield {}

    def follow(self, pose_at, seconds, phase):
        """Hand poses pose_at(u) on a smoothstep clock of `seconds`, then a quarter-second hold."""
        self.phase = phase
        start = self.time()
        while True:
            t = min(1., (self.time()-start+self.dt)/seconds)
            u = float(np.clip(t**3*(10+t*(-15+6*t)), 0., 1.))
            targets = pose_at(u)
            yield {s: dict(position_world_m=p.tolist(), orientation_wxyz=r.as_quat()[[3, 0, 1, 2]].tolist(),
                           gripper_position_m=self.grip) for s, (p, r) in targets.items()}
            if t >= 1.-1e-8:
                break
        yield from self.hold(.25)

    def move(self, targets, seconds, phase):
        """Straight hand moves from the measured TCPs to `targets` = {side: (position, Rotation)}."""
        initial = self.tcp()
        turns = {s: Slerp([0, 1], Rotation.from_quat([initial[s][1].as_quat(), r.as_quat()]))
                 for s, (_, r) in targets.items()}
        yield from self.follow(lambda u: {s: (initial[s][0]+u*(p-initial[s][0]), turns[s](u))
                                         for s, (p, _) in targets.items()}, seconds, phase)

    def attached_targets(self, desired, lead=None):
        """Hand targets that carry the pieces to `desired`, applied to the `lead` pose: the grasp offsets plus an
        integral and a proportional correction of the measured piece pose error (servo deflection under load)."""
        lead = desired if lead is None else lead
        actual = self.bodies()
        targets = {}
        for side, (position, turn) in desired.items():
            error = position-actual[side][0]
            angle = (turn*actual[side][1].inv()).as_rotvec()
            self.bias[side] = np.clip(self.bias[side]+4*self.dt*error, -.10, .10)
            self.angle_bias[side] = np.clip(self.angle_bias[side]+4*self.dt*angle, -.3, .3)
            commanded = Rotation.from_rotvec(self.angle_bias[side]+.5*angle)*lead[side][1]
            targets[side] = (lead[side][0]+self.bias[side]+.5*error+commanded.apply(self.offsets[side][0]),
                             commanded*self.offsets[side][1])
        return targets

    def follow_bodies(self, pose_at, seconds, phase, *, relative_budget_m=None, absolute_budget_m=.004):
        """Advance the path clock while the absolute (and, when given, relative) error is inside its budget; finish
        once both are inside half of it. The command leads the path by LEAD_S."""
        self.phase = phase
        start = self.time()
        elapsed, last_advance, ahead = 0., start, 0.

        def smooth(x):
            return float(np.clip(x**3*(10+x*(-15+6*x)), 0., 1.))
        while True:
            fraction = min(1., elapsed/seconds)
            desired = pose_at(smooth(fraction))
            absolute, relative = self.tracking_error(desired)
            self.tracking = dict(phase=phase, progress=smooth(fraction), absolute_material_error_m=absolute,
                                 relative_material_error_m=relative, relative_budget_m=relative_budget_m)
            inside = absolute < absolute_budget_m and (relative_budget_m is None or relative < relative_budget_m)
            if fraction >= 1 and absolute < absolute_budget_m/2 and (relative_budget_m is None or relative < relative_budget_m/2):
                self.samples.append(dict(time_s=self.time(), **self.tracking))
                return
            if inside:
                elapsed = min(seconds, elapsed+self.dt)
                last_advance = self.time()
            if self.time()-start > max(30., seconds*24) or self.time()-last_advance > 12.:
                raise RuntimeError(f"Coil tracking stalled: {self.tracking}")
            # The lead grows and shrinks by one control period per step: switching straight between the leading and the
            # current pose jumped the command where the path is fast.
            ahead = min(LEAD_S, ahead+self.dt) if inside else max(0., ahead-self.dt)
            lead = pose_at(smooth(min(1., fraction+ahead/seconds)))
            targets = self.attached_targets(desired, lead)
            yield {s: dict(position_world_m=p.tolist(), orientation_wxyz=r.as_quat()[[3, 0, 1, 2]].tolist(),
                           gripper_position_m=self.grip) for s, (p, r) in targets.items()}

    def along(self, poses, u):
        """`along` measured by the material travel at the block centre: its displacement plus radius x angle."""
        return along(poses, u, self.centre, self.radius)

    def tracking_error(self, desired):
        """Largest material displacement (piece pose error at the block centre plus radius x angle), absolute and
        of silver relative to gold."""
        actual = self.bodies()

        def at_centre(pose):
            return pose[0] + pose[1].apply(self.centre)

        absolute = max(float(np.linalg.norm(at_centre(desired[s])-at_centre(actual[s]))) +
                       self.radius*float((desired[s][1]*actual[s][1].inv()).magnitude()) for s in desired)

        def relative(poses):
            return M.compose(M.inverse(poses["right"]), poses["left"])
        wanted, measured = relative(desired), relative(actual)
        relative_error = float(np.linalg.norm(at_centre(wanted)-at_centre(measured))) + \
            self.radius*float((wanted[1]*measured[1].inv()).magnitude())
        return absolute, relative_error

    def plan(self, rest, arms=None):
        """Piece poses per phase and every TCP target of each hand, from the measured resting poses. `arms`
        (official_arms.OracleArms) is needed only by a record with anchor "arms"."""
        record, sides = self.record, tuple(PARTS)

        def lowest(s, pose):
            return float((pose[1].apply(self._points(s)) + pose[0])[:, 2].min())
        table_top = min(lowest(s, rest[s]) for s in sides)     # the resting pieces' lowest points touch the table
        if record.get("anchor") == "arms":
            # the pair's placement found by both arms together, the pieces kept ASSEMBLY_CLEARANCE_M above the table
            assembly = arms.placement(record, self.tcp_in_piece, {s: self._points(s) for s in sides},
                                      table_top + ASSEMBLY_CLEARANCE_M)
        else:
            assembly = M.assembly(record)
        if record.get("anchor") == "hands":
            # holding the hands' midpoint lowers one piece by half of what the other rises: lift the whole assembly
            # until both pieces keep ASSEMBLY_CLEARANCE_M above the table
            low = min(min(lowest("right", g), lowest("left", s_)) for _, g, s_ in assembly)
            up = max(0., table_top + ASSEMBLY_CLEARANCE_M - low)
            assembly = [(k, (g[0] + [0., 0., up], g[1]), (s_[0] + [0., 0., up], s_[1])) for k, g, s_ in assembly]
        start = {"right": assembly[0][1], "left": assembly[0][2]}
        turns = {"right": [g[1] for _, g, _ in assembly], "left": [s[1] for _, _, s in assembly]}
        lifted = {s: (rest[s][0] + [0., 0., LIFT_M], rest[s][1]) for s in sides}
        carry1, carry2 = {}, {}
        for s in sides:
            carry1[s], carry2[s] = M.carry(lifted[s], start[s], turns[s], self.centroids[s], 72)
        steps, last = {}, {"right": assembly[0][1], "left": assembly[0][2]}
        for step, gold, silver in assembly[1:]:
            if step not in steps:                 # each step's path starts where the previous one ended
                steps[step] = {s: [last[s]] for s in sides}
            steps[step]["right"].append(gold)
            steps[step]["left"].append(silver)
            last = {"right": gold, "left": silver}
        assembled = {"right": assembly[-1][1], "left": assembly[-1][2]}
        drop = min(lowest(s, assembled[s]) for s in sides) - (table_top + .0005)
        placed = {s: (assembled[s][0] - [0., 0., drop], assembled[s][1]) for s in sides}
        grasp = {s: M.compose(rest[s], self.tcp_in_piece[s]) for s in sides}
        pre = {s: (grasp[s][0] - grasp[s][1].apply([.08, 0., 0.]) + [0., 0., .03], grasp[s][1]) for s in sides}
        # the hands come down onto the pre-grasp poses from pregrasp_rise_m above them (default 12 cm). The lattice
        # designs' longer pieces put silver's grasp nearer the left shoulder: from 12 cm the wrist angle is 0.09 rad
        # from its limit, from 6 cm 0.18 rad (s43, 2026-10-10)
        high = {s: (pre[s][0] + [0., 0., record.get("pregrasp_rise_m", .12)], pre[s][1]) for s in sides}
        # a joint-space carry starts with the piece raised straight up until its lowest point is RAISE_MARGIN_M above
        # the higher of its lifted and its start lowest points: the joint move in between may dip
        raised = {s: (lifted[s][0] + [0., 0., max(lowest(s, lifted[s]), lowest(s, carry1[s][-1])) + RAISE_MARGIN_M
                                      - lowest(s, lifted[s])], lifted[s][1]) for s in sides}
        bodies = dict(lifted=lifted, raised=raised, carry1=carry1, carry2=carry2, steps=steps, assembled=assembled,
                      placed=placed)

        def hand(s, poses):
            return [M.compose(p, self.tcp_in_piece[s]) for p in poses]

        def line(a, b, n):
            return [M.interpolate(a, b, k/n) for k in range(1, n+1)]
        targets, segments = {}, {}
        for s in sides:
            path = [high[s]] + line(high[s], pre[s], 12) + line(pre[s], grasp[s], 24)
            path += hand(s, line(rest[s], lifted[s], 24))
            lifted_at = len(path)
            path += hand(s, carry1[s]) + hand(s, carry2[s])
            for step in sorted(steps):
                path += hand(s, steps[step][s][1:])
            path += hand(s, line(assembled[s], placed[s], 24))
            placed_tcp = M.compose(placed[s], self.tcp_in_piece[s])
            back = (placed_tcp[0] - placed_tcp[1].apply([.06, 0., 0.]), placed_tcp[1])
            path += line(placed_tcp, back, 24) + line(back, (back[0] + [0., 0., .06], back[1]), 12)
            targets[s] = path
            # a joint-space carry (record "carry": "joint"): the piece first rises straight to carry1's last height
            # (the start's), then the hand poses up to carry1's last pose (the start's height and orientation) are
            # left free, so the arm plans the pickup and the rest apart
            segments[s] = (path[:lifted_at] + hand(s, line(lifted[s], raised[s], 24)), path[lifted_at+len(carry1[s])-1:])
        return dict(bodies=bodies, grasp=grasp, pre=pre, high=high, targets=targets, table_top=table_top,
                    segments=segments if record.get("carry") == "joint" else None)

    def carry_moves(self, arms, start, ends, table_top, clearance_m=CARRY_CLEARANCE_M, depth=0):
        """arms.joint_move from the arm joint angles `start` to `ends` = {side: angles}; where a held piece (placed
        through tcp_in_piece) would come lower than clearance_m above the table, or than at either end if that is
        lower, through a via configuration: at the lowest such sample that hand rises straight up until its piece
        clears by clearance_m (IK from the sample), and each half is checked again, down to CARRY_VIA_DEPTH levels.
        Without such a sample it is joint_move's own path."""
        points = {s: self._points(s) for s in PARTS}

        def height(s, q):
            piece = M.compose(arms.tcp(s, arms.full({s: q})), M.inverse(self.tcp_in_piece[s]))
            return float((piece[1].apply(points[s]) + piece[0])[:, 2].min()) - table_top
        moves = arms.joint_move(start, ends)
        worst = None
        for s in PARTS:
            floor = min(clearance_m, height(s, start[s]), height(s, ends[s])) - .001
            for k, q in enumerate(moves[s]):
                low = height(s, q)
                if low < floor and (worst is None or low - floor < worst[0]):
                    worst = (low - floor, s, k, low)
        if worst is None or depth == CARRY_VIA_DEPTH:
            return moves
        _, s, k, low = worst
        tcp = arms.tcp(s, arms.full({s: moves[s][k]}))
        via = {x: moves[x][k] for x in PARTS}
        via[s] = arms.solve(s, (tcp[0] + [0., 0., clearance_m - low + .01], tcp[1]), {s: moves[s][k]})
        first = self.carry_moves(arms, start, via, table_top, clearance_m, depth+1)
        second = self.carry_moves(arms, via, ends, table_top, clearance_m, depth+1)
        return {x: first[x] + second[x] for x in PARTS}

    def _points(self, side):
        from topogym.assets.geometry.cast_coil_official import piece_points, piece_primitives
        return piece_points(piece_primitives(self.record, PIECES[side]))

    def actions(self):
        from .official_arms import OracleArms
        self.phase, self.grip = "open_at_home", OPEN
        yield {s: dict(gripper_position_m=self.grip) for s in PARTS}
        yield from self.hold(.5)
        rest = self.bodies()
        arms = self.arms = OracleArms()
        plan = self.plan(rest, arms)
        self.flip, paths, carry_end = {}, {}, {}
        for s in PARTS:
            if plan["segments"] is not None:
                hold, (pickup, rest_of_plan), margin = arms.plan_segments(s, plan["segments"][s])
                paths[s], carry_end[s] = pickup, rest_of_plan[0]
            else:
                hold, paths[s], margin = arms.plan(s, plan["targets"][s])
            self.flip[s] = Rotation.from_euler("x", 90*hold, degrees=True)
        self.tcp_in_piece = {s: (p, r*self.flip[s]) for s, (p, r) in self.tcp_in_piece.items()}

        self.phase = "home_to_pregrasp"
        moves = arms.joint_move(self.command, {s: paths[s][0] for s in PARTS})
        for k in range(len(moves["left"])):
            yield {s: dict(joint_position_rad=moves[s][k].tolist(), gripper_position_m=self.grip) for s in PARTS}
        yield from self.hold(.25)
        rolled = {key: {s: (p, r*self.flip[s]) for s, (p, r) in plan[key].items()} for key in ("high", "pre", "grasp")}
        yield from self.follow(lambda u: {s: M.interpolate(rolled["high"][s], rolled["pre"][s], u) for s in PARTS},
                               2., "descend")
        yield from self.follow(lambda u: {s: M.interpolate(rolled["pre"][s], rolled["grasp"][s], u) for s in PARTS},
                               2.5, "approach")
        self.phase, self.grip = "close_and_acquire", CLOSED
        yield {s: dict(gripper_position_m=self.grip) for s in PARTS}
        yield from self.hold(1.)
        hands = self.obs["privileged"]["world"]["grasp"]["hands"]
        if any(hands[s]["phase"] != "holding" for s in PARTS):
            raise RuntimeError("Bilateral automatic grasp not acquired: "+str({s: hands[s]["phase"] for s in PARTS}))
        held, tcp = self.bodies(), self.tcp()
        self.offsets = {s: M.compose(M.inverse(held[s]), tcp[s]) for s in PARTS}

        bodies = plan["bodies"]
        yield from self.follow_bodies(lambda u: {s: M.interpolate(held[s], bodies["lifted"][s], u) for s in PARTS},
                                      3., "lift")
        if plan["segments"] is not None:
            # a joint-space carry: the pieces rise to the start's height, then each arm moves in joint space to the
            # first solution of the rest of its plan
            yield from self.follow_bodies(lambda u: {s: M.interpolate(bodies["lifted"][s], bodies["raised"][s], u)
                                                     for s in PARTS}, 3., "raise")
            self.phase = "carry_joints"
            moves = self.carry_moves(arms, self.command, carry_end, plan["table_top"])
            for k in range(len(moves["left"])):
                yield {s: dict(joint_position_rad=moves[s][k].tolist(), gripper_position_m=self.grip) for s in PARTS}
            yield from self.hold(.25)
        else:
            yield from self.follow_bodies(lambda u: {s: self.along([bodies["lifted"][s]] + bodies["carry1"][s], u)
                                                     for s in PARTS}, 6., "carry_turn")
        yield from self.follow_bodies(lambda u: {s: self.along([bodies["carry1"][s][-1]] + bodies["carry2"][s], u) for s in PARTS},
                                      3., "carry_close_in")
        for step, route_step in zip(sorted(bodies["steps"]), reversed(self.record["route"])):
            path = bodies["steps"][step]
            relative = [M.compose(M.inverse(g), s) for g, s in zip(path["right"], path["left"])]
            if route_step["kind"] == "slide":
                cells = float(np.linalg.norm(route_step["vector"]))
                seconds = STEP_SECONDS["long_slide"] if cells > 1.5 else STEP_SECONDS["slide_per_cell"]*cells
            else:
                seconds = STEP_SECONDS["turn"]
            seconds *= self.slowdown[step]

            def target(u, path=path, relative=relative):
                gold = self.along(path["right"], u)
                return {"right": gold, "left": M.compose(self.bodies()["right"], self.along(relative, u))}
            yield from self.follow_bodies(target, seconds, f"assembly_step_{step}",
                                          relative_budget_m=self.relative_budget[step])
        assembled = self.bodies()
        yield from self.follow_bodies(lambda u: {s: M.interpolate(assembled[s], bodies["placed"][s], u) for s in PARTS},
                                      4., "place_assembly", relative_budget_m=self.relative_budget[len(self.record["route"])])
        self.phase, self.grip = "release", OPEN
        yield {s: dict(gripper_position_m=self.grip) for s in PARTS}
        yield from self.hold(.5)
        tcp = self.tcp()
        back = {s: (p - r.apply([.06, 0., 0.]), r) for s, (p, r) in tcp.items()}
        yield from self.move(back, 2., "retreat_along_fingers")
        yield from self.move({s: (p + [0., 0., .06], r) for s, (p, r) in back.items()}, 1.5, "retreat_up")
        self.phase = "settle"
        yield from self.hold(4.)
