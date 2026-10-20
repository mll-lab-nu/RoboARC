"""The hedgehog_in_cage easy -i01 reference: the released rows up to their release, then the ball carried out while held.

The released route (tools/references/oracles_genesis/hedgehog-in-cage-easy-i00) opens the gripper at row 7126 with the ball
88 mm from the cage axis and three of its spikes still inside the 0 deg window (a 60 mm slot between parallel walls,
180 mm tall): the radial spike and two side spikes 30 deg either side, whose tips reach the slot walls. The free fall
(0.1 m) runs down the slot onto the mount, and the ball comes out only by rolling there; on the exact constraint
diagonal it lands 31 mm away from where the released regrasp (row 7615) closes, and the regrasp closes on air.

This route keeps the grasp instead. From the measured state at row 7125 (ball pose, the radial direction u from the
cage axis through the ball centre, the grasp = the ball in the TCP frame):
1. shift the ball 3 mm toward the cage axis and 2 mm sideways (its clearance to the slot grows from 0.37 to 2.6 mm);
2. roll it -85 deg about u through its centre: the side spikes' sideways spread turns into a vertical one, which the
   tall slot admits (the offset of step 1 is kept to 30 deg and returned to zero by 60 deg);
3. pull it 70 mm out along u;
4. carry it past the cage on the -y side, over the goal volume's centre, and lower its centre to 1.04 m;
5. hold until the episode ends (the success rule: the whole ball in the goal volume, still for 2 s; holding allowed).
Each step is sampled densely (<= 1 mm, 1 deg); the held hand follows the samples by IK seeded from the previous one
(Session robot.ik), and the arm is commanded at most 0.006 rad per control step (the window_change oracle's held step).

usage: easy_held_exit.py <instance.json> <released.npz> <out.npz> [--release-row N]
       easy_held_exit.py <instance.json> <released.npz> --check <state.npz>   (the route from a saved row-7125 state,
       kinematics only: per sample the ball's clearance to cage and mount, IK residual, joint step, the robot's
       clearance; nothing is stepped)
Run it with the RoboTangle image's Python, as the stress tools (tools/stress/_env.py)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stress"))
import _env  # noqa: F401,E402  (before numpy: SIMD masking, threads, this checkout first on sys.path)
import argparse, hashlib, json, time  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from rcb_topogym import Session  # noqa: E402

RELEASE_ROW = 7126                  # the released route opens the gripper here
SHIFT_MM = (-3., 2.)                # step 1: along u, along t (sideways)
ROLL_DEG, KEEP_DEG, FREE_DEG = -85., 30., 60.
PULL_MM = 70.
GOAL_XY = (-0.13568, -0.23344)     # the goal volume's centre (instance objects.goal_region)
GOAL_Z = 1.04                       # the ball centre over the goal (tabletop 0.920, volume top 1.157)
PAST_CAGE_M = .0675 + .080          # the carry's y line: cage radius + the ball's reach
HELD_STEP_RAD, SETTLE_TOL_RAD, HOLD_ROWS = .006, .001, 800


def route(p0, r0, u, cage_c):
    """Ball poses (position, Rotation) from the row-7125 pose."""
    t = np.cross([0., 0., 1.], u)
    shift = 1e-3 * (SHIFT_MM[0] * u + SHIFT_MM[1] * t)
    out = []
    for k in range(1, 9):                                           # 1. shift, 0.45 mm steps
        out.append((p0 + shift * k / 8, r0))
    for k in range(1, int(abs(ROLL_DEG)) + 1):                      # 2. roll, 1 deg steps
        a = abs(k)
        w = 1. if a <= KEEP_DEG else max(0., (FREE_DEG - a) / (FREE_DEG - KEEP_DEG))
        out.append((p0 + shift * w, R.from_rotvec(np.radians(np.sign(ROLL_DEG) * a) * u) * r0))
    r1 = out[-1][1]
    for k in range(1, int(PULL_MM) + 1):                            # 3. pull, 1 mm steps
        out.append((p0 + u * k * 1e-3, r1))

    def line(a, b, step=.001):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        return [(a + (b - a) * k / n, r1) for k in range(1, n + 1)]
    pa = out[-1][0]
    pb = np.r_[pa[0], cage_c[1] - PAST_CAGE_M, pa[2]]               # 4. past the cage (-y), over the goal, down
    pc = np.r_[GOAL_XY, pa[2]]
    pd = np.r_[GOAL_XY, GOAL_Z]
    return out + line(pa, pb) + line(pb, pc) + line(pc, pd)


class Exit:
    def __init__(self, s):
        self.s, w = s, s.env._world
        self.m, self.d = w.model, w.data
        B = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.ball, self.cage = B("hedgehog_ball::hedgehog"), B("hedgehog_cage::cage")
        self.frame = s.robot.tcp_frames["arm"]
        self.qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k}")] for k in range(1, 8)]

    def plan(self):
        """The route's arm joint targets from the current (row-7125) state."""
        d = self.d
        p0, r0 = d.xpos[self.ball].copy(), R.from_quat(d.xquat[self.ball][[1, 2, 3, 0]])
        cage_c = d.xpos[self.cage].copy()
        u = np.r_[(p0 - cage_c)[:2], 0.]
        u /= np.linalg.norm(u)
        pt, qt = self.s.ee_pose("arm")
        tcp_r = R.from_quat(np.asarray(qt, float)[[1, 2, 3, 0]])
        g_p, g_r = tcp_r.inv().apply(p0 - np.asarray(pt, float)), tcp_r.inv() * r0      # the ball in the TCP frame
        poses = route(p0, r0, u, cage_c)
        q = np.array([d.qpos[a] for a in self.qadr])
        qs, worst = [], dict(ik_mm=0., step_rad=0.)
        for p, r in poses:
            rt = r * g_r.inv()
            q7, err = self.s.robot.ik({self.frame: np.r_[p - rt.apply(g_p), rt.as_quat()[[3, 0, 1, 2]]]}, q, 1e-4, 1e-3, 200)  # the Session.ik tolerances (tighter ones restart from home: another branch)
            q7 = np.asarray(q7, float)[:7]
            worst = dict(ik_mm=max(worst["ik_mm"], err[self.frame][0] * 1e3), step_rad=max(worst["step_rad"], float(np.abs(q7 - q).max())))
            qs.append(q7)
            q = q7
        return poses, qs, dict(samples=len(poses), u=u.round(4).tolist(), **{k: round(v, 3) for k, v in worst.items()})

    def execute(self, qs, grip, q_cmd):
        """Command the joint path at most HELD_STEP_RAD per control step; True when the episode ended."""
        for q7 in qs:
            while True:
                delta = q7 - q_cmd
                n = np.abs(delta).max()
                q_cmd = q7.copy() if n <= HELD_STEP_RAD else q_cmd + delta * (HELD_STEP_RAD / n)
                if self.s.step(np.r_[q_cmd, grip].astype(np.float32)):
                    return True, q_cmd
                if n <= HELD_STEP_RAD:
                    break
        return False, q_cmd


def check(s, ex, state):
    """The route from a saved state, kinematics only."""
    import trimesh
    from topogym.solvers.collision.triangle_meshes import TriangleMeshPair
    m, d = ex.m, ex.d
    z = np.load(state)
    d.qpos[:] = z["qpos"]
    mujoco.mj_forward(m, d)
    poses, qs, summary = ex.plan()

    def mesh_of(b, world):
        parts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            if m.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_BOX):
                mesh = trimesh.creation.box(extents=2 * m.geom_size[g])
            else:
                k = m.geom_dataid[g]
                mesh = trimesh.Trimesh(m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float),
                                       m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k] + m.mesh_facenum[k]], process=False)
            T = np.eye(4)
            if world:
                T[:3, :3] = d.geom_xmat[g].reshape(3, 3); T[:3, 3] = d.geom_xpos[g]
            else:
                Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[g]); T[:3, :3] = Rg.reshape(3, 3); T[:3, 3] = m.geom_pos[g]
            mesh.apply_transform(T)
            parts.append(mesh)
        return trimesh.util.concatenate(parts)
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    pair = TriangleMeshPair((trimesh.util.concatenate([mesh_of(ex.cage, True), mesh_of(B("hedgehog_mount"), True),
                                                       mesh_of(B("table"), True)]), mesh_of(ex.ball, False)))
    ent = {}
    for name, rec in s.env._world.manifest["entities"].items():
        for link in rec["links"]:
            ent[B(link["body"])] = name
    rob = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and ent.get(int(m.geom_bodyid[g]), "robot") == "robot"
           and (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "") not in ("link0", "base", "robot")]
    obst = [g for g in range(m.ngeom) if ent.get(int(m.geom_bodyid[g])) in ("hedgehog_cage", "hedgehog_mount", "table")]
    fadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hedgehog_ball::root")]
    lim = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k}")] for k in range(1, 8)])
    d2, ft, rows = mujoco.MjData(m), np.zeros(6), []
    for i, ((p, r), q7) in enumerate(zip(poses, qs)):
        T = np.eye(4); T[:3, :3] = r.as_matrix(); T[:3, 3] = p
        d2.qpos[:] = d.qpos; d2.qpos[ex.qadr] = q7; d2.qpos[fadr:fadr + 3] = p; d2.qpos[fadr + 3:fadr + 7] = r.as_quat()[[3, 0, 1, 2]]
        mujoco.mj_forward(m, d2)
        rows.append(dict(i=i, ball_mm=round(1e3 * pair.clearance((np.eye(4), T)), 2),
                         robot_mm=round(1e3 * min(mujoco.mj_geomDistance(m, d2, a, b, .03, ft) for a in rob for b in obst), 1),
                         margin=round(float(np.min(np.minimum(q7 - lim[:, 0], lim[:, 1] - q7))), 3)))
    seg = [("shift", 0, 8), ("roll", 8, 8 + int(abs(ROLL_DEG))), ("pull", 8 + int(abs(ROLL_DEG)), 8 + int(abs(ROLL_DEG)) + int(PULL_MM)),
           ("carry", 8 + int(abs(ROLL_DEG)) + int(PULL_MM), len(rows))]
    steps = np.r_[np.abs(qs[0] - np.array([d.qpos[a] for a in ex.qadr])).max(), np.abs(np.diff(qs, axis=0)).max(1)]
    out = dict(summary, segments={n: dict(ball_mm=min(r["ball_mm"] for r in rows[a:b]), robot_mm=min(r["robot_mm"] for r in rows[a:b]),
                                          margin=min(r["margin"] for r in rows[a:b]), step_rad=round(float(steps[a:b].max()), 3),
                                          step_at=int(a + np.argmax(steps[a:b]))) for n, a, b in seg})
    print(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance"); ap.add_argument("released"); ap.add_argument("out", nargs="?")
    ap.add_argument("--check", default=None)
    ap.add_argument("--release-row", type=int, default=RELEASE_ROW,
                    help="the row at which <released.npz> opens the gripper (the released route: 7126; the family's "
                         "oracle re-run closed loop on the Robotiq spring references: 7300, its release_for_regrasp)")
    a = ap.parse_args()
    release_row = a.release_row
    s = Session(json.loads(Path(a.instance).read_text())["task"], instance=a.instance, strict=True)
    ex = Exit(s)
    if a.check:
        check(s, ex, a.check)
        return s.close()
    birth = s.status()["state_sha"]
    ref = np.load(a.released)
    rows = ref["actions"].astype(np.float32)
    assert ref["action_layout"].tolist() == s.action_layout, (ref["action_layout"].tolist(), s.action_layout)
    t0 = time.time()
    s.record_start()
    ended = False
    for row in rows[:release_row]:
        if s.step(row):
            ended = True
            break
    assert not ended, "the episode ended inside the released rows"
    poses, qs, summary = ex.plan()
    ended, q_cmd = ex.execute(qs, float(rows[release_row - 1][7]), rows[release_row - 1][:7].astype(float))
    for _ in range(HOLD_ROWS):
        if ended:
            break
        ended = s.step(np.r_[q_cmd, rows[release_row - 1][7]].astype(np.float32))
    acts = s.record_stop()
    st = s.status()
    meta = dict(source="hedgehog_in_cage easy reference: the rows of `route` to their release (`released_rows`), then "
                       "the ball rolled out of the slot and carried to the goal while held",
                route=str(a.released), route_sha256=hashlib.sha256(Path(a.released).read_bytes()).hexdigest(),
                released_rows=release_row, plan=summary, wall_s=round(time.time() - t0, 1), success=st["success"],
                episode_end=st["episode_end"])
    s.save_trajectory(acts, a.out, meta=meta)
    print(json.dumps(dict(instance=a.instance, birth_sha=birth, rows=int(len(acts)), plan=summary, success=st["success"],
                          episode_end=st["episode_end"], review=st["review"], forcing=st["forcing"], state_sha=st["state_sha"],
                          out=a.out, sha256=hashlib.sha256(Path(a.out).read_bytes()).hexdigest())))
    s.close()


if __name__ == "__main__":
    main()
