"""horseshoe_ring on MuJoCo: try shoe grasp candidates from the frozen birth, closed loop (planning tool, no reference).

From the instance's birth the oracle's own opening sequence runs (HorseshoePolicy: open, joint approach above the
shoes, Cartesian descent, close to 2 mm, 1 s hold, the grasp binding check) with the tier's recorded grasp for one hand
and a candidate for the other (or both candidates). A candidate passes when both hands hold their own shoe. Passing
candidates are then screened against the tier's recorded fold (fold.json object frames): HorseshoeIK.plan_fold with the
candidate's shoe-to-TCP offsets (native IK and collision geometry at every 4th frame), as the recorded fold's dense
audit was made.

  python grasp_probe.py <instance.json> <tier> <out.json> [--side left|right|both] [--angles 180,170,...]
      [--openings 0.027,0.018] [--max 6]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/roboarc/shared"))
import numpy as np
from scipy.spatial.transform import Rotation

DATA = Path(__file__).resolve().parents[3] / "source/topogym/planning/harbor_bridge/reference_data/horseshoe_ring"


class StopAfterConfirm:
    """Delegate to the oracle's policy until its grasp check has passed (phase lift_assembly) or raised."""
    def __init__(self, inner):
        self.inner, self.outcome = inner, None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def act(self, observation):
        try:
            action = self.inner.act(observation)
        except RuntimeError as error:
            self.outcome = dict(passed=False, error=str(error)[:400])
            return None
        if self.inner.policy.phase == "lift_assembly":
            self.outcome = dict(passed=True)
            return None
        return action


def run(instance, grasps):
    from rcb_topogym import Session
    from topogym.planning.harbor_bridge.actions import SessionPolicyRunner
    from topogym.planning.horseshoe_ring.kinematics import CheckedJointPolicy
    from topogym.planning.horseshoe_ring.policy import HorseshoePolicy
    s = Session("horseshoe_ring", instance=instance, strict=True)
    policy = StopAfterConfirm(CheckedJointPolicy(HorseshoePolicy(grasps, part="fold"), s.env._robot))
    t0 = time.time()
    s.record_start()
    SessionPolicyRunner(s).run_policy(request_id="probe", policy=policy, submission_hash="probe", state_fields=("world",))
    hands = s.env._world.read_state()["grasp"]["hands"]
    w = s.env._world
    names = {int(l.idx): f"{e}/{l.name}" for e, ent in w.entities.items() for l in ent.links}
    c = w.contacts()
    fingers = [f"{names.get(int(a), a)}|{names.get(int(b), b)}|{1000 * float(p):.2f}mm"
               for a, b, p in zip(c["link_a"], c["link_b"], c["penetration"])
               if "finger" in names.get(int(a), "") + names.get(int(b), "")]
    out = dict(policy.outcome or dict(passed=False, error="policy ended"), rows=s.n_steps, wall_s=round(time.time() - t0, 1),
               bindings={k: (v["binding"]["target"][0] if v["binding"] else None) for k, v in hands.items()},
               rejection={k: v["rejection"] for k, v in hands.items()}, finger_contacts=sorted(set(fingers)))
    s.close()
    return out


def fold_screen(instance, tier, grasps):
    from rcb_topogym import Session
    from topogym.planning.horseshoe_ring.kinematics import HorseshoeIK
    from topogym.planning.horseshoe_ring.oracle import load_reference, expand_coupled_plan
    from topogym.planning.rigid_reference.feedback import body_pose, relative_pose
    s = Session("horseshoe_ring", instance=instance, strict=True)
    env = s.env
    reference, meta = load_reference(DATA / tier / "reference.json", env._task.assembly)
    fold = expand_coupled_plan(reference, json.loads((DATA / tier / "fold.json").read_text()))
    state = env._world.read_state()
    offsets = {k: relative_pose(body_pose(state["bodies"]["horseshoe_" + k]),
                                (np.asarray(g["tcp_position_m"]), Rotation.from_quat(np.roll(g["tcp_quaternion_wxyz"], -1))))
               for k, g in grasps.items()}
    checker = HorseshoeIK(env)
    q = checker.seed.copy()
    for k, g in grasps.items():
        idx = env._robot._q_indices[k]
        q[idx[:6]] = np.asarray(g["grasp_qpos"])[idx[:6]]
        q[idx[6:]] = .0162
    t0 = time.time()
    result = checker.plan_fold(fold, offsets, seed=q, stride=4)
    s.close()
    return dict(passed=result["passed"], failure=result["failure"], frames=len(result["frames"]), of=len(fold),
                wall_s=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance"); ap.add_argument("tier"); ap.add_argument("out")
    ap.add_argument("--side", default="left", choices=("left", "right", "both"))
    ap.add_argument("--angles", default="180,170,190,160,200,150,210")
    ap.add_argument("--openings", default="0.027,0.018")
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--no-fold", action="store_true")
    ap.add_argument("--dz", default=None, help="instead of new candidates: the recorded grasp raised by these heights (m)")
    ap.add_argument("--extra", default=None, help="instead of new candidates: the recorded grasp with these extra fields "
                    "(JSON objects separated by |, e.g. {\"open_position_m\": 0.018})")
    a = ap.parse_args()
    from rcb_topogym import Session
    from topogym.planning.horseshoe_ring.kinematics import HorseshoeIK
    recorded = json.loads((DATA / a.tier / "grasps.json").read_text())
    recorded = {k: recorded[k]["candidates"][0] for k in ("left", "right")}
    s = Session("horseshoe_ring", instance=a.instance, strict=True)
    ik = HorseshoeIK(s.env)
    state = s.env._world.read_state()
    sides = ("left", "right") if a.side == "both" else (a.side,)
    angles = [int(x) for x in a.angles.split(",")]
    pools = {}
    for side in sides:
        if a.extra:
            pools[side] = [dict(recorded[side], **json.loads(spec)) for spec in a.extra.split("|")]
        elif a.dz:
            pools[side] = [raised(ik, recorded[side], side, float(dz)) for dz in a.dz.split(",")]
        else:
            pools[side] = candidates_at(ik, s.env._task, state, side, angles,
                                        tuple(float(x) for x in a.openings.split(",")))
        print(side, "candidates", [(r["angle_deg"], r["flip"], r.get("open_position_m", .027)) for r in pools[side]], flush=True)
    s.close()
    report = dict(tier=a.tier, recorded={k: dict(angle=v["angle_deg"], flip=v["flip"], opening=v.get("open_position_m"))
                                         for k, v in recorded.items()}, trials=[])
    trials = 0
    for side in sides:
        for cand in pools[side]:
            if trials >= a.max:
                break
            grasps = dict(recorded)
            grasps[side] = cand
            row = dict(side=side, angle=cand["angle_deg"], flip=cand["flip"], opening=cand.get("open_position_m", .027),
                       raised_m=cand.get("raised_m", 0.),
                       static_checks=[c["passed"] for c in cand["checks"]])
            row.update(run(a.instance, grasps))
            if row["passed"] and not a.no_fold:
                row["fold"] = fold_screen(a.instance, a.tier, grasps)
            if row["passed"]:
                row["candidate"] = cand
            report["trials"].append(row)
            trials += 1
            print(json.dumps({k: v for k, v in row.items() if k != "candidate"}), flush=True)
            Path(a.out).write_text(json.dumps(report, indent=1))


def raised(ik, candidate, side, dz):
    """The recorded candidate with its TCP dz higher (pads above a ring lying against the band's lower part), its
    approach pose 8 cm above it, both solved by the same checked IK as grasp_candidates."""
    row = dict(candidate)
    turn = Rotation.from_quat(np.roll(candidate["tcp_quaternion_wxyz"], -1))
    tcp = np.asarray(candidate["tcp_position_m"]) + [0, 0, dz]
    pre = ik.solve(side, (tcp + [0, 0, .08], turn), np.asarray(candidate["pre_qpos"]))
    target = ik.solve(side, (tcp, turn), pre)
    for q in (pre, target):
        q[ik.robot._q_indices[side][6:]] = candidate.get("open_position_m", .027)
    row.update(tcp_position_m=tcp.tolist(), pre_qpos=pre.tolist(), grasp_qpos=target.tolist(), raised_m=dz,
               checks=[ik.check(pre), ik.check(target, contacts={side: "horseshoe_" + side})])
    return row


def candidates_at(ik, task, state, side, angles, openings):
    """HorseshoeIK.grasp_candidates at the given band angles (both jaw flips, every opening), in angle order."""
    rows, _ = ik.grasp_candidates(task, state, side, openings=openings, angles=angles)
    return sorted(rows, key=lambda r: (angles.index(r["angle_deg"]), r["flip"] != 1, -r["open_position_m"]))


if __name__ == "__main__":
    main()
