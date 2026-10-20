"""horseshoe_ring on MuJoCo: rewrite a tier's hash-bound grasp inputs for a new shoe grasp (planning tool).

The prefix stage reads two of the tier's planning inputs that depend on the shoe grasps:
- grasps.json: per hand the selected candidate (HorseshoeIK.grasp_candidates' row);
- approach.json: the joint approach from home to 8 cm above both grasps (101 samples) and the Cartesian descent to
  them (60 samples), each with its collision check; the prefix rechecks every row before it starts.
This tool takes the new candidates (a grasp_probe.py report's passing trial for one hand, the other hand's recorded
candidate kept), recomputes approach.json on the MuJoCo instance's birth exactly as the released inputs were made
(scripts/oracle_runners/run_horseshoe_policy.py, part plan), and writes both next to the tier's recipe.json with the
recipe's sha256 entries updated. The reference, fold and support inputs are object-space and stay as they are.

  python regrasp_inputs.py <instance.json> <recipe dir> <probe report.json> <trial index>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/roboarc/shared"))
import numpy as np
from scipy.spatial.transform import Rotation


def main(instance, recipe_dir, report, index):
    from rcb_topogym import Session
    from topogym.planning.horseshoe_ring.kinematics import HorseshoeIK
    from topogym.planning.rigid_reference.feedback import tcp_pose
    from topogym.planning.rigid_reference.geometry import interpolate
    from topogym.utils.io import sha256_file
    recipe_dir = Path(recipe_dir)
    trial = json.loads(Path(report).read_text())["trials"][int(index)]
    if not trial["passed"] or (trial.get("fold") and not trial["fold"]["passed"]):
        raise ValueError("the trial did not pass")
    selection = json.loads((recipe_dir / "grasps.json").read_text())
    selection[trial["side"]] = dict(candidates=[trial["candidate"]],
                                    note="re-selected on MuJoCo (grasp_probe.py): " + json.dumps(
                                        {k: trial[k] for k in ("angle", "flip", "opening")}))
    grasps = {s: selection[s]["candidates"][0] for s in ("left", "right")}
    s = Session("horseshoe_ring", instance=instance, strict=True)
    env = s.env
    ik = HorseshoeIK(env)
    q = ik.seed.copy()
    for side, g in grasps.items():
        q[env._robot._q_indices[side][6:]] = g.get("open_position_m", .027)
    paths = []
    q_start, q_end = q.copy(), q.copy()
    for side, g in grasps.items():
        idx = env._robot._q_indices[side][:6]
        q_end[idx] = np.asarray(g["pre_qpos"])[idx]
    for u in np.linspace(0, 1, 101):
        q = q_start + u * (q_end - q_start)
        checked = ik.check(q)
        paths.append(dict(phase="above", fraction=float(u), qpos=q.tolist(), **checked))
        if not checked["passed"]:
            raise ValueError(f"Joint approach collision: {paths[-1]}")
    initial = {k: (np.asarray(g["tcp_position_m"]) + [0, 0, .08], Rotation.from_quat(np.roll(g["tcp_quaternion_wxyz"], -1)))
               for k, g in grasps.items()}
    target = {k: (np.asarray(g["tcp_position_m"]), Rotation.from_quat(np.roll(g["tcp_quaternion_wxyz"], -1)))
              for k, g in grasps.items()}
    for u in np.linspace(0, 1, 61)[1:]:
        for k in grasps:
            q = ik.solve(k, interpolate(initial[k], target[k], u), q)
        checked = ik.check(q, contacts={k: "horseshoe_" + k for k in grasps} if u > .85 else None)
        paths.append(dict(phase="descend", fraction=float(u), qpos=q.tolist(), **checked))
        if not checked["passed"]:
            raise ValueError(f"Approach collision: {paths[-1]}")
    s.close()
    (recipe_dir / "grasps.json").write_text(json.dumps(selection, indent=1))
    (recipe_dir / "approach.json").write_text(json.dumps(paths, indent=1))
    recipe = json.loads((recipe_dir / "recipe.json").read_text())
    for name in ("grasps", "approach"):
        recipe["selection"][0]["parameters"][name]["sha256"] = sha256_file(recipe_dir / f"{name}.json")
    (recipe_dir / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(json.dumps({n: recipe["selection"][0]["parameters"][n]["sha256"][:16] for n in ("grasps", "approach")}))


if __name__ == "__main__":
    main(*sys.argv[1:5])
