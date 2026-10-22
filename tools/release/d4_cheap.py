#!/usr/bin/env python3
"""D4 cheap checks of one base's appearance seeds, in ONE container of the release image (RoboARC 1.2.0):

    python d4_cheap.py <base> <family> --tests <task tests dir> --oracles <oracles/<base> dir> [--seeds 1,2]

For seed 0 and each seed S, in fresh processes: the seed staged exactly as the privileged verifier stages it
(/opt/robotangle/stage_instance.py --seed-file against the task's runtime_binding[.s<S>].json: the canonical instance
and every asset sha = the binding check), then `state_parts.py birth` (strict Session). Per seed S one line
`D4 {json}` with:
  a_files     seeds/<S>/instance_assets/ byte-identical to seed 0's (every file: MJCF, URDF, meshes, cloth / rope
              states, settled births; a seed may change no file, its looks are build parameters)
  a_fields    instance.json's physical fields (derive_seed_instance.PHYSICAL) equal seed 0's
  a_model     the built MjModel's physics arrays (state_parts.MODEL_PHYSICS, opt) equal seed 0's (visuals excluded)
  b_strict    staged against its binding, strict Session: instance_exact, assets_exact, runtime binding, matches
  c_birth     the birth tick record's parts equal seed 0's but `task` (world, robot, targets, state, ...)
  d_restamp   tools/release/restamp.py --check privileged.npz privileged.s<S>.npz --sha <the seed's built birth>
              ("n/a": a reference without a birth stamp serves every seed; "none": no reference)
  ok          all of them
"""
import json, os, pathlib, subprocess, sys, zipfile

PHYSICAL = ("objects", "soft_state", "reset_joint_targets", "action_dim", "action_layout", "gripper_space", "control_hz",
            "resolved_variation_id")
HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path("/opt/robotangle/instances")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def files(d):
    import hashlib
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


def main():
    base, fam = sys.argv[1], sys.argv[2]
    tests = pathlib.Path(sys.argv[sys.argv.index("--tests") + 1])
    orc = pathlib.Path(sys.argv[sys.argv.index("--oracles") + 1])
    seeds = [int(x) for x in (sys.argv[sys.argv.index("--seeds") + 1] if "--seeds" in sys.argv else "1,2").split(",")]
    births = {}
    for s in [0] + seeds:
        dst = pathlib.Path(f"/tmp/s{s}")
        (pathlib.Path("/tmp") / f"seed{s}").write_text(f"{s}\n")
        rc, out = run([sys.executable, "/opt/robotangle/stage_instance.py", base, str(dst), "--binding",
                       str(tests / "runtime_binding.json"), "--seed-file", f"/tmp/seed{s}"])
        staged = rc == 0
        rc, out2 = run([sys.executable, str(HERE / "state_parts.py"), "birth", "--task", fam, "--instance", str(dst / "instance.json")])
        line = [l for l in out2.splitlines() if l.startswith("PARTS ")]
        births[s] = dict(staged=staged, stage_log=out.strip().splitlines()[-1:] if out.strip() else [],
                         **(json.loads(line[-1][6:]) if line else dict(error=out2[-800:])))
    i0 = json.loads((SRC / base / "instance.json").read_text())
    ref = orc / "privileged.npz"
    stamped = ref.is_file() and "instance_sha.npy" in zipfile.ZipFile(ref).namelist()
    b0 = births[0]
    for s in seeds:
        sd, b = SRC / base / "seeds" / str(s), births[s]
        iS = json.loads((sd / "instance.json").read_text())
        r = dict(base=base, seed=s, birth=b.get("state_sha"), seed0_birth=b0.get("state_sha"))
        r["a_files"] = files(sd / "instance_assets") == files(SRC / base / "instance_assets")
        r["a_fields"] = all(iS.get(k) == i0.get(k) for k in PHYSICAL)
        r["a_fields_differing"] = [k for k in PHYSICAL if iS.get(k) != i0.get(k)]
        r["a_model"] = b.get("model_physics_sha") is not None and b.get("model_physics_sha") == b0.get("model_physics_sha")
        r["b_strict"] = bool(b.get("staged") and b.get("instance_exact") and b.get("assets_exact") and b.get("runtime_binding")
                             and b.get("instance_matches") and b.get("state_sha") == iS.get("state_sha"))
        pa, pb = b0.get("parts") or {}, b.get("parts") or {}
        r["c_birth_parts_differing"] = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
        r["c_birth"] = bool(pa) and set(pa) == set(pb) and set(r["c_birth_parts_differing"]) <= {"task"}
        if not ref.is_file():
            r["d_restamp"] = "none"
        elif not stamped:
            r["d_restamp"] = "n/a"
        else:
            rc, out = run([sys.executable, str(HERE / "restamp.py"), "--check", str(ref), str(orc / f"privileged.s{s}.npz"),
                           "--sha", str(b.get("state_sha"))])
            r["d_restamp"] = rc == 0
            if rc:
                r["d_restamp_log"] = out[-400:]
        if "error" in b:
            r["error"] = b["error"]
        r["ok"] = r["a_files"] and r["a_fields"] and r["a_model"] and r["b_strict"] and r["c_birth"] and r["d_restamp"] in (True, "n/a", "none")
        print("D4 " + json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
