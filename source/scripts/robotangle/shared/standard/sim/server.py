"""eai-standard/2.2 sim service for TopoGym (Genesis 1.2.1 on CUDA, Nyx RGB; docs/rcb/STANDARD_MODE_2_2.md): the TopoGym
adapter of the shared layer rcb_limited.py (scripts/eai_standard/, copied unchanged into every standard task). Adapted from
robot_coding_bench's RoboWits service. HTTP surface (rcb_limited.py): GET /spec, GET /status, POST /observe, POST /submit
(RCB1 bodies, RCB1 responses), POST /kinematics (stateless FK / IK on the robot's own model). Logs every call to
/data/episode.jsonl and keeps /data/final.json current. Nothing about objects ever leaves this process.

Crash recovery (rcb_limited.Episode, STANDARD_MODE_2_2.md section 13, resume class `approx`: Genesis' GPU contact solver is
not bit-deterministic): every executed control step is journaled (/data/journal.f64, the exact float32 row given to
Session.step), every completed call checkpointed (rows, state hash, proprio row). After a crash run.sh exits non-zero,
compose restarts the container (`restart: on-failure:6`) and this service rebuilds the frozen scene, replays the journal
up to the last checkpoint and continues the SAME episode, never a reset; a hash mismatch is recorded, not fatal, unless a
TCP is more than RCB_RESUME_MAX_DEV_M from its checkpoint.

Episode end and grading (two passes: the live episode, then one fresh replay that also records the video): when the
episode ends (success, the task's failure predicate, the wall clock, an error) the service writes /data/trajectory.npz and
/data/episode.json (the live result and final-state hash, `live_rows`: the rows the live episode executed). The collect
hook's SIGUSR1 makes it exit, releasing the GPU; environment/sim/finalize.sh then replays the trajectory once in a fresh
process, rendering /data/replay.mp4 during that replay, and completes /data/final.json.

Differences from the RoboWits service, all from the simulator: the physics runs on the GPU, so `deterministic` (live and
replay state hashes equal) is reported and not required; rows are streamed at the task's control rate (control_hz,
physics_substeps physics steps each); the joint action is per arm the arm joint targets plus the frozen instance gripper target;
`ee_pose` is solved by bounded least squares over the robot's Genesis forward kinematics; a task whose own failure
predicate fires (e.g. touching the buzz wire) ends the episode with reason "task_failed"; a family may enable passive grasp
assistance (the gripper `command` text of /spec says the rule). Genesis' CUDA context belongs to the thread that built the
scene, so every simulator call runs on the main thread: the HTTP layer serves from a thread and hands submit / observe /
kinematics, the per-call checkpoint's state hash and proprio row, and the episode end to the main thread's job queue."""
from __future__ import annotations
import json, math, os, queue, re, sys, threading, time, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcb_limited as L  # noqa: E402

PROTOCOL = L.PROTOCOL
DATA = os.environ.get("RCB_DATA", "/data"); os.makedirs(DATA, exist_ok=True); os.makedirs(os.path.join(DATA, "calls"), exist_ok=True)
TASK = os.environ.get("RCB_TASK", "cube_pick_place"); INSTANCE = os.environ.get("RCB_INSTANCE", os.path.join(os.path.dirname(__file__), "instance.json"))
OBS_CALIBRATION = os.environ.get("RCB_OBS_CALIBRATION", "1") == "1"
MAX_CHUNK_S = float(os.environ.get("RCB_MAX_CHUNK_S", "10")); MAX_WATCH_S = float(os.environ.get("RCB_MAX_WATCH_S", "30"))
LOCK = threading.Lock()
E = {"ready": False, "step": 0, "calls": 0, "done": False, "reason": None, "success": False, "t0": time.time(), "error": None}
JOBS: queue.Queue = queue.Queue()      # simulator work, executed in order by the main thread
PID_FILE = os.path.join(DATA, "server.pid")
# Crash-safe episode bookkeeping, set up before the scene is built (start count, back-off, startup watchdog); the journal row
# is the action row (its width from the frozen instance: the scene does not exist yet)
EP = L.Episode(DATA, dim=int(json.load(open(INSTANCE))["action_dim"]), resume="approx")
EP.startup()
log = EP.log


def on_main(fn, *a):
    """Run fn on the main thread (the owner of the CUDA context) and return its result or raise its exception."""
    if threading.current_thread() is threading.main_thread():
        return fn(*a)
    done, box = threading.Event(), {}

    def job():
        try:
            box["r"] = fn(*a)
        except BaseException as exc:  # noqa: BLE001 — handed back to the HTTP thread
            box["e"] = exc
        finally:
            done.set()
    JOBS.put(job); done.wait()
    if "e" in box:
        raise box["e"]
    return box["r"]


def ctrl_steps() -> int:
    return int(S.n_steps) if E["ready"] else 0


# Episode ends that are infrastructure failures, never the agent's: a simulator error, and a numerically unstable simulation
# (rcb_topogym.SIMULATOR_INSTABILITY: a cloth particle faster than PARTICLE_SPEED_LIMIT_M_S, before the task's own failure
# predicates; finalize.sh applies the same rule to the replay).
INFRA_REASONS = ("error", "simulator_instability")


# Public schemas: PROTOCOL.md §§4.2–4.8, STANDARD_MODE_2_2.md §§1–11 and
# STANDARD_MODE_2_4.md §§2–3 at main 82c7436a1. Diagnostics remain sidecar-private.
PUBLIC_SPEC_KEYS = ("protocol", "protocol_compat", "simulator", "robot", "task_text", "control_hz", "physics_substeps",
                    "physics_hz", "max_chunk_s", "max_watch_s", "cameras", "default_cameras", "action_types",
                    "proprio_layout", "proprio_groups", "static", "budget", "robot_spec", "world", "kinematics",
                    "depth", "on_service_error", "max_call_wall_s", "episode")
PUBLIC_META_KEYS = ("step", "calls", "control_steps", "physics_steps", "sim_time_s", "time_s", "time_left_s",
                    "proprio", "cameras", "done", "reason", "stopped", "degraded", "operation_id", "chunk_id",
                    "type", "executed_rows", "control_steps_chunk", "physics_steps_chunk", "clipped", "n_clipped",
                    "plan", "ok", "obs_id", "seconds", "control_steps_watch", "physics_steps_watch")
PUBLIC_CAMERA_KEYS = ("intrinsic_cv", "extrinsic_cv", "cam2world_gl", "image_size")


def _project(value, keys) -> dict:
    return {k: value[k] for k in keys if k in value} if isinstance(value, dict) else {}


def public_reason(reason):
    if reason is None or reason in ("success", "timeout", "task_failed", "error"):
        return reason
    category = str(reason).split(":", 1)[0]
    if category in ("call_watchdog", "resume_failed", "sim_crashed", "sim_crashed_on_call"):
        return category
    if category in INFRA_REASONS or category == "instance_mismatch":
        return "error"
    return "task_failed"


def public_status(value) -> dict:
    out = _project(value, ("step", "calls", "control_steps", "physics_steps", "done", "reason", "elapsed_s", "resumes"))
    if "reason" in out: out["reason"] = public_reason(out["reason"])
    return out


def public_operation(value):
    if value is None: return None
    out = _project(value, ("operation_id", "status", "rows_before", "committed_rows", "stopped",
                           "execution_uncertain", "restored_committed_rows"))
    if out.get("stopped") not in (None, "max_call_wall_s", "client_disconnected"):
        out["stopped"] = None
    return out


def public_spec(value) -> dict:
    out = _project(value, PUBLIC_SPEC_KEYS)
    if "action_types" in out:
        out["action_types"] = {name: _project(action, ("dim", "layout", "units", "limits", "semantics", "dt", "interpolate",
                                                      "frame", "quaternion", "gripper_units", "note"))
                               if isinstance(action, dict) else action for name, action in out["action_types"].items()
                               if name in ("joint_position", "ee_pose", "not_yet")}
    if "proprio_groups" in out:
        out["proprio_groups"] = _project(out["proprio_groups"], ("joint_pos", "tcp_pose", "joint_cmd", "joint_vel",
                                                               "gripper_opening", "joint_effort"))
    if "cameras" in out:
        out["cameras"] = {c: _project(v, ("rgb", "depth", "mounted", "resolution", "intrinsic_cv", "mount",
                                        "calibration_in_obs", "depth_range_m")) for c, v in out["cameras"].items()}
        for camera in out["cameras"].values():
            if "mount" in camera:
                camera["mount"] = _project(camera["mount"], ("type", "parent_link", "link_T_cam_cv", "extrinsic_cv", "cam2world_gl"))
    if "robot_spec" in out:
        robot = _project(out["robot_spec"], ("name", "base", "joints", "grippers", "controller", "tcp", "description"))
        robot["base"] = _project(robot.get("base"), ("type", "pose"))
        robot["joints"] = [_project(j, ("name", "type", "unit", "position_limits", "velocity_limit", "effort_limit",
                                       "group", "note", "rated_velocity", "rated_effort", "rated_source"))
                           for j in robot.get("joints", [])]
        robot["grippers"] = {a: _project(g, ("joints", "opening_m", "tcp_frame", "command", "command_space", "command_unit",
                                             "command_limits", "open_target", "closed_target"))
                             for a, g in robot.get("grippers", {}).items()}
        controller = _project(robot.get("controller"), ("type", "control_hz", "physics_hz", "gains", "note"))
        if "gains" in controller:
            controller["gains"] = {j: _project(g, ("stiffness", "damping")) for j, g in controller["gains"].items()}
        robot["controller"] = controller
        robot["tcp"] = {f: _project(v, ("parent_link", "link_T_frame", "definition")) for f, v in robot.get("tcp", {}).items()}
        out["robot_spec"] = robot
    if "world" in out:
        out["world"] = _project(out["world"], ("frame", "up", "units", "quaternion_order", "camera_axes"))
        if "units" in out["world"]: out["world"]["units"] = _project(out["world"]["units"], ("length", "angle", "time"))
    if "budget" in out: out["budget"] = _project(out["budget"], ("reset", "steps", "wall_clock_s"))
    if "depth" in out:
        out["depth"] = _project(out["depth"], ("modes", "default", "kind", "unit", "invalid", "max_m", "frames_encoding", "noise"))
    if "kinematics" in out:
        kin = _project(out["kinematics"], ("endpoint", "joint_names", "frame", "quaternion_order", "links", "tcp_frames",
                                          "ee_frames", "ik_targets", "base_pose_proprio", "ik", "stateless", "note"))
        if "ik" in kin:
            kin["ik"] = _project(kin["ik"], ("solver", "seed", "joints", "collision", "tol_pos_m", "tol_rot_rad", "max_iters"))
        out["kinematics"] = kin
    return out


def public_kinematics(value) -> dict:
    out = _project(value, ("ok", "op", "frame", "poses") if value.get("op") == "fk" else ("ok", "op", "frame", "q", "error"))
    if "poses" in out: out["poses"] = {f: p for f, p in out["poses"].items() if f in K_SPEC["links"]}
    if "error" in out:
        out["error"] = {f: _project(e, ("pos_m", "rot_rad")) for f, e in out["error"].items() if f in K_SPEC["ik_targets"]}
    return out


def _robot_limit_sentence(index, report):
    names = {"joint_velocity": ("target speed", "rad/s", 1),
             "joint_acceleration": ("target acceleration", "rad/s^2", 1),
             "joint_jump": ("target change", "rad", 1),
             "gripper_width": ("closing", "mm", 1000),
             "drive_fraction": ("drive", "%", 100)}
    item = names.get(report.get("limit")); cap = report.get("cap")
    if item is None or not isinstance(cap, (int, float)) or isinstance(cap, bool) or not math.isfinite(cap):
        return f"{LAYOUT[index]} command limited"
    label, unit, scale = item
    return f"{LAYOUT[index]} {label} limited at {cap * scale:g} {unit}"


def _limit_clipped(reports, row) -> list:
    """Only named robot drives can create public notices; task geometry and causes remain private."""
    if row is None or not reports: return []
    indices = {name: j for j, name in enumerate(LAYOUT)}
    for arm in ARMS:
        for j, name in zip(range(*ARM[arm][0].indices(len(LAYOUT))), S.robot.groups[arm].arm_joints):
            indices[name] = j
    out = []
    for report in reports:
        index = indices.get(report.get("entity"))
        if index is None and report.get("limit") == "gripper_width" and report.get("entity") in ARMS:
            index = ARM[report["entity"]][1]
        if index is None: continue
        own = {"limit": report.get("limit"), "cap": report.get("cap")}
        out.append({"row": row, "index": index, "reason": _robot_limit_sentence(index, own), "_robot_limit": own})
    return out


def _public_clipped(notes) -> list:
    out = []
    for note in notes:
        row, index = note.get("row"), note.get("index")
        if not isinstance(row, int) or isinstance(row, bool) or row < 0:
            continue
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(LAYOUT):
            continue
        # Never reuse a detector/exception's free-form reason. These sentences depend only on robot limits.
        name, reason = LAYOUT[index], str(note.get("reason", ""))
        if isinstance(note.get("_robot_limit"), dict):
            sentence = _robot_limit_sentence(index, note["_robot_limit"])
        elif reason.startswith("table guard:"):
            sentence = f"{name} target limited by the table guard"
        elif reason.startswith("speed:"):
            sentence = f"{name} target speed limited to {M.JOINT_STEP_LIMIT_RAD * CTRL_HZ:g} rad/s"
        elif reason.startswith(name + " limit"):
            lo, hi = LIMITS[name]
            sentence = f"{name} command limited to [{lo:g}, {hi:g}]"
        else:
            sentence = f"{name} command limited"
        out.append({"row": row, "index": index, "reason": sentence})
    return out


def public_meta(value) -> dict:
    out = _project(value, PUBLIC_META_KEYS)
    if "obs_id" in out and "chunk_id" not in out:
        for key in ("type", "executed_rows", "control_steps_chunk", "physics_steps_chunk", "clipped", "n_clipped", "plan", "ok"):
            out.pop(key, None)
    if "reason" in out: out["reason"] = public_reason(out["reason"])
    if "proprio" in out: out["proprio"] = _project(out["proprio"], ("layout", "now"))
    if "cameras" in out:
        out["cameras"] = {c: _project(v, PUBLIC_CAMERA_KEYS) for c, v in out["cameras"].items() if c in CAMS}
    if "clipped" in out: out["clipped"] = _public_clipped(out["clipped"])
    if "plan" in out:
        out["plan"] = [{"row": p["row"], **{a: "Success" if p[a] == "Success" else "failed: arm holds"
                       for a in ARMS if a in p}} for p in out["plan"] if "row" in p]
    if out.get("stopped") not in (None, "max_call_wall_s", "client_disconnected"):
        out["stopped"] = None
    return out


def public_parts(parts) -> list:
    out = []
    for name, kind, data, extra in parts:
        prefix, _, camera = name.partition("/")
        if name not in ("proprio", "frame_step", "frame_time_s") and not (
                camera in CAMS and prefix in ("video", "frames", "last", "extrinsic_cv", "last_depth", "depth_frames", "depth")):
            continue
        expected = "mp4" if prefix == "video" else "bytes" if prefix == "depth_frames" else "array"
        if kind != expected: continue
        keys = ("fps", "n") if kind == "mp4" else ("dtype", "shape", "codec", "unit") if prefix == "depth_frames" else ("dtype", "shape")
        out.append((name, kind, data, _project(extra, keys)))
    return out


def public_error(code, value, path) -> dict:
    value = value if isinstance(value, dict) else {}
    operation = value.get("operation_id")
    if code == 409 and isinstance(operation, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,80}", operation):
        return {"reason": "operation already admitted; query /operations/" + operation,
                "operation_id": operation, "operation": public_operation(value.get("operation"))}
    messages = {400: "invalid request", 404: "no such endpoint", 408: "request expired before execution",
                409: "episode finished", 413: "request too large", 500: "kinematics error" if path == "/kinematics" else "simulator error"}
    out = {"error": messages.get(code, "service error")}
    if code == 409: out["reason"] = public_reason(value.get("reason"))
    if "detail" in value: out["detail"] = "internal details retained in private log"
    return out


def public_handler(svc):
    """Project the task's HTTP output without modifying the pinned shared transport/client."""
    base = L.make_handler(svc)

    class Handler(base):
        def _json(self, code, value):
            svc.log({"ev": "private_response", "path": self.path, "code": code, "body": value})
            if code >= 400:
                out = None if value is None and self.path.startswith("/operations/") else public_error(code, value, self.path)
            elif self.path == "/spec": out = public_spec(value)
            elif self.path == "/status": out = public_status(value)
            elif self.path.startswith("/operations/"): out = public_operation(value)
            else: out = public_kinematics(value)
            return super()._json(code, out)

        def _rcb1(self, meta, parts):
            svc.log({"ev": "private_response", "path": self.path, "code": 200, "meta": meta,
                     "parts": [{"name": n, "kind": k, **(x or {})} for n, k, _, x in parts]})
            return super()._rcb1(public_meta(meta), public_parts(parts))

    return Handler


def serve_public(svc, port=8000) -> None:
    from http.server import HTTPServer
    svc.spec.setdefault("on_service_error", L.ON_SERVICE_ERROR)
    if svc.episode is not None:
        svc.spec.setdefault("max_call_wall_s", svc.episode.max_call_wall_s)
    L.check_spec(svc.spec)
    HTTPServer(("0.0.0.0", port), public_handler(svc)).serve_forever()


def instability_fields() -> dict:
    """The breakdown evidence and the live episode's evidence: its review events (deep task contacts: a success with any
    is held for review by the grader, rcb_topogym.CONTACT_RULE) and its forcing events (rcb_topogym.FORCING: evidence only)."""
    ready = E.get("ready")
    return {"simulator_instability": E["reason"] == "simulator_instability", "instability": getattr(S, "instability", None),
            "review": S.monitor.review() if ready else None, "forcing": S.monitor.forcing() if ready else None,
            **(getattr(S, "last_step_meta", {}) if ready else {}),
            **({"episode_limits": S._episode_limit_meta["limits"], "episode_notices": S._episode_limit_meta["notices"]}
               if ready and hasattr(S, "_episode_limit_meta") else {})}


def write_final() -> None:
    if L.final_verdict(DATA):           # finalize.sh's verdict, or a give-up after crashes: never overwritten
        return
    n = int(E.get("episode_control_steps") or ctrl_steps())
    L._write_json(os.path.join(DATA, "final.json"), {
        "success": bool(E["success"]), "infra_error": int(E["reason"] in INFRA_REASONS), **instability_fields(), "steps": E["step"], "calls": E["calls"], "control_steps": n, "physics_steps": S.physics_steps, "sim_time_s": S.sim_time_s,
        "elapsed_s": round((E.get("t_end") or time.time()) - E["t0"], 1), "reason": E["reason"], "done": E["done"], "task": TASK,
        "protocol": PROTOCOL, "finalized": False, "n_actions": E.get("n_actions"), "live_rows": E.get("live_rows"),
        "live_sha": E.get("live_sha"), **EP.final_fields()})


def dump() -> None:
    """The episode is over: save the recorded trajectory and the live result for the collect hook (main thread). Rows
    recorded after the live end (the rest of the chunk that ended it, watch holds) were never commanded live: `live_rows`
    tells the replay where the live episode stopped."""
    if E.get("dumped"): return
    E["dumped"] = True
    try:
        acts = S.record_stop(); E["live_sha"] = S.state_sha(); E["n_actions"] = int(len(acts))
        n_live = E.get("end_row"); E["live_rows"] = int(len(acts) if n_live is None else min(int(n_live), len(acts)))
        S.save_trajectory(acts, os.path.join(DATA, "trajectory.npz"), meta={"source": "standard-mode episode (eai-standard/2.2)", "steps": E["step"],
                                                                            "reason": E["reason"], "live_rows": E["live_rows"], "resumes": EP.resumes})
    except Exception as exc:  # noqa
        traceback.print_exc(); log({"ev": "dump_error", "error": repr(exc)})
    n = int(E.get("episode_control_steps") or 0)
    L._write_json(os.path.join(DATA, "episode.json"), {
        "success": bool(E["success"]), "infra_error": int(E["reason"] in INFRA_REASONS), **instability_fields(), "steps": E["step"], "calls": E["calls"], "control_steps": n, "physics_steps": S.physics_steps, "sim_time_s": S.sim_time_s,
        "elapsed_s": round((E.get("t_end") or time.time()) - E["t0"], 1), "reason": E["reason"], "task": TASK, "protocol": PROTOCOL,
        "n_actions": E.get("n_actions"), "live_rows": E.get("live_rows"), "live_sha": E.get("live_sha"), **EP.final_fields()})
    write_final()


def finish(reason: str) -> None:
    if E["done"]: return
    E["done"] = True; E["reason"] = reason; E["t_end"] = time.time(); E["episode_control_steps"] = ctrl_steps()
    E.setdefault("end_row", len(S._recording) if S._recording is not None else None)   # timeout / error / guard ends
    if not E["success"]:
        try: E["success"] = check_success()
        except Exception: pass
    log({"ev": "end", "reason": reason, "step": E["step"], "control_steps": ctrl_steps(), "elapsed_s": round(E["t_end"] - E["t0"], 1), "success": bool(E["success"])})
    dump()


def on_submit_error(exc: Exception) -> None:
    """Keep the released error-ending behavior; the public response is redacted separately."""
    finish("error")


# ---------------------------------------------------------------- scene
import rcb_topogym as M  # noqa: E402
try:
    S = M.Session(TASK, instance=INSTANCE, strict=True)
    M.atomic_json(os.path.join(DATA, "startup_binding.json"), dict(
        instance_exact=S.instance_exact, runtime_binding=S.runtime_binding,
        instance_generated=bool(getattr(S, "instance_generated", False)), instance_sha=S.instance_sha,
        instance_realized=bool(getattr(S, "instance_realized", False)), assets_exact=bool(getattr(S, "assets_exact", True)),
        initial_state=getattr(S, "initial_state", None)))
except M.InstanceMismatch as exc:
    # This frozen instance is not evaluable. Do not restart it as a GPU crash
    # or wait for a trajectory that could never have been executed.
    EP.scene_built()  # cancel the startup watchdog before clean exit
    M.atomic_json(os.path.join(DATA, "final.json"), {
        "success": False, "replay_success": False, "done": True, "finalized": True,
        "instance_mismatch": True, "infra_error": 0, "reason": "instance_mismatch",
        "error": str(exc), "task": TASK, "protocol": PROTOCOL, **EP.final_fields()})
    raise SystemExit(0)
EP.scene_built()                        # the startup watchdog stands down: a long resume replay is not a hang
E["ready"] = True                       # S exists: ctrl_steps() counts its control steps (a resume replay included)
CTRL_HZ = float(S.control_hz); SUBSTEPS = int(S.physics_substeps)
CAMS = list(S.VIDEO_CAMERAS)
CAM_INFO = S.camera_info(CAMS)
MOUNT = {c: ("static" if CAM_INFO[c]["mount"]["type"] == "static" else f"on the robot link {CAM_INFO[c]['mount']['parent_link']}") for c in CAMS}
ARMS = list(S.robot.arms)
LAYOUT = list(S.action_layout)
ARM = {a: (S.robot.arm_slice[a], S.robot.grip_index[a]) for a in ARMS}


def joint_limits() -> dict:
    return {n: [float(l[0]), float(l[1])] for n, l in zip(LAYOUT, S.robot.limits)}   # float32-exact (rcb_topogym): a row at a limit is not clipped
LIMITS = joint_limits(); LIM_ARR = np.array([LIMITS[n] for n in LAYOUT], dtype=np.float64)

R_SPEC = S.robot_spec(); K_SPEC = S.kinematics_spec()

# The agent-facing frame: the robot frame, origin at the robot's
# kinematic root link on its mounting, axes as the simulator's world (+z up). Everything this service gives or takes (proprio
# TCP poses, ee_pose targets, /kinematics, camera extrinsics) is in it; the simulator's own world frame (origin on the floor)
# stays in this process. Physics, the recorded joint rows, graders and state hashes do not depend on it.
BASE_LINK = S.robot.entity.links[0].name
T_WR = M._T(R_SPEC["base"]["pose"][:3], R_SPEC["base"]["pose"][3:])        # robot frame -> world (the root link's pose)
T_RW = np.eye(4); T_RW[:3, :3] = T_WR[:3, :3].T; T_RW[:3, 3] = -T_WR[:3, :3].T @ T_WR[:3, 3]
Q_WR = M.mat_to_quat(T_WR[:3, :3]); Q_RW = Q_WR * np.array([1.0, -1.0, -1.0, -1.0])
R_SPEC["base"] = {"type": "fixed", "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  "note": f"the origin of the robot frame (its kinematic root link {BASE_LINK}; x y z qw qx qy qz)"}


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _moved(T, pos, quat, q_T) -> tuple[np.ndarray, np.ndarray]:
    return T[:3, :3] @ np.asarray(pos, dtype=np.float64) + T[:3, 3], _qmul(q_T, np.asarray(quat, dtype=np.float64))


def to_robot(pos, quat):
    """A world pose (position, quaternion wxyz) -> the robot frame."""
    return _moved(T_RW, pos, quat, Q_RW)


def to_world(pos, quat):
    """A robot-frame pose (position, quaternion wxyz) -> the simulator's world frame."""
    return _moved(T_WR, pos, quat, Q_WR)


def camera_to_robot(info: dict) -> dict:
    """A camera_info entry (or a static mount) with its world extrinsics re-expressed in the robot frame: extrinsic_cv
    (world -> camera) becomes robot -> camera, cam2world_gl (camera -> world) camera -> robot. Intrinsics and a link
    mount's link_T_cam_cv are relative already."""
    out = dict(info)
    if "extrinsic_cv" in out:
        E = np.eye(4); E[:3] = np.asarray(out["extrinsic_cv"], dtype=np.float64)
        out["extrinsic_cv"] = M._r((E @ T_WR)[:3])
    if "cam2world_gl" in out:
        out["cam2world_gl"] = M._r(T_RW @ np.asarray(out["cam2world_gl"], dtype=np.float64))
    if isinstance(out.get("mount"), dict):
        out["mount"] = camera_to_robot(out["mount"])
    return out


EE = [f"{a}_{k}" for a in ARMS for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
N = len(LAYOUT)
PROPRIO_LAYOUT = (LAYOUT + EE + [f"{n}_cmd" for n in LAYOUT] + [f"{n}_vel" for n in LAYOUT]
                  + [f"{a}_gripper_opening_m" for a in ARMS] + [f"{n}_effort" for n in LAYOUT])
PROPRIO_GROUPS = {   # name: [first column, count, meaning]
    "joint_pos": [0, N, "measured encoder positions in the joint_position layout: arm rad; gripper in its declared native or width units"],
    "tcp_pose": [N, 7 * len(ARMS), "measured tool-frame (TCP) pose per arm, in the order of the arms: x y z (m) + qw qx qy qz, robot frame (spec world.frame); "
                                   f"the ee_pose action's frame (kinematics tcp_frames {K_SPEC['tcp_frames']})"],
    "joint_cmd": [N + 7 * len(ARMS), N, "the commanded joint targets being tracked (joint_position layout): the last streamed row step"],
    "joint_vel": [2 * N + 7 * len(ARMS), N, "measured encoder velocities: arm rad/s; gripper coordinate rate in its declared units/s"],
    "gripper_opening": [3 * N + 7 * len(ARMS), len(ARMS), "encoder-derived nominal fingertip gap of each gripper (m)"],
    "joint_effort": [3 * N + 8 * len(ARMS), N, "the queried PD drive-effort estimate: arm and rotary finger Nm, linear finger N; "
                                               "a motor-current analogue, not an object contact force"],
}
WORLD_FRAME = (f"robot frame: origin at the robot's root link ({BASE_LINK}) on its mounting, +z up, the table top horizontal; "
               "every pose, camera extrinsic and kinematics answer of this service is in it (robot_spec.base)")
SPEC = {
    "protocol": PROTOCOL, "protocol_compat": L.COMPAT, "simulator": f"topogym ({S.engine_name(True)}, Nyx RGB)", "robot": R_SPEC["name"],
    "task_text": S.spec["text"], "control_hz": CTRL_HZ, "physics_substeps": SUBSTEPS, "max_chunk_s": MAX_CHUNK_S, "max_watch_s": MAX_WATCH_S,
    "cameras": {c: {"rgb": True, "depth": True, "mounted": MOUNT[c], "resolution": list(CAM_INFO[c]["image_size"]), "calibration_in_obs": OBS_CALIBRATION,
                    "intrinsic_cv": CAM_INFO[c]["intrinsic_cv"], "mount": camera_to_robot(CAM_INFO[c]["mount"]), "depth_range_m": CAM_INFO[c]["depth_range_m"]} for c in CAMS},
    "default_cameras": list(CAMS),
    "action_types": {
        "joint_position": {"dim": N, "layout": LAYOUT, "units": ("rad for arm joints; native gripper drive targets (ALOHA m, Robotiq rad), see robot_spec.grippers" if S.gripper_space == "native" else "rad for the arm joints; gripper opening width in metres, 0 = closed .. the limit = open"),
                           "limits": LIMITS,
                           "semantics": f"absolute joint targets tracked by the robot's PD controllers, streamed at the {CTRL_HZ:g} Hz control rate; rows "
                                        f"`dt` apart (a multiple of 1/{CTRL_HZ:g} s), hold or linear interpolation between rows (the gripper switches "
                                        "at the row); out-of-limit values are clipped and reported. An arm joint target moves at most "
                                        f"{M.JOINT_STEP_LIMIT_RAD:g} rad per control step ({M.JOINT_STEP_LIMIT_RAD * CTRL_HZ:g} rad/s, as ee_pose rows): "
                                        "a row asking for more takes the control steps it needs and is reported in clipped[]. "
                                        f"No gripper target goes lower than {M.table_guard_words()}: a target "
                                        "that would is stopped there (reported in clipped[])"},
        "ee_pose": {"dim": 8 * len(ARMS), "layout": [f"{a}_{k}" for a in ARMS for k in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")],
                    "frame": WORLD_FRAME, "quaternion": "wxyz, unit norm",
                    "gripper_units": "same command space and limits as joint_position / robot_spec.grippers",
                    "semantics": "each row is solved by robot-model inverse kinematics (no scene collision checking) and driven to in joint space; "
                                 "`dt` is ignored (the joint distance sets the duration); an arm whose IK "
                                 "fails holds and is reported in plan[]. A solution more than "
                                 f"{M.IK_JUMP_LIMIT_RAD:g} rad from the current joints on any joint is refused the same way (for a small "
                                 "pose change that is another IK branch, which would swing the arm): split long moves into rows. The "
                                 f"table guard applies as for joint_position (no gripper target lower than {M.table_guard_words()}, "
                                 "reported in clipped[])",
                    "note": "the pose is each gripper's tool frame (TCP; axes in robot_spec.tcp); same convention as the proprio TCP poses, so "
                            "an observed pose is a valid target"},
        "not_yet": ["joint_delta", "joint_velocity", "ee_delta", "joint_torque"]},
    "proprio_layout": PROPRIO_LAYOUT,
    "proprio_groups": PROPRIO_GROUPS,
    "static": {},                       # nothing about where things are (standard mode); the task text names the objects
    "world": {"frame": WORLD_FRAME, "up": "+z", "units": {"length": "m", "angle": "rad", "time": "s"}, "quaternion_order": "wxyz",
              "camera_axes": "intrinsic_cv / extrinsic_cv: OpenCV camera axes (x right, y down, z forward); cam2world_gl: OpenGL axes "
                             "(x right, y up, z backward)"},
    "budget": {"reset": 0, "steps": None, "wall_clock_s": float(os.environ.get("RCB_WALL_CLOCK_S", "10800"))},
    "episode": ("ends at the first success (done, reason success), when a task rule fires (reason task_failed), "
                "when the wall-clock budget is used up (timeout), or on an unrecovered simulator error (error). "
                "Robot command limits are reported in clipped; a recovered command keeps the same episode and budget. "
                "The recorded trajectory is replayed in a fresh process for grading"),
    "robot_spec": R_SPEC,
    "kinematics": K_SPEC,
    "depth": {"modes": list(L.DEPTH_MODES), "default": "none", "kind": "z-depth", "unit": "m", "invalid": 0, "max_m": L.DEPTH_MAX_M,
              "frames_encoding": "uint16 mm, zlib", "noise": L.DEPTH_NOISE,
              "source": "Genesis geometric depth at the same camera rig and pixels as the Nyx RGB (the distance along the optical axis); "
                        "it may differ from the RGB at transparent surfaces"},
}
L.check_spec(SPEC)                      # refuse to start with an incomplete spec or a forbidden field


def check_success() -> bool:
    return S.success()


def public_joint_positions():
    """Encoder readings and the robot's nominal linkage; no task-object or grasp-assist lookup."""
    robot = S.robot; q = robot.measured_full(); row = np.empty(N, dtype=np.float64); openings = []
    for arm in ARMS:
        group = robot.groups[arm]
        row[robot.arm_slice[arm]] = [q[robot.joint[n].qs_idx_local[0]] for n in group.arm_joints]
        native = q[robot.joint[group.actuated_fingers[0]].qs_idx_local[0]]
        width = robot.native_to_width(arm, native)
        row[robot.grip_index[arm]] = native if robot.gripper_space == "native" else width
        openings.append(width)
    return row, openings


def proprio_row() -> np.ndarray:
    q, openings = public_joint_positions()
    ep = [np.concatenate(to_robot(*S.ee_pose(a))) for a in ARMS]
    return np.concatenate([q, *ep, S.hold_action(), S.joint_velocities(), openings, S.joint_efforts()]).astype(np.float32)


def grab(cameras: list[str], want_depth: bool) -> tuple[dict, dict]:
    f = S.env.sensor_frame(depth=want_depth)
    rgb = {c: np.ascontiguousarray(np.asarray(f.rgb[c])[..., :3]) for c in cameras}
    return rgb, ({c: np.asarray(f.depth[c], dtype=np.float32) for c in cameras} if want_depth else {})


def extrinsics(cameras: list[str]) -> dict:
    meta = S.env.sensor_frame().camera_metadata
    return {c: (np.asarray(meta[c]["T_camera_cv_from_world"], dtype=np.float64) @ T_WR)[:3] for c in cameras}


def calib(cameras: list[str]) -> dict:
    return {c: camera_to_robot({k: v[k] for k in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl", "image_size")})
            for c, v in S.camera_info(cameras).items()}


def recorder(cams: list[str], fps: int, meta: dict) -> L.Recorder:
    return L.Recorder(cams, int(round(CTRL_HZ / max(1, fps))), grab, proprio_row, extrinsics, ctrl_steps, depth=L.depth_mode(meta), clock_hz=CTRL_HZ, time_clock=lambda: S.sim_time_s, fps=fps)


def meta_common(cams) -> dict:
    n = ctrl_steps()
    out = {"step": E["step"], "calls": E["calls"], "control_steps": n, "physics_steps": S.physics_steps, "sim_time_s": S.sim_time_s,
            "time_s": round(time.time() - E["t0"], 2), "time_left_s": max(0.0, round(SPEC["budget"]["wall_clock_s"] - (time.time() - E["t0"]), 1)),
            "proprio": {"layout": SPEC["proprio_layout"], "now": proprio_row().tolist()}, "done": E["done"], "reason": E["reason"]}
    diagnostics = M.merge_step_meta(E.get("_call_step_meta", [])) if hasattr(M, "merge_step_meta") else {}
    E["_call_private"] = {"safety": S.safety(), **diagnostics}
    log({"ev": "private_telemetry", "call": E["calls"], **E["_call_private"]})
    if OBS_CALIBRATION: out["cameras"] = calib(cams)
    return out


def save_call_evidence(idx: int, meta: dict, rec: L.Recorder) -> None:
    d = os.path.join(DATA, "calls", f"{idx:06d}"); os.makedirs(d, exist_ok=True)
    json.dump({**meta, **E.get("_call_private", {})}, open(os.path.join(d, "meta.json"), "w"))
    try:
        from PIL import Image
        for c in rec.cams:
            if rec.frames[c] and rec.frames[c][-1] is not None: Image.fromarray(np.ascontiguousarray(rec.frames[c][-1])).save(os.path.join(d, f"last_{c}.jpg"), quality=85)
    except Exception:  # noqa
        pass


# ---------------------------------------------------------------- executing chunks
def clip_joint_row(row: np.ndarray, clipped: list, ri: int) -> np.ndarray:
    r = row.copy()
    for j, name in enumerate(LAYOUT):
        lo, hi = LIM_ARR[j]
        if r[j] < lo or r[j] > hi:
            clipped.append({"row": ri, "index": int(j), "reason": f"{name} limit [{lo:.3f},{hi:.3f}]"}); r[j] = float(np.clip(r[j], lo, hi))
    return r


def ctrl_step(a: np.ndarray, rec: L.Recorder, *, row_start=False, notes: list | None = None, ri: int | None = None) -> bool:
    """One recorded control step. True when the call must stop here: the episode is over (runaway guard, the task's own
    failure predicate, the wall clock), or the call ran past max_call_wall_s / its client hung up (EP.journal). A gripper
    target the table guard stopped (rcb_topogym._TableGuard) is reported in `notes` (meta["clipped"]), once per row and arm."""
    if deadline_reached():
        finish("timeout")
        return True
    row = np.asarray(a, dtype=np.float32)   # exactly what the simulator gets (Session.step casts to float32): the journal row
    if EP.poll_stop():
        return True
    before = S.n_steps
    guard = S.step(row)
    E.setdefault("_call_step_meta", []).append(getattr(S, "last_step_meta", {}))
    rejection = getattr(S, "last_step_rejection", None)
    if rejection is not None:
        E["_step_rejection"] = {**rejection, "row": ri, "control_step": int(S.n_steps)}
        log({"ev": "simulator_rejection", **E["_step_rejection"]})
        return True                      # restored control period: no journal row, episode and budget remain live
    if S.n_steps != before and row_start:
        E["step"] += 1
    if S.n_steps != before and notes is not None:
        for note in _limit_clipped(getattr(S, "last_step_meta", {}).get("limits", []), ri):
            if note not in notes: notes.append(note)
    if S.n_steps != before and S.last_guard and notes is not None:
        for arm in ARMS:
            e = S.last_guard.get(arm)
            reason = f"table guard: the {arm} gripper"
            if e and not any(n["row"] == ri and n["reason"].startswith(reason + " ") for n in notes[-8:]):
                notes.append({"row": ri, "index": int(ARM[arm][1]), "reason": (
                    f"{reason} would have reached {e['asked_mm']:+.1f} mm from the tabletop; its target stopped at "
                    f"{e['stopped_mm']:+.1f} mm (no gripper target goes lower than {M.table_guard_words()})")})
    if S.n_steps != before and getattr(S, "last_static_guard", None) and notes is not None:
        for arm in ARMS:
            e = S.last_static_guard.get(arm)
            reason = f"static-contact guard: the {arm} arm"
            if e and not any(n["row"] == ri and n["reason"].startswith(reason + " ") for n in notes[-8:]):
                notes.append({"row": ri, "index": int(ARM[arm][0].start), "reason": (
                    f"{reason} was {e['depth_mm']:.1f} mm deep in a fixed object; its joint targets stopped where it was "
                    f"(an arm more than {M.STATIC_GUARD_DEPTH_M * 1000:g} mm deep in a fixed object stops pushing)")})
    stop = EP.journal(S.hold_action(), physics_steps=S.last_step_ticks) if S.n_steps != before else EP.poll_stop()
    rec.tick()
    if S.episode_end is not None and "end_row" not in E and S._recording is not None:
        E["end_row"] = len(S._recording)      # the row at which the live episode ended (success or a failure predicate)
    if S.episode_end is not None: finish(S.episode_end)
    elif guard: finish("step_guard")
    return E["done"] or stop


def run_joint_position(A: np.ndarray, dt: float, interp: str, rec: L.Recorder, clipped: list) -> None:
    """Rows `dt` apart (linear interpolation or hold). Joint-speed limit (rules 2026-10-03, as ee_pose rows): an arm joint
    target moves at most rcb_topogym.JOINT_STEP_LIMIT_RAD per control step; a row that asks for more takes the control
    steps it needs (reported in meta["clipped"]), so every row is still reached."""
    steps_per_row = L.control_periods(dt, CTRL_HZ)
    q_prev = np.asarray(S.hold_action(), dtype=np.float64)
    grips = [ARM[a][1] for a in ARMS]
    arm_cols = [j for j in range(N) if j not in grips]
    limit = M.JOINT_STEP_LIMIT_RAD
    for ri, row in enumerate(A):
        target = clip_joint_row(row.astype(np.float64), clipped, ri)
        reach = np.abs(target[arm_cols] - q_prev[arm_cols])
        n = max(steps_per_row, int(math.ceil(float(reach.max(initial=0.)) / limit - 1e-6)))
        slowed = False
        for k in range(1, n + 1):
            a = target.copy() if interp == "hold" or k > steps_per_row else q_prev + (target - q_prev) * (k / steps_per_row)
            a[grips] = target[grips]                          # the gripper switches at the row
            cmd = np.asarray(S.hold_action(), dtype=np.float64)
            d = a[arm_cols] - cmd[arm_cols]
            if np.any(np.abs(d) > limit + 1e-6):
                a[arm_cols] = cmd[arm_cols] + np.clip(d, -limit, limit)
                slowed = True
            if ctrl_step(a, rec, row_start=k == 1, notes=clipped, ri=ri): return
        if slowed:
            j = arm_cols[int(np.argmax(reach))]
            clipped.append({"row": ri, "index": int(j), "reason": (
                f"speed: {LAYOUT[j]} asked to move {float(reach.max()):.3f} rad "
                f"{'at once' if interp == 'hold' else f'in {steps_per_row} control step(s)'}; joint "
                f"targets move at most {limit:g} rad per control step ({limit * CTRL_HZ:g} rad/s), so this row took {n} "
                "control steps")})
        q_prev = target


def run_ee_pose(A: np.ndarray, rec: L.Recorder, plan: list, clipped: list) -> None:
    """IK per row from the measured joints. A solution farther than rcb_topogym.IK_JUMP_LIMIT_RAD from them on any joint
    is refused (rules 2026-10-03: for a small pose change that is another IK branch, which swings the arm): that arm holds
    and plan[] says why, as for a failed IK."""
    for ri, row in enumerate(A):
        cur = np.asarray(S.hold_action(), dtype=np.float64); goal = cur.copy(); st = {"row": ri, "ik": {}}
        for i, arm in enumerate(ARMS):
            sl, gi = ARM[arm]; off = 8 * i
            q = row[off + 3:off + 7]; q = q / max(1e-9, float(np.linalg.norm(q)))
            try:
                qa, ep, er = S.ik(arm, *to_world(row[off:off + 3], q))   # the row is in the robot frame
                info = getattr(S.robot, "last_ik_info", {})
                st["ik"][arm] = {**info, "pos_m": float(ep), "rot_rad": float(er)}
                if ep > 0.02 or er > 0.35:
                    raise ValueError(f"no IK solution within 2 cm / 20 deg (residual {ep * 1000:.0f} mm / {math.degrees(er):.0f} deg)")
                qa = np.clip(qa, LIM_ARR[sl, 0], LIM_ARR[sl, 1])
                jump, joint = S.ik_jump(arm, qa)
                st["ik"][arm].update(max_dq=jump, joint=joint)
                if jump > M.IK_JUMP_LIMIT_RAD:
                    st[arm] = (f"refused: the IK solution is {jump:.2f} rad from the current joints on {joint} (limit "
                               f"{M.IK_JUMP_LIMIT_RAD:g} rad per ee_pose row): for a small pose change that is another IK "
                               "branch, which would swing the arm. Split the motion into rows that change each joint by at "
                               "most that much (for example a few cm and degrees apart), or move the joints with "
                               "joint_position. This arm holds.")
                else:
                    goal[sl] = qa; st[arm] = "Success"
            except Exception as exc:  # noqa: BLE001
                st[arm] = f"failed: {str(exc)[:90]}"
            goal[gi] = float(np.clip(row[off + 7], LIM_ARR[gi, 0], LIM_ARR[gi, 1]))
        plan.append(st)
        n = int(np.clip(math.ceil(float(np.max(np.abs(goal - cur))) / 0.04), 4, 400))   # at most ~0.04 rad per control step
        for k in range(1, n + 1):
            a = cur + (goal - cur) * (k / n)
            for arm in ARMS: a[ARM[arm][1]] = goal[ARM[arm][1]]
            if ctrl_step(a, rec, row_start=k == 1, notes=clipped, ri=ri): return


def do_submit(meta: dict, A: np.ndarray) -> tuple[dict, list]:
    typ = meta.get("type"); dt = float(meta.get("dt", 1.0 / CTRL_HZ)); interp = meta.get("interpolate", "linear")
    cams = [c for c in (meta.get("cameras") or CAMS) if c in CAMS] or CAMS
    fps = int(meta.get("video_fps", 10)); video = meta.get("video", "mp4")
    if typ not in ("joint_position", "ee_pose"): raise ValueError(f"type must be one of joint_position, ee_pose (not yet: {SPEC['action_types']['not_yet']})")
    dim = SPEC["action_types"][typ]["dim"]
    if A.ndim != 2 or A.shape[1] != dim: raise ValueError(f"{typ} actions must be (T, {dim}), got {list(A.shape)}")
    if not np.all(np.isfinite(A)): raise ValueError("actions contain NaN/inf")
    if typ == "joint_position":
        L.control_periods(dt, CTRL_HZ)
    if typ == "joint_position" and A.shape[0] * dt > MAX_CHUNK_S + 1e-9: raise ValueError(f"chunk longer than max_chunk_s={MAX_CHUNK_S}: {A.shape[0]} rows x {dt} s")
    if typ == "ee_pose" and A.shape[0] > 50: raise ValueError("ee_pose: at most 50 rows per submit")
    if interp not in ("hold", "linear"): raise ValueError("interpolate must be hold|linear")
    if video not in ("mp4", "frames", "both", "none"): raise ValueError("video must be mp4|frames|both|none")
    E["_step_rejection"] = None
    E["_call_step_meta"] = []
    rec = recorder(cams, fps, meta); clipped: list = []; plan: list = []
    E["calls"] += 1; step0 = E["step"]
    before = ctrl_steps(); before_ticks = S.physics_steps; rec.tick(force=True)
    if typ == "joint_position": run_joint_position(A, dt, interp, rec, clipped)
    else: run_ee_pose(A, rec, plan, clipped)
    rec.tick(force=True, last=True)
    succ = check_success(); E["success"] = E["success"] or succ
    out = {**meta_common(cams), "chunk_id": E["calls"], "type": typ, "executed_rows": E["step"] - step0, "control_steps_chunk": ctrl_steps() - before,
           "physics_steps_chunk": S.physics_steps - before_ticks, "clipped": clipped[:200], "n_clipped": len(clipped), "plan": plan,
           "rejected": E["_step_rejection"], "ok": E["_step_rejection"] is None}
    log({"ev": "submit", "call": E["calls"], "type": typ, "rows": E["step"] - step0, "dt": dt, "interpolate": interp, "cameras": cams, "note": (meta.get("note") or "")[:500],
         "control_steps": ctrl_steps() - before, "n_clipped": len(clipped), "plan": plan[:50], "success": succ})
    save_call_evidence(E["calls"], {k: v for k, v in out.items() if k != "cameras"}, rec)
    if succ: finish("success")
    out["done"] = E["done"]; out["reason"] = E["reason"]
    return out, rec.parts(video, fps)


def do_observe(meta: dict) -> tuple[dict, list]:
    cams = [c for c in (meta.get("cameras") or CAMS) if c in CAMS] or CAMS
    seconds = float(meta.get("seconds", 0)); fps = int(meta.get("video_fps", 10)); video = meta.get("video", "frames")
    if not math.isfinite(seconds) or seconds < 0 or seconds > MAX_WATCH_S: raise ValueError(f"seconds must be in [0, {MAX_WATCH_S}]")
    if video not in ("mp4", "frames", "both", "none"): raise ValueError("video must be mp4|frames|both|none")
    E["_step_rejection"] = None
    E["_call_step_meta"] = []
    rec = recorder(cams, fps, meta); before = ctrl_steps(); before_ticks = S.physics_steps
    watch = seconds > 0 and not E["done"]
    rec.tick(force=True, last=not watch)
    if watch:
        hold = np.asarray(S.hold_action(), dtype=np.float64)
        for _ in range(int(round(seconds * CTRL_HZ))):
            if ctrl_step(hold, rec): break
        rec.tick(force=True, last=True)
        succ = check_success(); E["success"] = E["success"] or succ
        if succ: finish("success")
    E["calls"] += 1
    out = {**meta_common(cams), "obs_id": E["calls"], "seconds": seconds, "control_steps_watch": ctrl_steps() - before,
           "physics_steps_watch": S.physics_steps - before_ticks, "ok": E["_step_rejection"] is None,
           **({"rejected": E["_step_rejection"]} if E["_step_rejection"] is not None else {})}
    log({"ev": "observe", "call": E["calls"], "seconds": seconds, "cameras": cams, "control_steps": ctrl_steps() - before})
    if seconds > 0: save_call_evidence(E["calls"], {k: v for k, v in out.items() if k != "cameras"}, rec)
    return out, rec.parts(video if seconds > 0 else ("frames" if video == "mp4" else video), fps)


def do_kinematics(req: dict) -> dict:
    """/kinematics (validated by rcb_limited.parse_kinematics): the tooling's stateless FK / IK, never the simulator; poses
    in the robot frame (world.frame), mapped to and from the simulator's world here."""
    if req.get("op") == "ik":
        req = {**req, "targets": {f: np.r_[to_world(p[:3], p[3:])] for f, p in req["targets"].items()}}
    out = S.kinematics(req)
    if out.get("op") == "fk":
        out["poses"] = {l: [float(v) for v in np.r_[to_robot(p[:3], p[3:])]] for l, p in out["poses"].items()}
    return out


def status() -> dict:
    return {"step": E["step"], "calls": E["calls"], "control_steps": ctrl_steps(), "done": E["done"], "reason": E["reason"],
            "elapsed_s": round(time.time() - E["t0"], 1)}


def deadline_reached():
    return E["ready"] and not E["done"] and time.monotonic() >= E["deadline"]


def end_from_signal() -> None:
    """The collect hook's SIGUSR1 (rcb_limited.Episode runs this in a thread): the main thread ends the episode, dumps it
    and exits 0 (_end_and_exit), so the GPU is free for finalize.sh's fresh-process replay."""
    JOBS.put(_end_and_exit)


def _end_and_exit() -> None:
    """Main thread, between simulator jobs. No svc.lock here: an HTTP call holds it while it waits for the main thread."""
    E["exiting"] = True                     # the HTTP layer answers 409 from now on
    if not E["done"]: finish("timeout")
    else: dump()
    log({"ev": "exit", "reason": "collect hook: episode dumped, releasing the GPU for the fresh-process replay"})
    if not os.path.exists(PID_FILE):        # a SIGUSR1 that came in during a resume: the hook must still see this process end
        open(PID_FILE, "w").write(str(os.getpid()))
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def replay_row(row) -> None:
    """One journaled row during a resume (main thread, no rendering asked for). As live: after the task's own episode end
    Session.step records the row without moving the scene, and the row of that end is remembered (`live_rows`)."""
    if not S.episode_done:
        S.step(np.asarray(row, dtype=np.float32))
        if S.episode_end is not None and "end_row" not in E and S._recording is not None:
            E["end_row"] = len(S._recording)


# ---------------------------------------------------------------- start, or resume after a crash
E["deadline"] = float("inf")
S.record_start()                        # before begin(): a resume replay of the journal is recorded too
EP.begin(step_ticks=lambda row, ticks: S.step(row, _physics_steps=ticks), state=E, step=replay_row, state_sha=S.state_sha, proprio=proprio_row, spec=SPEC, on_end=end_from_signal,
         start_event={"task": TASK, "eval_index": S.eval_index})
E["deadline"] = time.monotonic() + SPEC["budget"]["wall_clock_s"] - (time.time() - E["t0"])   # the wall clock runs from the first start
write_final()
open(PID_FILE, "w").write(str(os.getpid()))
print(f"{PROTOCOL} sim ready (startup {EP.n_startups}, resumes {EP.resumes})", flush=True)
threading.Thread(target=serve_public, daemon=True, args=(L.Service(
    spec=SPEC, lock=LOCK, state=E, submit=lambda meta, A: on_main(do_submit, meta, A), observe=lambda meta: on_main(do_observe, meta),
    kinematics=lambda req: on_main(do_kinematics, req), status=status, on_submit_error=lambda exc: on_main(on_submit_error, exc), log=log,
    episode=EP, state_sha=lambda: on_main(S.state_sha), proprio=lambda: on_main(proprio_row)),)).start()
while True:                                 # the main thread: simulator jobs in order; signals are handled between them
    if deadline_reached():
        finish("timeout")
    try:
        job = JOBS.get(timeout=0.5)
    except queue.Empty:
        continue
    job()
