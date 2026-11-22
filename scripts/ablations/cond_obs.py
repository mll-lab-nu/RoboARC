"""E5's observation conditions, edits of a Standard task: is the Standard regime realistic enough?

- V1 (`obsv1`): C00 (the per-call log patch, scripts/ablations/cond_info.py) without the contact-safety readings: no
  response of the service carries `meta["safety"]` (forcing, contact depths, review events, volume ratios, table-guard
  stops), and neither the instruction nor /spec promises them. The service still measures and records them exactly as
  the Standard task does (episode.jsonl, the call evidence, the grader's review of a success with a deep contact): only
  the agent does not get them. A real robot has its joint efforts (still in the proprio) but no contact sensor.
- V2 (`obsv2`): V1 plus the errors of a real RGB-D camera, on the observation side only:
  - depth: per-pixel Gaussian noise whose standard deviation grows with the square of the range, Intel's stereo
    RMS-error model for the D400 cameras, sigma(z) = z^2 * subpixel / (f_px * baseline), then rounded to the camera's
    depth unit, with the datasheet's Min-Z and fill rate; the wrist cameras as a RealSense D405, the static cameras as
    a D435i (the real-robot setup);
  - calibration: every camera's stated intrinsics and extrinsics carry a fixed error, drawn from a fixed seed and the
    camera's name (the same in every episode and every task: one rig, calibrated once): the stated pose is off by a rotation of CALIB["rot_deg"] about a random axis
    and a translation of CALIB["trans_mm"] in a random direction (in the camera frame, so a wrist camera's hand-eye
    offset `link_T_cam_cv` and a static camera's pose alike), the stated focal lengths by CALIB["focal_rel"] (relative,
    random sign). The images are rendered by the true camera; every calibration the service states (spec cameras, meta
    cameras, the per-frame extrinsic_cv) is the erroneous one, consistently.
  The robot, the physics, the recorded rows, the grader and the fresh-process replay are unchanged (environment/sim/
  replay.py never imports the service).

The sensor numbers and their sources are in SENSORS and CALIB below and in docs/ABLATIONS.md."""
from __future__ import annotations

import math
import pathlib
import re

import cond_info

LEVELS = (1, 2)

# ---------------------------------------------------------------- V1: no contact-safety readings
V1_FN_ANCHOR = "# ---------------------------------------------------------------- start, or resume after a crash\n"
V1_FN = '''# ---------------------------------------------------------------- ablation E5 V1: no contact-safety readings
def unfelt(res):
    """ablation E5 V1: a /submit or /observe response without meta["safety"]. The service still measures and records the
    readings (episode.jsonl, the call evidence, the grader's review), exactly as the Standard task does; only the agent
    does not get them."""
    out, parts = res
    out.pop("safety", None)
    return out, parts


'''
V1_SERVICE_OLD = "submit=lambda meta, A: on_main(do_submit, meta, A), observe=lambda meta: on_main(do_observe, meta),"
V1_SERVICE_NEW = ("submit=lambda meta, A: unfelt(on_main(do_submit, meta, A)), observe=lambda meta: unfelt(on_main(do_observe, meta)),"
                  "   # ablation E5 V1")
V1_SPEC_OLD = ('"it (meta safety records them), but a success with a deep task contact (meta safety review) is held for review "')
V1_SPEC_NEW = ('"it (the service records them; this variant does not report them), but a success with a deep task contact (a review '
               'event in that record) is held for review "')
V1_RULE_OLD = (r"The success implementation is not exposed, and of the simulator's internal\s+measurements only the safety readings of "
               r"`meta\[\"safety\"\]` are \(listed under Action types, below\); the task\s+requirements above still apply in full\.")
V1_RULE_NEW = ("The success implementation is not exposed, and none of the simulator's internal\n  measurements is: in this variant "
               "not even its contact-safety readings (no response reports contact forces,\n  contact depths or forcing); the task "
               "requirements above still apply in full.")
V1_ACTION_OLD = (r"Every response's `meta\[\"safety\"\]` reports, since the\s+previous call, forcing \(the arms' pressing loads and "
                 r"forcing events\),.*?against the limits above\.")
V1_ACTION_NEW = ("In this variant no response\n  reports the contact-safety readings (forcing, contact depths, review events, "
                 "volume ratios, table-guard stops):\n  the service still measures and records them for the grader and the "
                 "reviewer, as described above, but you do\n  not see them.")
V1_EVIDENCE = ("a forcing event in the evidence (`meta[\"safety\"]`), shown to the reviewer",
               "a forcing event in the evidence (recorded by the service; this variant does not report it to you), shown to the "
               "reviewer")
V1_GUARD = ("(the safety readings report these stops as static_guard)",
            "(the service records these stops as static_guard; this variant does not report them)")
V1_LEFT = r"meta\[\"safety\"\]|safety readings of|meta safety"


def patch_v1(src: pathlib.Path, dst: pathlib.Path) -> None:
    name = src.name
    srv = dst / "environment" / "sim" / "server.py"
    t = srv.read_text()
    t = cond_info.sub1(t, V1_FN_ANCHOR, V1_FN + V1_FN_ANCHOR, f"{name} server.py V1 block")
    t = cond_info.sub1(t, V1_SERVICE_OLD, V1_SERVICE_NEW, f"{name} server.py Service")
    t = cond_info.sub1(t, V1_SPEC_OLD, V1_SPEC_NEW, f"{name} server.py spec episode")
    compile(t, str(srv), "exec")
    srv.write_text(t)
    ins = (dst / "instruction.md").read_text()
    ins = cond_info.sub1(ins, V1_RULE_OLD, V1_RULE_NEW, f"{name} instruction (Rules)", regex=True)
    ins = re.sub(V1_ACTION_OLD, lambda m: V1_ACTION_NEW, ins, count=1, flags=re.S) if len(
        re.findall(V1_ACTION_OLD, ins, flags=re.S)) == 1 else _fail(f"{name} instruction (Action types): no safety paragraph")
    ins = cond_info.sub1(ins, *V1_EVIDENCE, f"{name} instruction (forcing evidence)")
    if V1_GUARD[0] in ins:                       # not every family's text names the static guard's record
        ins = cond_info.sub1(ins, *V1_GUARD, f"{name} instruction (static guard)")
    left = re.findall(V1_LEFT, ins)
    if left:
        raise SystemExit(f"{name}: the instruction still promises safety readings: {left}")
    (dst / "instruction.md").write_text(ins)
    toml = (dst / "task.toml").read_text()
    toml = cond_info.sub1(toml, "no privileged state", "no privileged state, no contact-safety readings",
                          f"{name} task.toml description (V1)")
    (dst / "task.toml").write_text(toml)


def _fail(msg: str):
    raise SystemExit(msg)


# ---------------------------------------------------------------- V2: a real RGB-D camera's depth noise and calibration error
# Sources (docs/ABLATIONS.md, "E5: the sensor model", has the quotes):
# [DS]  RealSense D400 Series Datasheet, rev. 023 (Mar 2026; document 337029): baselines 50 mm (D435/D435i) and 18 mm
#       (D405), Table 3-49; imager HFOV 90 deg, Tables 3-13 / 3-15; Min-Z at 848x480 195 mm (D43x) and 70 mm (D405),
#       Table 4-12; fill rate >= 99 % (D43x) and >= 99.5 % (D405), Table 4-15.
# [BKM] Grunnet-Jepsen et al., Best-Known-Methods for Tuning Intel RealSense D400 Depth Cameras, rev. 1.9, sec. 3e: depth
#       RMS error = distance^2 x subpixel / (focal length [px] x baseline), focal length = (Xres / 2) / tan(HFOV / 2); the
#       example "D435 with HFOV=90deg, Xres=848, and baseline=50mm ... subpixel=0.08"; sec. 8a: the D4 depth unit, 1 mm.
# [SDK] librealsense src/ds/d400/d400-device.cpp: the D405's default depth unit, 0.0001 m.
# [SC]  Grunnet-Jepsen et al., Intel RealSense Self-Calibration for D400 Series, rev. 2.7, Addendum A: focal-length errors
#       "less than ~0.3%".
# [HEC] hand-eye calibration of a RealSense D435 on a robot arm: 0.48 deg / 2.04 mm (Wang et al., PlaneHEC,
#       arXiv:2507.19851, sec. IV); ViSP's extrinsic-calibration tutorial (Panda / UR5 + D435), 0.60 deg / 1.1 mm.
SENSORS = {
    "D405": {"baseline_m": 0.018, "hfov_deg": 90.0, "subpixel": 0.08, "unit_m": 0.0001, "min_z_m": 0.070, "fill": 0.995},
    "D435i": {"baseline_m": 0.050, "hfov_deg": 90.0, "subpixel": 0.08, "unit_m": 0.001, "min_z_m": 0.195, "fill": 0.99},
}
WIDTH_PX = 848
# The calibration error of a calibrated real camera ([SC], [HEC]): one fixed draw per camera from V2_SEED.
CALIB = {"rot_deg": 0.5, "trans_mm": 2.0, "focal_rel": 0.003}
V2_SEED = 20261118


def sigma_m(sensor: str, z_m: float) -> float:
    """The datasheet's depth RMS error (m) of `sensor` at range z_m (m)."""
    s = SENSORS[sensor]
    f_px = (WIDTH_PX / 2) / math.tan(math.radians(s["hfov_deg"]) / 2)
    return z_m * z_m * s["subpixel"] / (f_px * s["baseline_m"])


V2_FN_ANCHOR = '\nEE = [f"{a}_{k}" for a in ARMS for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]\n'
V2_FN = r'''
# ---------------------------------------------------------------- ablation E5 V2: a real RGB-D camera (observation only)
# Depth: per-pixel Gaussian noise, sigma(z) = z^2 * subpixel / (f_px * baseline) (Intel's D400 RMS-error model), then
# rounded to the camera's depth unit; no reading closer than the camera's Min-Z, and the datasheet's fill rate (a valid
# pixel reads 0 with probability 1 - fill); the wrist cameras (on a robot link) as a RealSense D405, the static cameras
# as a D435i. Calibration: each camera's stated intrinsics and extrinsics carry a fixed error drawn once from V2_SEED
# (the same in every episode): the stated pose is the true one moved, in the camera frame, by a rotation of V2_CALIB
# rot_deg about a random axis and a translation of trans_mm in a random direction; the stated focal lengths are scaled
# by 1 + focal_rel or 1 - focal_rel. Images are rendered by
# the true camera; every calibration this service states (spec cameras, meta cameras, the per-frame extrinsic_cv) is the
# erroneous one. Nothing here touches physics, the recorded rows, the grader or the fresh-process replay.
import zlib
V2_SEED = @SEED@
V2_SENSORS = @SENSORS@
V2_WIDTH_PX = @WIDTH@
V2_CALIB = @CALIB@
V2_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def _v2_sensor(c: str) -> dict:
    s = V2_SENSORS["D435i" if CAM_INFO[c]["mount"]["type"] == "static" else "D405"]
    return {**s, "f_px": (V2_WIDTH_PX / 2) / math.tan(math.radians(s["hfov_deg"]) / 2)}


def _v2_draw(c: str) -> dict:
    rng = np.random.default_rng([V2_SEED, zlib.crc32(c.encode())])
    axis = rng.standard_normal(3); axis /= np.linalg.norm(axis)
    move = rng.standard_normal(3); move /= np.linalg.norm(move)
    th = math.radians(V2_CALIB["rot_deg"])
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    T = np.eye(4); T[:3, :3] = np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)
    T[:3, 3] = move * V2_CALIB["trans_mm"] / 1000.0
    sign = 1.0 if rng.random() < 0.5 else -1.0
    Ti = np.eye(4); Ti[:3, :3] = T[:3, :3].T; Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return {"T": T, "T_inv": Ti, "focal": 1.0 + sign * V2_CALIB["focal_rel"]}


V2_ERR = {c: _v2_draw(c) for c in CAMS}
V2_SENSOR = {c: _v2_sensor(c) for c in CAMS}
_V2_FRAMES = {c: 0 for c in CAMS}


def v2_K(c: str, K) -> list:
    """The stated intrinsics: the true ones with the camera's focal-length error (fx, fy)."""
    K = np.array(K, dtype=np.float64); e = V2_ERR[c]
    K[0, 0] *= e["focal"]; K[1, 1] *= e["focal"]
    return M._r(K)


def v2_extrinsic(c: str, E) -> np.ndarray:
    """A stated extrinsic_cv (3x4, frame -> camera): the true one with the camera's pose error (world_T_cam @ T)."""
    E4 = np.eye(4); E4[:3] = np.asarray(E, dtype=np.float64)[:3]
    return (V2_ERR[c]["T_inv"] @ E4)[:3]


def v2_info(c: str, info: dict) -> dict:
    """A camera_info entry (or a mount) as stated: intrinsic_cv, extrinsic_cv, cam2world_gl and link_T_cam_cv with the
    camera's calibration error; everything else as it is."""
    out = dict(info)
    if "intrinsic_cv" in out:
        out["intrinsic_cv"] = v2_K(c, out["intrinsic_cv"])
    if "extrinsic_cv" in out:
        out["extrinsic_cv"] = M._r(v2_extrinsic(c, out["extrinsic_cv"]))
    if "cam2world_gl" in out:
        out["cam2world_gl"] = M._r(np.asarray(out["cam2world_gl"], dtype=np.float64) @ V2_GL2CV @ V2_ERR[c]["T"] @ V2_GL2CV)
    if "link_T_cam_cv" in out:
        L4 = np.eye(4); a = np.asarray(out["link_T_cam_cv"], dtype=np.float64); L4[:a.shape[0]] = a
        out["link_T_cam_cv"] = M._r((L4 @ V2_ERR[c]["T"])[:a.shape[0]])
    if isinstance(out.get("mount"), dict):
        out["mount"] = v2_info(c, out["mount"])
    return out


def v2_depth(c: str, z) -> np.ndarray:
    """A depth map as the camera measures it: Gaussian noise of the RMS error at each valid pixel's range, drawn anew for
    every frame (reproducible: the camera and its frame count seed it), then the camera's depth unit; no reading closer
    than Min-Z or at a dropped pixel (1 - fill rate); a pixel with no reading stays without one."""
    z = np.array(z, dtype=np.float32)
    ok = np.isfinite(z) & (z > 0)
    s = V2_SENSOR[c]; _V2_FRAMES[c] += 1
    rng = np.random.default_rng([V2_SEED, zlib.crc32(c.encode()), _V2_FRAMES[c]])
    zz = z[ok].astype(np.float64)
    near = zz < s["min_z_m"]
    zz = zz + rng.standard_normal(zz.shape) * (zz * zz * s["subpixel"] / (s["f_px"] * s["baseline_m"]))
    zz = np.round(zz / s["unit_m"]) * s["unit_m"]
    zz[near | (zz <= 0) | (rng.random(zz.shape) >= s["fill"])] = 0.0   # no reading
    z[ok] = zz.astype(np.float32)
    return z

'''
def v2_block() -> str:
    """The V2 code that goes into server.py, with this module's numbers."""
    return (V2_FN.replace("@SEED@", repr(V2_SEED)).replace("@SENSORS@", repr(SENSORS)).replace("@WIDTH@", repr(WIDTH_PX))
            .replace("@CALIB@", repr(CALIB)))


V2_SPEC_CAM_OLD = ('"intrinsic_cv": CAM_INFO[c]["intrinsic_cv"], "mount": camera_to_robot(CAM_INFO[c]["mount"]), '
                   '"depth_range_m": CAM_INFO[c]["depth_range_m"]} for c in CAMS},')
V2_SPEC_CAM_NEW = ('"intrinsic_cv": v2_K(c, CAM_INFO[c]["intrinsic_cv"]), "mount": v2_info(c, camera_to_robot(CAM_INFO[c]["mount"])), '
                   '"depth_range_m": CAM_INFO[c]["depth_range_m"],\n'
                   '                    "sensor_model": f"RealSense {\'D435i\' if CAM_INFO[c][\'mount\'][\'type\'] == \'static\' else \'D405\'} '
                   '(ablation E5 V2: depth noise and a calibration error, see the task text)"} for c in CAMS},')
V2_DEPTH_SPEC_OLD = '"noise": L.DEPTH_NOISE,'
V2_DEPTH_SPEC_NEW = ('"noise": "ablation E5 V2: RealSense-like, sigma(z) = z^2 * subpixel / (f_px * baseline) per pixel and frame, then the '
                     'camera\'s depth unit (see the task text)",')
V2_EXT_OLD = 'return {c: (np.asarray(meta[c]["T_camera_cv_from_world"], dtype=np.float64) @ T_WR)[:3] for c in cameras}'
V2_EXT_NEW = 'return {c: v2_extrinsic(c, (np.asarray(meta[c]["T_camera_cv_from_world"], dtype=np.float64) @ T_WR)[:3]) for c in cameras}   # V2'
V2_CALIB_OLD = '    return {c: camera_to_robot({k: v[k] for k in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl", "image_size")})'
V2_CALIB_NEW = '    return {c: v2_info(c, camera_to_robot({k: v[k] for k in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl", "image_size")}))   # V2'
V2_GRAB_OLD = 'return rgb, ({c: np.asarray(f.depth[c], dtype=np.float32) for c in cameras} if want_depth else {})'
V2_GRAB_NEW = 'return rgb, ({c: v2_depth(c, f.depth[c]) for c in cameras} if want_depth else {})   # ablation E5 V2'
V2_INS_ANCHOR = "\n### Action types\n"


def v2_text() -> str:
    """The instruction's paragraph on the camera model, with the numbers of SENSORS and CALIB."""
    ex = lambda s, z: f"{sigma_m(s, z) * 1000:.2g} mm at {z:g} m"
    return (
        "\n**Camera model (this variant).** The cameras behave like real RGB-D cameras: the wrist cameras like an Intel "
        "RealSense D405, the static cameras like a RealSense D435i. Their depth has those cameras' noise: every valid pixel's "
        "depth z gets Gaussian noise with the standard deviation the camera's datasheet gives for its RMS error, "
        "sigma(z) = z² · 0.08 / (f · B) with f = "
        f"{(WIDTH_PX / 2) / math.tan(math.radians(90.0) / 2):.0f} px and the stereo baseline B = 18 mm (D405) or 50 mm "
        f"(D435i) ({ex('D405', 0.2)} and {ex('D405', 0.5)} for the D405; {ex('D435i', 0.5)} and {ex('D435i', 1.0)} for the "
        "D435i), drawn anew for every frame, and is then rounded to the camera's depth unit (0.1 mm for the D405, 1 mm for "
        "the D435i). As on those cameras, there is no reading (0) closer than 70 mm (D405) or 195 mm (D435i), and 0.5 % "
        "(D405) or 1 % (D435i) of the other pixels have none either; a pixel with no reading stays 0. The camera calibration "
        "this service states (`spec[\"cameras\"]`, "
        "`meta[\"cameras\"]`, the per-frame `extrinsic_cv`) is that of a calibrated real camera, not the exact one: each "
        f"camera's stated pose (a wrist camera's `link_T_cam_cv`, a static camera's pose) is off by a rotation of "
        f"{CALIB['rot_deg']:g}° about an unknown axis and {CALIB['trans_mm']:g} mm in an unknown direction, its stated "
        f"focal lengths by {CALIB['focal_rel'] * 100:g} %, the same in every call. The images themselves come from the true cameras. The robot, the physics and the grading are "
        "those of the standard task.\n")


V2_LEFT = r"(?i)\bexact by default\b"


def patch_v2(src: pathlib.Path, dst: pathlib.Path) -> None:
    name = src.name
    srv = dst / "environment" / "sim" / "server.py"
    t = srv.read_text()
    t = cond_info.sub1(t, V2_FN_ANCHOR, v2_block() + V2_FN_ANCHOR, f"{name} server.py V2 block")
    for old, new, what in ((V2_SPEC_CAM_OLD, V2_SPEC_CAM_NEW, "spec cameras"), (V2_DEPTH_SPEC_OLD, V2_DEPTH_SPEC_NEW, "spec depth"),
                           (V2_EXT_OLD, V2_EXT_NEW, "extrinsics"), (V2_CALIB_OLD, V2_CALIB_NEW, "calib"),
                           (V2_GRAB_OLD, V2_GRAB_NEW, "grab")):
        t = cond_info.sub1(t, old, new, f"{name} server.py {what}")
    if not re.search(r"(?m)^import [^\n]*\bmath\b", t):      # the V2 block uses math and np, as the service does
        raise SystemExit(f"{name}: server.py does not import math")
    compile(t, str(srv), "exec")
    srv.write_text(t)
    ins = (dst / "instruction.md").read_text()
    ins = cond_info.sub1(ins, V2_INS_ANCHOR, v2_text() + V2_INS_ANCHOR, f"{name} instruction (camera model)")
    (dst / "instruction.md").write_text(ins)
    toml = (dst / "task.toml").read_text()
    toml = cond_info.sub1(toml, "no contact-safety readings", "no contact-safety readings, RealSense-like depth noise and "
                          "calibration error", f"{name} task.toml description (V2)")
    (dst / "task.toml").write_text(toml)


def make(src: pathlib.Path, dst: pathlib.Path, level: int) -> None:
    """Edit dst (a copy of the Standard task src) into V1 (level 1) or V2 (level 2)."""
    if level not in LEVELS:
        raise ValueError(f"E5 observation level must be 1 or 2, not {level!r}")
    cond_info.patch_log(src, dst)
    patch_v1(src, dst)
    if level == 2:
        patch_v2(src, dst)


def leak_check(dst: pathlib.Path) -> list[str]:
    """What V1 / V2 must not have: a promise of safety readings in the text, a response path that keeps them."""
    srv = (dst / "environment" / "sim" / "server.py").read_text()
    ins = (dst / "instruction.md").read_text()
    out = []
    if re.search(V1_LEFT, ins):
        out.append("the instruction promises meta['safety']")
    if srv.count("unfelt(on_main(") != 2:
        out.append("server.py: /submit and /observe are not both stripped of meta['safety']")
    return out
