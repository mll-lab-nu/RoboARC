"""The information conditions of the RoboARC ablation (E1, E2): C00, C10, C0G and C1G, edits of a Standard task.

Ported from the 2026-11 ablation's variant tool (`make_variant.py`, variants q1c00, q1c10, q1c0g, q1c1g). Every
string below that ends up in a generated task is the tool's own, character for character, so a variant made here is
byte-identical to the one the original ablation ran (scripts/ablations/tests/test_cond_info.py checks the edits;
`digests/<token>.tsv` holds every variant's DIGEST).

- C00 (`q1c00`): the Standard task with the per-call log patch only: episode.jsonl also records each call's depth mode,
  video, fps, the observe note and the call's safety readings. The agent sees nothing new; every condition carries it,
  so the conditions differ only in the factor under test.
- C10 (`q1c10`): C00 + the task objects' poses in every response's `meta["objects"]` (Session.objects() moved into the
  agent-facing robot frame: never the simulator's world frame, which would disclose the table height; a fixed object
  by its bounds only).
- C0G (`q1c0g`): C00 + `GET /geometry` (rcb_client `Robot.geometry()`): every task part's collision shapes, each in
  the part's own frame, so the answer holds no position of the scene.
- C1G (`q1c1g`): C10 + `GET /geometry` in the agent-facing frame (a moving link where it stands at the start); no
  internal state (`meta["state"]` is never sent).

Physics, grader, frozen instance and seeds stay the base task's. Every edit must apply exactly once: anything
unexpected raises, and scripts/ablations/make_variants.py then removes the half-made directory."""
from __future__ import annotations

import pathlib
import re

CONDITIONS = ("q1c00", "q1c10", "q1c0g", "q1c1g")


def sub1(text: str, old: str, new: str, what: str, regex: bool = False) -> str:
    n = len(re.findall(old, text)) if regex else text.count(old)
    if n != 1:
        raise SystemExit(f"{what}: expected 1 match, found {n}")
    return re.sub(old, lambda m: new, text) if regex else text.replace(old, new)


# ---- the Standard-service patches every condition is built from ("probing v2" in the generated code: the 2026-11
# ablation's name for these conditions, kept so that the generated files stay byte-identical to the ones it ran)
# LOG: every condition, C00 included, logs the same per-call detail (depth mode, video, fps, the observe note, the
#      call's safety readings) in episode.jsonl, so the conditions differ only in the factor under test.
# OBJ: C10 / C1G: every response's meta["objects"] = Session.objects() in the agent-facing frame (the robot frame that
#      every other pose of this service uses: to_robot), never the simulator's world frame (that would be off by the root
#      link's pose and disclose the table height).
LOG_SUBMIT_OLD = '"cameras": cams, "note": note,'
LOG_SUBMIT_NEW = ('"cameras": cams, "note": note, "depth": L.depth_mode(meta), "video": video, "video_fps": fps,\n'
                  '         "safety": out.get("safety"),   # probing v2 log patch')
LOG_OBS_FIELDS_OLD = "    cams, fps, _ = request_fields(meta)\n    seconds = number(meta"
LOG_OBS_FIELDS_NEW = "    cams, fps, note = request_fields(meta)   # probing v2 log patch: the note is logged\n    seconds = number(meta"
LOG_OBS_OLD = 'log({"ev": "observe", "call": E["calls"], "seconds": seconds, "cameras": cams, "control_steps": ctrl_steps() - before})'
LOG_OBS_NEW = ('log({"ev": "observe", "call": E["calls"], "seconds": seconds, "cameras": cams, "control_steps": ctrl_steps() - before,\n'
               '         "note": note, "depth": L.depth_mode(meta), "video": video, "video_fps": fps,\n'
               '         "safety": out.get("safety")})   # probing v2 log patch')
OBJ_META_OLD = '"safety": robot_felt(S.safety()), **({"cameras": calib(cams)} if OBS_CALIBRATION else {})}'
OBJ_META_NEW = ('"safety": robot_felt(S.safety()), **({"cameras": calib(cams)} if OBS_CALIBRATION else {}),\n'
                '            "objects": objects_robot()}   # probing v2 C10: the task objects\' poses, robot frame')
OBJ_FN_ANCHOR = "\ndef meta_common(cams) -> dict:\n"
OBJ_FN = r"""
def objects_robot() -> dict:
    # probing v2 C10: Session.objects() moved into the agent-facing frame (the robot frame, as every pose here): position
    # and orientation by to_robot, the axis-aligned bounds as the box round the moved corners, velocity rotated.
    from scipy.spatial.transform import Rotation
    rot, out = T_RW[:3, :3], {}
    for n, o in S.objects().items():
        o = dict(o)
        q = o.get("orientation_wxyz")
        p, q2 = to_robot(o["position"], q if q is not None else [1.0, 0.0, 0.0, 0.0])
        o["position"] = [round(float(x), 5) for x in p]
        if q is not None:
            o["orientation_wxyz"] = [round(float(x), 6) for x in q2]
            o["euler_xyz"] = [round(float(x), 5) for x in Rotation.from_quat(np.asarray(q2)[[1, 2, 3, 0]]).as_euler("xyz")]
        if "bounds" in o:
            lo, hi = np.asarray(o["bounds"], dtype=np.float64)
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ rot.T + T_RW[:3, 3]
            o["bounds"] = [[round(float(x), 5) for x in c.min(0)], [round(float(x), 5) for x in c.max(0)]]
            if "size" in o:
                o["size"] = [round(float(x), 5) for x in c.max(0) - c.min(0)]
        if "velocity" in o:
            o["velocity"] = [round(float(x), 5) for x in rot @ np.asarray(o["velocity"], dtype=np.float64)]
        if o.get("fixed") and "bounds" in o:
            # a fixed body's origin can lie anywhere (the goal marker's is on the floor: its z would disclose the table
            # height): a fixed object is reported by its bounds only, with their centre as its position
            lo2, hi2 = np.asarray(o["bounds"], dtype=np.float64)
            o["position"] = [round(float(x), 5) for x in (lo2 + hi2) / 2]
            for k in ("orientation_wxyz", "euler_xyz", "velocity"):
                o.pop(k, None)
        out[n] = o
    return out

"""
OBJ_RULE_OLD = r"Nothing tells you where the\s+objects are \(no object poses or positions\)\. "
OBJ_RULE_NEW = ("In this variant the task objects' poses are given as well: every response's `meta[\"objects\"]` "
                "(name -> position, orientation_wxyz, euler_xyz, bounds, size, velocity, fixed; for a cloth, rope or "
                "other deformable object its points' centroid, bounding box, mean velocity and n_points; a fixed object "
                "only its bounds, size and the bounds' centre as its position), in the world "
                "frame of `spec[\"world\"]`, the frame of the TCP poses and camera extrinsics. ")
OBJ_PARA_OLD = r" As everywhere in this mode, nothing reports their poses[^\n]*"
OBJ_PARA_NEW = (" Their poses are in every response's `meta[\"objects\"]` (see Rules); nothing else about them is reported "
                "(no shapes, no internal states such as a thread's or a mechanism's): the rest comes from the cameras.")


def patch_log(src: pathlib.Path, dst: pathlib.Path) -> None:
    srv = dst / "environment" / "sim" / "server.py"
    t = srv.read_text()
    t = sub1(t, LOG_SUBMIT_OLD, LOG_SUBMIT_NEW, f"{src.name} server.py submit log")
    t = sub1(t, LOG_OBS_FIELDS_OLD, LOG_OBS_FIELDS_NEW, f"{src.name} server.py observe fields")
    t = sub1(t, LOG_OBS_OLD, LOG_OBS_NEW, f"{src.name} server.py observe log")
    srv.write_text(t)


def patch_objects(src: pathlib.Path, dst: pathlib.Path) -> None:
    srv = dst / "environment" / "sim" / "server.py"
    t = srv.read_text()
    t = sub1(t, OBJ_FN_ANCHOR, OBJ_FN + OBJ_FN_ANCHOR.lstrip("\n"), f"{src.name} server.py objects_robot")
    t = sub1(t, OBJ_META_OLD, OBJ_META_NEW, f"{src.name} server.py meta_common")
    srv.write_text(t)
    ins = (dst / "instruction.md").read_text()
    ins = sub1(ins, OBJ_RULE_OLD, OBJ_RULE_NEW, f"{src.name} instruction (Rules)", regex=True)
    if re.search(OBJ_PARA_OLD, ins):          # the "Objects here" paragraph, where the family has one
        ins = sub1(ins, OBJ_PARA_OLD, OBJ_PARA_NEW, f"{src.name} instruction (Objects here)", regex=True)
    ins = sub1(ins, "**No privileged state.**", "**No privileged state, except the object poses (this variant).**",
               f"{src.name} instruction (Rules heading)")
    if re.search(r"(?i)no object poses|nothing reports their poses|where the\s+objects are", ins):
        raise SystemExit(f"{src.name}: a no-poses statement is left in the instruction")
    (dst / "instruction.md").write_text(ins)


# ---- the part shapes (GET /geometry): the pieces of the 2026-11 tool's full-information variant (q1c1s) that the
# geometry conditions are built from; FULL_FN is the source of q1c1g's geometry block (_geo_only_fn)
FULL_FN = r'''
# ---------------------------------------------------------------- probing v2 C1s: part shapes and internal state (local)
# Reads only (read_state, as Session.objects(); the model's arrays): nothing here writes to the simulator, so the live
# episode's state hash and its fresh-process replay, which never reads any of this, must stay equal (DEMO.md, check 5).
# Frames: a link welded to the world is given directly in the agent-facing frame (never by a link origin: a fixed body's
# origin can lie anywhere, the goal marker's is on the floor); a movable part's base link relative to its meta["objects"]
# pose; any other moving link relative to its pose in meta["state"]["links"], moved by to_robot.
STATE_LINKS: dict = {}     # part -> [moving link names]: the links reported in meta["state"]["links"]
TASK_STATE_PUBLISHED = ("cursor_m", "started", "traversal_complete", "coverage_complete")   # buzz_wire_3d's route tracker
TASK_STATE_WITHHELD = ("wrong_bin_since_s", M._RULES_KEY)   # caliper: the sorting answer; the rule counters are in meta["safety"]
_withheld_seen: set = set()


def _rr(a, nd: int = 6) -> list:
    return [round(float(x), nd) for x in np.asarray(a, dtype=np.float64).reshape(-1)]


def geometry_robot() -> dict:
    """GET /geometry's answer, built once from the frozen start state before anything can step."""
    import mujoco
    from scipy.spatial import ConvexHull
    w = S.env._world; m, d = w.model, w.data; T = mujoco.mjtGeom

    def form(g: int) -> dict:
        t, s = int(m.geom_type[g]), m.geom_size[g]
        if t == T.mjGEOM_BOX: return {"type": "box", "half_extents": _rr(s)}
        if t == T.mjGEOM_SPHERE: return {"type": "sphere", "radius": _rr(s[:1])[0]}
        if t == T.mjGEOM_CAPSULE: return {"type": "capsule", "radius": _rr(s[:1])[0], "half_length": _rr(s[1:2])[0]}
        if t == T.mjGEOM_CYLINDER: return {"type": "cylinder", "radius": _rr(s[:1])[0], "half_length": _rr(s[1:2])[0]}
        if t == T.mjGEOM_ELLIPSOID: return {"type": "ellipsoid", "radii": _rr(s)}
        if t == T.mjGEOM_MESH:             # MuJoCo collides a mesh as its convex hull: the hull's corners are the whole shape
            i = int(m.geom_dataid[g]); a = int(m.mesh_vertadr[i])
            v = np.asarray(m.mesh_vert[a:a + int(m.mesh_vertnum[i])], dtype=np.float64)
            try:
                v = v[ConvexHull(v).vertices]
            except Exception:              # noqa: BLE001 -- a flat mesh has no 3D hull: all its points
                pass
            return {"type": "convex", "vertices": [_rr(x) for x in v]}
        raise RuntimeError(f"geometry: geom {g} has an unsupported type {t}")

    parts, none = {}, {}
    for name, ent in S._entities().items():
        fixed, links, base = bool(getattr(getattr(ent, "morph", None), "fixed", False)), {}, int(ent.base_link.idx)
        for l in ent.links:
            b = int(l.idx); a = int(m.body_geomadr[b])
            gs = [g for g in range(a, a + int(m.body_geomnum[b])) if m.geom_contype[g] or m.geom_conaffinity[g]]
            if not gs:
                continue
            welded = int(m.body_weldid[b]) == 0
            frame = "world" if welded else ("part" if b == base and not fixed else "link")
            if frame == "link":
                STATE_LINKS.setdefault(name, []).append(l.name)
            shapes = []
            for g in gs:
                s = form(g); n = m.geom(g).name.split("::", 1)[-1]
                if n: s["name"] = n
                pos, quat = np.asarray(m.geom_pos[g], dtype=np.float64), np.asarray(m.geom_quat[g], dtype=np.float64)
                if welded:                 # the body's pose in the simulator's world, then the agent-facing frame
                    q_b = np.asarray(d.xquat[b], dtype=np.float64)
                    pos, quat = to_robot(np.asarray(d.xpos[b], dtype=np.float64) + M.quat_to_mat(q_b) @ pos, _qmul(q_b, quat))
                s["pos"], s["quat"] = _rr(pos), _rr(quat)
                shapes.append(s)
            links[l.name] = {"frame": frame, "shapes": shapes}
        if links:
            parts[name] = {"fixed": fixed, "links": links}
        else:
            none[name] = "no collision shape (a marker): its bounds in meta[\"objects\"] only"
    grasp = S.env._world.read_state().get("grasp") or {}
    regions = {}
    for t in (grasp.get("config") or {}).get("targets") or ():
        regions.setdefault(t["entity"], []).append({"link": t["link"], "center": _rr(t["center_m"]), "radius": round(float(t["radius_m"]), 6)})
    return {"frame": "the world frame of spec['world'], as every pose of this service",
            "shape_frames": {"world": "pos/quat in the world frame directly: the link never moves",
                             "part": "pos/quat relative to the part's pose, meta['objects'][part] (position, orientation_wxyz)",
                             "link": "pos/quat relative to the link's pose, meta['state']['links'][part][link]"},
            "shape_types": {"box": "half_extents along the shape's axes", "sphere": "radius",
                            "capsule": "radius; half_length of the straight section along the shape's z (the caps extend it by radius)",
                            "cylinder": "radius; half_length along the shape's z", "ellipsoid": "radii along the shape's axes",
                            "convex": "vertices in the shape's own frame: it collides as the convex hull of these points"},
            "point_formula": "a point x of a shape is at T(frame) * (pos + R(quat) * x); quat is wxyz",
            "parts": parts, "no_shape": none, "grasp_regions": regions}


GEOMETRY_BODY = json.dumps(geometry_robot(), separators=(",", ":")).encode()


def state_robot() -> dict:
    """meta["state"]: what the simulator holds beyond the poses, in the agent-facing frame."""
    ws = S.env._world.read_state(); arts = ws.get("articulations") or {}
    out = {"progress": round(float(S._reward), 4), "joints": {}, "links": {}}
    for name in S._entities():                  # 1-dof joints only: a free joint's position is the body's simulator-world pose
        js = {n: {"position": round(float(np.asarray(v["position"]).reshape(-1)[0]), 7),
                  "velocity": round(float(np.asarray(v["velocity"]).reshape(-1)[0]), 6)}
              for n, v in (arts.get(name) or {}).get("joints", {}).items()
              if np.size(v["position"]) == 1 and np.size(v["velocity"]) == 1}
        if js:
            out["joints"][name] = js
    for name, links in STATE_LINKS.items():
        out["links"][name] = {}
        for n in links:
            rec = arts[name]["links"][n]
            p, q = to_robot(rec["position_m"], rec["quaternion_wxyz"])
            out["links"][name][n] = {"position": _rr(p), "orientation_wxyz": _rr(q)}
    g = ws.get("grasp")
    if g:
        hands = {}
        for arm, h in g["hands"].items():
            b, c = h.get("binding"), h.get("candidate")
            hands[arm] = {"intent": h["intent"], "phase": h["phase"], "holding": list(b["target"]) if b else None,
                          "candidate": M._plain(c), "rejection": h["rejection"], "stable_time_s": round(float(h["stable_time_s"]), 4),
                          "closing_fraction": round(float(h["last_fraction"]), 4)}
        out["grasp"] = {"active": bool(g["active"]), "hands": hands}
    if ws.get("mechanism"):
        out["mechanism"] = {k: M._plain(v) for k, v in ws["mechanism"].items() if k != "angles"}   # angles: the rate's raw sample window
    st = S.env._task.task_state or {}
    ts = {k: M._plain(st[k]) for k in TASK_STATE_PUBLISHED if k in st}
    if "previous_position_m" in st:             # the wand's pose at the last tick: simulator world -> agent-facing frame
        p, q = to_robot(st["previous_position_m"], st["previous_quaternion_wxyz"])
        ts["previous_position_m"], ts["previous_quaternion_wxyz"] = _rr(p), _rr(q)
    if ts:
        out["task_state"] = ts
    left = set(st) - set(TASK_STATE_PUBLISHED) - set(TASK_STATE_WITHHELD) - {"previous_position_m", "previous_quaternion_wxyz"}
    if left - _withheld_seen:                   # fail closed: a key nobody has looked at stays here, and is logged once
        _withheld_seen.update(left); log({"ev": "state_withheld", "keys": sorted(left)})
    return out

'''
FULL_GET_OLD = "        def do_GET(self):\n            touch()\n            return base.do_GET(self)\n"
FULL_GET_NEW = ('        def do_GET(self):\n            touch()\n'
                '            if self.path == "/geometry":   # probing v2 C1s: read-only, built at startup, no simulator call\n'
                '                log({"ev": "geometry_read", "time_s": round(agent_time(), 2), "bytes": len(GEOMETRY_BODY)})\n'
                '                return self._send(200, "application/json", GEOMETRY_BODY)\n'
                '            return base.do_GET(self)\n')
FULL_FRAME_OLD = '"no positions of the scene are given; every pose, camera extrinsic and kinematics answer of this service is in it "'
FULL_SPEC_OLD = '    "static": {},                       # nothing about where things are (standard mode); the task text names the objects\n'
FULL_CLI = [   # environment/rcb_client.py: Robot.geometry()
    ('    r.status()                                   # calls, control steps, done, reason, elapsed time\n',
     '    r.status()                                   # calls, control steps, done, reason, elapsed time\n'
     '    geo = r.geometry()                           # this variant: the collision shapes of every task part and fixture (read-only)\n',
     "docstring"),
    ('    def operation_status(self, operation_id=None):\n',
     '''    def geometry(self) -> dict:
        """This variant: GET /geometry, the collision shapes of every task part and fixture. Read-only: it moves nothing and
        advances no time, and its answer never changes, so one call is enough. Its keys explain its own conventions."""
        r = requests.get(self.base + "/geometry", timeout=(5, 120))
        if r.status_code == 409: raise EpisodeOver(_over(r))
        r.raise_for_status(); return r.json()

    def operation_status(self, operation_id=None):
''', "Robot.geometry"),
]
FULL_RULE_END_OLD = (r"The success implementation is not exposed, and of the simulator's internal\s+measurements only the safety readings of "
                     r"`meta\[\"safety\"\]` are \(listed under Action types, below\); the task\s+requirements above still apply in full\.")
FULL_PARA_OLD = "nothing else about them is reported (no shapes, no internal states such as a thread's or a mechanism's): the rest comes from the cameras."
FULL_CLIENT_ANCHOR = 'res.save("/app/runs/003")                     # meta.json'


# ---- C1G (q1c1g): C10's poses plus GET /geometry, and no meta["state"] at all ---------------------------------------
# The missing step between C10 (poses) and C1s (poses + shapes + internal state): the part shapes alone. Without the state,
# a moving link (a slider, a hinge, a spring contact, a ratchet bit) has no reported pose, so its shapes (and any grasp
# region on it) are given where they are at the frozen start: in the part's frame, or in the world frame for a fixed part.
GEO_HELPER = """
LINK_AT_START: dict = {}   # (part, link) -> (body, base body, fixed, base link name): q1c1g's moving links


def _at_start(b, base, fixed, pos, quat):
    # a point/frame of moving link b at the frozen start: world (agent-facing) for a fixed part, else its part's frame
    d = S.env._world.data
    q_l = np.asarray(d.xquat[b], dtype=np.float64)
    p_w = np.asarray(d.xpos[b], dtype=np.float64) + M.quat_to_mat(q_l) @ np.asarray(pos, dtype=np.float64)
    q_w = _qmul(q_l, np.asarray(quat, dtype=np.float64))
    if fixed:
        return to_robot(p_w, q_w)
    q_p = np.asarray(d.xquat[base], dtype=np.float64)
    return (M.quat_to_mat(q_p).T @ (p_w - np.asarray(d.xpos[base], dtype=np.float64)),
            _qmul(q_p * np.array([1.0, -1.0, -1.0, -1.0]), q_w))

"""


def _geo_only_fn() -> str:
    t = FULL_FN
    a = '            if frame == "link":\n                STATE_LINKS.setdefault(name, []).append(l.name)\n'
    assert t.count(a) == 1
    t = t.replace(a, '            moving_at_start = frame == "link"   # q1c1g: no meta["state"], so given at the start configuration\n'
                     '            if moving_at_start:\n                frame = "world" if fixed else "part"\n'
                     '                LINK_AT_START[(name, l.name)] = (b, base, fixed, ent.base_link.name)\n')
    a = '                s["pos"], s["quat"] = _rr(pos), _rr(quat)\n'
    assert t.count(a) == 1
    t = t.replace(a, '                if moving_at_start and not welded:\n'
                     '                    pos, quat = _at_start(b, base, fixed, pos, quat)\n' + a)
    a = ('        regions.setdefault(t["entity"], []).append({"link": t["link"], "center": _rr(t["center_m"]), '
         '"radius": round(float(t["radius_m"]), 6)})\n')
    assert t.count(a) == 1
    t = t.replace(a, '        key = (t["entity"], t["link"])\n'
                     '        if key in LINK_AT_START:   # q1c1g: a region on a moving link, at the start configuration\n'
                     '            b, base, fixed, base_name = LINK_AT_START[key]\n'
                     '            c, _ = _at_start(b, base, fixed, np.asarray(t["center_m"], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))\n'
                     '            regions.setdefault(t["entity"], []).append({"link": "world" if fixed else base_name, "center": _rr(c), '
                     '"radius": round(float(t["radius_m"]), 6), "at_start": True})\n'
                     '        else:\n    ' + a)
    a = "                             \"link\": \"pos/quat relative to the link's pose, meta['state']['links'][part][link]\"},"
    assert t.count(a) == 1, 'shape_frames link'
    t = t.replace(a, "                             \"moving links\": \"a part's moving link (slider, hinge, spring contact, ratchet bit) "
                     "is given in the part's frame (or the world frame for a fixed part) as it stands at the start; its current "
                     "pose is not reported\"},")
    a = '\ndef geometry_robot() -> dict:\n'
    assert t.count(a) == 1
    return t.replace(a, GEO_HELPER + a)


GEO_SPEC_NEW = ('    "static": {},                       # nothing about where things are in /spec itself; the task text names the objects\n'
                '    "part_shapes": {"endpoint": "GET /geometry", "client": "Robot.geometry()", "note": "read-only; the collision shapes of '
                'every task part and fixture as they stand at the start; the answer never changes"},   # q1c1g\n')
GEO_FRAME_NEW = ('"the scene\'s poses are in every response\'s meta (objects) and the part shapes behind GET /geometry; every pose, '
                 'camera extrinsic and kinematics answer of this service is in it "')
GEO_RULE_HEAD = ("**No privileged state, except the object poses (this variant).**",
                 "**Object poses and part shapes are given (this variant).**")
GEO_RULE_END_NEW = ("The part shapes are behind `r.geometry()` (under `Part shapes`, below). The simulator's internal state is not "
                    "given: no joint positions or moving links' current poses, no grasp phase, no mechanism or task history. The safety "
                    "readings are in `meta[\"safety\"]` (listed under Action types, below). The simulator itself, its source, the success "
                    "implementation and its verdicts, forward rollouts and the reference solution are not exposed; the task requirements "
                    "above still apply in full.")
GEO_PARA_NEW = ("their shapes are given as well (`Part shapes`, below), but no internal states such as a thread's or a mechanism's: "
                "the rest comes from the cameras.")
GEO_POSITIONS = ("no positions of the scene are given", "the scene's poses are in `meta[\"objects\"]`")
GEO_CLIENT_NEW = ('geo = r.geometry()                            # this variant: the collision shapes of every task part and fixture (read-only)\n'
                  'obs.meta["objects"]                           # this variant, every response: the object poses\n')
GEO_SECTION = """### Part shapes (this variant)
**Part shapes.** `geo = r.geometry()` (`GET /geometry`) gives the collision shapes of every task part and fixture, exactly as the simulator collides them. It is read-only (it moves nothing and advances no time) and its answer never changes, so one call is enough; it may be some hundred kilobytes, so keep it in a variable.
- `geo["parts"][part]["links"][link]`: `frame` and `shapes`. `frame` says what the shapes' `pos` and `quat` are relative to: `"world"` the world frame of `spec["world"]` directly, `"part"` the part's pose in `meta["objects"][part]`. A part's moving link (a slider, a hinge, a spring contact, a ratchet bit) is given where it stands at the start; its current pose is not reported.
- A shape has `type`, `pos`, `quat` (wxyz), `name` when it has one, and its sizes in metres: `box`: `half_extents`; `sphere`: `radius`; `capsule`: `radius` and `half_length` of the straight section along the shape's z; `cylinder`: `radius` and `half_length` along z; `ellipsoid`: `radii`; `convex`: `vertices` in the shape's own frame, and it collides as the convex hull of these points. A point x of a shape is at T(frame) · (`pos` + R(`quat`) · x).
- `geo["no_shape"]` names what has no collision shape (a marker: only its bounds in `meta["objects"]`); `geo["grasp_regions"][part]`: the assisted grasp's regions, each `link`, `center` (in that link's frame; with `at_start`, a region on a moving link in the part's frame at the start) and `radius`.

Only the task's parts and fixtures are there: not the robot's links (`spec["robot_spec"]` and `r.fk` tell about the robot) and not the table. No masses, friction coefficients or other physical parameters are given.

**Not given.** The simulator's internal state is not reported: no joint positions or velocities, no current poses of moving links, no grasp phase, no mechanism readings or task history. The rest comes from the cameras and your own measurements: contacts and contact forces are not reported, beyond the safety readings, and neither is how far any success condition is met.

"""
GEO_LEFT = (r"(?i)no privileged state|no shapes|nothing reports|no positions of the scene|only the safety readings|"
            r"where the\s+objects are|no object poses")


def patch_geo(src: pathlib.Path, dst: pathlib.Path) -> None:
    name = src.name
    srv = dst / "environment" / "sim" / "server.py"
    t = srv.read_text()
    t = sub1(t, OBJ_FN_ANCHOR, _geo_only_fn() + OBJ_FN_ANCHOR.lstrip("\n"), f"{name} server.py C1g block")
    t = sub1(t, FULL_GET_OLD, FULL_GET_NEW, f"{name} server.py /geometry")
    t = sub1(t, FULL_FRAME_OLD, GEO_FRAME_NEW, f"{name} server.py world frame")
    t = sub1(t, FULL_SPEC_OLD, GEO_SPEC_NEW, f"{name} server.py spec")
    if '"state": state_robot()' in t:
        raise SystemExit(f"{name}: q1c1g must not report meta state")
    compile(t, str(srv), "exec")
    srv.write_text(t)
    cli = (dst / "environment" / "rcb_client.py").read_text()
    for old, new, what in FULL_CLI:
        cli = sub1(cli, old, new, f"{name} rcb_client.py {what}")
    compile(cli, str(dst / "environment" / "rcb_client.py"), "exec")
    (dst / "environment" / "rcb_client.py").write_text(cli)
    ins = (dst / "instruction.md").read_text()
    ins = sub1(ins, *GEO_RULE_HEAD, f"{name} instruction (Rules heading)")
    ins = sub1(ins, FULL_RULE_END_OLD, GEO_RULE_END_NEW, f"{name} instruction (Rules end)", regex=True)
    if FULL_PARA_OLD in ins:
        ins = sub1(ins, FULL_PARA_OLD, GEO_PARA_NEW, f"{name} instruction (Objects here)")
    if GEO_POSITIONS[0] in ins:
        ins = sub1(ins, *GEO_POSITIONS, f"{name} instruction (spec world)")
    ins = sub1(ins, FULL_CLIENT_ANCHOR, GEO_CLIENT_NEW + FULL_CLIENT_ANCHOR, f"{name} instruction (client)")
    ins = sub1(ins, "## External material\n", GEO_SECTION + "## External material\n", f"{name} instruction (section)")
    left = re.findall(GEO_LEFT, ins)
    if left:
        raise SystemExit(f"{name}: statements against the given shapes left in the instruction: {left}")
    if 'meta["state"]' in ins:
        raise SystemExit(f"{name}: the instruction still points at meta state")
    (dst / "instruction.md").write_text(ins)
    toml = (dst / "task.toml").read_text()
    toml = sub1(toml, "no privileged state", "object poses and part shapes given", f"{name} task.toml description (geometry)")
    (dst / "task.toml").write_text(toml)


# ---- C0G (q1c0g): geometry without poses: the Standard task plus GET /geometry,
# each part's collision shapes in the part's OWN frame (its base link at the frozen start), fixed parts and fixtures
# included, so the answer holds no position of the scene at all: the agent has the parts' models, not where they are.
# No meta["objects"], no meta["state"]. A moving link (slider, hinge, spring contact, ratchet bit) is given where it
# stands at the start, relative to its part's base link.
GEO0_FN = r"""
def geometry_parts() -> dict:
    # q1c0g: GET /geometry's answer without any position of the scene, built once from the frozen start state
    import mujoco
    from scipy.spatial import ConvexHull
    w = S.env._world; m, d = w.model, w.data; T = mujoco.mjtGeom
    rr = lambda a, nd=6: [round(float(x), nd) for x in np.asarray(a, dtype=np.float64).reshape(-1)]
    def form(g: int) -> dict:
        t, s = int(m.geom_type[g]), m.geom_size[g]
        if t == T.mjGEOM_BOX: return {"type": "box", "half_extents": rr(s)}
        if t == T.mjGEOM_SPHERE: return {"type": "sphere", "radius": rr(s[:1])[0]}
        if t == T.mjGEOM_CAPSULE: return {"type": "capsule", "radius": rr(s[:1])[0], "half_length": rr(s[1:2])[0]}
        if t == T.mjGEOM_CYLINDER: return {"type": "cylinder", "radius": rr(s[:1])[0], "half_length": rr(s[1:2])[0]}
        if t == T.mjGEOM_ELLIPSOID: return {"type": "ellipsoid", "radii": rr(s)}
        if t == T.mjGEOM_MESH:
            i = int(m.geom_dataid[g]); a = int(m.mesh_vertadr[i])
            v = np.asarray(m.mesh_vert[a:a + int(m.mesh_vertnum[i])], dtype=np.float64)
            try:
                v = v[ConvexHull(v).vertices]
            except Exception:              # noqa: BLE001 -- a flat mesh has no 3D hull: all its points
                pass
            return {"type": "convex", "vertices": [rr(x) for x in v]}
        raise RuntimeError(f"geometry: geom {g} has an unsupported type {t}")
    def rel(base: int, b: int, pos, quat):   # a frame on body b at the start, relative to body base at the start
        q_b = np.asarray(d.xquat[b], dtype=np.float64)
        p_w = np.asarray(d.xpos[b], dtype=np.float64) + M.quat_to_mat(q_b) @ np.asarray(pos, dtype=np.float64)
        q_w = _qmul(q_b, np.asarray(quat, dtype=np.float64))
        q_p = np.asarray(d.xquat[base], dtype=np.float64)
        return (M.quat_to_mat(q_p).T @ (p_w - np.asarray(d.xpos[base], dtype=np.float64)),
                _qmul(q_p * np.array([1.0, -1.0, -1.0, -1.0]), q_w))
    parts, none, body = {}, {}, {}
    for name, ent in S._entities().items():
        fixed, base, links = bool(getattr(getattr(ent, "morph", None), "fixed", False)), int(ent.base_link.idx), {}
        for l in ent.links:
            b = int(l.idx); body[(name, l.name)] = (b, base); a = int(m.body_geomadr[b])
            gs = [g for g in range(a, a + int(m.body_geomnum[b])) if m.geom_contype[g] or m.geom_conaffinity[g]]
            if not gs:
                continue
            shapes = []
            for g in gs:
                s = form(g); n = m.geom(g).name.split("::", 1)[-1]
                if n: s["name"] = n
                pos, quat = rel(base, b, m.geom_pos[g], m.geom_quat[g])
                s["pos"], s["quat"] = rr(pos), rr(quat)
                shapes.append(s)
            rec = {"shapes": shapes}
            if int(m.body_weldid[b]) != int(m.body_weldid[base]):
                rec["moving_at_start"] = True   # moves relative to the part's base link; given where it stands at the start
            links[l.name] = rec
        if links:
            parts[name] = {"fixed": fixed, "base_link": ent.base_link.name, "links": links}
        else:
            none[name] = "no collision shape (a marker)"
    grasp = S.env._world.read_state().get("grasp") or {}
    regions = {}
    for t in (grasp.get("config") or {}).get("targets") or ():
        b, base = body[(t["entity"], t["link"])]
        c, _ = rel(base, b, np.asarray(t["center_m"], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0]))
        regions.setdefault(t["entity"], []).append({"center": rr(c), "radius": round(float(t["radius_m"]), 6)})
    return {"frame": "each part's own frame: its base link as it stands at the start; where the parts are is not given",
            "shape_types": {"box": "half_extents along the shape's axes", "sphere": "radius",
                            "capsule": "radius; half_length of the straight section along the shape's z (the caps extend it by radius)",
                            "cylinder": "radius; half_length along the shape's z", "ellipsoid": "radii along the shape's axes",
                            "convex": "vertices in the shape's own frame: it collides as the convex hull of these points"},
            "point_formula": "a point x of a shape is at T(part) * (pos + R(quat) * x); quat is wxyz; T(part), the part's pose, is not given",
            "parts": parts, "no_shape": none, "grasp_regions": regions}


GEOMETRY_BODY = json.dumps(geometry_parts(), separators=(",", ":")).encode()

"""
GEO0_SPEC_NEW = ('    "static": {},                       # nothing about where things are (standard mode); the task text names the objects\n'
                 '    "part_shapes": {"endpoint": "GET /geometry", "client": "Robot.geometry()", "note": "read-only; the collision shapes of '
                 'every task part and fixture, each in its own frame; no positions; the answer never changes"},   # q1c0g\n')
GEO0_RULE_HEAD = ("**No privileged state.**", "**No privileged state, except the part shapes (this variant).**")
GEO0_RULE_END_NEW = ("The part shapes are behind `r.geometry()` (under `Part shapes`, below), each in its part's own frame: where the "
                     "parts are is not given. The simulator's internal state is not given either: no joint positions or moving links' "
                     "current poses, no grasp phase, no mechanism or task history. The safety readings are in `meta[\"safety\"]` (listed "
                     "under Action types, below). The simulator itself, its source, the success implementation and its verdicts, "
                     "forward rollouts and the reference solution are not exposed; the task requirements above still apply in full.")
GEO0_PARA = (" As everywhere in this mode, nothing reports their poses", " As everywhere in this mode, nothing reports their poses "
             "(their shapes are given, under `Part shapes` below, but not where they are)")
GEO0_CLIENT_NEW = ('geo = r.geometry()                            # this variant: every task part\'s collision shapes, in its own frame (read-only)\n')
GEO0_SECTION = """### Part shapes (this variant)
**Part shapes.** `geo = r.geometry()` (`GET /geometry`) gives the collision shapes of every task part and fixture, exactly as the simulator collides them, each in its part's own frame. It does not say where any part is: the parts' poses still have to come from the cameras. It is read-only (it moves nothing and advances no time) and its answer never changes, so one call is enough; it may be some hundred kilobytes, so keep it in a variable.
- `geo["parts"][part]`: `fixed` (the part never moves), `base_link`, and `links[link]["shapes"]`. Every shape's `pos` and `quat` are relative to the part's own frame: its base link as it stands at the start. A link marked `moving_at_start` (a slider, a hinge, a spring contact, a ratchet bit) moves relative to the base link; it is given where it stands at the start, and its current pose is not reported.
- A shape has `type`, `pos`, `quat` (wxyz), `name` when it has one, and its sizes in metres: `box`: `half_extents`; `sphere`: `radius`; `capsule`: `radius` and `half_length` of the straight section along the shape's z; `cylinder`: `radius` and `half_length` along z; `ellipsoid`: `radii`; `convex`: `vertices` in the shape's own frame, and it collides as the convex hull of these points. A point x of a shape is at T(part) · (`pos` + R(`quat`) · x), where T(part), the part's pose, is not given.
- `geo["no_shape"]` names what has no collision shape (a marker); `geo["grasp_regions"][part]`: the assisted grasp's regions, each `center` (in the part's frame, at the start) and `radius`.

Only the task's parts and fixtures are there: not the robot's links (`spec["robot_spec"]` and `r.fk` tell about the robot) and not the table. No masses, friction coefficients or other physical parameters are given.

**Not given.** No part's position or orientation, and not the simulator's internal state: no joint positions or velocities, no poses of moving links, no grasp phase, no mechanism readings or task history. The rest comes from the cameras and your own measurements: contacts and contact forces are not reported, beyond the safety readings, and neither is how far any success condition is met.

"""


def patch_geo0(src: pathlib.Path, dst: pathlib.Path) -> None:
    name = src.name
    srv = dst / "environment" / "sim" / "server.py"
    t = srv.read_text()
    t = sub1(t, OBJ_FN_ANCHOR, GEO0_FN + OBJ_FN_ANCHOR.lstrip("\n"), f"{name} server.py C0g block")
    t = sub1(t, FULL_GET_OLD, FULL_GET_NEW, f"{name} server.py /geometry")
    t = sub1(t, FULL_SPEC_OLD, GEO0_SPEC_NEW, f"{name} server.py spec")
    if "objects_robot()" in t or "state_robot()" in t or "to_robot(" in t.split("def geometry_parts", 1)[1].split("GEOMETRY_BODY", 1)[0]:
        raise SystemExit(f"{name}: q1c0g must report no poses")
    compile(t, str(srv), "exec")
    srv.write_text(t)
    cli = (dst / "environment" / "rcb_client.py").read_text()
    for old, new, what in FULL_CLI:
        cli = sub1(cli, old, new.replace("the collision shapes of every task part and fixture (read-only)",
                                         "every task part's collision shapes, in its own frame (read-only)"), f"{name} rcb_client.py {what}")
    compile(cli, str(dst / "environment" / "rcb_client.py"), "exec")
    (dst / "environment" / "rcb_client.py").write_text(cli)
    ins = (dst / "instruction.md").read_text()
    ins = sub1(ins, *GEO0_RULE_HEAD, f"{name} instruction (Rules heading)")
    ins = sub1(ins, FULL_RULE_END_OLD, GEO0_RULE_END_NEW, f"{name} instruction (Rules end)", regex=True)
    if GEO0_PARA[0] in ins:
        ins = sub1(ins, *GEO0_PARA, f"{name} instruction (Objects here)")
    ins = sub1(ins, FULL_CLIENT_ANCHOR, GEO0_CLIENT_NEW + FULL_CLIENT_ANCHOR, f"{name} instruction (client)")
    ins = sub1(ins, "## External material\n", GEO0_SECTION + "## External material\n", f"{name} instruction (section)")
    if 'meta["objects"]' in ins or 'meta["state"]' in ins:
        raise SystemExit(f"{name}: the instruction points at poses or state")
    if not re.search(r"Nothing tells you where the\s+objects are", ins):
        raise SystemExit(f"{name}: the no-poses rule is missing")
    (dst / "instruction.md").write_text(ins)
    toml = (dst / "task.toml").read_text()
    toml = sub1(toml, "no privileged state", "part shapes given, no poses", f"{name} task.toml description (geometry only)")
    (dst / "task.toml").write_text(toml)


def make(src: pathlib.Path, dst: pathlib.Path, condition: str) -> None:
    """Edit dst (a copy of the Standard task src) into the condition (CONDITIONS)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown information condition {condition!r} (one of {', '.join(CONDITIONS)})")
    patch_log(src, dst)
    if condition in ("q1c10", "q1c1g"):
        patch_objects(src, dst)
    if condition == "q1c1g":
        patch_geo(src, dst)
    elif condition == "q1c0g":
        patch_geo0(src, dst)
