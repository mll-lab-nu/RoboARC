#!/usr/bin/env python3
"""The cstate field check: the deformable state a CS variant's service gives, against the simulator's own state at the
same step. Runs INSIDE the task's simulator image (the `sim` service's image: topogym, MuJoCo, the staged frozen
instance), started by scripts/ablations/check_cstate.py; not meant to be run by hand.

It builds the frozen scene the way the service does (rcb_topogym.Session, strict), executes the variant's own code from
its environment/sim/server.py (the robot-frame lines and everything the variant inserted before meta_common: objects,
geometry, deformable state) against that scene, steps the task's reference rows (solution/oracle_qpos.npy, one row per
control step; a task without one: the robot holding its start targets) and at every checkpoint compares what meta["objects"] and meta["deformables"] would carry with the
simulator's own readings, through other code paths:
  - the frame: the robot frame is the robot's root link pose in MjData, and it is not the identity (so a world value
    would not pass);
  - a rope_chain rope: its nodes (RopeChain.node_positions) and its FEM view (state["fem"], the rope's skinned surface,
    must lie one radius from the reported centreline);
  - an articulated chain: every link's capsule axis from MjData's own geom poses (geom_xpos, geom_xmat), the chain's
    contiguity, and the rigid links' poses from read_state()["articulations"];
  - a cloth: the grader's particles (read_state()["particles"]) from the reported vertices by the cloth's own
    barycentric binding; the grid's shape and triangulation against the mesh;
  - meta["objects"] against Session.objects();
  - no key that names a grader quantity, JSON without NaN, and reading changes nothing (the state hash before and after);
  - GET /geometry: for the rigid parts, exactly C1G's answer (cond_info's code, when /ablations is mounted), the
    deformable bodies apart.
Prints one JSON report line (prefix CSTATE_REPORT) and exits 0 when every check passes."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np

FRAME_FROM = "R_SPEC = S.robot_spec(); K_SPEC = S.kinematics_spec()\n"
FRAME_TO = "\ndef camera_to_robot("
BLOCK_FROM = "\ndef objects_robot() -> dict:\n"
BLOCK_TO = "\ndef meta_common(cams) -> dict:\n"
META_KEYS = {"kind", "chains", "points", "radius", "links", "position", "orientation_wxyz", "vertices", "grid_shape"}
GRADER_WORDS = re.compile(r"(?i)success|progress|reward|crossing|stage|verdict|done|reason|tension|force|contact(?!_radius)|"
                          r"velocit|speed|holding|grasped|predicate|score")
TOL_POINT_M = 3e-5            # 0.01 mm rounding (sqrt 3 x 5e-6) and float noise
TOL_GAP_M = 1e-3              # a chain's joints: link k's axis end to link k + 1's start (joint compliance)


def _frame_independent(S) -> np.ndarray:
    """robot frame -> world: the robot's root link pose as MjData holds it."""
    d = S.env._world.data; b = int(S.robot.entity.links[0].idx)
    T = np.eye(4); T[:3, :3] = np.asarray(d.xmat[b], dtype=np.float64).reshape(3, 3); T[:3, 3] = np.asarray(d.xpos[b])
    return T


def _to_world(T, X) -> np.ndarray:
    return np.asarray(X, dtype=np.float64).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]


def _seg_dist(P, A, B) -> np.ndarray:
    """Distance of every point P (n, 3) to the polyline of segments A[i] -> B[i] (m, 3)."""
    best = np.full(len(P), np.inf)
    U = B - A; uu = np.maximum(np.einsum("ij,ij->i", U, U), 1e-30)
    for i0 in range(0, len(P), 2048):
        p = P[i0:i0 + 2048, None, :]
        s = np.clip(np.einsum("pmj,mj->pm", p - A[None], U) / uu[None], 0.0, 1.0)
        q = A[None] + s[..., None] * U[None]
        best[i0:i0 + 2048] = np.sqrt(((p - q) ** 2).sum(-1)).min(axis=1)
    return best


def _keys(obj, path="") -> list:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append((f"{path}.{k}", str(k)))
            out += _keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:4]):           # the lists are uniform: their first entries stand for the rest
            out += _keys(v, f"{path}[{i}]")
    return out


class Fail(list):
    def need(self, ok, what):
        if not ok:
            self.append(what)
        return ok


def check_point(ns, S, T_WR, label, fails, rep) -> None:
    w = S.env._world; d = w.data
    sha0 = S.state_sha()
    meta = {"objects": ns["objects_robot"](), "deformables": ns["deformables_robot"]()}
    fails.need(S.state_sha() == sha0, f"{label}: reading the state changed the state hash")
    blob = json.dumps(meta, allow_nan=False)
    rep.setdefault("meta_bytes", []).append(len(blob))
    for path, k in _keys(meta["deformables"]):
        named = path.count(".") == 1 or path.split(".")[-2:-1] == ["links"]   # an entity's or a link's name
        fails.need(k in META_KEYS or named, f"{label}: meta deformables key {path} not in the documented set")
        fails.need(not GRADER_WORDS.search(k) or named, f"{label}: grader-like key {path}")
    ws = w.read_state()
    worst = {}
    for name, kind in ns["DEFORMABLE"].items():
        rec = meta["deformables"].get(name)
        if not fails.need(rec is not None, f"{label}: meta deformables lacks {name}"):
            continue
        if kind == "cloth":
            c = w._cloths[name]
            V = _to_world(T_WR, rec["vertices"])
            fails.need(len(V) == len(c.rest), f"{label}: {name}: {len(V)} vertices, the mesh has {len(c.rest)}")
            ref = np.asarray(ws["particles"][name]["position_m"], dtype=np.float64)
            err = float(np.abs(c.particles_of(V) - ref).max())
            verr = float(np.abs(V - np.asarray(c.vertices(d), dtype=np.float64)).max())
            worst[name] = {"particles_from_vertices_max_m": err, "vertices_max_m": verr}
            fails.need(err <= TOL_POINT_M and verr <= TOL_POINT_M, f"{label}: {name}: vertices off the simulator's ({verr:.2e} m, particles {err:.2e} m)")
            continue
        chains = rec["chains"]; lay = ns["_LAYOUT"][name]
        fails.need(len(chains) == len(lay["chains"]), f"{label}: {name}: chain count")
        rope = getattr(w, "_ropes", {}).get(name)
        m_err = gap = 0.0
        for ch, L in zip(chains, lay["chains"]):
            pts = _to_world(T_WR, ch["points"]); B = L["bodies"]; n = len(B)
            fails.need(len(pts) == n + 1, f"{label}: {name}: {len(pts)} points for {n} links")
            # every link's capsule axis from MjData's own geom poses (an independent path: geom_xpos / geom_xmat)
            prox, dist = [], []
            for b in B:
                g = [g for g in range(int(w.model.body_geomadr[b]), int(w.model.body_geomadr[b]) + int(w.model.body_geomnum[b]))
                     if w.model.geom_contype[g] or w.model.geom_conaffinity[g]][0]
                c0 = np.asarray(d.geom_xpos[g], dtype=np.float64)
                z = np.asarray(d.geom_xmat[g], dtype=np.float64).reshape(3, 3)[:, 2] * (float(w.model.geom_size[g][1]) if int(w.model.geom_type[g]) == 3 else 0.0)
                a, e = c0 - z, c0 + z
                j, nj = int(w.model.body_jntadr[b]), int(w.model.body_jntnum[b])
                anchor = np.asarray(d.xanchor[j]) if nj == 1 and int(w.model.jnt_type[j]) != 0 else np.asarray(d.xpos[b])
                if np.linalg.norm(a - anchor) > np.linalg.norm(e - anchor):     # MjData's own joint anchor (mjJNT_FREE = 0)
                    a, e = e, a
                prox.append(a); dist.append(e)
            prox, dist = np.asarray(prox), np.asarray(dist)
            m_err = max(m_err, float(np.abs(pts[:-1] - prox).max()), float(np.abs(pts[-1] - dist[-1]).max()))
            if n > 1:
                gap = max(gap, float(np.linalg.norm(dist[:-1] - prox[1:], axis=1).max()))
            if rope is not None:
                nodes = np.asarray(rope.node_positions(d), dtype=np.float64)
                r = float(ch["radius"])
                e_mid = float(np.abs(pts[1:-1] - nodes[1:-1]).max())
                e_end = max(abs(float(np.linalg.norm(pts[0] - nodes[0])) - r), abs(float(np.linalg.norm(pts[-1] - nodes[-1])) - r))
                # the rope's FEM view (the grader's rope: skinned from the chain) lies within one radius of the reported
                # centreline (material end to material end), and the centreline lies inside the rope
                fem = np.asarray(ws["fem"][name]["position_m"], dtype=np.float64)
                A, Bp = np.vstack([nodes[:1], pts[1:-1]]), np.vstack([pts[1:-1], nodes[-1:]])
                dd = _seg_dist(fem, A, Bp)
                inside = max(float(np.sqrt(((fem - p) ** 2).sum(1)).min()) for p in pts)
                worst[name + ".rope"] = {"nodes_max_m": e_mid, "ends_vs_radius_m": e_end, "fem_max_dist_m": float(dd.max()),
                                         "points_to_fem_max_m": inside, "radius_m": r}
                fails.need(e_mid <= TOL_POINT_M and e_end <= TOL_POINT_M, f"{label}: {name}: points off the rope's nodes ({e_mid:.2e}, ends {e_end:.2e})")
                fails.need(float(dd.max()) <= 1.25 * r and inside <= 1.25 * r,
                           f"{label}: {name}: the rope is not within one radius of the reported centreline ({float(dd.max()):.2e} m, {inside:.2e} m)")
        fails.need(m_err <= TOL_POINT_M, f"{label}: {name}: points off MjData's capsule axes ({m_err:.2e} m)")
        fails.need(gap <= TOL_GAP_M, f"{label}: {name}: a chain is not contiguous ({gap:.2e} m)")
        worst[name] = {"axes_max_m": m_err, "joint_gap_max_m": gap}
        arts = (ws.get("articulations") or {}).get(name, {}).get("links", {})
        for ln, pose in (rec.get("links") or {}).items():
            ref = arts[ln]
            p = _to_world(T_WR, pose["position"])[0]
            fails.need(float(np.abs(p - np.asarray(ref["position_m"])).max()) <= TOL_POINT_M, f"{label}: {name}.{ln}: position off")
            q_rw = np.asarray(pose["orientation_wxyz"], dtype=np.float64)
            R = T_WR[:3, :3] @ ns["M"].quat_to_mat(q_rw / np.linalg.norm(q_rw))
            fails.need(float(np.abs(R - ns["M"].quat_to_mat(np.asarray(ref["quaternion_wxyz"], dtype=np.float64))).max()) <= 1e-4,
                       f"{label}: {name}.{ln}: orientation off")
    # meta["objects"] against Session.objects() (C10's code: the robot frame, a fixed object by its bounds' centre)
    objs = S.objects(); oerr = 0.0
    for n, o in meta["objects"].items():
        ref = objs[n]
        want = (np.asarray(ref["bounds"][0]) + np.asarray(ref["bounds"][1])) / 2 if (ref.get("fixed") and "bounds" in ref) else np.asarray(ref["position"])
        oerr = max(oerr, float(np.abs(_to_world(T_WR, o["position"])[0] - want).max()))
    worst["objects_max_m"] = oerr
    fails.need(oerr <= 2e-5, f"{label}: meta objects off Session.objects() ({oerr:.2e} m)")
    rep.setdefault("checkpoints", []).append({"label": label, "step": int(S.n_steps), **worst})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="/task", help="the variant task directory (mounted)")
    ap.add_argument("--steps", default="0,25,100,250", help="control steps at which to check (reference rows replayed)")
    ap.add_argument("--ablations", default="/ablations", help="scripts/ablations (mounted), for the C1G parity check")
    ap.add_argument("--cloth-on-cpu", action="store_true",
                    help="smoke test of the code path only: a VBD cloth on the CPU twin step (not the task's physics)")
    a = ap.parse_args()
    rep: dict = {"task": os.path.basename(os.path.normpath(a.task)), "cloth_on_cpu": bool(a.cloth_on_cpu)}
    try:
        rep["task"] = re.search(r'(?m)^name = "roboarc/([^"]+)"', open(os.path.join(a.task, "task.toml")).read()).group(1)
    except (OSError, AttributeError):
        pass
    fails = Fail()
    try:
        sys.path.insert(0, "/srv")
        if a.cloth_on_cpu:
            import topogym.world.mujoco.cloth_vbd as CV
            from topogym.world.mujoco.profiles import NAMED_PROFILES
            twin = NAMED_PROFILES["blanket_vbd_cpu_twin"]["cloth"]
            orig = CV.ClothVBD.__init__

            def init(self, world, name, record, scene_dir):
                prof = dict(world.profile or {})
                prof["cloth"] = dict(prof.get("cloth") or {}, device="cpu",
                                     solver=twin.get("solver", "topogym.world.mujoco.vbd_fast:FastSolverVBD"),
                                     solver_kwargs=twin["solver_kwargs"])
                world.profile = prof
                orig(self, world, name, record, scene_dir)
            CV.ClothVBD.__init__ = init
        import rcb_topogym as M
        t0 = time.time()
        S = M.Session(os.environ["RCB_TASK"], instance=os.environ["RCB_INSTANCE"], strict=True)
        rep["build_s"] = round(time.time() - t0, 1)
        T_ind = _frame_independent(S)
        src = open("/srv/server.py").read()
        frame = src[src.index(FRAME_FROM):src.index(FRAME_TO)]
        block = src[src.index(BLOCK_FROM):src.index(BLOCK_TO)]
        events = []
        ns = {"np": np, "json": json, "math": math, "os": os, "S": S, "M": M, "log": events.append, "__name__": "cstate_check"}
        exec(compile(frame, "server.py (robot frame)", "exec"), ns)
        T_srv = np.asarray(ns["T_WR"], dtype=np.float64)
        rep["robot_frame_vs_root_link"] = float(np.abs(T_srv - T_ind).max())
        fails.need(rep["robot_frame_vs_root_link"] <= 1e-6, "the service's robot frame is not the robot's root link pose")
        fails.need(float(np.linalg.norm(T_ind[:3, 3])) > 1e-3, "the robot frame is the world frame here: a world value would pass")
        rep["robot_frame_offset_m"] = [round(float(x), 4) for x in T_ind[:3, 3]]
        exec(compile(block, "server.py (variant block)", "exec"), ns)
        geo = json.loads(ns["GEOMETRY_BODY"])
        rep["geometry_bytes"] = len(ns["GEOMETRY_BODY"])
        deform = ns["DEFORMABLE"]
        fails.need(set(geo.get("deformables", {})) == set(deform), "GET /geometry deformables != DEFORMABLE")
        fails.need(not (set(geo["parts"]) & set(deform)), "a deformable body is among geo parts")
        for path, k in _keys(geo["deformables"]):
            fails.need(not GRADER_WORDS.search(k), f"geometry: grader-like key {path}")
        lay = {}
        for name, g in geo["deformables"].items():
            if g["kind"] == "cloth":
                A, Bn = g["grid_shape"] or (None, None)
                ok = A is not None and A * Bn == g["n_vertices"]
                fails.need(ok, f"{name}: no regular grid")
                if ok:
                    rule = sorted(tuple(sorted(t)) for a_ in range(A - 1) for b_ in range(Bn - 1)
                                  for t in ((a_ * Bn + b_, (a_ + 1) * Bn + b_, (a_ + 1) * Bn + b_ + 1),
                                            (a_ * Bn + b_, (a_ + 1) * Bn + b_ + 1, a_ * Bn + b_ + 1)))
                    fails.need(sorted(tuple(sorted(t)) for t in g["triangles"]) == rule, f"{name}: triangles do not follow the documented split")
                    fails.need((A - 1) * g["spacing_m"][0] >= (Bn - 1) * g["spacing_m"][1], f"{name}: the first grid axis is not the long side")
                lay[name] = {"grid_shape": g["grid_shape"], "spacing_m": g["spacing_m"], "triangles": len(g["triangles"]),
                             "contact_radius_m": g.get("contact_radius_m")}
            else:
                lay[name] = {"chains": [[c["links"][0], c["links"][-1], len(c["links"]), c["parent"], c["radius"]] for c in g["chains"]],
                             "rigid_links": sorted(g.get("rigid_links", {})), "welded_links": sorted(g.get("welded_links", {})),
                             "grasp_regions": len(g.get("grasp_regions", ()))}
        rep["layout"] = lay
        # C1G parity: the rigid parts exactly as C1G gives them (cond_info's block, run on the same scene)
        if os.path.isdir(a.ablations) and not any(k == "cloth" for k in deform.values()):
            sys.path.insert(0, a.ablations)
            import cond_info as ci
            ns2 = {"np": np, "json": json, "math": math, "os": os, "S": S, "M": M, "log": events.append, "__name__": "c1g_check"}
            exec(compile(frame, "server.py (robot frame)", "exec"), ns2)
            exec(compile(ci.OBJ_FN + ci._geo_only_fn(), "C1G block", "exec"), ns2)
            g1 = json.loads(ns2["GEOMETRY_BODY"])
            parts1 = {k: v for k, v in g1["parts"].items() if k not in deform}
            regions1 = {k: v for k, v in g1["grasp_regions"].items() if k not in deform}
            same = parts1 == geo["parts"] and regions1 == geo["grasp_regions"] and \
                {k: v for k, v in g1["no_shape"].items() if k not in deform} == geo["no_shape"]
            fails.need(same, "GET /geometry's rigid parts differ from C1G's")
            rep["c1g_parity"] = bool(same)
        elif any(k == "cloth" for k in deform.values()):
            rep["c1g_parity"] = "n/a (C1G's block has no cloth support; the blanket scenes' rigid parts are compared by the CPU tests)"
        ref = os.path.join(a.task, "solution", "oracle_qpos.npy")
        Q = np.load(ref).astype(np.float32) if os.path.exists(ref) else None   # a task without a reference: hold still
        rep["rows"] = "reference" if Q is not None else "hold (the task has no reference solution)"
        steps = sorted({int(x) for x in a.steps.split(",") if x.strip()})
        t1 = time.time(); i = 0
        for target in steps:
            while S.n_steps < target and not S.episode_done and (Q is None or i < len(Q)):
                S.step(Q[i] if Q is not None else np.asarray(S.hold_action(), dtype=np.float32)); i += 1
            check_point(ns, S, T_ind, f"step {S.n_steps}", fails, rep)
        rep["step_s"] = round(time.time() - t1, 1)
        rep["episode_end"] = S.episode_end
    except Exception as exc:  # noqa: BLE001 -- the report says what broke
        fails.append(f"error: {exc!r}")
        rep["traceback"] = traceback.format_exc()[-2000:]
    rep["fails"] = list(fails); rep["ok"] = not fails
    print("CSTATE_REPORT " + json.dumps(rep, sort_keys=True), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
