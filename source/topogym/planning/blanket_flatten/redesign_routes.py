"""Robot routes for the redesigned blanket tiers (blanket redesign coordinator, 2026-10-11).

The redesign's oracles are object-level lab programs (topogym/planning/blanket_flatten/redesign/reverse.py,
topogym/planning/blanket_fold/redesign/oracle.py): Python functions that read the live cloth (measured corrections,
re-centrings, fold layouts) and call run_stage(sim, stage) for each grasp-and-move stage. Here they run unchanged
against the robot: a planner (route["planner"], called by run_route_oracle after a settle while the task is not yet
successful) drives the lab program in a worker thread on a ProxyLab that mirrors the live cloth. Each run_stage call
of the program becomes ONE route-format stage (planned on the cloth measured at that moment), handed to the robot;
the program resumes on the next planner call with the cloth the robot left.

Stage conversion (lab -> route, routes.py format):
  - grasp: the lab's pick rule (nearest exposed particle within search_m, the assist's exposure rule with
    remote_from_patch) on the live cloth; grasp_uv is that particle's material point in the task's uv;
  - segments: every hand target becomes an absolute {"to": (x, y, h above the table)} (the lab's 'by' / 'uv' resolved
    on the live cloth); a 'path' (several waypoints, one smoothstep in the lab) becomes consecutive segments split at
    the union of both hands' waypoint fractions, durations by arc-length fraction; a hand without a target holds;
  - settle_s carries over. Stages that keep a grasp across stages (relays) are not used by these tiers and raise.
The ProxyLab reports backend "vbd" so the fold oracles use their VBD heights (relative lifts / releases).
"""
from __future__ import annotations

import math
import queue
import threading

import numpy as np

SHEET_CENTRE_XY = (-0.1356786, 0.05)        # the redesigned tiers' flat sheet centre (lab centre_shift (0, -0.10))
# lane A (10-11): the robot tries yaw 0/90/180/270 and prefers the one whose two fingertips stand on the same layer
# the quiet settle before each planner call, at most (2026-10-13 speed-up, user-approved route shortening; it ends
# earlier once the sheet is quiet; robot_oracle's default is 6 s)
PLANNER_SETTLE_S = 3.
# the routes that take the 2026-10-13 speed-ups (the 3 s planner settle, the 1.5 s fold drags on VBD, the faster
# finishing relays): only those recorded after it. The tiers recorded before it on d38cdf4bfb (flatten very_easy4c,
# easier4, easy4f, medium4e; fold very_easy, easy) keep the slower motions, so their oracles reproduce their records
# (the route sets its lab sim's `speedup_1013`; the fold drags and the finishing relays read it, also inside a fold
# route's flatten part)
FAST_FLATTEN = ("hard4b",)
FAST_FOLD = ("easier", "medium", "hard")
FOOTING = dict(half_span_m=.010, radius_m=.006, tolerance_m=.003)
PATCH_RADIUS_M, CONTACT_DISTANCE_M = .018, .006


class ProxyLab:
    """The lab-program surface over the robot's live cloth (no physics here)."""
    backend = "vbd"
    sphere_r = .003
    PAD_HALF_Z = .01

    def __init__(self, uv_task, table, centre_xy=SHEET_CENTRE_XY):
        uv_task = np.asarray(uv_task, float)
        self.uv_task = uv_task
        self.uv = uv_task - (uv_task.min(0) + uv_task.max(0)) / 2       # the lab's centred material coordinates
        self.table_top = float(table)
        self.centre_xy = np.asarray(centre_xy, float)
        self.hand_names = ("left", "right")
        self.held = dict.fromkeys(self.hand_names)
        self.events = []
        self.time = 0.
        self._P = None
        self._plan = None          # set by the planner: callable(route_stage) -> next cloth

    # ---------------------------------------------------------------- state
    def particles(self, X=None):
        return self._P.copy()

    def nearest_particle(self, uv):
        return int(np.argmin(np.sum((self.uv - np.asarray(uv, float)) ** 2, axis=1)))

    def patch(self, centre, P=None):
        P = self._P if P is None else P
        return np.flatnonzero((np.linalg.norm(P - P[centre], axis=1) <= PATCH_RADIUS_M)
                              & (np.linalg.norm(self.uv - self.uv[centre], axis=1) <= PATCH_RADIUS_M))

    def exposed(self, patch, P=None):
        """The assist's exposure rule with remote_from_patch (lane A 4a05e10ab)."""
        from scipy.spatial import cKDTree
        x = self._P if P is None else np.asarray(P)
        idx = np.asarray(patch)
        if not len(idx):
            return True
        r = PATCH_RADIUS_M
        low, high = x[idx, :2].min(axis=0) - r - 1e-9, x[idx, :2].max(axis=0) + r + 1e-9
        near = np.flatnonzero(np.all((x[:, :2] >= low) & (x[:, :2] <= high), axis=1))
        remote = cKDTree(self.uv[idx]).query(self.uv[near])[0] > 2 * r
        for i in idx:
            local = np.linalg.norm(x[near, :2] - x[i, :2], axis=1) <= r
            if np.any(local & remote & (x[near, 2] > x[i, 2] + CONTACT_DISTANCE_M)):
                return False
        return True

    def guard_ratio(self, P=None):
        return 1.

    def hand_pos(self, hand):
        return np.array([0., 0., 2.])          # parked (fold_layouts then assumes the arms' home TCPs)

    def evaluate(self, X=None, *, coverage_target=.95, speed=None):
        """The flatten predicate's object part (the lab's FlexSim.evaluate gates; the robot's Task decides success)."""
        from scipy.spatial import Delaunay
        from topogym.evaluation.cloth_metrics import footprint_area_slabs
        P = self._P
        if not hasattr(self, "_faces"):
            self._faces = Delaunay(self.uv).simplices
        cov = float(footprint_area_slabs(P, self._faces) / (.60 * .48))
        h = P[:, 2] - self.table_top
        gates = dict(covered=cov >= coverage_target, flat=float(h.max()) <= .03, above_table=float(h.min()) >= -.008)
        return dict(coverage=round(cov, 4), max_height_m=round(float(h.max()), 4), gates=gates, goal=all(gates.values()))

    def settle(self, max_s=10., **kw):
        """The robot settles before every planner call; a lab settle between stages is that settle."""
        return 0.

    # ---------------------------------------------------------------- the stage hooks (programs.run_stage / run_stages)
    def robot_stage(self, stage):
        return self.robot_stages([stage])[0]

    def robot_stages(self, stages):
        """Stages fixed by one measurement (programs.run_stages) -> ONE planner call: all are built and converted on
        the cloth measured now; the robot runs them back to back (no settle-and-replan between them). A stage the robot
        did not admit / reach / hold is re-planned once on the live cloth (4 cm grasp search), in the next call."""
        built = [st() if callable(st) else st for st in stages]
        for st in built:
            if st.get("keep") or any(seg.get("release") for seg in st.get("segments", ())) or st.get("bar") \
                    or st.get("fold") or any(seg.get("rot") for seg in st.get("segments", ())):
                raise NotImplementedError(f"stage {st.get('name')}: relays / bars / rigid folds have no robot route")
        recs = [dict(name=st.get("name"), ok=True, robot=True, attempts=[]) for st in built]
        todo, outcome = list(range(len(built))), {}
        for attempt in range(2):
            planned = []
            for j in todo:
                st = built[j] if attempt == 0 else dict(built[j], search_m=max(float(built[j].get("search_m", 0.) or 0.), .04))
                # a later stage of the call keeps the lab's grasp-relative targets ('by'): its grasp point may have
                # moved under the earlier stages, and the robot locates it live
                rs = to_route_stage(self, st, keep_relative=bool(planned))
                if not rs["grasp_uv"]:
                    recs[j]["reason"] = "no grasp (a wait): left to the robot's settle"
                    continue
                planned.append((j, rs))
            if not planned:
                return recs
            cloth, last = self._plan([rs for _, rs in planned])          # blocks until the robot ran them and settled
            self._P = np.asarray(cloth, float)
            # per-stage outcomes: last["stages"] (every stage of the call, by name) when the runner gives them, else
            # `last` is the call's final stage and the earlier ones count as done
            by_name = {o.get("name"): o for o in ((last or {}).get("stages") or [last] if last else [])}
            failed = []
            for j, rs in planned:
                o = by_name.get(rs["name"])
                recs[j]["attempts"].append(dict(grasp_uv=rs["grasp_uv"], last=o))
                if o is not None and (o.get("skipped") or not all((o.get("admitted") or {}).values()) or o.get("grasp_lost")):
                    failed.append(j)
                    outcome[j] = o
                else:
                    recs[j]["grasps"] = {h: dict(ok=True, uv=list(g)) for h, g in rs["grasp_uv"].items()}
            todo = failed
            if not todo:
                break
        for j in todo:
            recs[j] = dict(recs[j], ok=False, reason=f"robot: {outcome[j]}")
        return recs


def _resolve(sim, hand, spec, origin):
    from topogym.planning.blanket_flatten.redesign.programs import _target
    return _target(sim, hand, spec, origin)


def to_route_stage(sim, stage, keep_relative=False):
    """One lab stage -> one route stage on the cloth now (see the module note). keep_relative: a lab segment whose
    moving hands all name a single grasp-relative 'by' target stays a route 'by' segment (located live by the robot)."""
    from topogym.planning.blanket_flatten.redesign.programs import pick_material, pick_top, pick_top_band
    P = sim.particles()
    picks = {}
    for hand, uv in (stage.get("grasp") or {}).items():
        picks[hand] = pick_material(sim, uv, stage.get("search_m", 0.), False)
    for hand, xy in (stage.get("grasp_xy") or {}).items():
        picks[hand] = pick_top(sim, xy, stage.get("search_m", .02))
    for hand, band in (stage.get("grasp_top") or {}).items():
        picks[hand] = pick_top_band(sim, band)
    origin = {h: P[c].copy() for h, c in picks.items()}         # the lab's 'by' origin: the grasp point at stage start
    grasp_uv = {h: tuple(round(float(v), 6) for v in sim.uv_task[c]) for h, c in picks.items()}
    cur = dict(origin)                                           # each hand's patch position at the segment start
    segments = []
    for seg in stage.get("segments", ()):
        tracks = {}
        for hand, spec in (seg.get("hands") or {}).items():
            if spec is None or spec == "hold" or hand not in picks:
                continue
            specs = spec["path"] if "path" in spec else [spec]
            tracks[hand] = [_resolve(sim, hand, x, origin) for x in specs]
        if not tracks:
            segments.append(dict(name=seg.get("name", "hold"), duration_s=float(seg["duration_s"])))
            continue
        specs = {h: seg["hands"][h] for h in tracks}
        if keep_relative and all(isinstance(x, dict) and "by" in x for x in specs.values()):
            segments.append(dict(name=seg.get("name", "move"), duration_s=float(seg["duration_s"]),
                                 by={h: tuple(round(float(v), 5) for v in x["by"]) for h, x in specs.items()}))
            for h in tracks:
                cur[h] = np.asarray(tracks[h][-1], float)
            continue
        # one lab segment = one smoothstep along each hand's polyline: split at the union of the hands' waypoint
        # fractions (arc length), each piece a route segment of the matching share of the duration
        poly, cuts = {}, set()
        for h, pts in tracks.items():
            q = np.vstack([np.asarray(cur[h])[None]] + [np.asarray(x)[None] for x in pts])
            L = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
            f = L / max(L[-1], 1e-12)
            poly[h] = (q, f)
            cuts.update(round(float(x), 9) for x in f[1:] if x > 0)
        cuts = sorted(cuts) or [1.]
        prev = 0.
        for k, c in enumerate(cuts):
            to = {}
            for h, (q, f) in poly.items():
                j2 = min(max(int(np.searchsorted(f, c, side="right")) - 1, 0), len(q) - 2)
                a = (c - f[j2]) / max(f[j2 + 1] - f[j2], 1e-12)
                p = q[j2] + min(max(a, 0.), 1.) * (q[j2 + 1] - q[j2])
                to[h] = (round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2] - sim.table_top), 5))
            segments.append(dict(name=seg.get("name", "move") + (f"_{k + 1}" if len(cuts) > 1 else ""),
                                 duration_s=round(float(seg["duration_s"]) * (c - prev), 4), to=to))
            prev = c
        for h in tracks:
            cur[h] = np.asarray(tracks[h][-1], float)
    return dict(name=stage.get("name"), grasp_uv=grasp_uv, segments=segments,
                settle_s=float(stage.get("settle_s", 0.) or 0.))


class LabProgramPlanner:
    """route["planner"]: runs program(proxy, *args) in a worker thread; each call returns the next stage, or the next
    stages that one measurement fixes (programs.run_stages) ([] = done)."""

    def __init__(self, program, args_fn=lambda planner: (), centre_xy=SHEET_CENTRE_XY, name="lab_program", proxy_attrs=None):
        self.program, self.args_fn, self.centre_xy, self.name = program, args_fn, centre_xy, name
        self.proxy_attrs = dict(proxy_attrs or {})      # route settings the lab program reads off its sim (speedup_1013)
        self.reset()

    def reset(self):
        self.out, self.inp = queue.Queue(), queue.Queue()
        self.thread, self.proxy, self.calls, self.error = None, None, 0, None

    def _worker(self):
        try:
            self.program(self.proxy, *self.args_fn(self))
        except BaseException as e:            # reported to the robot runner on the next call
            self.error = e
        finally:
            self.out.put(None)

    def __call__(self, cloth, uv, table, target=None, task=None, last=None):
        """last (run_route_oracle): the previous planned stage's outcome (skipped, admitted, grasp_lost) or None."""
        self.calls += 1
        if self.thread is None:
            self.proxy = ProxyLab(uv, table, self.centre_xy)
            for k, val in self.proxy_attrs.items():
                setattr(self.proxy, k, val)
            self.proxy._P = np.asarray(cloth, float).copy()
            self.target, self.task = target, task

            def plan(route_stages):
                self.out.put(route_stages)
                return self.inp.get()
            self.proxy._plan = plan
            self.thread = threading.Thread(target=self._worker, name=self.name, daemon=True)
            self.thread.start()
        else:
            self.inp.put((np.asarray(cloth, float).copy(), last))
        nxt = self.out.get()
        if nxt is None:
            if self.error is not None:
                raise RuntimeError(f"{self.name}: the lab program failed: {self.error!r}") from self.error
            return []
        return list(nxt)


# ---------------------------------------------------------------------------------------------- the flatten tiers
# the lab program a flatten route runs, where it is not the tier's own (2026-10-13, d38cdf4bfb: hard4b's program ended
# after its fold layouts with the sheet not yet covered or flat, 8166 rows; hard4b_fin2 adds the finishing relays, the
# flatten that blanket_fold hard's route runs successfully on the same head). hard4b_fin2 failed on its own too (twice:
# the far hem folded back on top, coverage 0.94 of 0.95); hard4b_hem re-centres before the fold layouts and lays the far
# hem back out before and after the finishing relays (reverse.with_hem_finish). blanket_fold hard keeps hard4b_fin2 as
# its flatten part (its tier names it).
FLATTEN_PROGRAM = dict(hard4b="hard4b_hem")


def _flatten_route(name):
    from topogym.planning.blanket_flatten.redesign.reverse import ROUTES_FIXUP
    from topogym.planning.blanket_flatten.redesign.tiers import TIERS
    prog = ROUTES_FIXUP[FLATTEN_PROGRAM.get(name, name)]
    return dict(stages=(), max_strain_ratio=float(TIERS[name].get("strain_guard", 1.5)),
                planner=LabProgramPlanner(prog, lambda planner: (TIERS[name],), name=f"redesign_{name}",
                                          proxy_attrs=dict(speedup_1013=name in FAST_FLATTEN)),
                planner_rounds=80, footing=dict(FOOTING), fast=True, redesign=dict(tier=name, start=TIERS[name].get("start")),
                **(dict(planner_settle_s=PLANNER_SETTLE_S) if name in FAST_FLATTEN else {}))


REDESIGN_FLATTEN = ("very_easy4c", "easier4", "easy4f", "medium4e", "hard4b")


# ---------------------------------------------------------------------------------------------- the blanket_fold tiers
FOLD_TIERS = dict(very_easy="b_very_easy", easier="b_easier", easy="b_easy", medium="b_medium", hard="b_hard")


def _fold_route(difficulty):
    """blanket_fold_<difficulty>: the ladder-B oracle; its target is the Task's frozen fold target (task=...)."""
    from topogym.planning.blanket_fold.redesign.oracle import ORACLES
    from topogym.tasks.blanket_fold.redesign.tiers import TIERS
    lab = FOLD_TIERS[difficulty]

    def args(planner):
        if planner.task is None or getattr(planner.task, "fold_target_particles", None) is None:
            raise ValueError("a blanket_fold route needs the Task (planner(..., task=task)) for its fold target")
        return (TIERS[lab], np.asarray(planner.task.fold_target_particles, float))
    fast = difficulty in FAST_FOLD
    planner = LabProgramPlanner(ORACLES[lab], args, name=f"blanket_fold_{lab}",
                                proxy_attrs=dict(speedup_1013=fast))
    return dict(stages=(), max_strain_ratio=1.5, planner=planner, planner_rounds=80, footing=dict(FOOTING), fast=True,
                redesign=dict(tier=lab), **(dict(planner_settle_s=PLANNER_SETTLE_S) if fast else {}))


def redesign_routes():
    """ROUTES entries for routes.py (fresh planners per call: a planner holds one episode's state): the redesigned
    blanket_flatten tiers by their lab names, the blanket_fold tiers as blanket_fold_<difficulty>."""
    out = {n: _flatten_route(n) for n in REDESIGN_FLATTEN}
    out.update({f"blanket_fold_{d}": _fold_route(d) for d in FOLD_TIERS})
    return out
