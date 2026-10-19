"""Start states of the redesigned scenes on the MuJoCo chain (inverse generation, step 2): the authored (lifted) knot is
settled, tightened by a simulated pull (the virtual hands pinch both sides of each knot `side_offsets_m` outside its
outermost crossings and pull apart along the rope until the pinch force reaches `force_n`), lowered, the anchored ends
are welded to the world (the clamps), the hands release and the rope settles. Deterministic (single-thread MuJoCo)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .mjlab import Lab, to_local


def knot_spans(lab, rope, gap=.08):
    """Material spans [s_first, s_last] of the knots of `rope`: its self-crossing stations clustered along s (a gap
    longer than `gap` of rope without a self-crossing visit separates two knots)."""
    names = list(lab.rods)
    k = names.index(rope)
    ss = []
    for c in lab.crossings():
        if c["a"][0] == k and c["b"][0] == k:
            ss += [(c["a"][1], c["b"][1])]
    if not ss:
        return []
    # union of [s_i, s_j] intervals of each crossing (a knot's crossings interleave), then merge
    iv = sorted([sorted(x) for x in ss])
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def knot_span(lab, rope):
    spans = knot_spans(lab, rope)
    return (spans[0][0], spans[-1][1]) if spans else None


def tighten(lab, rope, span, *, anchored_start=True, pin_offset=.05, force_n=3., lift=.008, max_pull=.9,
            relay_end=True, log=None):
    """Tighten the knot(s) of `rope` by a simulated pull: the left virtual hand pins the rope `pin_offset` before the
    (first) knot on its anchored side; the right one pinches the free end and pulls it away from the knot (along the
    knot -> free end direction) in 4 / 2 / 1 mm steps (finer near `force_n`) until its 20 ms mean force reaches
    force_n: the free tail straightens behind the hand (a pull on the end never pushes rope into loops) and the knots
    between pin and hand cinch. Hold 0.3 s, lower, release; relay_end: carry the free end back to where it lay before
    (the tail relaxes on the table in a curve instead of staying a taut straight line)."""
    L = lab.length(rope)
    if anchored_start:
        s_pin, s_end = max(.03, span[0] - pin_offset), L - .02
    else:
        s_pin, s_end = min(L - .03, span[1] + pin_offset), .02
    end0 = lab.point(rope, s_end).copy()
    centre = lab.point(rope, .5 * (span[0] + span[1]))
    for h in lab.hands:
        lab.hands[h]["limit"] = 1e9
    lab.grasp("left", rope, s_pin, check_pinch=False)
    lab.grasp("right", rope, s_end, check_pinch=False)
    up = np.array([0., 0., lift])
    lab.move({"left": lab.hand_pos("left") + .5 * up, "right": lab.hand_pos("right") + up}, speed=.03)
    axis = (lab.hand_pos("right") - centre) * [1, 1, 0]
    axis /= np.linalg.norm(axis)
    trace, pulled, k = [], 0., 0
    while pulled < max_pull:
        f = float(np.mean(lab.hands["right"]["window"]))
        trace.append((round(lab.t, 3), round(f, 3)))
        if f >= force_n:
            break
        step = .004 if f < .4 * force_n else (.002 if f < .75 * force_n else .001)
        lab.move({"right": lab.hand_pos("right") + step * axis}, speed=.05, min_time=.02)
        pulled += step
        k += 1
    lab.wait(.3)
    # slacken first (the pin still holds): releasing the pin under tension would yank the knot toward the hand
    lab.move({"right": lab.hand_pos("right") - .03 * axis}, speed=.03)
    lab.move({"left": lab.hand_pos("left") - .5 * up, "right": lab.hand_pos("right") - up}, speed=.02)
    if relay_end:
        # carry the free end back to where it lay, on an arc (the slack tail lies down as a curve, not doubled back)
        p = lab.hand_pos("right")
        target = end0.copy()
        target[2] = p[2]
        chord = target - p
        normal = np.array([-chord[1], chord[0], 0.]) / max(np.linalg.norm(chord[:2]), 1e-9)
        if np.dot(normal[:2], (p - centre)[:2]) < 0:
            normal = -normal
        mid = .5 * (p + target) + .3 * np.linalg.norm(chord[:2]) * normal
        lab.move({"right": p + [0, 0, .02]}, speed=.05)
        lab.move({"right": mid + [0, 0, .02]}, speed=.08)
        lab.move({"right": target + [0, 0, .02]}, speed=.08)
        lab.move({"right": target}, speed=.03)
    lab.release("right")
    lab.place_hand("right", lab.hand_pos("right") + [0, 0, .25], lab.d.mocap_quat[lab.hands["right"]["mocap"]].copy())
    lab.wait(.2)
    lab.release("left")
    lab.place_hand("left", lab.hand_pos("left") + [0, 0, .25], lab.d.mocap_quat[lab.hands["left"]["mocap"]].copy())
    rec = dict(span_m=list(span), stations_m=[s_pin, s_end], steps=k, pulled_m=round(pulled, 3),
               final_force_n=trace[-1][1], crossings_after=len(lab.crossings()))
    if log is not None:
        log.append(dict(op="tighten", rope=rope, **rec))
    return rec


def run_birth(scene, out_dir, *, quiet_limit=.001):
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lab = Lab(dict(ropes=[dict(name=r["name"], centerline=np.asarray(r["centerline"]), radius=r["radius"])
                          for r in scene["ropes"]], fixtures=scene.get("fixtures", [])))
    log = []
    for st in scene["birth"]["steps"]:
        op = st["op"]
        if op == "settle":
            ok = lab.settle(max_s=st.get("max_s", 2.), limit=st.get("limit", .002), hold=.2)
            log.append(dict(op="settle", t=lab.t, quiet=ok, crossings=len(lab.crossings())))
        elif op == "tighten":
            anchored = {r["name"]: (r.get("anchor") or {}).get("end") for r in scene["ropes"]}
            start_side = anchored.get(st["rope"]) != "end"
            spans = knot_spans(lab, st["rope"])
            if spans:
                # one pull from the free end cinches every knot between the pin (before the first knot) and the hand
                whole = (spans[0][0], spans[-1][1])
                f = st.get("force_n", 3.)
                f = max(f) if isinstance(f, (list, tuple)) else f
                tighten(lab, st["rope"], whole, anchored_start=start_side, force_n=f, log=log)
                lab.settle(max_s=1., limit=.003, hold=.1)
        elif op == "anchor":
            lab.anchor(st["rope"], st.get("end", "start"), st.get("links", 2))
            log.append(dict(op="anchor", rope=st["rope"], end=st.get("end", "start")))
        elif op == "release_all":
            for h in lab.hands:
                lab.release(h)
            for h in lab.hands:            # park the hands away from the rope
                lab.place_hand(h, lab.hand_pos(h) + [0, 0, .25], lab.d.mocap_quat[lab.hands[h]["mocap"]].copy())
            log.append(dict(op="release_all", t=lab.t))
        print(json.dumps(log[-1], default=float), round(time.time() - t0, 1), "s", flush=True)
    quiet = lab.settle(max_s=3., limit=quiet_limit, hold=.5)
    cls = lab.centerlines_local()
    state = dict(t=lab.t, quiet=quiet, crossings=len(lab.crossings()), goal=lab.goal_metrics(scene.get("regions") or None))
    lab.save_state(out / "start_state.npz")
    start = dict(scene)
    start["start_state"] = dict(centerlines={k: v.tolist() for k, v in cls.items()}, **state,
                                birth_log=log, wall_s=time.time() - t0)
    (out / "scene_start.json").write_text(json.dumps(start, default=float))
    print("start", json.dumps(state, default=float), flush=True)
    return lab, start
