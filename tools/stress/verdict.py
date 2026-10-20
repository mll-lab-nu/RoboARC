"""Verdicts of stress results (docs/simulator/STRESS_VALIDATION.md, criteria v2.11) and one table per instance.

usage: verdict.py <results dir> [<results dir> ...] [--reading v2.13b|v2.12|v2.11|v2.10|v2.9|v2.8|v2.7|v2.6|v2.5|v2.4|v2.3|v2.2|v2] [--json out.json]
                  [--md out.md]
Verdict per stream:
  BREAKDOWN  any of: a world fault, a non-finite state, a MuJoCo instability or buffer warning; the Session's own
             breakdown end (simulator_instability, forced_breakdown); a contact deeper than 10 mm in any MuJoCo step;
             tunnelling (a task link's centre of mass below the tabletop, inside the table); an uncaused speed jump; a
             stored-energy pop whose kick (the change of the link's velocity vector within the tick) is above 3 m/s;
             energy created, globally (X > 0.1 J) or locally (all three local signals above 0.25 J; the culprit is a
             contact pair's or an equality's / limit's net work). Pops with a kick at or below 3 m/s are legitimate
             dynamics: listed ("pop"), not a verdict.
  VIOLENT    no breakdown, but a forensic trigger (task speed > 3 m/s, a link spinning above 60 rad/s whose surface moves
             faster than 3 m/s, any contact force > 2000 N, a task link leaving the workspace, a causal whip or a
             chain's redistribution that leaves its link above 3 m/s): violent but causal, reported. Slower whips are
             listed ("whip <= 3 m/s"), as pops at or below 3 m/s are.
  CLEAN      neither
--reading gives superseded readings for comparison: v2.10 (every fast pop the rod does not pay for is a breakdown, whatever its constraint-stored share), v2.9 (rule 6's link clause strict; every fast pop a breakdown, a rod's elastic release included), v2.8 (a pop fails on its after-speed above 3 m/s; a chain's
redistribution balances within its tick only; the local rule's culprit is a contact pair only), v2.7 (every whip and
redistribution a trigger, whatever its speed; the same BREAKDOWN rules as v2.8), v2.6 (also the angular trigger on any link above 60 rad/s, whatever its size), v2.5
(a jump causal only through gravity's term or through a tie's work
or a stored-potential drop counts as uncaused), v2.4 (also a chain's redistribution), v2.3 (also a
robot-powered jump), v2.2 (also the local energy test X rise + pair net) and v2 (X + S rise + pair net). Every acceptance
report shows the readings side by side."""
import json, sys
from collections import defaultdict
from pathlib import Path

# F5 energy created. X = E - E0 - W: kinetic + potential energy minus the work of the drives,
# gravity compensation and passive forces. Global: a passive scene born at rest has X <= 0 at all times (rigorous).
# Local (creation after earlier dissipation), CRITERIA v2.3 (2026-10-09, frozen): all three signals above ENERGY_LOCAL_J:
#   the X rise within 0.1 s (the exact books show it), the rise of X + S above its running minimum (not a release of
#   stored constraint energy; S = sum max(0, -f.pos/2)), and one contact pair's net work over its contact episode (a
#   culprit). Each alone has a known false-positive mode (X rise: releases; X + S: S inflated at impacts, flickering in
#   squeezes; pair net: circulation in constraint loops). LOCAL_RULE (env VALIDATE_LOCAL_RULE) reproduces the earlier
#   readings: "x_plus_s" = the v2 letter (X + S and pair), "x_rise" = v2.2 (X rise and pair), "all3" = v2.3.
ENERGY_GLOBAL_J = 0.1
ENERGY_LOCAL_J = 0.25
PRESS_KE_CALM_J = 0.05     # v2.12 pressed pile: the cloth stays calm under a press (directed_e6: KE 1-2 mJ, W2b: see docs)
import os as _os
# reading -> (local-energy rule, robot-powered jumps causal, a chain's redistribution causal, the v2.6 extensions causal,
# the angular trigger's form, the whip speed floor). v2.11 (below) is in force; the others are for side-by-side reports. Both
# last parts are VIOLENT triggers only, never failures. The angular trigger: "surface" (v2.7) = a link spinning above
# 60 rad/s whose surface moves faster than 3 m/s (omega * r_link, r_link = the farthest point of its geoms from its
# centre of mass); "omega" (up to v2.6) = any link above 60 rad/s. The whip floor (v2.8): a causal whip or a chain's
# redistribution (rule 6) is a trigger only when its link ends above 3 m/s, the speed trigger's own threshold; up to
# v2.7 every one was. A cloth corner snapping back at 1.2 m/s on a grasp's release reads alike from v2.8 on.
READINGS = {"v2.8": ("all3", True, True, True, "surface", True), "v2.7": ("all3", True, True, True, "surface", False),
            "v2.6": ("all3", True, True, True, "omega", False), "v2.5": ("all3", True, True, False, "omega", False),
            "v2.4": ("all3", True, False, False, "omega", False), "v2.3": ("all3", False, False, False, "omega", False),
            "v2.2": ("x_rise", False, False, False, "omega", False), "v2": ("x_plus_s", False, False, False, "omega", False)}
# v2.9 (2026-10-10), on v2.8's rules: (a) the local rule's culprit signal also takes the equality and limit
# books (constraint_net_work_j); (c) a chain's redistribution (rule 6) balances over a 3-tick look-back window
# (whip_redistribution_v29; v2.8 counts it as uncaused); (d) a pop fails when its kick |v_after - v_before| exceeds
# 3 m/s (pop_kick_v29; v2.8: its after-speed, pop_fast_v28); (e) no peaks or events after the first mjWARN_BADQACC,
# MuJoCo's auto-reset to qpos0 (the observer's summary gives post_reset). The X + S signal stays -1/2 f.pos; two
# estimates are recorded for forensics only (energy_created_spring_j: the springs' energy; energy_created_win_j: S
# windowed over 20 ms) and can be read for comparison as v2.9s / v2.9w. Results recorded before v2.9 lack the books,
# the window and the kick: they are read as v2.8.
READINGS["v2.9"] = READINGS["v2.8"]
READINGS["v2.9s"] = READINGS["v2.8"]
READINGS["v2.9w"] = READINGS["v2.8"]
# v2.10 (2026-10-11), on v2.9's rules: rule 6's link clause tolerates LINK_OUT_EPS_J (1e-6 J) of positive
# outside work, contact round-off (whip_redistribution_v210, a whip; every earlier reading counts it as uncaused); the
# pop guard covers releases of energy stored in contacts, equalities and limits only: a fast pop the rod's own elastic
# energy pays for (pop_rod_v210_fast / _kick / _after, the suffix = its class up to v2.9) is no breakdown; it stays
# visible through the speed trigger and is listed as a note.
READINGS["v2.10"] = READINGS["v2.8"]
# v2.11 (2026-10-11, in force), on v2.10's rules: a fast pop is a breakdown only if the constraint-stored energy released into that
# link alone (contacts, equalities, limits; the rod excluded) could give it the guard's speed, v_c = sqrt(2 E_c / m) > 3 m/s
# (pop_share_v211_fast / _kick / _after, the suffix = its class up to v2.10; earlier readings count it by that class).
READINGS["v2.11"] = READINGS["v2.8"]
# v2.12 (in force from 2026-10-12, user): v2.11's rules (its energy measures unchanged) plus two VBD-cloth rules. A cloth
# particle kicked faster than 3 m/s fails only if the contact storage released into it alone gives v_c > 3 m/s
# (vbd.<cloth>.pop12.fast, v2.11's energy share); the cloth's own balance with its storage (X) may not rise above its
# running minimum by more than 0.25 J, judged apart from the system sum; with two-way coupling, the cloth and the
# coupling together (X + C, C = its work on the cloth plus its work on the robot) may not stay more than 0.25 J above
# their running minimum for 10 ms. Results without the v2.12 books read as v2.11.
READINGS["v2.12"] = READINGS["v2.8"]
# v2.13b (adopted by the user 2026-10-13, in force for rounds from then on; PASSed rounds keep their v2.12 verdicts):
# v2.12's rules on the integrator-side books (observer: contact work J(q0)' f . v_mean, armature energy in link KE) plus
# one fail class, energy_created(integrator): in one local event the excess and the integrator's own residual (the
# excess less every constraint row's work) both rose 0.25 J in 0.1 s and no other energy rule failed. No new threshold.
READINGS["v2.13b"] = READINGS["v2.8"]
BOOKS29 = {"v2.9", "v2.9s", "v2.9w", "v2.10", "v2.11", "v2.12", "v2.13b"}
BOOKS210 = {"v2.10", "v2.11", "v2.12", "v2.13b"}
BOOKS211 = {"v2.11", "v2.12", "v2.13b"}
BOOKS212 = {"v2.12", "v2.13b"}
BOOKS213 = {"v2.13b"}
Y29 = {"v2.9": "energy_created_j", "v2.9s": "energy_created_spring_j", "v2.9w": "energy_created_win_j",
       "v2.10": "energy_created_j", "v2.11": "energy_created_j", "v2.12": "energy_created_j",
       "v2.13b": "energy_created_j"}
DEFAULT_READING = _os.environ.get("STRESS_READING", "v2.13b")
BAD_WARNINGS = ("mjWARN_BADQACC", "mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_CONTACTFULL", "mjWARN_CNSTRFULL")


def pk(o, k):
    v = (o.get("peaks") or {}).get(k)
    return v["value"] if v else 0.


def classify(r, reading=None):
    """One result (replay.py or a search episode) -> its verdict, failures, triggers, notes and peaks. reading: None (v2.9,
    the criteria in force, unless STRESS_READING says otherwise), or "v2.10" / "v2.9" / "v2.8" / "v2.7" / "v2.6" / "v2.5" / "v2.4" /
    "v2.3" / "v2.2" / "v2" (or the forensic "v2.9s" / "v2.9w")."""
    rule, work_ok, redis_ok, v26_ok, ang_form, whip_floor = READINGS[reading or DEFAULT_READING]
    o = r.get("observer") or {}
    c = o.get("counts") or {}
    fails, trig = [], []
    if r.get("fault"):
        fails.append("fault")
    if o.get("observer_errors"):
        fails.append(f"observer_error x{o['observer_errors']} (a harness bug, not physics)")
    if o.get("nonfinite"):
        fails.append("nonfinite")
    w = [k for k in (o.get("warnings") or {}) if k in BAD_WARNINGS]
    if w:
        fails.append("warn:" + ",".join(w))
    end = r.get("episode_end") or ""
    if end == "simulator_instability" or end.startswith("forced_breakdown"):
        fails.append("session:" + end[:40])
    if c.get("deep_task_contact"):
        fails.append("task depth>10mm")
    if c.get("deep_robot_contact"):
        fails.append("robot depth>10mm")
    if c.get("tunnel"):
        fails.append("tunnel")
    v29_ok = (reading or DEFAULT_READING) in BOOKS29          # v2.9 (c): rule 6 over a 3-tick window
    v210_ok = (reading or DEFAULT_READING) in BOOKS210       # v2.10: rule 6's link clause with its round-off tolerance
    v211_ok = (reading or DEFAULT_READING) in BOOKS211       # v2.11: the pop guard's energy share
    n_unc = (c.get("jump_uncaused", 0) + (0 if v29_ok else c.get("whip_redistribution_v29", 0))
             + (0 if v210_ok else c.get("whip_redistribution_v210", 0))
             + (0 if work_ok else c.get("jump_work_causal", 0))
             + (0 if redis_ok else c.get("whip_redistribution", 0))
             + (0 if v26_ok else c.get("jump_causal_gravity", 0) + c.get("whip_redistribution_v26", 0)))
    rod_f, rod_k, rod_a = (c.get("pop_rod_v210_" + x, 0) for x in ("fast", "kick", "after"))
    sh_f, sh_k, sh_a = (c.get("pop_share_v211_" + x, 0) for x in ("fast", "kick", "after"))
    n_popfast = (c.get("pop_fast", 0) + (c.get("pop_kick_v29", 0) if v29_ok else c.get("pop_fast_v28", 0))
                 + (0 if v26_ok else c.get("jump_causal_gravity_pop", 0))
                 + (0 if v210_ok else rod_f + (rod_k if v29_ok else rod_a))
                 + (0 if v211_ok else sh_f + (sh_k if v29_ok else sh_a)))
    if n_unc:
        fails.append(f"uncaused_jump x{n_unc}")
    if n_popfast:
        fails.append(f"pop kick>3m/s x{n_popfast} ({pk(o, 'pop_kick'):.1f} m/s)" if v29_ok and "pop_kick" in (o.get("peaks") or {})
                     else f"pop>3m/s x{n_popfast} ({pk(o, 'pop_speed'):.1f} m/s)")
    notes = []
    n_pop_ok = (c.get("pop_stored_release", 0) + (c.get("pop_fast_v28", 0) if v29_ok else c.get("pop_kick_v29", 0))
                + (0 if v210_ok else (rod_a if v29_ok else rod_k))
                + (0 if v211_ok else (sh_a if v29_ok else sh_k)))
    n_rod = rod_f + rod_k + rod_a if v210_ok else 0
    n_share = sh_f + sh_k + sh_a if v211_ok else 0
    if n_pop_ok:
        notes.append(f"pop x{n_pop_ok} <= {pk(o, 'pop_speed'):.2f} m/s" + (f" (kick <= 3 m/s)" if v29_ok else ""))
    if n_rod:
        notes.append(f"rod release x{n_rod} (v2.10: rod elastic energy; kick {pk(o, 'pop_kick'):.2f} m/s)")
    if n_share:
        notes.append(f"pop below its energy share x{n_share} (v2.11: constraint-stored release could give <= 3 m/s)")
    slips = int(c.get("grasp_lost:grasp_slipped", 0))
    lost = {k.split(":", 1)[1]: v for k, v in c.items() if k.startswith("grasp_lost:")}
    if slips:
        notes.append(f"grasp slip x{slips}")
    X, Yr, P = pk(o, "energy_excess_j"), pk(o, "energy_created_j"), pk(o, "contact_net_work_j")
    Xr = pk(o, "energy_rise_0p1s_j")
    culprit = ((o.get("peaks") or {}).get("contact_net_work_j") or {}).get("where")
    if (reading or DEFAULT_READING) in BOOKS29 and "energy_created_win_j" in (o.get("peaks") or {}):
        Yr = pk(o, Y29[reading or DEFAULT_READING])
        if pk(o, "constraint_net_work_j") > P:
            P = pk(o, "constraint_net_work_j")
            culprit = (o["peaks"]["constraint_net_work_j"] or {}).get("where")
    books212 = [b for b in (o.get("vbd") or {}).values() if isinstance(b, dict) and "pop12" in b]
    warnings = []
    if (reading or DEFAULT_READING) in BOOKS212 and books212:
        n12 = sum(int((b.get("pop12") or {}).get("fast", 0)) for b in books212)
        if n12:
            fails.append(f"cloth pop x{n12} (v2.12: contact storage released into one particle gives > 3 m/s)")
        # the cloth's own guard, judged apart from the system sum (MuJoCo-side dissipation cannot hide a cloth
        # creation): its balance with its storage (the book's X: E incl. barrier + penalties, less the robot's work) may
        # not rise above its running minimum by more than the local threshold. Not the balance without storage: storage
        # that booked robot work charged rises it when released (blanket_vbd_gpu BLF2c random_e9: 1.05 J, X 0.5 mJ).
        # Pressed pile (user 2026-10-13): steps taken while a gripper was held on its cloth floor (the table guard's cloth
        # stop within the observer's PRESS_WINDOW_S) are booked apart (X_press). Their rise is the pile's stored
        # barrier / penalty energy under a press the agent commands, not a creation: a WARNING when the cloth stays calm
        # (KE <= PRESS_KE_CALM_J, no particle kick above 3 m/s, no pop), VIOLENT (agent-caused) when it throws cloth.
        # Steps outside a press (X_free) keep the guard. Books without the split read as before (X).
        if all("X_free_max_rise" in b for b in books212):
            xf = max(float(b["X_free_max_rise"]) for b in books212)
            xp = max(float(b["X_press_max_rise"]) for b in books212)
            if xf > ENERGY_LOCAL_J:
                fails.append(f"energy_created(cloth) X rose {xf:.2f}J outside a guarded press (v2.12)")
            elif xp > ENERGY_LOCAL_J:
                ke = max(float(b.get("KE_press_max", 0.)) for b in books212)
                kk = max(float(b.get("kick_press_max", 0.)) for b in books212)
                if ke <= PRESS_KE_CALM_J and kk <= 3.0 and not n12:
                    warnings.append(f"pressed-pile storage, agent-caused: X rose {xp:.2f}J under a guarded press "
                                    f"(cloth KE <= {ke * 1000:.1f} mJ, kick <= {kk:.2f} m/s)")
                else:
                    trig.append(f"pressed pile thrown {xp:.2f}J (KE {ke:.3f} J, kick {kk:.2f} m/s; agent-caused)")
        else:
            xc = max(float(b.get("X_max_rise", 0.)) for b in books212)
            if xc > ENERGY_LOCAL_J:
                fails.append(f"energy_created(cloth) X rose {xc:.2f}J (v2.12)")
        # two-way coupling (cloth_coupling.py, user 2026-10-12): the coupling's work is booked apart from both balances
        # (on the robot inside the MuJoCo side's xfrc work, on the cloth as the robot's work W_robot). The cloth and the
        # coupling together, XC = X + C (C = W_couple + W_react), may not rise more than the local threshold above its
        # running minimum for 4 consecutive cloth steps (10 ms): a contact's onset is booked on the cloth at its
        # end-of-step force while the robot pays the reaction a step later, a one-step spike of C the next steps repay
        # (BLF2a cem_e1: C +0.43 J in one step, ending at -16.6 J). Books from before XC_sustained_rise (C_max_rise only)
        # pass when X and C together cannot have risen that far, and otherwise fail as unresolved (re-run them).
        tw = [b for b in books212 if "C_max_rise" in b]
        if tw:
            if all("XC_sustained_rise" in b for b in tw):
                xcs = max(float(b["XC_sustained_rise"]) for b in tw)
                if xcs > ENERGY_LOCAL_J:
                    fails.append(f"energy_created(cloth+coupling) X+C held {xcs:.2f}J for 10 ms (v2.12)")
            else:
                bound = max(float(b.get("X_max_rise", 0.)) + float(b["C_max_rise"]) for b in tw)
                if bound > ENERGY_LOCAL_J:
                    fails.append(f"energy_created(cloth+coupling) unresolved: X+C rise bound {bound:.2f}J, books without "
                                 "XC_sustained_rise (re-run) (v2.12)")
    if X > ENERGY_GLOBAL_J:
        fails.append(f"energy_created(global) {X:.2f}J @tick {o['peaks']['energy_excess_j']['tick']}")
    elif rule == "x_plus_s" and Yr > ENERGY_LOCAL_J and P > ENERGY_LOCAL_J:
        w = (o["peaks"]["contact_net_work_j"] or {}).get("where")
        fails.append(f"energy_created(local) {Yr:.2f}J, contact {w} gave {P:.2f}J @tick {o['peaks']['energy_created_j']['tick']}")
    elif rule == "all3" and Xr > ENERGY_LOCAL_J and Yr > ENERGY_LOCAL_J and P > ENERGY_LOCAL_J:
        w = culprit
        fails.append(f"energy_created(local) X rose {Xr:.2f}J in 0.1 s, X+S {Yr:.2f}J, contact {w} gave {P:.2f}J "
                     f"@tick {o['peaks']['energy_rise_0p1s_j']['tick']}")
    elif rule == "x_rise" and Xr > ENERGY_LOCAL_J and P > ENERGY_LOCAL_J:
        w = (o["peaks"]["contact_net_work_j"] or {}).get("where")
        fails.append(f"energy_created(local) X rose {Xr:.2f}J in 0.1 s @tick {o['peaks']['energy_rise_0p1s_j']['tick']}, "
                     f"contact {w} gave {P:.2f}J")
    # v2.13b (observer STRESS_INTEGRATOR_BOOKS): energy the step made with no constraint row booking it. In one
    # local event (the 0.1 s window of energy_rise_0p1s_j) the excess rose above the local threshold and so did the
    # integrator's own residual (the excess less every contact / equality / limit row's work, J' f . v_mean): the step,
    # not a contact, created it. It only fills the culprit rules' gap: no other rule changes.
    if ((reading or DEFAULT_READING) in BOOKS213 and pk(o, "integrator_created_j") >= ENERGY_LOCAL_J
            and not any(f.startswith("energy_created") for f in fails)):
        ic = o["peaks"]["integrator_created_j"]
        fails.append(f"energy_created(integrator) X and the step's residual rose {ic['value']:.2f}J in 0.1 s @tick "
                     f"{ic.get('tick')} (residual episode max {pk(o, 'step_residual_rise_j'):.2f}J)")
    task_force = max(pk(o, "force_task_robot"), pk(o, "force_task_task"), pk(o, "force_task_static"))
    any_force = max(task_force, pk(o, "force_robot_static"), pk(o, "force_robot_robot"))
    if pk(o, "task_speed") > 3:
        trig.append(f"speed {pk(o, 'task_speed'):.1f}")
    # the angular trigger (see READINGS). A result recorded before v2.7 has no surface-speed peak (task_tip_speed); it
    # is read with the v2.6 trigger.
    if ang_form == "surface" and "task_tip_speed" in (o.get("peaks") or {}):
        if pk(o, "task_angular_fast_surface") > 60:
            trig.append(f"ang {pk(o, 'task_angular_fast_surface'):.0f} (surface > 3 m/s)")
    elif pk(o, "task_angular") > 60:
        trig.append(f"ang {pk(o, 'task_angular'):.0f}")
    if any_force > 2000:
        trig.append(f"force {any_force:.0f}")
    if c.get("left_workspace"):
        trig.append("left_ws")
    def whips(sfx=""):
        return (c.get("whip_causal" + sfx, 0) + (c.get("whip_redistribution" + sfx, 0) if redis_ok else 0)
                + (c.get("whip_redistribution_v26" + sfx, 0) if v26_ok else 0)
                + (c.get("whip_redistribution_v29" + sfx, 0) if v29_ok else 0)
                + (c.get("whip_redistribution_v210" + sfx, 0) if v210_ok else 0))
    nwhip = whips()
    # the whip floor (see READINGS). A result recorded before v2.8 has no whip speed (whip_speed); it is read without it.
    if whip_floor and nwhip and "whip_speed" in (o.get("peaks") or {}):
        fast = whips("_fast")
        if nwhip - fast:
            notes.append(f"whip <= 3 m/s x{nwhip - fast}")
        nwhip = fast
    if nwhip:
        trig.append(f"whip x{nwhip}")
    verdict = "BREAKDOWN" if fails else "VIOLENT" if trig else "CLEAN"
    task_depth = max(pk(o, "depth_task_robot"), pk(o, "depth_task_task"), pk(o, "depth_task_static"))
    return dict(id=r.get("id"), base=Path(r["instance"]).parent.name, rows=r.get("rows"), steps=r.get("control_steps"),
                end=end or None, verdict=verdict, fails=fails, triggers=trig, notes=notes + warnings, warnings=warnings,
                pops=list(o.get("pop_speeds") or []), grasp_slips=slips, grasp_lost=lost,
                speed=round(pk(o, "task_speed"), 3), ang=round(pk(o, "task_angular"), 1),
                task_force=round(task_force, 1), any_force=round(any_force, 1),
                task_depth_mm=round(1000 * task_depth, 3), robot_depth_mm=round(1000 * max(pk(o, "depth_robot_static"), pk(o, "depth_robot_robot")), 3),
                energy_rise=round(pk(o, "energy_rise_0p1s_j"), 4), energy_excess=round(pk(o, "energy_excess_j"), 4),
                energy_created=round(pk(o, "energy_created_j"), 4), contact_net=round(pk(o, "contact_net_work_j"), 4),
                x_rise=round(Xr, 4),
                robot_speed=round(pk(o, "robot_speed"), 3), slew=r.get("slew_limited_steps"),
                capped=o.get("solver_capped_substeps"), ms=r.get("ms_per_control"), success=r.get("success"),
                review=bool(r.get("review")), jumps_causal=c.get("jump_causal", 0),
                commit=(r.get("provenance") or {}).get("repo_commit") or r.get("tested_commit", "") or "")


def load(dirs, reading=None):
    rows = []
    for run in dirs:
        for p in sorted(Path(run).glob("*.json")):
            if p.name.startswith("_"):
                continue
            r = json.loads(p.read_text())
            if isinstance(r, dict) and "instance" in r:
                rows.append(dict(classify(r, reading), run=str(run)))
    return rows


def main():
    args = sys.argv[1:]
    out_json = args[args.index("--json") + 1] if "--json" in args else None
    out_md = args[args.index("--md") + 1] if "--md" in args else None
    reading = args[args.index("--reading") + 1] if "--reading" in args else None
    runs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--json", "--md", "--reading"))]
    rows = load(runs, reading)
    by = defaultdict(list)
    for r in rows:
        by[r["base"]].append(r)
    lines = []
    for base in sorted(by):
        rs = sorted(by[base], key=lambda r: (r["verdict"] != "BREAKDOWN", r["verdict"] != "VIOLENT", r["id"]))
        n = {v: sum(r["verdict"] == v for r in rs) for v in ("BREAKDOWN", "VIOLENT", "CLEAN")}
        lines.append(f"\n### {base}: {len(rs)} streams, {n['BREAKDOWN']} breakdown, {n['VIOLENT']} violent but causal, {n['CLEAN']} clean\n")
        lines.append("| stream | steps | end | verdict | why | v m/s | w rad/s | F task N | F any N | depth task mm | E created J | contact net J | ms |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            why = "; ".join(r["fails"] + r["triggers"] + r.get("notes", []))
            lines.append(f"| {r['id']} | {r['steps']} | {r['end'] or ''} | {r['verdict']} | {why} | {r['speed']} | {r['ang']} | "
                         f"{r['task_force']} | {r['any_force']} | {r['task_depth_mm']} | {r['energy_created']} | {r['contact_net']} | {r['ms']} |")
    text = "\n".join(lines)
    print(text)
    if out_md:
        Path(out_md).write_text(text + "\n")
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
