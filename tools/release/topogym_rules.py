"""Grader details for the TopoGym tier instructions (both modes), added to each family's public goal text (TopoGym's task descriptions).

The base text is his: topogym/tasks/goal_description.py describe_goal() + render_goal() (the instance's "public_goal" in
describe.json, rendered by the generators as "## Success requirements" + "## Task failure conditions"). It is copied
verbatim. This module adds ONLY what the grader checks and his text does not state (sizes, radii, distances, counts,
sampling, edge cases), read from his task code at 2ec4b8d6 (build78/predicates_table_2ec4b8d6.md has the file:line
citations), with the tier's values from the frozen instance.json. It repeats nothing his text says and contradicts none
of it. No positions or poses: conditions, thresholds, names and sizes only (the limited-mode rule)."""

RIGID = {"cast_coil", "cast_keyhole", "interlocked_claws", "hedgehog_in_cage", "horseshoe_ring", "bed_bolt",
         "clock_assembling", "battery_replacement", "caliper_sorting", "wire_splicing"}
SOFT = {"knot_untying", "rope_threading", "orbits", "slotted_post_ring"}
# The tooling's physical-validity policy (rcb_topogym.py, tooling v16; ci_checks/test-topogym-safety-rules.py checks the
# numbers match): it replaces the instant volume-failure wording of his text (the task text itself states the contact
# rule: rcb_topogym.rules_text).
CONTACT_RULE = dict(deep_m=.003, sustained_m=.002, held_deep_m=.004, held_sustained_m=.002, window_s=.25)   # review events
SOFT_RULE = dict(window_s=.25, residual_m=.002)
FORCING = dict(near_cap=.95, contact_n=20., event_s=.5, attribution_s=.25)
BREAKDOWN_PENETRATION_M = .010
PARTICLE_SPEED_LIMIT_M_S = 20.0
TABLE_GUARD_MARGIN_M = -.015
STATIC_GUARD_DEPTH_M = .005     # rcb_topogym R5: an arm deeper than this in a static body stops where it is (MuJoCo)
# How far the gripper's lowest collision vertex reaches beyond the TCP along the approach axis, measured with the robots'
# own forward kinematics at the home pose (2026-10-03, lab): ALOHA 16.3 mm open or closed; the Robotiq 2F-85's fingertips
# move forward as they close, 5.5 mm open, 18.7 mm closed.
FINGERTIP_WORDS = {"aloha": "about 16 mm beyond the tool frame (TCP) along its approach axis, open or closed",
                   "panda_robotiq": "about 6 mm beyond the tool frame (TCP) along its approach axis when open and about "
                                    "19 mm when closed (the Robotiq fingertips move forward as they close)"}
PLACEMENT = {"cast_keyhole", "interlocked_claws", "hedgehog_in_cage", "horseshoe_ring", "orbits", "slotted_post_ring"}
# remaining components of the placement families that are free bodies (the rest are fixed), as named in s.objects(); a
# variation's own words (by shape or role: the pieces' colours change with the appearance seed) override the family's
FREE_REMAINDER = {"cast_keyhole": "the fork plate", "interlocked_claws": "the other claw piece",
                  "horseshoe_ring": "the horseshoes and chain links", "slotted_post_ring": "the cord, beads and end discs",
                  "cast_keyhole_double_maze": "the left-hand plate", "interlocked_claws_easier": "the frame piece",
                  **{f"interlocked_claws_rd_{t}": "the lock piece" for t in ("very_easy", "easier", "easy", "medium", "hard")}}
# The grading replays of the GPU families (rcb_topogym.py GRADING_HOLD_S / REPLAY_TOLERANCES, tooling v16; the numbers
# must match: ci_checks/test-topogym-grading-replay.py). family -> {threshold: (task parameter, passing side, slack)}.
GRADING_HOLD_S = 2.0
REPLAY_TOLERANCES = {
    "blanket_flatten": {"coverage": ("minimum_coverage_ratio", "at_least", -.03),
                        "particle_speed": ("maximum_speed_m_s", "at_most", .05)},
    "knot_untying": {"rope_speed": ("terminal.maximum_speed_m_s", "at_most", .00025)},
    "rope_threading": {},
    "slotted_post_ring": {},
    "orbits": {},
    "lock_picking": {},
}
# threshold -> words for (live value, replay value)
TOLERANCE_WORDS = {
    "coverage": lambda live, replay: f"a blanket coverage of {replay:.0%} instead of {live:.0%}",
    "particle_speed": lambda live, replay: (f"cloth speeds up to {replay * 1000:g} mm/s instead of {live * 1000:g} mm/s (the "
                                            "displacement limit during the confirmation stays)"),
    "rope_speed": lambda live, replay: f"rope speeds up to {replay * 1000:g} mm/s instead of {live * 1000:g} mm/s",
}
# failure strings as `s.status()["episode_end"]` reports them (unlimited); limited mode reports `task_failed`
FAIL_STRINGS = {
    "all": ['"forced_breakdown: <what>"'],
    "rigid": ['"A task body fell below the table"'],
    "horseshoe_ring": ['"Physical chain/eye connection lost"'],
    "buzz_wire_3d": ['"wire_touched"', '"Wand fell below table"'],
    "lock_picking": ['"rotor_limit_exceeded"', '"Key fell below table"'],
    "fiber_bundling": ['"fiber_gripper_contact"', '"Fiber below table"'],
    "orbits": ['"connection_or_volume_invalid: <what>"'], "slotted_post_ring": ['"connection_or_volume_invalid: <what>"'],
    "knot_untying": ['"rope_volume_invalid: <what>"'], "rope_threading": ['"rope_volume_invalid: <what>"'],
    "blanket_flatten": ['"Cloth fell below the table"'], "blanket_fold": ['"Cloth fell below the table"'],
}


def _mm(m, nd=1):
    s = f"{float(m) * 1000:.{nd}f}"
    return (s.rstrip("0").rstrip(".") if "." in s else s) + " mm"


def _or(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


def table_guard_words():
    """The guard's limit in words (rcb_topogym.table_guard_words)."""
    mm = abs(TABLE_GUARD_MARGIN_M) * 1000
    return ("the tabletop" if mm == 0 else f"{mm:g} mm below the tabletop" if TABLE_GUARD_MARGIN_M < 0 else
            f"{mm:g} mm above the tabletop")


def _robot_of(inst):
    return ((inst or {}).get("simulator") or {}).get("robot_id")


def _square_mm(inst):
    """Side of the usable goal square: the inner edge of the 5 mm dashed stripe. The instance records the marker's outer
    extent as goal_region size, and the stripe lies wholly outside the containment limit (topogym/assets/geometry/
    goal_region.py), so the square is size - 10 mm. An instance made before the marker grew from 2 mm (2.1.0 and older)
    records a size 6 mm smaller; its texts were made with size - 4 mm."""
    size = inst.get("objects", {}).get("goal_region", {}).get("size")
    return None if not size else (float(size[0]) - .01) * 1000


def claws_separation(inst):
    """True for the interlocked_claws settings whose goal is the whole-piece separation (easy / medium / hard, Cast ABC,
    Rotor: InterlockedClawsTask.separation_only, terminal revision claws_separation_*; Cast Trinity's three pieces pair by
    pair, terminal revision trinity_pairwise_separation_*), not a placement in the goal square (very_easy / easier and
    the rd tiers)."""
    goal = (inst or {}).get("public_goal") or {}
    return goal.get("task_id") == "interlocked_claws" and str(goal.get("terminal_revision", "")).startswith(
        ("claws_separation", "trinity_pairwise_separation"))


def claws_three_pieces(inst):
    """Cast Trinity: the separation of three pieces, every two of them (trinity_terminal.pairwise_separation_candidate)."""
    goal = (inst or {}).get("public_goal") or {}
    return str(goal.get("terminal_revision", "")).startswith("trinity_pairwise_separation")


def claws_separation_words(minimum_separation_m=.01, three=False):
    """The grader-details line of the whole-piece separation (topogym/tasks/interlocked_claws/terminal.py
    separation_candidate; three pieces: trinity_terminal.pairwise_separation_candidate, every two of them): what it
    measures and his text does not state. It replaces the goal-square and separator lines of the placement families, which
    this goal does not check."""
    if three:
        return ("Separation as the grader measures it: for every two of the three pieces, every vertex of both pieces' "
                "visual and collision meshes counts, and the gap is measured along ten directions only: the three world "
                "axes, each of the two pieces' own three body axes, and the line between the two pieces' centres (the mean "
                "of each piece's vertices). For every two pieces, along at least one of them they must be at least "
                f"{_mm(minimum_separation_m, 0)} apart; a separating plane in any other direction does not count. The green "
                "dashed square on the table is not part of this goal: neither containment in it nor a separator plane is "
                "checked.")
    return ("Separation as the grader measures it: every vertex of both pieces' visual and collision meshes counts, and "
            "the gap is measured along ten directions only: the three world axes, each piece's own three body axes, and "
            "the line between the two pieces' centres (the mean of each piece's vertices). Along at least one of them the "
            f"two pieces must be at least {_mm(minimum_separation_m, 0)} apart; a separating plane in any other direction "
            "does not count. The green dashed square on the table is not part of this goal: neither containment in it nor "
            "a separator plane is checked.")


def buzz_aperture_mm(inst):
    """buzz_wire_3d: (loop inner radius, wire radius, usable opening) in mm, from the base's own layout, as the grader has
    it (BuzzWire3DTask.evaluate_transition: aperture = loop_radius - loop_tube_radius - wire_radius). The redesigned
    tiers (rd_*) have smaller loops than the first five; until 2.0.1 this text stated the first five's 12.67 mm for all."""
    from topogym.config.registry import task_definition
    task = task_definition("buzz_wire_3d", inst["variation_id"], 0)(None, seed=0)
    lay = task.build_layout()
    inner = (lay.loop_radius_m - lay.loop_tube_radius_m) * 1e3
    return inner, lay.wire_radius_m * 1e3, inner - lay.wire_radius_m * 1e3


def grader_details(fam, inst, *, limited):
    """Bullet lines (without the leading "- ") for one instance."""
    tp = inst.get("effective_task_parameters", {}) or {}
    g = tp.get("geometry", {}) or {}
    sim = inst.get("simulator", {}) or {}
    dt, decimation = float(sim.get("physics_dt_s", .0025)), int(sim.get("control_decimation", 10))
    substeps = int(sim.get("solver_substeps", 4))
    import math
    window_ticks = max(1, math.ceil(CONTACT_RULE["window_s"] / dt - 1e-12))
    out = []
    if claws_separation(inst):
        out.append(claws_separation_words(tp.get("minimum_separation_m", .01), three=claws_three_pieces(inst)))
    elif fam in PLACEMENT:
        side = _square_mm(inst)
        if side:
            out.append(f"Goal square in this instance: the inner edge of the dashed green line is the containment "
                       f"limit (the whole line lies outside it); it encloses a {side:.1f} mm x "
                       f"{side:.1f} mm square. Every point counts: for a rigid part every vertex of its visual and collision "
                       "meshes" + ("; for the cord every vertex of its simulated volume" if fam in ("orbits", "slotted_post_ring")
                                   else "") + ".")
        free = FREE_REMAINDER.get(inst.get("variation_id"), FREE_REMAINDER.get(fam))
        out.append("The separator is an invisible, fixed vertical plane parallel to the goal square's near edge, 5 mm beyond "
                   "the point of all remaining components that was closest to the goal square at the start; the square's "
                   "near edge is at least 30 mm beyond it. " + (
                       f"{free[0].upper() + free[1:]} are free to move (their speed is not checked), but no point of them "
                       "may cross that plane." if free and free.endswith("s") else
                       f"{free[0].upper() + free[1:]} is free to move (its speed is not checked), but no point of it may "
                       "cross that plane." if free else "The remaining components here are fixed."))
    if fam in RIGID:
        c = CONTACT_RULE
        out.append(f"Contact penetration as this benchmark grades it: penetration is recorded, it never ends the episode by "
                   f"itself. A success with deep interpenetration (a task contact deeper than {_mm(c['deep_m'], 0)}, or deeper "
                   f"than {_mm(c['sustained_m'], 0)} for at least {c['window_s']:g} s: {window_ticks} consecutive physics ticks; for a "
                   f"finger contact with a held piece {_mm(c['held_deep_m'], 0)}, or {_mm(c['held_sustained_m'], 0)} for "
                   f"{c['window_s']:g} s) is not credited automatically but reviewed, with the recorded evidence and frames; a "
                   f"clean success is credited automatically. A task contact deeper than {_mm(BREAKDOWN_PENETRATION_M, 0)} is a "
                   "simulator breakdown (below).")
        out.append(f"Contact penetration is read from the contacts at the end of every physics tick ({dt * 1000:g} ms; {decimation} ticks per "
                   "control step). \"Held\" is decided anew at every tick: a zero-force touch counts, and if one finger "
                   "loses contact (with no active assisted grasp), the other finger is back at the non-held limits in the same "
                   "tick." + (" The cage is fixed, so finger contact with it is never held (always the non-held limits)."
                              if fam == "hedgehog_in_cage" else ""))
    if fam in SOFT:
        what = "rope" if fam in ("knot_untying", "rope_threading") else "cord"
        w = SOFT_RULE["window_s"]
        modulus = float((tp.get("rope_material") or {}).get("E", 50000.))
        out.append(f"The {what}'s Young's modulus is {modulus / 1000:g} kPa. Gripper pressure can deform it. "
                   f"A crushed {what} volume (an inverted volume element)"
                   + (f" or a permanent cord attachment stretched by more than {_mm(SOFT_RULE['residual_m'], 0)}"
                      if what == "cord" else "")
                   + f" ends the episode only once it has lasted at least {w:g} s "
                   f"({max(1, math.ceil(w / dt - 1e-12))} consecutive physics ticks; while it lasts, the goal cannot be confirmed); an "
                   f"invalid soft-body solver state is a simulator breakdown (below). Close to about half the {what}'s "
                   "diameter and slow down in the last centimetre.")
    out.append(f"Robot safety (this benchmark's tooling): no gripper target goes lower than {table_guard_words()}; a commanded "
               "target that would bring any part of a gripper lower is stopped there, on the commanded path. The fingertips reach "
               + FINGERTIP_WORDS.get(_robot_of(inst), "beyond the tool frame (TCP)")
               + ". Arm drive torques are capped (the effort limits in robot_spec), so an arm pushing against an obstacle "
               "stalls at that torque."
               + (f" An arm with any part more than {_mm(STATIC_GUARD_DEPTH_M, 0)} deep in a fixed object (a fixture, the table) "
                  "is stopped where it is: its joint targets become its measured joints for that control step, so it stops "
                  "pushing (the safety readings report these stops as static_guard)."
                  if (inst or {}).get("physics_engine") == "mujoco" else ""))
    ps = FORCING
    out.append(f"Forcing is recorded, not judged: an arm drive (not a gripper drive) at {ps['near_cap']:.0%} of its torque cap "
               f"or more while that arm's gripper, or a part it holds, presses on anything else with {ps['contact_n']:g} N or "
               f"more in total is forcing; {ps['event_s']:g} s of it on every physics tick is a forcing event in the evidence "
               + ("(`meta[\"safety\"]`)" if limited else "(`s.safety()`, `s.status()[\"forcing\"]`)")
               + ", shown to the reviewer when a success is reviewed. Forcing does not end the episode and does not change the "
               "grade by itself.")
    broken = (["a non-finite state", f"a task contact deeper than {_mm(BREAKDOWN_PENETRATION_M, 0)}"] if fam in RIGID else
              ["an invalid soft-body solver state"] if fam in SOFT else
              [f"a cloth particle faster than {PARTICLE_SPEED_LIMIT_M_S:g} m/s"] if fam in ("blanket_flatten", "blanket_fold")
              else [])
    if broken:
        out.append(f"Simulator breakdown ({_or(broken)}): the episode ends as an infrastructure error, not your failure, "
                   f"unless an arm was forcing (even for one tick) within the {ps['attribution_s']:g} s before it: then it is "
                   "your failure (forced_breakdown: what broke and what forced it).")
    if fam == "clock_assembling":
        out.append("The rules (seating, the tooth-on-tooth gates, meshing, the screws' turns, the bridge's fixing, the "
                   "click and the run test's timing) are applied after every physics tick "
                   f"({dt * 1000:g} ms); the success conditions are checked at the end of every tick. The mainspring is a "
                   "spring between the barrel's arbor and drum; the click is a limit on the arbor's turn that the rules move "
                   "up as it is wound.")
    if fam == "horseshoe_ring":
        out.append("The four permanent connections (the upper and lower eye of each horseshoe with its chain link) are "
                   "checked at every tick by the linking number of the two loops.")
    if fam == "cast_coil" or fam == "buzz_wire_3d":
        out.append("The release clearance is measured between axis-aligned bounding boxes: one around each collision geometry "
                   "of the whole robot (every link of both arms) and one around each collision geometry of the "
                   + ("pieces" if fam == "cast_coil" else "tool") + ", so the true gap usually has to be larger than 5 mm.")
    if fam == "buzz_wire_3d":
        term = tp.get("terminal", {}) or {}
        inner, wire, usable = buzz_aperture_mm(inst)
        out += [
            f"The loop's usable opening: the wire's centreline must cross the loop's disk less than {usable:.2f} mm from the "
            f"loop's centre (loop inner radius {inner:.2f} mm minus the {wire:.2f} mm wire radius).",
            "Continuous traversal: from one physics tick to the next, the crossing point may move along the wire by at most "
            f"twice the loop's own motion in that tick (its displacement plus {usable:.2f} mm times its rotation angle) plus "
            f"{_mm(term.get('route_progress_tolerance_m', .003))}. A larger jump, no crossing, two crossings or a tangency "
            "cancels the traversal; it must then start again within "
            f"{_mm(term.get('start_route_tolerance_m', .01))} of the starting end, the end where the tool rests at the start.",
            "The finish support is the deck, and the finish resting pose is the start resting pose repeated at the wire's "
            "finishing end: the tool lies on the deck with its loop at that end of the wire and its handle pointing inward, "
            "the way it lies at the start.",
            f"Metal contact is sampled at all {substeps} solver substeps of every tick. If any tick since the start had incomplete "
            "contact coverage (a simulator-side condition), the episode can no longer succeed.",
        ]
    if fam == "lock_picking":
        rotors = len(g.get("slots", [None]*3))
        faces = 2*(rotors+1)
        chain = "shallow_slots" in g      # chain-lock drafts (tasks/lock_picking/chain.py): two tab and two slot depths
        rotor_word = {2: "two", 3: "three", 4: "four"}[rotors]
        tilt_word = "0.38" if rotors == 2 else "0.25"
        bore, shaft = float(g.get("bore_radius", .0075)), float(g.get("shaft_radius", .0065))
        tab_r, tab_h, slot_h = float(g.get("tab_radius", .036)), float(g.get("tab_half_deg", 4)), float(g.get("slot_half_deg", 8))
        tol = float((tp.get("terminal", {}) or {}).get("shaft_contact_tolerance_m", 1e-5))
        rot_tol = float((tp.get("terminal", {}) or {}).get("rotor_tolerance_deg", .5))
        if chain:
            # A centred shaft tilted about the middle of the stack (first rotor's front face to END's back face).
            span = rotors*float(g.get("pitch", .12)) + float(g.get("disc_thickness", .01))
            tilt_word = f"{math.degrees(math.atan((bore - shaft + tol)/(span/2))):.2f}"
        out += [
            f"Sizes: bore radius {_mm(bore)}, shaft radius {_mm(shaft, 2)}. At each of the {faces} plate faces ({rotor_word} "
            f"{_mm(g.get('disc_thickness', .01))} rotor plates and END, {_mm(g.get('pitch', .15), 0)} apart), the shaft axis' "
            f"distance from the bore axis plus {_mm(shaft, 2)} / cos(tilt) may exceed {_mm(bore)} by at most {_mm(tol, 2)}: "
            f"about {_mm(bore - shaft + tol, 2)} of offset when the key is straight, which over the stack allows only about "
            f"{tilt_word} degrees of tilt.",
            f"The shaft reaches from {_mm(float(g.get('grip_offset', .495)) + .02, 0)} behind the leading tab to 15 mm in "
            f"front of it; all {faces} plate faces must lie within that length.",
        ]
        if chain:
            out.append(
                f"Tabs: half-width {tab_h:g} degrees; the leading tab reaches radius {_mm(tab_r, 0)} and the second tab, "
                f"{_mm(g.get('pitch', .12), 0)} behind it, {_mm(g.get('second_tab_radius', .022), 0)}. Slots: half-width "
                f"{slot_h:g} degrees, reaching radius {_mm(g.get('slot_radius', .045), 0)} (deep) or "
                f"{_mm(g.get('shallow_radius', .028), 0)} (shallow). A tab that overlaps a plate must sit inside one of that "
                "plate's slots that reaches beyond the tab, with an angular margin of at least asin(axis offset / tab "
                "radius) + the tilt.")
        else:
            out.append(
                f"Tabs: radius {_mm(tab_r, 0)}, half-width {tab_h:g} degrees; slots: half-width {slot_h:g} degrees. A tab that "
                f"overlaps a plate must sit inside one of that plate's slots with an angular margin of at least asin(axis offset / "
                f"{_mm(tab_r, 0)}) + the tilt.")
        stops = tp.get("rotor_stops") or []
        lim = []
        if chain:
            initial = g.get("initial_disc_deg", [0.]*rotors)
            for k, st in enumerate(stops):
                lo, hi = (st["bounds_deg"] if isinstance(st, dict) else st)
                lim.append(f"rotor {k+1} {hi-initial[k]:g} degrees counterclockwise and {initial[k]-lo:g} degrees clockwise")
        for k, st in enumerate(stops[:3] if not chain else ()):
            lo, hi = (st["bounds_deg"] if isinstance(st, dict) else st)
            if rotors == 2:
                initial = g.get("initial_disc_deg", [0., 0.])[k]
                if k == 0 and initial > 0:
                    lim.append(f"rotor 1 can turn {initial-lo:g} degrees toward the second rotor's slot "
                               f"({initial:g} degrees aligns the slots), or {hi-initial:g} degrees away")
                else:
                    lim.append(f"rotor {k+1} can turn {hi-initial:g} degrees one way and {initial-lo:g} degrees the other")
                continue
            if hi - lo < 360:
                lim.append(f"rotor {k + 1} {abs(lo):g} degrees one way and {abs(hi):g} degrees the other"
                           if lo < 0 < hi and abs(lo) != abs(hi) else f"rotor {k + 1} {abs(hi):g} degrees either way")
            else:
                lim.append(f"rotor {k + 1} turns freely")
        if lim:
            reference = ("Rotor stops (from each rotor's start angle, seen from the side the key enters, looking along the key "
                         "toward END; rotor 1 is the first plate the key reaches)" if chain else
                         "Rotor stops (from each rotor's start angle)" if rotors == 2 else
                         "Rotor stops (from each rotor's start angle; \"the other way\" is the sense that turns rotor 2's slot "
                         "toward the tabs' starting angle)")
            out.append(f"{reference}: {'; '.join(lim)}. Going more than {rot_tol:g} degrees beyond a "
                       "stop ends the episode.")
    if fam == "fiber_bundling":
        n = g.get("fiber_count")
        out += [
            f"Sizes: ring inner radius {_mm(g.get('ring_inner_radius_m', .0135))}, ring tube radius 4 mm; "
            f"{n} fibers of radius 1.2 mm; sheath radius 12 mm." if n else
            f"Sizes: ring inner radius {_mm(g.get('ring_inner_radius_m', .0135))}, ring tube radius 4 mm; fiber radius "
            "1.2 mm; sheath radius 12 mm.",
            "Each fiber is tracked at 11 points from tip to root. A point counts when it crosses the ring's plane less than "
            "17.48 mm from the ring's axis, in tip-to-root order; a crossing out of order or outside that radius resets that "
            "point and every point behind it, and crossing back loses the credit.",
            "At the end every fiber point must be more than 5.2 mm past the ring's plane (tube + fiber radius) and the centre "
            "of the sheath's front face more than 16 mm past it (tube + sheath radius). The exit side is the side opposite "
            "to where the fibers start.",
            "The failure counts contact between a fine fiber and the gripper (fingers and hand, housing included); arm links "
            f"touching a fiber do not count. Contact is sampled at all {substeps} solver substeps of every tick; if any tick since the "
            "start had incomplete contact coverage (a simulator-side condition), the episode can no longer succeed.",
        ]
    if fam == "knot_untying":
        out += [
            "The rope is 8 mm thick (radius 4 mm), so its highest point may be at most 11 mm above the tabletop.",
            "\"Nonlocal\" means at least 16 mm apart along the rope: such parts of the rope's centreline must be at least "
            "8 mm apart in the top view. Closer parts are exempt, but an actual crossing always counts.",
            knot_release_words(tp),
        ]
    if fam == "rope_threading":
        if g.get("ring_shape") == "lower_semicircle":
            out.append("The cyan fixture is the lower semicircle of a 75 mm centreline-radius ring, with a 6 mm tube "
                       "radius. Its upper half is physically absent; the rope radius is 4 mm. This is a draping task.")
        else:
            from topogym.assets.geometry.rope_threading import ThreadingLayout
            from topogym.tasks.rope_threading.common import PRESETS, usable_rope_aperture_radius
            vid = (inst or {}).get("resolved_variation_id") or (inst or {}).get("variation_id") or ""
            if vid.startswith("rope_threading_rd_"):   # the redesigned tiers' own rings (bulkhead lacing), not the family's
                from topogym.tasks.rope_threading.redesign import frozen as _F
                from topogym.tasks.rope_threading.redesign.task import RedesignLayout
                layout = RedesignLayout(_F.load(vid[len("rope_threading_rd_"):]))
            else:
                layout = ThreadingLayout(**(PRESETS[fam]["geometry"] | g))
            allowance = float(tp.get("aperture_contact_allowance_m", 0.))
            profile = tp.get("aperture_check_profile", "cylindrical")
            terminal = {"aperture_margin_m": .0001, **(tp.get("terminal") or {})}
            sizes = []
            for ring in layout.rings():
                middle = usable_rope_aperture_radius(ring, layout.rope_radius_m, terminal, allowance)
                side = usable_rope_aperture_radius(ring, layout.rope_radius_m, terminal, allowance,
                                                   axial_offset_m=ring["tube_radius_m"], profile=profile)
                sizes.append(f"{ring['name']}: inner radius {_mm(ring['radius_m'] - ring['tube_radius_m'])}, "
                             f"tube radius {_mm(ring['tube_radius_m'])}, centreline passage radius "
                             f"{_mm(middle)} on the middle plane and {_mm(side)} on the two tube-offset planes")
            out += [f"Sizes (rope radius {_mm(layout.rope_radius_m)}): " + "; ".join(sizes) + ".",
                    f"Aperture check: {profile}, compression allowance {_mm(allowance)}. "
                    "This allowance changes the geometric passage test; the collision geometry stays as declared.",
                    "The material start is the rope end that is farther from the rings at the start."]
        if tp.get("rope_robot_contact") == "fingers":
            # the environment profile's contact set (topogym/tasks/rope_threading/common.py finalize_spec: IPC
            # robot_collision_links = the finger links, for all IPC contacts): an environment fact the agent must know
            out.append("Only the grippers' fingers collide with the rope and the rings: every other robot part (palms, wrists, "
                       "arm links, bases) passes through them without contact, so only the fingers can push, pinch or hold "
                       "the rope.")
    if fam in ("blanket_flatten", "blanket_fold"):     # blanket_fold: blanket_flatten's cloth, release and clearance
        L, W = float(tp.get("length_m", .6)), float(tp.get("width_m", .48))
        cov, pad = float(tp.get("minimum_coverage_ratio", .9)), float(tp.get("particle_size_m", .008))
        out += [
            (f"The blanket is {L:g} m x {W:g} m ({L * W:.3f} m^2), so the coverage needs at least {cov * L * W:.4f} m^2 of "
             "its area in the top view. Heights are measured at the cloth particles' centres." if fam == "blanket_flatten" else
             f"The blanket is {L:g} m x {W:g} m. Heights are measured at the cloth particles' centres."),
            f"The release clearance is measured from one box around all cloth particles, grown by {_mm(pad, 0)}: the collision "
            f"box of every robot part that can move (arm links, wrists, grippers, fingers) must stay at least 5 mm from it, i.e. "
            f"at least {_mm(pad + .005, 0)} outside the cloth's bounding box, above its highest point too. The two arm bases are "
            "fixed to the world and are not counted: the cloth may lie against them.",
        ]
    g = grasp_regions(fam, inst, limited=limited)
    if g:
        out.append(g)
    if fam in REPLAY_TOLERANCES:
        out.append(grading_replay_words(fam, inst, limited=limited))
    if not limited:
        strings = (FAIL_STRINGS["rigid"] if fam in RIGID else []) + FAIL_STRINGS.get(fam, []) + FAIL_STRINGS["all"]
        out.append("A failure is reported in `s.status()[\"episode_end\"]` as " + _or(strings) + " (<what>: the "
                   "readable reason); `s.safety()` reports forcing, contact depths, volume ratios and table-guard stops "
                   "against these limits" + (", and `s.status()[\"review\"]` the deep task contacts that would hold a success "
                   "for review" if fam in RIGID else "") + ".")
    return out


def _parameter(params, path):
    for key in path.split("."):
        params = params[key]
    return params


def replay_tolerances(fam, inst):
    """[(threshold, live value, replay value)] of this instance's grading replays (its frozen task parameters)."""
    tp = (inst or {}).get("effective_task_parameters") or {}
    return [(name, float(_parameter(tp, path)), round(float(_parameter(tp, path)) + slack, 12))
            for name, (path, _, slack) in REPLAY_TOLERANCES.get(fam, {}).items()]


def knot_release_words(tp):
    """knot_untying's release check in words, from the task's own settling spec (terminal v5 since tooling v16: 1 mm from the
    links that can move, world-fixed bases may touch, no interior overlap with any robot part)."""
    from types import SimpleNamespace
    from topogym.tasks.knot_untying.common import KnotUntyingTask
    terminal = {"maximum_speed_m_s": .001, **(tp.get("terminal") or {})}
    spec = KnotUntyingTask.final_settling_spec(SimpleNamespace(config=dict(tp, terminal=terminal)))
    if spec.robot_contact_policy == "fixed_contact_active_clearance_v1":
        return ("The release check bounds every rope tetrahedron against the robot's collision geometries and decides the "
                f"close ones with the exact solids: every robot part that can move (grippers, arm links) needs at least "
                f"{_mm(spec.clearance_m)} of clearance from the rope; a world-fixed base may touch it. An overlap with the "
                "interior of any robot part means the release is not certified: no success while it lasts, never a failure.")
    return ("The release clearance is measured between axis-aligned bounding boxes of the robot's collision geometries and "
            "the rope (refined per tetrahedron where they are close).")


def control_hz(inst) -> float:
    """The instance's control rate (simulator physics_dt_s x control_decimation; 40 Hz by default, 25 Hz for the merged-v16
    rope_threading profile's 40 ms tick)."""
    sim = (inst or {}).get("simulator") or {}
    return 1.0 / (float(sim.get("physics_dt_s", .0025)) * int(sim.get("control_decimation", 10)))


def grading_replay_words(fam, inst, *, limited):
    """The grading-replay rule of a GPU family in plain words (rcb_topogym.py, tooling v16; the instance's control rate)."""
    steps = int(round(GRADING_HOLD_S * control_hz(inst)))
    tol = [TOLERANCE_WORDS[name](live, replay) for name, live, replay in replay_tolerances(fam, inst)]
    hold = (f"runs the rows with the thresholds above and, if the episode has not ended after the last row, holds the final "
            f"targets for up to {GRADING_HOLD_S:g} s ({steps} control steps), stopping at the first success or failure, so a "
            "confirmation that starts a few physics ticks later than in " + ("the live episode" if limited else "your own run")
            + " can still complete")
    during = (f"; during that hold it accepts {' and '.join(tol)}" if tol else
              "; the hold uses the thresholds above too (no tolerance for this family)")
    tail = (" Nothing else is relaxed: every other condition, the failure conditions and the confirmation time are the "
            "same." if tol else " Nothing else changes.")
    if limited:
        return ("Grading: success needs the live episode to succeed (with the thresholds above) AND one fresh-process replay of "
                "its rows to succeed; a live failure is final whatever the replay shows. This family's physics runs on the GPU, "
                f"which is not bit-deterministic, so that replay can differ slightly from the live episode: it {hold}{during}."
                + tail)
    return ("Grading: the verifier replays your trajectory twice in a fresh process. This family's physics runs on the GPU, "
            f"which is not bit-deterministic, so a replay can differ slightly from your own run: each replay {hold}{during}. "
            "Success needs both replays to succeed AND at least one of them to reach the goal with the thresholds above "
            "within your own rows (not only in the hold)." + tail + " A recording stops at your own success, so a replay "
            "whose confirmation ends a few ticks later needs more rows: you may append copies of the final row (holding "
            "still) before saving. `s.replay(actions, grading=True)` replays as the verifier does.")


# where his declared grasp regions sit, by construction of his recipes (76583213 topogym/tasks/<family>/common.py
# recommended_grasp_assist and the tooling's explicit lock_picking entry): words only, never a centre (a position)
GRASP_WHERE = {("buzz_wire_3d", "wand"): "on the wand's grip", ("lock_picking", "lock_key"): "on the key's handle",
               ("battery_replacement", "stick"): "around the stick's handle"}
GRASP_WHOLE = {"hedgehog_in_cage", "horseshoe_ring"}      # rigid_grasp_region: the part's whole mesh + 5 mm


def grasp_regions(fam, inst, *, limited):
    """The sizes of his grasp regions (the frozen simulator.grasp_assist): one sphere per graspable part. His grasp text
    (Grippers) names the parts and the tool-frame gate but not the regions' sizes."""
    ga = ((inst or {}).get("simulator") or {}).get("grasp_assist") or {}
    targets = [t for t in ga.get("targets", []) if isinstance(t, dict)]
    if ga.get("model", "none") == "none" or not targets:
        return None
    if ga["model"] in {"ipc_soft_band_v1", "pbd_local_patch_v1"}:
        regs = []
        for t in targets:
            regs.append(f"{t['entity']}: spatial radius {_mm(t.get('spatial_radius_m', .014))}, "
                        f"material radius {_mm(t.get('material_radius_m', .018))}, "
                        f"finger proximity {_mm(t.get('contact_distance_m', .006))}, "
                        f"maximum material strain ratio {t.get('max_strain_ratio', 1.3):g}, "
                        f"exposed patch {'required' if t.get('require_exposed_patch', True) else 'not required'}")
        return ("Soft grasp patches: " + "; ".join(regs) + ". Both fingers must be near the same material patch; "
                "penetration and native constraint health checks still apply. Two hands may hold disjoint patches "
                "of one object. Material bounds, when present, restrict the admissible region.")
    regs, per = [], {}
    for t in targets:       # clock_assembling: every part is a link of the one entity "clock", named by its link
        per.setdefault(t["link"] if fam == "clock_assembling" else t["entity"], []).append(t)
    several = any(len(ts) > 1 for ts in per.values())   # e.g. the claws redesign: a grip block at each end of a piece
    for ent, ts in per.items():
        where = "enclosing the whole part" if fam in GRASP_WHOLE else GRASP_WHERE.get((fam, ent))
        radii = sorted({_mm(t["radius_m"]) for t in ts})
        size = (f"radius {radii[0]}" if len(ts) == 1 else
                f"{len(ts)} spheres of radius {radii[0]}" if len(radii) == 1 else f"{len(ts)} spheres of radii {', '.join(radii)}")
        regs.append(f"{ent}: {size}" + (f", {where}" if where else ""))
    out = (f"Grasp regions (assisted grasping, see Grippers above; {'one or more spheres' if several else 'one sphere'} "
           "per graspable part): " + "; ".join(regs) + ". Both finger contacts must lie inside "
           + ("one of the part's spheres" if several else "the part's sphere") + "; a part touched elsewhere is grasped physically.")
    if not limited:
        out += " Each region's centre (in its link's frame) is in `/app/instance.json` under `simulator.grasp_assist.targets`."
    return out


def details_section(fam, inst, *, limited):
    """Markdown section placed right after his requirements; "" if there is nothing to add. Starts and ends with a
    newline so that it fits between his text and the next section."""
    lines = grader_details(fam, inst, limited=limited)
    if not lines:
        return ""
    return ("\n## Grader details (in addition to the requirements above)\n\n"
            + "\n".join(f"- {x}" for x in lines) + "\n")


def _secs(x) -> str:
    """A wall time for the texts: two significant digits below 10 s (0.053, 0.11, 1.9), whole seconds above."""
    x = float(x)
    return f"{x:.2g}" if x < 10 else f"{x:.0f}"


def step_seconds(d: dict) -> float:
    """The per-control-step wall time the texts quote: the measured arm-motion value where the build measured one
    (describe.json sec_per_step_motion: FEM / IPC families, 2026-10-01: motion steps on an unloaded GPU, the slower of our
    GPU types), else describe's sec_per_step."""
    return float(d.get("sec_per_step_motion") or d["sec_per_step"])


def step_time_words(d: dict) -> str:
    """'One control step takes about ...' for the instruction texts (no claim about checkpoints: the simulator keeps a
    complete checkpoint only at the episode start). With describe.json `sec_per_step_unloaded` (a hold step measured on an
    unloaded GPU), the texts give both ends: the unloaded time and the measured time on a shared GPU, which the chunk advice
    keeps using (the safe end)."""
    if d.get("sec_per_step_motion"):
        return (f"One control step with the arms moving takes about {step_seconds(d):.0f} s of wall clock here (measured on an "
                f"unloaded GPU, the slower of our GPU types; holding still is faster)")
    if d.get("sec_per_step_unloaded"):
        return (f"One control step takes about {_secs(d['sec_per_step_unloaded'])} s of wall clock on an unloaded GPU, and "
                f"up to about {_secs(step_seconds(d))} s when other simulations share the GPU; the call metadata reports the time")
    return f"One control step takes about {_secs(step_seconds(d))} s of wall clock here"


def limited_timing(sec_per_step, control_hz=40.0, submit_timeout_s=1800, observe_timeout_s=600, max_chunk_s=10.0, safety=2.0,
                   motion=False, unloaded=None):
    """Client-side timeouts (rcb_client.py: submit 1800 s, get_obs 600 s) against this instance's measured speed (motion:
    the measured arm-motion step time, describe.json sec_per_step_motion)."""
    per = max(float(sec_per_step), 1e-3)
    chunk_steps = int(submit_timeout_s / (safety * per)); watch_steps = int(observe_timeout_s / (safety * per))
    chunk_line = (f"keep each `submit` to at most {chunk_steps} control steps ({chunk_steps / control_hz:.2f} s of simulated "
                  f"time: rows x dt)" if chunk_steps < max_chunk_s * control_hz else
                  f"a `submit` of the maximum {max_chunk_s:g} s chunk fits")
    watch_line = (f"keep each `get_obs(seconds=...)` watch to at most {watch_steps} control steps "
                  f"({watch_steps / control_hz:.2f} s)" if watch_steps < 30 * control_hz else "any watch up to 30 s fits")
    return ("## Time per call\n"
            + (f"- One control step with the arms moving takes about {per:.0f} s of wall clock here (holding still is faster). "
               if motion else f"- One control step takes about {_secs(unloaded)} s of wall clock on an unloaded GPU and up to about "
               f"{_secs(per)} s when other simulations share it; the advice below assumes the slower figure. "
               if unloaded else f"- One control step takes about {_secs(per)} s of wall clock here. ")
            + "A request returns only when all of its steps "
            f"have run, and the client gives up waiting after {submit_timeout_s} s for `submit` and {observe_timeout_s} s "
            f"for `get_obs`: {chunk_line}; {watch_line} (both with a 2x margin).\n"
            "- If a request times out in the client anyway, the service still runs it to the end: do NOT send it again "
            "(it would run twice). Call `r.get_obs()` (repeat it if it also times out): it answers once the earlier "
            "request has finished and shows the state after it.\n")
