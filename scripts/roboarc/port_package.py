#!/usr/bin/env python3
"""Port a package written from Genesis-era texts into a final package (scripts/roboarc/mk_tasks.py, "packages/").

tools/release/release_rebuild.py writes a new base's packages/<base>-{privileged,standard} from the Genesis-era texts it
starts from (tools/release/templates/, with the design's own texts from tools/release/package_text/ over them). Those
texts still say Genesis, TopoGym, the old image and the review rounds' old wording. This tool applies the rewrites
that turned the released Genesis-era texts into RoboARC's (the rounds of old -> new replacements the generator used to
apply on every render, each asserting that its source text is there), then puts mk_tasks.fields() back into the
result as {{FIELD}}s, and checks that mk_tasks fills them back to exactly that text. A package that is final already
(its task.toml has {{AGENT_EGRESS}}) is left alone.

usage: python3 scripts/roboarc/port_package.py [--check] base ...
  --check   write nothing; exit 1 if a named base still has Genesis-era packages

The base must be in release.json (its record gives the facts the rewrites use: grasp rules, chain, step time, tier).
Proof that the rewrites are the ones the 1.0.0 tasks were rendered with: run on the 1.0.0 Genesis-era packages
(git show 153eb8cf01:scripts/roboarc/packages/...), it gives today's final packages byte for byte.
"""
from __future__ import annotations
import argparse, json, math, pathlib, re, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mk_tasks as M  # noqa: E402
from mk_tasks import (BENCH, PKG, REL, IMAGE, TC, CUTOFF_SEC, AGENT_EGRESS, MAX_STEPS, MAX_CALL_WALL_S, CHUNK_S,  # noqa: E402,F401
                      WATCH_S, TIERS, GOAL_DIR, MUJOCO_ENV, MAX_FRAMES, DEPTH_FRAME, put, tooling_constant, step_time,
                      base_step, row_cap, time_per_call, toml_str, NO_REFERENCE)

EGRESS_OLD = "# egress: Harbor default (public). `scripts/set_agent_egress.py allowlist` closes it to the model APIs.\n"
EGRESS_NEW = AGENT_EGRESS + "\n"
STAGED = "/tmp/robotangle-instance"                             # where the standard sim stages its instance
DIFFICULTY = {v["new"].removeprefix(BENCH + "-"): v["difficulty"] for v in M.NAMES.values()}

ROBOT = {"aloha": "aloha-2", "panda_robotiq": "franka-robotiq-2f85"}
ROBOT_WORDS = {"(aloha, Genesis)": "(ALOHA 2, Genesis)", "(panda_robotiq, Genesis)": "(Franka Panda + Robotiq 2F-85, Genesis)"}
IMAGE_RX = re.compile(r"(?:[\w.-]+/)+rcb-topogym:[\w.-]+")
PROTO = ("eai-standard/2.3", "eai-standard/2.4")   # the release's name of the layer -> its registered version (#89)


def text(s: str) -> str:
    """User-facing rename (README, instruction, task.toml); code identifiers (topogym, rcb_topogym) stay."""
    s = re.sub(r"\btopogym-([a-z0-9-]+-i\d\d-(?:privileged|standard))\b", BENCH + r"-\1", s)
    s = s.replace("docs/benchmarks/robotangle.md", f"docs/benchmarks/{BENCH}.md")   # the released texts' page
    s = IMAGE_RX.sub(IMAGE, s)
    for a, b in ROBOT_WORDS.items():
        s = s.replace(a, b)
    s = s.replace("TopoGym", "RoboARC").replace(*PROTO)
    s = re.sub(r"\bhis (?=[a-z]+(?:_[a-z0-9]+)*_v\d)", "the ", s)
    return s.replace("his qualification record", "the author's qualification record")


# ---------------------------------------------------------------- MuJoCo bases
MUJOCO_WORDS = (("(ALOHA 2, Genesis)", "(ALOHA 2, MuJoCo)"),
                ("(Franka Panda + Robotiq 2F-85, Genesis)", "(Franka Panda + Robotiq 2F-85, MuJoCo)"),
                ("RoboARC on Genesis 1.2.1", "RoboARC on MuJoCo 3.14.0"),
                ("`topogym`) on Genesis 1.2.1 physics", "`topogym`) on MuJoCo 3.14.0 physics"),
                ('"genesis",', '"mujoco",'), ("(Genesis initialises once)", "(one simulator per process)"))
STEP_RX = re.compile(r"One control step takes about ([0-9.]+) s of wall clock")
# The released texts of the families graded with grading replays (topogym_rules.grading_replay_words, README's
# determinism sentence) say their physics runs on the GPU and is not bit-deterministic; on MuJoCo it runs on the CPU in
# float64 (a cloth's step on a deterministic GPU solver) and a fresh-process replay reproduces the run on the same host.
# The grading-replay rule itself (the hold, its tolerances) is unchanged in the tooling and stays in the text.
def determinism_words(gpu: bool) -> tuple:
    phys = ("Its cloth step runs on the GPU on a deterministic solver and the rest on the CPU (MuJoCo, float64)" if gpu else
            "This family's physics runs on the CPU (MuJoCo, float64)")
    return (("This family's physics runs on the GPU, which is not bit-deterministic, so a replay can differ slightly from "
             "your own run:", phys + ", so a replay reproduces your own run on the same host; the rule is kept as a margin:"),
            ("This family's physics runs on the GPU, which is not bit-deterministic, so that replay can differ slightly from "
             "the live episode:", phys + ", so that replay reproduces the live episode on the same host; the rule is kept as "
             "a margin:"),
            ("if this GPU family's episode has not ended", "if this family's episode has not ended"),
            ("(GPU physics is not bit-deterministic; the CPU rigid families' is, on one host)",
             "(the deterministic GPU cloth step and MuJoCo replay bit for bit on one host; the rule does not rely on it)" if gpu
             else "(MuJoCo on the CPU replays bit for bit on one host; the rule does not rely on it)"))
README_GPU_RX = re.compile(r"The physics runs on the GPU, which is not bit-deterministic: the\s+final-state hashes \(`deterministic`\) "
                           r"are diagnostic")


CONTACT_RX = re.compile(r"A success with deep interpenetration \(a task contact deeper than (?P<deep>[0-9.]+) mm, or deeper than "
                        r"(?P<sus>[0-9.]+) mm for at least (?P<win>[0-9.]+) s: (?P<ticks>\d+) consecutive physics ticks; for a "
                        r"finger contact with a held piece (?P<hdeep>[0-9.]+) mm, or (?P<hsus>[0-9.]+) mm for (?P<hwin>[0-9.]+) s\)")
BREAKDOWN_RX = re.compile(r"(task contact deeper than )([0-9.]+)( mm(?:\)|: the episode| is a simulator breakdown))")


def contact_text(s: str, rec: dict) -> str:
    """The contact-review sentence with the values of the shipped tooling's CONTACT_RULE and BREAKDOWN_PENETRATION_M (owner
    decision 2026-10-12, option a: the released texts said 1 mm where the MuJoCo tooling grades 2 mm). The window in
    physics ticks: window_s over the base's physics tick (release.json physics_dt_s, recorded from its instance), else
    the released count, which assumes the released 2.5 ms tick."""
    c, mm = tooling_constant("CONTACT_RULE"), (lambda v: f"{v * 1000:g}")
    def sub(m: re.Match) -> str:
        if rec.get("physics_dt_s"):
            ticks = max(1, math.ceil(c["window_s"] / float(rec["physics_dt_s"]) - 1e-12))
        else:
            assert float(m["win"]) == c["window_s"], "the released tick count assumes the released window"
            ticks = int(m["ticks"])
        return (f"A success with deep interpenetration (a task contact deeper than {mm(c['deep_m'])} mm, or deeper than "
                f"{mm(c['sustained_m'])} mm for at least {c['window_s']:g} s: {ticks} consecutive physics ticks; for a finger "
                f"contact with a held piece {mm(c['held_deep_m'])} mm, or {mm(c['held_sustained_m'])} mm for {c['window_s']:g} s)")
    s, n = CONTACT_RX.subn(sub, s)
    assert n <= 1, "more than one contact-review sentence"
    if n:
        s = BREAKDOWN_RX.sub(lambda m: m[1] + mm(tooling_constant("BREAKDOWN_PENETRATION_M")) + m[3], s)
    return s


GRASP_RX = re.compile(r"(?m)^((?:- )?(?:Grasping|Grippers): )((?:assisted grasping \(|physical grasping)[^\n]*)\.$")


def grasp_text(s: str, rec: dict) -> str:
    """A MuJoCo base's grasping rule in the words the tooling's Session.grasp_rules() emits for its instance (release.json
    grasp_rules, read in the release image by release_rebuild.py): on MuJoCo the weld also lets go above the robot's
    slip force averaged over its window, and the Robotiq after losing finger contact (owner decision 2026-10-12)."""
    if not rec.get("grasp_rules"):
        return s
    s, n = GRASP_RX.subn(lambda m: m[1] + rec["grasp_rules"] + ".", s)
    assert n <= 2, f"{n} grasping sentences"
    return s


SAFETY_OLD = ('Every response\'s `meta["safety"]` reports, since the\n  previous call, forcing (the arms\' pressing loads and forcing '
              'events), the deepest task contact, the longest time above\n  its sustained limit and the review events (rigid families), '
              'the lowest rope / cord volume ratio (soft families) and the\n  table-guard stops, against the limits above.')
SAFETY_NEW = ('Every response\'s `meta["safety"]` reports, since the\n  previous call, what the robot itself can feel: its forcing '
              'record (the largest force an arm or a part it holds pressed\n  with, the drive load meanwhile, the longest forcing run '
              'and the forcing events, naming only "a held part" and "the\n  environment"), the table-guard stops of its gripper '
              'targets and its IK jumps. Task contacts, review events and\n  soft-body readings are the grader\'s evidence and are not '
              'reported.')


REGIONS_RX = re.compile(r"(?m)^- (?:Grasp regions|Soft grasp patches)[ (:].*$")
GRASP_ROW_RX = re.compile(r"(?m)^\| Grasping \| [^|\n]* \|$")
DIFFICULTY_GRASP_RX = re.compile(r"(?m)^((?:Two|A) \d+-dof .* Grasping: )(?:assisted grasping \([a-z0-9_]+\)|physical grasping)\.$")


def instance_facts(s: str, rec: dict, mode: str) -> str:
    """A MuJoCo base's statements of its instance's grasp, from the instance (release.json, read in the release image by
    release_rebuild.py: grasp = the [metadata] grasp words, grasp_regions = the grader-details line per mode, grasp_rules):
    a port whose grasp model changed on MuJoCo (rope_threading: the released IPC soft band -> contact_weld_v1) must not keep
    the released line. Records without these facts keep the released lines."""
    if rec.get("grasp"):
        s = re.sub(r'(?m)^grasp = ".*"$', lambda m: f"grasp = {json.dumps(rec['grasp'])}", s)
    new = rec.get("grasp_regions", {}).get(mode)
    if new:                 # only a line of the other grasp form (soft patches <-> rigid regions): a released line of the
        form = new.split(" ")[0]           # same form keeps its own words (e.g. hedgehog cast_cage's rod "box handle")
        s, n = REGIONS_RX.subn(lambda m: m[0] if m[0][2:].split(" ")[0] == form else "- " + new, s)
        assert n <= 2, f"{n} grasp-region lines"
    if rec.get("grasp_rules"):
        head = rec["grasp_rules"].split(":")[0]
        s = GRASP_ROW_RX.sub(lambda m: f"| Grasping | {head} |", s)
        s = DIFFICULTY_GRASP_RX.sub(lambda m: m[1] + head + ".", s)
    return s


FEM_RX = re.compile(r"(?m)^- The (rope|cord)'s Young's modulus is ([0-9.]+) kPa\.[^\n]*$")
CHAIN_ANCHOR = "- Robot safety (this benchmark's tooling):"


def chain_words(c: dict, name: str, young_kpa: float | None) -> str:
    """The rope / cord sentence of a MuJoCo chain base, from its profile and frozen scene (release.json chain, read in the
    release image by release_rebuild.py): rigid capsule links on joints, their springs, bend stops, plasticity, speed
    clamp and tension guard as the profile has them. It replaces the released FEM sentence (Young's modulus, inverted
    volume elements), whose rule never fires on a chain (topogym/world/mujoco/profiles.py)."""
    size = f"{c['link_mm']:g} mm long, {2 * c['radius_mm']:g} mm thick" if c.get("link_mm") and c.get("radius_mm") else None
    links = f"a chain of {c['links']} rigid capsule links" if c.get("links") else "a chain of rigid capsule links"
    if c["kind"] == "rope":
        spring = (f"; each joint bends and twists against a spring set by the {name}'s material"
                  + (f" (Young's modulus {young_kpa:g} kPa)" if young_kpa else "") + ", with damping")
        stop = (f", and stops bending at {c['bend_stop_rad']:g} rad, where neighbouring links touch"
                if c.get("bend_stop_rad") else "")
    else:
        k = float(c.get("stiffness_nm_rad") or 0.)
        spring = (f" on ball joints that bend and twist against springs (about {k:.2g} N m/rad per joint), with damping" if k > 0
                  else " on ball joints that bend and twist freely (no spring), with damping")
        stop = ""
    plastic = c.get("plastic")
    plastic = (f"; past {plastic['yield_angle_deg']:g} deg a joint's rest angle follows its bend (the {name} stays bent)"
               if plastic else "")
    out = (f"- The {name} is {links}" + (f" ({size})" if size else "") + (" joined end to end" if c["kind"] == "rope" else "")
           + spring + stop + plastic + f". The links do not deform: nothing about the {name}'s shape or volume ends the "
           "episode.")
    sc = c.get("speed_clamp")
    if sc:
        who = {"slotted_chain": "A part of the chain (a link, bead, end disc or the ring)", "cord": "A cord link or the handle"
               }.get(c.get("entity"), "A link")
        out += (f" {who} moving faster than {sc['v_m_s']:g} m/s or spinning faster than {sc['w_rad_s']:g} rad/s is slowed by "
                "extra damping (a speed clamp); this is not reported and does not end the episode.")
    tg = c.get("tension_guard")
    if tg and young_kpa and c.get("radius_mm") and c.get("link_mm"):
        r, l = c["radius_mm"] / 1000, c["link_mm"] / 1000
        t = tg["t_on"] * young_kpa * 1000 * math.pi * r ** 4 / 4 / l ** 2
        out += (f" Pulled harder than about {t:.3g} N, the {name}'s links get extra rotational inertia (a tension guard), "
                "which only removes energy and is not reported.")
    return out


def chain_text(s: str, rec: dict) -> str:
    """The released FEM sentence of a chain family replaced by its chain's own (chain_words), or the sentence added
    before the robot-safety line of a text that has none (a new design drafted without it)."""
    c = rec.get("chain")
    if not c:
        return s
    s = (s.replace("Simulator breakdown (an invalid soft-body solver state)",
                   "Simulator breakdown (a MuJoCo instability: a non-finite or diverging acceleration, position or velocity)")
          .replace("every vertex of its simulated volume", "every vertex of its links (capsules)")
          # knot_untying's release check (topogym_rules.knot_release_words): the rope's volume mesh is carried rigidly by
          # its capsule links on MuJoCo (rope_chain.RopeChain.fem_state); the check bounds that mesh, not FEM tetrahedra
          .replace("The release check bounds every rope tetrahedron against",
                   "The release check bounds the rope's volume mesh (carried rigidly by its capsule links), element by element, against")
          .replace("(refined per tetrahedron where they are close)",
                   "(refined per element of the volume mesh its capsule links carry where they are close)"))
    ent = c.get("entity") or "rope"        # the scene object list: the chain is not an FEM body on MuJoCo
    s = s.replace(f"`{ent}` (deformable, fem)", f"`{ent}` (a chain of rigid capsule links)")
    m = FEM_RX.search(s)
    if m:
        return FEM_RX.sub(lambda m: chain_words(c, m[1], float(m[2])), s, count=1)
    if "rigid capsule links" not in s and CHAIN_ANCHOR in s:
        name = "rope" if c["kind"] == "rope" else "cord"
        return s.replace(CHAIN_ANCHOR, chain_words(c, name, None) + "\n" + CHAIN_ANCHOR, 1)
    return s


def safety_text(s: str) -> str:
    """The standard instruction's meta["safety"] paragraph stays the released one (SAFETY_OLD): the service publishes
    main's declared fields again, with part names (user decision 2026-10-13; the sim service's robot_felt,
    SAFETY_PUBLISHED). A text still carrying the reduced paragraph of 2026-10-12 (SAFETY_NEW) gets the released one back."""
    return s.replace(SAFETY_NEW, SAFETY_OLD)


def mujoco_text(s: str, rec: dict) -> str:
    """A MuJoCo base's texts name the engine it runs on, and the wall time per control step measured on MuJoCo
    (release.json step_s: its reference's oracle run) where the released text stated Genesis'."""
    for a, b in MUJOCO_WORDS + determinism_words(bool(rec.get("gpu_physics"))):
        s = s.replace(a, b)
    s = README_GPU_RX.sub(lambda m: ("The physics (a deterministic GPU cloth step, the rest on the CPU) replays bit for bit on "
                                     "one host; the\nfinal-state hashes (`deterministic`) are still diagnostic" if rec.get("gpu_physics")
                                     else "The physics runs on the CPU and is bit-deterministic on one host; the\nfinal-state "
                                     "hashes (`deterministic`) are still diagnostic"), s)
    s = s.replace("RoboARC on MuJoCo 3.14.0 (CUDA physics, Nyx RGB)", "RoboARC on MuJoCo 3.14.0 ("
                  + ("CPU physics, CUDA cloth step" if rec.get("gpu_physics") else "CPU physics") + ", Nyx RGB)")
    if rec.get("step_s"):
        s = STEP_RX.sub(f"One control step takes about {rec['step_s']:g} s of wall clock", s)
    return s


FLAT_OLD = "Keep every cloth point at most 30 mm above the tabletop and no more than 8 mm below it."
FLAT_NEW = ("The sheet must lie flat on the tabletop, fully in contact with the table surface{fence}: keep every cloth "
            "point at most 30 mm above the tabletop and no more than 8 mm below it.")   # = the task code's (goal_description)
FOLD_FLAT_OLD = ("Flat: every point of the blanket within 35 mm of its height in the target fold, and no point more than "
                 "30 mm above the target's highest point.")
FOLD_FLAT_NEW = ("Flat: the folded cloth must lie flat on the tabletop, its bottom layer fully in contact with the table "
                 "surface: every point of the blanket within 35 mm of its height in the target fold, and no point more than "
                 "30 mm above the target's highest point.")   # = blanket_fold's own words (common.py)
GOAL_DIR = "environment/goal"   # a package's goal-state images (fold: the target shape is shown, never given as coordinates)


# The world frame without the table height or the robot base's position (user 2026-10-13: "in the wild nobody tells you
# how tall the table is"; the base sits on the table top, so its z is the table height). Text only: the axes stay.
FRAME_PRIV = re.compile(r"- World frame: robot base at \([^)]*\)(?:, table top at z = [-0-9.]+)?, \+z up\. Quaternions are \*\*WXYZ\*\*\.")
FRAME_PRIV_NEW = ("- World frame: +z up. Quaternions are **WXYZ**. The ee_pose action and the proprio TCP pose are given in this "
                  "world frame.")
FRAME_STD = re.compile(r"world frame \(robot base at \([^)]*\),\s*\+z up\)")
FRAME_STD_NEW = ("world frame (origin at the robot's root link on its mounting, +z up; the frame of the proprio TCP pose and the "
                 "camera extrinsics)")   # standard mode's frame (user 2026-10-13, option B): the service's world frame has its origin
#                                          at the robot's root link; named "world" as main's other services name theirs
FRAME_STD_RX = ((re.compile(r"world frame \(\+z up; the frame of the proprio TCP pose\)"), FRAME_STD_NEW),   # 10-13 first wording
                (re.compile(r"`spec\[\"world\"\]` the frame conventions\."), "`spec[\"world\"]` the frame conventions (the world "
                 "frame: origin at the robot's root link on its mounting, +z up; no positions of the scene are given)."))
FRAME_LEAKS = ("robot base at", "table top at z")


def frame_words(s: str, where: str, mode: str = "privileged") -> str:
    """The frame lines without the table height and the base position (privileged: the sim's world frame, +z up; standard:
    the service's world frame, whose origin is the robot's root link); no text may still name either."""
    s = FRAME_STD.sub(FRAME_STD_NEW, FRAME_PRIV.sub(FRAME_PRIV_NEW, s))
    if mode == "standard":
        for rx, new in FRAME_STD_RX:
            s = rx.sub(new, s)
        # the texts written on dev (package_text) say "robot frame" / robot_T_cam for the same frame: one name in what ships
        s = s.replace("robot frame", "world frame").replace("robot_T_cam", "world_T_cam")
    left = [w for w in FRAME_LEAKS if w in s]
    assert not left, f"{where}: still names {left} (the table height is never disclosed)"
    return s


# R5 (dev 84c360ff28, user 2026-10-13): the static-contact guard, the Robot safety line's sentence from
# topogym_rules for MuJoCo instances; the released texts predate it. Standard mode reports the stops in clipped[].
R5_ANCHOR = "stalls at that torque."
R5_SENTENCE = (" An arm with any part more than 5 mm deep in a fixed object (a fixture, the table) is stopped where it is: its "
               "joint targets become its measured joints for that control step, so it stops pushing ({report}).")
R5_REPORT = {"privileged": "the safety readings report these stops as static_guard",
             "standard": "the response does not report these stops"}   # the standard service has no clipped[] entry for them (audit S21)


def r5_words(s: str, mode: str) -> str:
    """The static-contact guard's sentence after the Robot safety line's torque sentence, once (a no-op where it is)."""
    if "deep in a fixed object" in s or R5_ANCHOR not in s:
        return s
    return s.replace(R5_ANCHOR, R5_ANCHOR + R5_SENTENCE.format(report=R5_REPORT[mode]), 1)


CLOCK_OLD = "The episode starts when the service comes up and ends on"


def clock_words(s: str, mode: str) -> str:
    """Standard mode: the service's clock starts at the agent's first request (main's #163 rule; server.py touch()), so the
    body no longer says the episode starts when the service comes up; the Time limit section states the clock."""
    if mode != "standard":
        return s
    s = s.replace(CLOCK_OLD, "The episode ends on", 1)
    assert "service comes up" not in s and "since the service first started" not in s, "standard text still dates the clock from the service start"
    return s


TIERS = ("very_easy", "easier", "easy", "medium", "hard")


NAMES = json.loads((HERE / "roboarc_names.json").read_text())   # owner's mapping: old base -> new base, family, number, difficulty
DIFFICULTY = {v["new"].removeprefix(BENCH + "-"): v["difficulty"] for v in NAMES.values()}


def difficulty_words(s: str, rec: dict, base: str = "") -> str:
    """task.toml `difficulty` (TAXONOMY), in both modes: the owner's value for the base (roboarc_names.json; the names
    carry no difficulty), else the variation's tier: a text templated on another tier kept the template's value, and
    some packages had no key (audits S2 / S3)."""
    tier = DIFFICULTY.get(base) or re.sub(r"^(rd|v2)_|_chain$", "", rec.get("tier") or "")
    assert tier in TIERS, f"{base}: no difficulty (not in roboarc_names.json, and its tier {rec.get('tier')!r} names none)"
    line = f'difficulty = "{tier}"'
    if re.search(r'(?m)^difficulty = ', s):
        return re.sub(r'(?m)^difficulty = "[^"]*"', line, s, count=1)
    s, n = re.subn(r'(?m)^(family = "[^"]*"[^\n]*\n)', lambda m: m.group(1) + line + "\n", s, count=1)
    assert n == 1, "task.toml: no family line to put difficulty after"
    return s


VOLUME_RULES = re.compile(r"(?m)^- (?:Keep the rope volume intact|Invalid rope volume ends the episode)[^\n]*\n")
STIFFNESS = (re.compile(r" \(Young's modulus [0-9.]+ kPa\)"), re.compile(r" \([^()\n]*N m/rad per joint\)"))


def audit_words(s: str, rec: dict, mode: str, readme: bool) -> str:
    """The text audits of 2026-10-13 (the work notes, AUDIT_PACKAGING.md), as rules:
    B4  a chain base has no rope volume, so the released volume rules go (the chain sentence says nothing about volume
        ends the episode);
    B5  a standard text carries no privileged grading line (two replays of a submitted trajectory);
    S19 standard texts state no physical parameter of the task objects (stiffness numbers); privileged keeps them;
    S2  a README's lower oracle row agrees with its key table."""
    if rec.get("chain"):
        s = VOLUME_RULES.sub("", s)
        s = s.replace('as "rope_volume_invalid: <what>" or "forced_breakdown: <what>"', 'as "forced_breakdown: <what>"')
    # a dated decision note inside a quoted source excerpt (blanket_flatten rd's predicate): not for the agent (2026-10-13)
    s = s.replace("        # (user 2026-10-12: that read-out-only union, ~150-190 ms per control step, is dropped; until then the last tick\n"
                  "        # always computed it, complete_metrics). The birth (physics step 0) still reports it: the frozen instances'\n",
                  "        # The birth (physics step 0) still reports it: the frozen instances'\n")
    assert "(user 20" not in s, "a dated user decision note in an agent-facing text"
    s = re.sub(r"(?m)^## Task failure conditions\n\s*\n(?=## )", "", s)   # no empty failure section (audit V6, 2026-10-17)
    if mode == "standard":
        s = re.sub(r"(?m)^- Grading: the verifier replays your trajectory twice[^\n]*\n", "", s)
        for rx in STIFFNESS:
            s = rx.sub("", s)
    if readme and rec.get("oracle") != "none" and rec.get("evidence"):
        s = s.replace('| oracle | none yet (`oracle = "none"`, no `solution/`) |', '| oracle | `full` (`solution/`) |')
    return s


# interlocked_claws, the whole-piece separation goal (easy / medium / hard, Cast ABC, Rotor; user 2026-10-13). Its predicate
# (topogym/tasks/interlocked_claws/terminal.py separation_candidate) checks a gap between the two complete pieces and that
# BOTH are still, for the confirmation time; it checks no goal square and no separator. The texts ported from the Genesis
# release kept the placement families' two grader-details lines ("Goal square in this instance ...", "The separator is
# ... free to move (its speed is not checked) ..."), which contradict the requirement above them. They go; in their place
# stands the separation line of dev's scripts/topogym_rules.py claws_separation_words() (word for word), unless the
# package's text already carries it. A placement text (very_easy / easier, the rd tiers) has no SEPARATION_MARK: no-op.
SEPARATION_MARK = "release and transport into the green region are not required."
SEPARATION_GAP = re.compile(r"Leave at least (\S+ mm) between the complete pieces")
SEPARATION_FALSE = re.compile(r"(?m)^- (?:Goal square in this instance: |The separator is an invisible, fixed vertical plane )[^\n]*\n")
SEPARATION_LINE = ("- Separation as the grader measures it: every vertex of both pieces' visual and collision meshes counts, and "
                   "the gap is measured along ten directions only: the three world axes, each piece's own three body axes, and "
                   "the line between the two pieces' centres (the mean of each piece's vertices). Along at least one of them the "
                   "two pieces must be at least {gap} apart; a separating plane in any other direction "
                   "does not count. The green dashed square on the table is not part of this goal: neither containment in it nor "
                   "a separator plane is checked.\n")


def separation_words(s: str, where: str) -> str:
    """The grader details of a claws separation text say what its predicate checks (instruction.md and README.md)."""
    if SEPARATION_MARK not in s:
        return s
    line = SEPARATION_LINE.format(gap=SEPARATION_GAP.search(s).group(1))
    false = SEPARATION_FALSE.findall(s)
    if line in s:
        assert not false, f"{where}: the separation line next to a goal-square / separator line"
    else:
        assert len(false) == 2, f"{where}: expected the goal-square and the separator line, found {len(false)}"
        s = SEPARATION_FALSE.sub("", s.replace(false[0], line, 1))
    left = [w for w in ("free to move", "speed is not checked", "Goal square", "The separator is") if w in s]
    assert not left and s.count(line) == 1, f"{where}: a separation text still says {left}"
    return s


def family_words(name: str, s: str, rec: dict) -> str:
    """A new base templated on another family's package (slotted / orbits on horseshoe_ring, blanket_fold on
    blanket_flatten): its grader and its standard sim name its own family, never the template's."""
    fam = rec["family"]
    if name == "tests/test.sh":
        s = re.sub(r"--family \S+", f"--family {fam}", s)      # privileged's grader; standard's reads its sim's result
    elif name.endswith("docker-compose.yaml"):
        s = re.sub(r"RCB_TASK: \S+", f"RCB_TASK: {fam}", s)
    return s


def flat_words(s: str, fence: bool) -> str:
    """The cloth families' flat condition led by the user's words (a no-op on texts without it)."""
    new = FLAT_NEW.format(fence=" (cloth resting on the fence does not count)" if fence else "")
    if fence:
        s = s.replace(FLAT_NEW.format(fence=""), new)
    return s.replace(FLAT_OLD, new).replace(FOLD_FLAT_OLD, FOLD_FLAT_NEW)


def goal_images(up: pathlib.Path, t: pathlib.Path, instruction: str) -> None:
    """A package's goal-state images: copied into the agent container at /app/goal/ (both modes); the instruction must
    name each one (an image nobody points to is a mistake)."""
    goal = up / GOAL_DIR
    if not goal.is_dir():
        return
    files = sorted(p for p in goal.iterdir() if p.is_file())
    assert files, f"{up}: empty {GOAL_DIR}"
    for p in files:
        assert f"/app/goal/{p.name}" in instruction, f"{up.name}: instruction.md does not name /app/goal/{p.name}"
        put(t / GOAL_DIR / p.name, p.read_bytes())
    df = t / "environment/Dockerfile"
    s, anchor = df.read_text(), "RUN mkdir -p /app/output\n"
    assert s.count(anchor) == 1, f"{t.name}: environment/Dockerfile has no single {anchor!r}"
    if "COPY goal/ /app/goal/" not in s:
        s = s.replace(anchor, "COPY goal/ /app/goal/\n" + anchor)
    df.write_text(s)


def swap(s: str, old: str, new: str, what: str, count: int = 1) -> str:
    assert s.count(old) == count, f"{what}: expected {count} x {old[:60]!r}, found {s.count(old)}"
    return s.replace(old, new)


R_HOLD_STRICT = ("Success needs both replays to succeed AND at least one of them to reach the goal with the thresholds above "
                 "within your own rows (not only in the hold).")
VERIFY_STRICT = ("at least one replay must\nreach the goal with the instance's own thresholds within the trajectory's rows.")
R_HOLD_WORDS = ("`strict_goal_reached`, not required", "each of them may complete its confirmation within that hold",
                "every replay may confirm\nwithin that hold")


SERVICE_END = ("\n  This service also ends the episode itself (`meta[\"reason\"]` is `timeout`; it is graded as it stands) when its own\n"
               "  count of that limit, `meta[\"time_left_s\"]`, reaches 0. That count starts at your first request to the service, so it\n"
               "  never reaches 0 before your time is up.\n")   # PROTOCOL §2.12: a service that ends the episode on its clock says so


def review_fixes(s: str, name: str, base: str, mode: str) -> str:
    """The text fixes for one rendered file (name: instruction.md | README.md)."""
    up = PKG / f"{base}-privileged" / "instruction.md"
    step = base_step(base); hz = int(re.search(r"Control runs at (\d+) Hz", up.read_text()).group(1))
    if mode == "privileged":
        # the families with a grading hold state the strict rule (R_HOLD_STRICT; README: VERIFY_STRICT), as the grader applies it
        assert not any(w in s for w in R_HOLD_WORDS), f"{base} {name}: R-hold wording"
        if name == "instruction.md" and "final targets for up to" in s:
            s = swap(s, R_HOLD_STRICT, R_HOLD_STRICT[:-2] + "; `s.replay(actions, grading=True)` reports it as "
                     "`[\"grading\"][\"strict_success\"]`).", f"{base} the privileged grading rule")
        if name == "instruction.md":
            cap = row_cap(base)
            s = swap(s, "There is no step budget; your\n  wall-clock budget is",
                     f"There is no step budget, but a trajectory may have at most\n  {cap} rows ({cap / hz / 60:.1f} min of simulated time): "
                     "the verifier has to replay it twice within its own time limit, so `s.step` stops\n  after that many control steps "
                     "since the last reset, and a longer trajectory is rejected as invalid (reward 0). Your\n  wall-clock budget is",
                     f"{base} step budget")
            # the agent's limit is stated once, by the closing `## Time limit` section normalize writes (PROTOCOL §2.12)
            s, n = re.subn(r"Your\n  wall-clock budget is \d+ min \(\d+ s\)\. ", "The time limit\n  (`## Time limit`) is your only budget. ", s)
            assert n == 1, f"{base}: the privileged budget sentence"
    elif name == "instruction.md":
        assert step_time(s) == step, f"{base}: the two modes state different step times"
        # the agent's limit is stated once, in `## Time limit` (PROTOCOL §2.12): the service's clock counts from the agent's
        # first request (server.py touch(), main's #163 rule), so its `time_left_s` never runs out before the cutoff. The
        # service still ends the episode at 0 and the body says so (SERVICE_END); the Genesis release's "has run since the
        # service first started" is gone
        s, n = re.subn(r'- \*\*Your wall-clock budget is \d+ min \(\d+ s\)\*\* \(`meta\["time_left_s"\]` counts it down\); ',
                       "- **Your only budget is the time limit (`## Time limit`)**; ", s)
        assert n == 1, f"{base}: the standard budget sentence"
        s, n = re.subn(r" One control step takes about [0-9.]+ s of wall clock here\.\n(?=- There is nothing to hand in)",
                       lambda m: SERVICE_END, s)        # the step time stays once, under "Time per call" (audit S8)
        assert n == 1, f"{base}: the standard budget bullet's end"
        s = swap(s, "steps are unlimited. Actions and\n  simulated time are reported as cost.",
                 f"there is no step budget, but the\n  episode ends after {MAX_STEPS} control steps, watches included ({MAX_STEPS / hz / 60:.0f} min "
                 "of simulated time; a technical cap,\n  `spec[\"max_control_steps\"]`: `meta[\"reason\"]` is `step_guard`, and the episode is graded as it stands). "
                 "Actions and simulated time are reported as cost.", f"{base} step guard")
        m = re.search(r"(?ms)^## Time per call\n.*?(?=^## )", s)
        assert m and "the client gives up waiting after 1800 s" in m.group(0), f"{base}: Time per call"
        s = s[:m.start()] + time_per_call(step, hz) + "\n" + s[m.end():]
    return s


def limits_env(name: str, s: str, base: str, mode: str) -> str:
    """Privileged Dockerfiles: the row cap for the agent's tooling and the verifier (rcb_topogym reads RCB_MAX_STEPS),
    and the verifier's own timeout for its grader (it skips the review-frame pass when the time left would not cover it)."""
    if mode != "privileged" or name not in ("environment/Dockerfile", "tests/Dockerfile"):
        return s
    env = f"ENV RCB_MAX_STEPS={row_cap(base)}"
    if name == "tests/Dockerfile":
        env += f" RCB_VERIFIER_BUDGET_S={M.verifier_timeout(base):.0f}"
    line = "ENV PYTHONPATH=/opt/tools:/opt/topogym\n" if name == "environment/Dockerfile" else "ENV PYTHONPATH=/tests:/opt/topogym\n"
    return swap(s, line, line + "# the longest trajectory the verifier replays within its timeout (scripts/roboarc/mk_tasks.py row_cap)\n" + env + "\n",
                f"{base} {name}")


def rewrite_toml(s: str, base: str, rec: dict) -> str:
    s = TC.set_agent_timeout(mujoco_text(text(instance_facts(s, rec, "privileged")), rec), CUTOFF_SEC, base)
    s = re.sub(r'(?m)^simulator = "topogym"', f'simulator = "{BENCH}"', s)
    s = re.sub(r'(?m)^robot = "([a-z_]+)"', lambda m: f'robot = "{ROBOT.get(m.group(1), m.group(1))}"', s)
    s = re.sub(r'(?m)^author_name = "jianwen"', 'author_name = "Jianwen Lyu"', s)
    s = s.replace('authors = [{ name = "jianwen" }]', 'authors = [{ name = "Jianwen Lyu" }]')
    s = s.replace('"topogym",', f'"{BENCH}",')
    s = re.sub(r'(?m)^tooling_version = "\d+"', f'tooling_version = "{REL["tooling_version"]}"', s)
    s = swap(s, EGRESS_OLD, EGRESS_NEW, f"{base} agent egress")
    s = re.sub(r'(?m)^oracle_evidence = .*\n', "", s)
    return re.sub(r'(?m)^oracle = .*\n', lambda m: M.oracle_lines(base, rec) + "\n", s, count=1)


def rewrite_readme(s: str, base: str, mode: str, rec: dict) -> str:
    s = grasp_text(contact_text(chain_text(mujoco_text(text(instance_facts(s, rec, mode)), rec), rec), rec), rec).replace("from `tests/instance.json`", "from the frozen instance (`/tests/instance.json`, staged from the image and "
                        "checked against `tests/runtime_binding.json`)")
    s = re.sub(r"(?ms)^## Reference solution\n.*?(?=^## )", "## Reference solution\n" + M.ref_solution(mode, rec) + "\n\n", s)
    nop = M.nop_words(base, mode, rec)
    s = review_fixes(s, "README.md", base, mode)
    cap = (f"\nA trajectory may have at most {row_cap(base)} rows (`RCB_MAX_STEPS`: two replays must fit the verifier's "
           "timeout; a longer one is\n`bad_trajectory`)." if mode == "privileged" else "")
    s = re.sub(r"(?ms)^(## Verification\n.*?)(?=^## )", lambda m: m.group(1).rstrip() + cap + "\n\n" + nop + "\n\n", s)
    return s.rstrip() + "\n\n### Provenance\n" + M.provenance(base, rec) + "\n\n### Change Log\n" + M.change_log(base, mode, rec) + "\n"


GRIP_OLD = "native drive target (m, limits 0 .. 0.8; open/closed targets in robot_spec)"
GRIP_NEW = ("native drive target (rad, limits 0 .. 0.8: 0 = open, 0.8 = closed, the Robotiq 2F-85's driver joint; "
            "open/closed targets in robot_spec)")
EXPOSED_OLD = ("The success implementation and internal measurements are not exposed;\n"
               "  the task requirements above still apply in full.")
EXPOSED_NEW = ("The success implementation is not exposed, and of the simulator's internal\n"
               "  measurements only the safety readings of `meta[\"safety\"]` are (listed under Action types, below); the task\n"
               "  requirements above still apply in full.")
WAIT_S = 3600             # v2_4/rcb_client.Robot.__init__'s default timeout (600 s until 2026-10-10)
WAIT_OLD = ("(default connection budget: 600 seconds,", "(default timeout 600 s)")
WAIT_NEW = (f"(default connection budget: {WAIT_S} seconds,", f"(default timeout {WAIT_S} s)")
# the call's image budget of the standard service (review round 3), read from its source so that the texts state what is enforced
FRAMES_AT = "- Both action types: no gripper target goes lower than"
FRAMES_NEW = (f"- One call returns at most {MAX_FRAMES} images (`spec[\"max_frames_per_call\"]`: cameras x frames; a frame with\n"
              f"  `depth=\"frames\"` counts {DEPTH_FRAME}). A request that would exceed it is answered 400 before anything moves: ask for\n"
              "  fewer cameras, a lower `video_fps` or a shorter call. Rows that take longer than planned (the speed limit, `ee_pose`)\n"
              "  stop adding frames at that number; the motion is executed in full and `meta[\"frames_skipped\"]` counts them.\n")
OPENING_NEW = ("\n  The proprio group `gripper_opening` (`res.group(\"gripper_opening\")`) is the measured gap between each gripper's finger\n"
               "  pads in metres (0 = the pads touch); the gripper entries of `joint_pos` are in the unit of the gripper command above.")


def panda(base: str) -> bool:
    return 'robot = "panda_robotiq"' in (PKG / f"{base}-privileged" / "task.toml").read_text()


def native(base: str) -> bool:
    return "_native_gripper" in (PKG / f"{base}-standard" / "instruction.md").read_text()


def review_fixes_r2(s: str, base: str, mode: str) -> str:
    """Round-2 text fixes of an instruction.md. The Robotiq 2F-85's native drive coordinate is its driver joint angle
    (topogym/robot/panda_robotiq: gripper_position_rad, limits 0 .. 0.8 rad, GraspDescription open 0 / closed 0.8); the
    release's texts gave it in metres. ALOHA's native coordinate is the finger rail in metres, as its texts say."""
    assert (GRIP_OLD in s) == (panda(base) and native(base)), f"{base} {mode}: the gripper unit text and the robot disagree"
    if GRIP_OLD in s:
        s = swap(s, GRIP_OLD, GRIP_NEW, f"{base} {mode} Robotiq unit")
    if mode == "standard":
        s = swap(s, EXPOSED_OLD, EXPOSED_NEW, f"{base} safety readings")
        for a, b in zip(WAIT_OLD, WAIT_NEW):
            s = swap(s, a, b, f"{base} Robot() wait")
        s = swap(s, "`meta[\"clipped\"]` says so.\n- `ee_pose`", "`meta[\"clipped\"]` says so." + OPENING_NEW + "\n- `ee_pose`",
                 f"{base} gripper_opening")
        s = swap(s, FRAMES_AT, FRAMES_NEW + FRAMES_AT, f"{base} image budget")      # round 3
    return s


GPU_OLD = ("# Nyx renders through Vulkan, so the container needs the NVIDIA graphics libraries as well as compute: `runtime: nvidia`\n"
           "# with NVIDIA_DRIVER_CAPABILITIES=all (`capabilities: [gpu]` alone injects compute only).\n")
GPU_NEW = ("# Nyx renders through Vulkan: the device reservation below passes exactly one GPU (Nyx needs exactly one), and\n"
           "# NVIDIA_DRIVER_CAPABILITIES=all makes the container toolkit mount the driver's graphics libraries, which the image's\n"
           "# Vulkan ICD and EGL vendor files name (images/roboarc/Dockerfile). No `runtime: nvidia`: with the image's\n"
           "# NVIDIA_VISIBLE_DEVICES=all it can expose every GPU of the host.\n")


def gpu_lines(name: str, s: str) -> str:
    """Compose files: one GPU through the device reservation only (no `runtime: nvidia`)."""
    if not name.endswith("docker-compose.yaml"):
        return s
    assert s.count(GPU_OLD) == 1 and len(re.findall(r"(?m)^\s*runtime: nvidia\n", s)) == 1, f"{name}: unexpected GPU block"
    s = re.sub(r"(?m)^\s*runtime: nvidia\n", "", s.replace(GPU_OLD, GPU_NEW))
    assert "count: 1" in s and "capabilities: [gpu]" in s and "NVIDIA_DRIVER_CAPABILITIES: all" in s, name
    return s


def image_lines(s: str) -> str:
    s = re.sub(r"(?m)^FROM\s+" + IMAGE_RX.pattern, "FROM " + IMAGE, s)
    return re.sub(r"(?m)^(\s*image:\s*)" + IMAGE_RX.pattern, lambda m: m.group(1) + IMAGE, s)


STAGE = "RUN python /opt/robotangle/stage_instance.py {base} {dest} --purge   # the frozen instance, from the image (scripts/roboarc/assets.json), checked against runtime_binding.json"
ENTRY = ('    # The frozen instance comes from the image (scripts/roboarc/assets.json): the entrypoint stages it next to a copy of\n'
         '    # runtime_binding.json and checks it (images/roboarc/stage_instance.py), then runs the command.\n'
         '    entrypoint: ["sh", "-c", "python /opt/robotangle/stage_instance.py {base} ' + STAGED + ' --binding /srv/runtime_binding.json && exec \\"$$@\\"", "stage"]\n')


def staged(name: str, s: str, base: str) -> str:
    """Task files that stage the frozen instance from the image instead of copying it from the task directory."""
    def sub(a: str, b: str) -> str:
        assert s.count(a) == 1, f"{base} {name}: expected one {a!r}"
        return s.replace(a, b)
    if name == "environment/Dockerfile" and "COPY instance.json /app/instance.json\n" in s:
        s = sub("COPY instance.json /app/instance.json\n", "")
        s = sub("COPY instance_assets/ /app/instance_assets/", STAGE.format(base=base, dest="/app"))
    elif name == "tests/Dockerfile" and "instance_assets" in s:
        s = sub("COPY grade.py instance.json rcb_topogym.py test.sh /tests/", "COPY grade.py rcb_topogym.py test.sh /tests/")
        s = sub("COPY instance_assets/ /tests/instance_assets/", STAGE.format(base=base, dest="/tests"))
    elif name == "environment/docker-compose.yaml" and "RCB_INSTANCE: /srv/instance.json" in s:
        s = sub("RCB_INSTANCE: /srv/instance.json", f"RCB_INSTANCE: {STAGED}/instance.json")
        s = sub("    command: sh /srv/run.sh\n", ENTRY.format(base=base) + "    command: sh /srv/run.sh\n")
    return s


CPU_PHYSICS = (("# The verifier replays in the simulator, which needs the GPU (CUDA physics, Nyx camera images for the replay video).",
                "# The verifier replays in the simulator: MuJoCo physics on the CPU; the GPU renders the Nyx camera images of the replay video."),
               ("# TopoGym runs its physics on the GPU (CUDA) and renders camera images with Nyx.",
                "# RoboARC runs its physics on MuJoCo (CPU) and renders camera images with Nyx on the GPU."))
GPU_PHYSICS = (("# The verifier replays in the simulator, which needs the GPU (CUDA physics, Nyx camera images for the replay video).",
                "# The verifier replays in the simulator, which needs the GPU (the cloth's CUDA step, Nyx camera images for the replay video)."),
               ("# TopoGym runs its physics on the GPU (CUDA) and renders camera images with Nyx.",
                "# RoboARC runs its physics on MuJoCo with the cloth's CUDA step on the GPU, and renders camera images with Nyx."))


def mujoco_lines(name: str, s: str, rec: dict) -> str:
    """A MuJoCo task: the SIMD masking in the images that run a Session (privileged agent and verifier: the RoboARC
    image) and in the standard sim service; render_bridge.py (and nyx_direct.py for its families) copied next to the
    privileged tooling; the compose comments say where the physics runs (the GPU reservation itself is unchanged: Nyx,
    and a GPU physics profile, need it)."""
    def sub(a: str, b: str) -> str:
        assert s.count(a) == 1, f"{name}: expected one {a!r}"
        return s.replace(a, b)
    direct = rec["family"] in M.DIRECT
    env = ("# MuJoCo task: numpy / OpenBLAS SIMD masking, so every CPU class generates the frozen bytes\n"
           "ENV " + " ".join(f"{k}={v}" for k, v in MUJOCO_ENV) + "\n")
    if name in ("environment/Dockerfile", "tests/Dockerfile") and f"FROM {IMAGE}\n" in s:
        s = sub(f"FROM {IMAGE}\n", f"FROM {IMAGE}\n" + env)
        if name == "environment/Dockerfile":
            s = sub("COPY rcb_topogym.py /opt/tools/rcb_topogym.py\n",
                    "COPY rcb_topogym.py /opt/tools/rcb_topogym.py\nCOPY render_bridge.py /opt/tools/render_bridge.py\n"
                    + ("COPY nyx_direct.py /opt/tools/nyx_direct.py\n" if direct else ""))
        else:
            s = sub("COPY grade.py rcb_topogym.py test.sh /tests/", "COPY grade.py rcb_topogym.py render_bridge.py "
                    + ("nyx_direct.py " if direct else "") + "test.sh /tests/")
    if name == "environment/docker-compose.yaml" and "\n  sim:\n" in s:
        s = sub("      RCB_TASK: ", "".join(f"      {k}: {v}\n" for k, v in MUJOCO_ENV) + "      RCB_TASK: ")
    if name.endswith("docker-compose.yaml"):
        for a, b in (GPU_PHYSICS if rec.get("gpu_physics") else CPU_PHYSICS):
            s = s.replace(a, b)
        s = s.replace("# TopoGym runs its physics on the GPU (CUDA) and renders camera images with Nyx: the GPU goes to the sidecar.",
                      "# RoboARC renders camera images with Nyx" + (" and steps its cloth on CUDA" if rec.get("gpu_physics")
                                                                        else "") + ": the GPU goes to the sidecar.")
        assert "CUDA physics" not in s and "TopoGym runs its physics" not in s, f"{name}: a Genesis-era GPU comment is left"
    return s


def legacy_texts(base: str, out: pathlib.Path) -> dict[tuple[str, str], str]:
    """The base's texts as the rewrites give them, before normalize: {(mode, file): text}."""
    rec = REL["bases"][base]
    texts = {}
    for mode in ("privileged", "standard"):
        t = out / f"{BENCH}-{base}-{mode}"; up = PKG / f"{base}-{mode}"
        rd = lambda n: (up / n).read_text()  # noqa: E731
        put(t / "task.toml", difficulty_words(rewrite_toml(rd("task.toml"), base, rec), rec, base).encode())
        fence = "`fence`" in rd("instruction.md")
        put(t / "README.md", audit_words(r5_words(frame_words(flat_words(rewrite_readme(separation_words(rd("README.md"), f"{t.name}/README.md"), base, mode, rec), fence), f"{t.name}/README.md", mode), mode), rec, mode, True).encode())
        ins = grasp_text(contact_text(chain_text(mujoco_text(text(instance_facts(separation_words(rd("instruction.md"), f"{t.name}/instruction.md"), rec, mode)), rec), rec), rec), rec).replace("RoboARC", "RoboARC (simulator package `topogym`)", 1)
        ins = flat_words(review_fixes_r2(review_fixes(ins, "instruction.md", base, mode), base, mode), fence)
        ins = clock_words(r5_words(frame_words(safety_text(ins) if mode == "standard" else ins, f"{t.name}/instruction.md", mode), mode), mode)
        put(t / "instruction.md", audit_words(ins, rec, mode, False).encode())
        for n in ("environment/Dockerfile", "environment/docker-compose.yaml", "tests/Dockerfile", "tests/docker-compose.yaml"):
            if (up / n).is_file():
                assert "RCB_MAX_CALL_WALL_S" not in rd(n) and "RCB_MAX_STEPS" not in rd(n), f"{base} {n}: the texts assume the defaults"
                text_n = limits_env(n, staged(n, gpu_lines(n, image_lines(rd(n))), base), base, mode)
                text_n = text_n.replace("docs/benchmarks/robotangle.md", f"docs/benchmarks/{BENCH}.md")   # the benchmark's page
                put(t / n, family_words(n, mujoco_lines(n, text_n, rec), rec).encode())
        goal_images(up, t, (t / "instruction.md").read_text())
        put(t / "tests/test.sh", family_words("tests/test.sh", (up / "tests/test.sh").read_text(), rec).encode(), exe=True)
        texts.update({(mode, n): (t / n).read_text() for n in M.TEXTS if (t / n).is_file()})
    return texts


# Where each field stands in the texts, for one mode (None: both): every one must be found exactly once.
FIELD_SPOTS = {
    ("README.md", None): ["## Reference solution\n{{REFERENCE_SOLUTION}}\n\n", "\n\n{{NOP}}\n\n",
                          "### Provenance\n{{PROVENANCE}}\n\n### Change Log\n{{CHANGE_LOG}}\n"],
    ("README.md", "privileged"): ["\nA trajectory may have at most {{ROW_CAP}} rows (`RCB_MAX_STEPS`"],
    ("instruction.md", "privileged"): ["a trajectory may have at most\n  {{ROW_CAP}} rows ({{ROW_CAP_MIN}} min of simulated time)"],
    ("instruction.md", "standard"): ["{{TIME_PER_CALL}}"],
    ("task.toml", None): ["{{AGENT_EGRESS}}\n", 'tooling_version = "{{TOOLING_VERSION}}"', "{{ORACLE}}\n"],
    ("environment/Dockerfile", "privileged"): ["ENV RCB_MAX_STEPS={{ROW_CAP}}\n"],
    ("tests/Dockerfile", "privileged"): ["ENV RCB_MAX_STEPS={{ROW_CAP}} RCB_VERIFIER_BUDGET_S={{VERIFIER_BUDGET_S}}\n"],
}


def with_fields(base: str, mode: str, name: str, s: str) -> str:
    """A text with mk_tasks.fields() put back as {{FIELD}}s; mk_tasks.fill() must give the text back."""
    values = M.fields(base, mode, REL["bases"][base])
    assert "{{" not in s and "}}" not in s, f"{base} {mode} {name}: braces in the text"
    p = s
    for spot in FIELD_SPOTS.get((name, None), []) + FIELD_SPOTS.get((name, mode), []):
        f = M.fill(spot, values, name)
        assert p.count(f) == 1, f"{base} {mode} {name}: {spot[:40]!r} found {p.count(f)} times"
        p = p.replace(f, spot)
    if name == "README.md":
        assert p.endswith(FIELD_SPOTS[("README.md", None)][2]), f"{base} {mode}: README does not end with provenance and change log"
    p = p.replace(IMAGE, "{{IMAGE}}")
    assert mode == "standard" or f"{values['ROW_CAP']} rows" not in p, f"{base} {mode} {name}: a row cap outside its field"
    assert M.fill(p, values, name) == s, f"{base} {mode} {name}: the fields do not give the text back"
    return p


def is_final(base: str) -> bool:
    return "{{AGENT_EGRESS}}" in (PKG / f"{base}-privileged" / "task.toml").read_text()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("bases", nargs="+")
    a = ap.parse_args(argv)
    todo = [b for b in a.bases if not is_final(b)]
    if a.check:
        for b in todo:
            print(f"FAIL packages/{b}-*: Genesis-era texts (run scripts/roboarc/port_package.py {b})")
        return 1 if todo else 0
    for b in todo:
        with tempfile.TemporaryDirectory() as tmp:
            out = {k: with_fields(b, k[0], k[1], s) for k, s in legacy_texts(b, pathlib.Path(tmp)).items()}
        for (mode, name), p in out.items():
            (PKG / f"{b}-{mode}" / name).write_text(p)
        print(f"ported packages/{b}-privileged, -standard")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
