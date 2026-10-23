"""wire_splicing's packages (scripts/roboarc/packages/wire-splicing-01-i01-<mode>/), made from bed-bolt-01-i01's (the
same template, interlocked-claws-easy-i00): the environment/ and tests/ files with the base's and family's names, the
texts (task.toml, README.md, instruction.md) with every task line replaced by wire_splicing's: names, the task sentence,
the public goal (WireSplicingTask.goal_words via goal_description.describe_goal), the grader details
(tools/release/topogym_rules.py, wire_splicing in RIGID), the grasp and objects lines, the implementation reference, the
success rule and the per-step wall clock. The {{FIELDS}} stay for mk_tasks.py. Ported from TopoGym's
mk_package_text_wire.py (runs/wire_splice_design_20261014/out/l3/harbor/tools, 2026-10-21).

usage (repository root, in the release image, the repository's source/ on PYTHONPATH):
    python tools/release/package_text_wire.py <wire instance.json> <step_s>"""
import inspect
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "tools/release")]
from topogym.config.registry import task_definition                      # noqa: E402
from topogym.tasks.wire_splicing.common import WireSplicingTask          # noqa: E402
from topogym.tasks.goal_description import describe_goal, render_goal     # noqa: E402
from topogym_rules import details_section                                 # noqa: E402

PKG = ROOT / "scripts/roboarc/packages"
OLD, NEW = "bed-bolt-01-i01", "wire-splicing-01-i01"
inst = json.loads(pathlib.Path(sys.argv[1]).read_text())
step_s = float(sys.argv[2])
task = task_definition("wire_splicing", "wire_splicing_eye_wrap", 0)(None, seed=0)
goal = describe_goal(task)
summary = goal["summary"]
reqs = render_goal(goal, include_summary=False)
old_summary = ("Drive the long bolt into the blind hole in the bed frame's corner block with the L-shaped hex key until its "
               "head is within 1.5 mm of the block face.")
GRASP_OLD = re.compile(r"assisted grasping \(contact_weld_v1\): a gripper whose command moves towards closed takes hold of a "
                       r"graspable part \(bolt, key\).*?then close it on a part to take hold\.")
grasp_new = ("assisted grasping (contact_weld_v1): a gripper whose command moves towards closed takes hold of a wire "
             "(wire_a, wire_b) once both fingers touch it inside its links' grasp regions, the tool frame within 6 cm of "
             "it and the penetration at most 2 mm, for 0.02 s: of each wire it touches, the touched link nearest the tool "
             "frame and every other link whose centre lies between the two pads then follow the gripper rigidly. Both "
             "fingers on the wires with every contact between the pads also holds, the same way (the red wire's two legs "
             "side by side, one pad on each; the eye's two sides with the blue wire beside them). Any command that opens "
             "the gripper (by more than 0.2 % of its travel) lets it go. A hold that breaks (the fingers open more than "
             "15 %, a held link drifts 1 cm / 0.15 rad from its held pose, the hold has to push or pull with more than "
             "60 N averaged over 20 ms, or a weld has fed more than 0.1 J of work into the wire) is lost, and the gripper "
             "must open before it can take hold again; that force is reached when, for example, a held wire is pulled "
             "hard against the other hand or a fixture. The jig is fixed; the grippers start empty and "
             "nearly closed: open a gripper, then close it on a wire to take hold.")
objects_priv = ("The scene objects are the task's entities (`s.objects()`: `wire_a` (red) and `wire_b` (blue), each fixed "
                "at its terminal, and `jig` (fixed: the bending post with its yellow rod)).")
objects_std = ("Objects here: `wire_a` (red) and `wire_b` (blue), each fixed at its terminal, and `jig` (fixed: the "
               "bending post with its yellow rod). As everywhere in this mode, nothing reports their poses: what you know "
               "of them comes from the cameras.")
predicate = inspect.getsource(WireSplicingTask.evaluate_transition)
details_of = lambda mode: details_section("wire_splicing", inst, limited=mode == "standard").strip("\n")  # noqa: E731


def bullets_end(text, g):
    """The end of the bullet list that starts two lines below the heading at g (or right under it)."""
    lines = text[g:].split("\n")
    n = 1
    while n < len(lines) and (lines[n].startswith("- ") or lines[n] == "") and not (lines[n] == "" and n + 1 < len(lines)
                                                                                  and not lines[n + 1].startswith("- ")):
        n += 1
    return g + len("\n".join(lines[:n]))


def rewrite(text, mode, kind):
    text = text.replace(OLD, NEW)
    text = text.replace("· bed_bolt", "· wire_splicing").replace(old_summary, summary)
    text = text.replace("RoboARC bed_bolt (", "RoboARC wire_splicing (")
    text = text.replace("RoboARC bed_bolt on the same frozen instance", "RoboARC wire_splicing on the same frozen instance")
    text = text.replace('Session("bed_bolt",', 'Session("wire_splicing",').replace("--task bed_bolt ", "--task wire_splicing ")
    text = text.replace("`bed_bolt` / seed 0", "`wire_splicing` / seed 0")
    text = text.replace("variation bed_bolt_crossbar", "variation wire_splicing_eye_wrap")
    text = text.replace("terminal revision bed_bolt_seated_v1", f"terminal revision {task.terminal_revision}")
    text = re.sub(r"One control step takes about [0-9.]+ s of wall clock here",
                  f"One control step takes about {step_s:g} s of wall clock here", text)
    if kind == "instruction":
        i = text.index("## Success requirements")
        g = text.index("## Grader details", i)
        text = text[:i] + reqs + "\n\n" + details_of(mode) + text[bullets_end(text, g):]
    if kind == "readme":
        i = text.index("### Success requirements")
        g = text.index("### Grader details", i)
        h3 = re.sub(r"^## ", "### ", reqs, flags=re.M).replace("\n\n- ", "\n- ")
        det = details_of(mode).replace("## Grader details", "### Grader details").replace("\n\n- ", "\n- ")
        text = text[:i] + h3 + "\n\n" + det + (text[bullets_end(text, g):] or "\n")
    text = GRASP_OLD.sub(grasp_new, text)
    if kind != "toml":
        text = re.sub(r"The scene objects are the task's entities \(`s\.objects\(\)`: [^\n]*\n", objects_priv + "\n", text)
        text = re.sub(r"Objects here: `bed_frame` [^\n]*", objects_std, text)
        text = re.sub(r"Implementation reference: `[^`]*` \(module `[^`]*`, class `[^`]*`\)\.",
                      "Implementation reference: `WireSplicingEyeWrapTask.evaluate_transition` (module "
                      "`topogym.tasks.wire_splicing.common`, class `WireSplicingTask`).", text)
        text = re.sub(r"```python\n    def evaluate_state\(self, state\):\n.*?\n```",
                      lambda _: "```python\n" + predicate.rstrip("\n") + "\n```", text, flags=re.S)
    if kind == "toml":
        text = text.replace('family = "bed_bolt"', 'family = "wire_splicing"')
        text = text.replace('"trajectory", "bed_bolt"]', '"trajectory", "wire_splicing"]')
        text = re.sub(r'grasp = "assisted \(contact_weld_v1 on the bolt and the key[^"]*"',
                      'grasp = "assisted (contact_weld_v1 on the links of both wires: a hand holds every link of a touched '
                      'wire between its pads; the grippers start empty)"', text)
    return text


for mode in ("privileged", "standard"):
    src, out = PKG / f"{OLD}-{mode}", PKG / f"{NEW}-{mode}"
    for d in ("environment", "tests"):
        shutil.rmtree(out / d, ignore_errors=True)
        for f in sorted((src / d).rglob("*")):
            if f.is_file():
                t = out / f.relative_to(src)
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_text(f.read_text().replace(OLD, NEW).replace("--family bed_bolt", "--family wire_splicing")
                             .replace("RCB_TASK: bed_bolt", "RCB_TASK: wire_splicing"))
                t.chmod(f.stat().st_mode)
    for name, kind in (("task.toml", "toml"), ("README.md", "readme"), ("instruction.md", "instruction")):
        text = rewrite((src / name).read_text(), mode, kind)
        (out / name).write_text(text)
        left = [w for w in ("bolt", "Bolt", " key", "thread", "bed_frame", "bed frame", "bed-bolt", "bed_bolt", "crossbar",
                            "hex") if w in text]
        print(out / name, "left-over words:", left)
