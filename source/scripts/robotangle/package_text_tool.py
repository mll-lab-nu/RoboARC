#!/usr/bin/env python3
"""Draft the task texts of a new MuJoCo base (release_rebuild.py package_text / new_bases) from a released template.

usage: scripts/robotangle/package_text_tool.py --template <released base> --template-describe <d0.json> --template-instance <i0.json>
                                              --new <new base> --describe <d1.json> --instance <i1.json> --out <package_text dir>
                                              [--difficulty <tier>] [--public-instructions <file.json>] [--task-note <text>]
                                              [--report <file.md>]
  --template           a released base of scripts/robotangle/packages/ (e.g. interlocked-claws-very-easy-i00): its task.toml,
                       README.md and instruction.md (both modes) are the starting point, as every new base's texts so far
  --template-describe  `rcb_topogym.py describe` of the template's scene on MuJoCo (its -i01 instance), with
  --template-instance  that instance.json; --describe / --instance the same for the new base's instance (made as the release
                       makes it). Run describe in the runtime the release uses (the release image, or a bound tree)
  --new                the new base's name (e.g. interlocked-claws-06-i01)
  --out                the family's package_text directory; writes <out>/<new>-<mode>/{task.toml,README.md,instruction.md}
  --public-instructions  {variation: {"public_instruction": Task.public_instruction()}} (run where the variation is
                       registered): a design whose task class adds words to the family's goal description (a scene note
                       after the task sentence, extra success conditions) states them; describe's public_goal has only the
                       family's rendering. Without it, the texts say what public_goal says
  --task-note          one sentence added after the task sentence (e.g. naming a design's parts); words only, no rule
  --difficulty         the new base's tier (very_easy .. hard; docs/rcb/TAXONOMY.md) for task.toml's difficulty line, e.g.
                       a redesigned tier set; default: the variation record's tier, and no line when it has none (as the
                       experimental new bases)

How: the generator of the released texts (scripts/topogym_mk_task.py: instruction, readme, task_toml) renders the
template's scene and the new scene from their describe records, in privileged mode. Every hunk between the two renders
is a line the scene changes (names, variation, task sentence, success requirements, goal volume and square, grader
details, grasp regions, object list, implementation reference and its evaluate_state excerpt); each is applied to the
template's released files of both modes: the whole hunk where it appears verbatim, else line by line, else the changed
words of a line where they appear. A hunk that applies nowhere is listed in the report (UNRESOLVED) for the drafter.
Lines robotangle-new's generator rewrites anyway (engine words, step time, budget, grasping sentence, contact review,
safety paragraph) are left as released: the step time of the new describe is set to the template's, so no step-time
hunk is made (the released sentence stays; `step_s` in the release manifest is a separate decision).
The drafter reads the report and the before/after diff (`--report`), fixes what is left by hand, and the family's README
in --out records the method (one section per new base). The user reviews new texts before release.
"""
import argparse, difflib, json, pathlib, re, sys

root = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(root / "scripts"), str(root)]
sys.dont_write_bytecode = True
import topogym_mk_task as G  # noqa: E402

FILES = ("instruction.md", "README.md", "task.toml")
TIMING = ("sec_per_step", "sec_per_step_unloaded", "sec_per_step_motion")


def render(d: dict, inst: dict, slug: str, level) -> dict:
    """The released generator's privileged texts for one scene (the same inputs for both scenes but the scene's own)."""
    fam = inst["task"]
    text, _ = G.goal_words(fam, "", d)
    difficulty = (f"{G.robot_words(d).capitalize()} with a {len(d['action_layout'])}-D joint-target action space at "
                  f"{d['control_hz']:g} Hz; success needs the task's predicate (above) held for {d['stable_duration_s']:g} s. "
                  f"Grasping: {d['grasp_rules'].split(':')[0]}.")
    ei = inst.get("eval_index", 0)
    return {"instruction.md": G.instruction(fam, text, d, inst),
            "README.md": G.readme(slug, fam, text, d, ei, inst["variation_id"], inst["seed"], difficulty, True,
                                  G.success_rule_of(inst), inst),
            "task.toml": G.task_toml(slug, fam, text, d, ei, inst["variation_id"], inst["seed"], 2.0, level, inst)}


def hunks(a: str, b: str):
    al, bl = a.splitlines(), b.splitlines()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, al, bl, autojunk=False).get_opcodes():
        if tag == "insert" and i1 > 0:     # an inserted line goes after its predecessor
            yield al[i1 - 1:i1], al[i1 - 1:i1] + bl[j1:j2]
        elif tag != "equal":
            yield al[i1:i2], bl[j1:j2]


def segment(old: str, new: str):
    """The changed words of one line: the shortest span, widened to word boundaries, that differs."""
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    s = 0
    while s < min(len(old), len(new)) - p and old[-1 - s] == new[-1 - s]:
        s += 1
    while p > 0 and old[p - 1] not in " ([`\"'":
        p -= 1
    while s > 0 and old[len(old) - s] not in " .,;:)]`\"'":
        s -= 1
    return old[p:len(old) - s], new[p:len(new) - s]


def apply(text: str, old: list, new: list):
    """Apply one hunk to a released file; returns (text, how) with how in block / lines / words / '' (not applied)."""
    if new and "\n".join(new) + "\n" in text and not ("\n".join(old) + "\n" in text if old else False):
        return text, "already"
    if old and "\n".join(old) + "\n" in text:
        return text.replace("\n".join(old) + "\n", "\n".join(new) + ("\n" if new else "")), "block"
    if not old:
        return text, ""
    if not new and len(old) == 1 and re.match(r"[a-z_]+ = ", old[0]):   # a deleted key line, another comment style
        key = old[0].split("#")[0].rstrip()
        hit = re.compile(rf"(?m)^{re.escape(key)}(\s*#.*)?\n")
        if hit.search(text):
            return hit.sub("", text), "key"
    if not any(difflib.SequenceMatcher(None, o, line).ratio() > .8 for o in old for line in text.splitlines()):
        return text, "n/a (not in this file)"
    if len(old) == len(new):
        out, how = text, []
        for o, n in zip(old, new):
            if o == n or (f"\n{n}\n" in f"\n{out}" and f"\n{o}\n" not in f"\n{out}"):
                continue
            if f"\n{o}\n" in f"\n{out}":
                out = re.sub(rf"(?m)^{re.escape(o)}$", lambda m: n, out); how.append("line")
                continue
            so, sn = segment(o, n)
            if len(so) >= 12 and so in out:
                out = out.replace(so, sn); how.append("words")
            else:
                lines = out.splitlines()
                score, k = max(((difflib.SequenceMatcher(None, o, line).ratio(), k) for k, line in enumerate(lines)),
                               default=(0., -1))
                if score <= .8:
                    how.append("n/a line")    # this line of the hunk is not in this file (the other mode's wording)
                    continue
                # the file's own version of the line (e.g. the released Genesis wording): replaced by the new line;
                # the report marks it for the drafter's check
                lines[k] = n
                out = "\n".join(lines) + ("\n" if out.endswith("\n") else ""); how.append("similar (check)")
        return out, "+".join(sorted(set(how))) or "same"
    # a block of another length (e.g. a new family's grader details from another family's template): replaced from the
    # file's close version of its first line to that of its last, when they stand in order about as far apart
    lines = text.splitlines()
    best = lambda o: max(((difflib.SequenceMatcher(None, o, x).ratio(), k) for k, x in enumerate(lines)), default=(0., -1))
    (s0, i), (s1, j) = best(old[0]), best(old[-1])
    if s0 > .8 and s1 > .8 and i <= j and abs((j - i + 1) - len(old)) <= max(3, len(old) // 3):
        lines[i:j + 1] = new
        return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), "block by anchors (check)"
    return text, ""


# Lines that state a fact of the instance and that a template's released file may state in its Genesis form (a port whose
# grasp model changed, e.g. rope_threading's IPC soft band -> contact_weld_v1 on MuJoCo): when the two renders agree, no
# hunk replaces them, so they are set from the new render directly. (prefixes a file may use, prefix in the render)
FACTS = ((("- Grasp regions", "- Soft grasp patches"), "instruction.md"),
         (("grasp = ",), "task.toml"), (("success_rule = ",), "task.toml"),
         (("Two 6-dof ", "A 7-dof "), "README.md"),     # the README's Difficulty paragraph (robot, rate, grasp model)
         (("| Grasping |",), "README.md"))
OBJECTS = ("The scene objects are the task's entities", "Objects here: ")
GRIPPERS = ("- Grippers: ", "Grasping: ")      # the grasping rule's sentence, in either mode's form (prefix kept)
REGION_CENTRES = re.compile(r" Each region's centre \(in its link's frame\) is in `/app/instance\.json` under "
                            r"`simulator\.grasp_assist\.targets`\.")


def instance_facts(text: str, r1: dict, mode: str):
    """Set the instance-fact lines (FACTS) from the new scene's render; standard mode drops the privileged-only sentence."""
    lines, done = text.splitlines(), []
    for prefixes, src in FACTS:
        new = next((x for x in r1[src].splitlines() if x.startswith(prefixes)), None)
        if new is None:
            continue
        if mode == "standard":
            new = REGION_CENTRES.sub("", new)
        for k, line in enumerate(lines):
            if line.startswith(prefixes) and line != new:
                lines[k] = new; done.append(f"{line[:60]!r} -> {new[:60]!r}")
    objs = next((x for x in r1["instruction.md"].splitlines() if x.startswith(OBJECTS[0])), None)
    if objs:                               # the scene's objects (privileged: the render's line; standard: their names)
        names = re.findall(r"`([^`]+)`", objs.split("s.objects()`:", 1)[1])
        for k, line in enumerate(lines):
            new = objs if line.startswith(OBJECTS[0]) else (OBJECTS[1] + ", ".join(f"`{n}`" for n in names) + "."
                                                           if line.startswith(OBJECTS[1]) else None)
            if new and line != new:
                lines[k] = new; done.append(f"{line[:60]!r} -> {new[:60]!r}")
    rule = next((x[len(GRIPPERS[0]):] for x in r1["instruction.md"].splitlines() if x.startswith(GRIPPERS[0])), None)
    for k, line in enumerate(lines):
        pre = next((p for p in GRIPPERS if line.startswith(p)), None)
        if rule and pre and line[len(pre):] != rule:
            lines[k] = pre + rule; done.append(f"{line[:60]!r} -> {(pre + rule)[:60]!r}")
    out = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    # the implementation excerpt (evaluate_state) as one block: the new scene's code, never a merge of two sources
    rx = re.compile(r"(Implementation reference: [^\n]*\n(?:[^\n]*\n){0,3}?```python\n).*?\n```", re.S)
    m1 = rx.search(r1["instruction.md"])
    if m1 and rx.search(out):
        new = rx.sub(lambda m: m1.group(0), out, count=1)
        if new != out:
            out = new; done.append("implementation excerpt from the new scene")
    return out, done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("template", "template-describe", "template-instance", "new", "describe", "instance", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--difficulty", choices=("very_easy", "easier", "easy", "medium", "hard"))
    ap.add_argument("--report", type=pathlib.Path)
    ap.add_argument("--public-instructions", type=pathlib.Path)
    ap.add_argument("--task-note")
    a = ap.parse_args(argv)
    d0, i0 = json.loads(pathlib.Path(a.template_describe).read_text()), json.loads(pathlib.Path(a.template_instance).read_text())
    d1, i1 = json.loads(pathlib.Path(a.describe).read_text()), json.loads(pathlib.Path(a.instance).read_text())
    if i0["task"] != i1["task"]:      # a new family's base from another family's template (same robot): every line the
        print(f"note: template family {i0['task']} -> new family {i1['task']} (the family name is a hunk like any other)")
    for k in TIMING:                       # no step-time hunk: the released sentence stays
        if k in d0:
            d1[k] = d0[k]
        else:
            d1.pop(k, None)
    lvl0 = G.tier_of(i0)
    lvl1 = a.difficulty or G.tier_of(i1)
    released = (root / "scripts/robotangle/packages" / f"{a.template}-privileged/task.toml").read_text()
    m = re.search(r"\(variation ([a-z0-9_]+), seed", released)
    r0 = render(d0, i0, f"topogym-{a.template}", lvl0)
    tvar = m.group(1) if m else i0["variation_id"]
    if m and m.group(1) != i0["variation_id"]:   # the template's released variation id (an alias of its MuJoCo scene's)
        v0, v = i0["variation_id"], m.group(1)
        r0 = {f: t.replace(f"variation {v0}", f"variation {v}").replace(f"variation `{v0}`", f"variation `{v}`")
              for f, t in r0.items()}
    r0 = {f: t.replace(f"topogym-{a.template}-", f"topogym-{a.new}-") for f, t in r0.items()}   # names: applied below
    r1 = render(d1, i1, f"topogym-{a.new}", lvl1)
    if a.public_instructions:              # the task class's own goal words (Task.public_instruction()) over public_goal's
        pi = json.loads(a.public_instructions.read_text())[i1["variation_id"]]["public_instruction"]
        summary, requirements = G.goal_words(i1["task"], "", d1)
        head, _, rest = pi.partition("\n\n")
        note = head[len(summary):].strip() if head.startswith(summary) else ""
        have = [x for x in requirements.splitlines() if x.startswith("- ")]
        theirs = [x for x in rest.splitlines() if x.startswith("- ")]
        extra = [x for x in theirs if x not in have]
        if extra and any(x not in theirs for x in have):
            # the class's default scene is not this instance's (instance-specific numbers): only the note is taken
            rep_note = f"public_instruction: its scene differs from the instance's; extra conditions not taken: {extra}"
            extra = []
        else:
            rep_note = f"public_instruction: note {note!r}; extra conditions {extra}"
        anchor = "- Keep all final conditions satisfied together"
        if extra and anchor not in requirements:
            raise SystemExit("no place for the extra success conditions (the confirmation line is missing)")
        new_req = requirements.replace(anchor, "\n".join(extra) + "\n" + anchor, 1) if extra else requirements
        r1 = {f: t.replace(requirements, new_req).replace(summary, f"{summary} {note}" if note else summary)
              for f, t in r1.items()}
    if a.task_note:                         # after the task sentence (and any note of the class), wherever it stands
        summary, _ = G.goal_words(i1["task"], "", d1)
        cur = next(x for x in (summary,) if x in r1["instruction.md"])
        tail = r1["instruction.md"].split(cur, 1)[1].split("\n", 1)[0]
        r1 = {f: t.replace(cur + tail, cur + tail + " " + a.task_note.strip()) for f, t in r1.items()}
    s0 = G.goal_words(i0["task"], "", d0)[0]
    s1 = re.search(r"\*\*Task:\*\* (.*)", r1["instruction.md"]).group(1)
    rep = [f"# {a.new}: texts from {a.template}", "", *([rep_note, ""] if a.public_instructions else []),
           f"template scene: {i0['variation_id']} (state {i0.get('state_sha', '')[:16]}); new scene: {i1['variation_id']} "
           f"(state {i1.get('state_sha', '')[:16]}); difficulty {lvl0!r} -> {lvl1!r}", ""]
    unresolved = 0
    for mode in ("privileged", "standard"):
        src = root / "scripts/robotangle/packages" / f"{a.template}-{mode}"
        dst = pathlib.Path(a.out) / f"{a.new}-{mode}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in FILES:
            before = (src / f).read_text()
            after = before.replace(f"topogym-{a.template}-", f"topogym-{a.new}-")   # the names (task, title, twin)
            rep.append(f"## {mode}/{f}")
            for old, new in hunks(r0[f], r1[f]):
                after, how = apply(after, old, new)
                if not how:
                    unresolved += 1
                    rep += ["- UNRESOLVED:", "```diff", *[f"- {x}" for x in old], *[f"+ {x}" for x in new], "```"]
                else:
                    rep.append(f"- {how}: {len(old)} -> {len(new)} line(s): {(old or new)[0][:100]!r}")
            after, facts = instance_facts(after, r1, mode)
            if i0["task"] != i1["task"]:           # a new family from another family's template: its name everywhere
                n = len(re.findall(rf"\b{i0['task']}\b", after))
                after = re.sub(rf"\b{i0['task']}\b", i1["task"], after)
                if n:
                    facts.append(f"family name {i0['task']} -> {i1['task']} ({n}x)")
            for v in {i0["variation_id"], tvar}:   # the variation named in a line no render hunk reached (README task line)
                if v != i1["variation_id"] and f"(variation {v}, seed" in after:
                    after = after.replace(f"(variation {v}, seed", f"(variation {i1['variation_id']}, seed")
                    facts.append(f"variation {v} -> {i1['variation_id']}")
            if s0 != s1 and s0 in after:          # the task sentence wherever the template's still stands (e.g. the
                # standard task.toml's description, which no render line has); where the new sentence (s0 + a note)
                # already stands, it is left alone, so a note is never doubled
                rest = s1[len(s0):] if s1.startswith(s0) else None
                rx = re.escape(s0) + (f"(?!{re.escape(rest)})" if rest else "")
                after, k = re.subn(rx, lambda m: s1, after)
                if k:
                    facts.append(f"task sentence {s0[:50]!r} -> {s1[:50]!r} ({k}x)")
            rep += [f"- instance fact: {x}" for x in facts]
            (dst / f).write_text(after)
            diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), f"{a.template}-{mode}/{f}",
                                             f"{a.new}-{mode}/{f}", lineterm="", n=1))
            rep += ["", "```diff", *diff, "```", ""]
    rep.insert(3, f"UNRESOLVED hunks: {unresolved} (fix by hand; lines robotangle-new's generator rewrites may stay)")
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text("\n".join(rep) + "\n")
    print(f"{a.new}: written to {a.out}; unresolved hunks {unresolved}" + (f"; report {a.report}" if a.report else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
