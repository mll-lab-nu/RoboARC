#!/usr/bin/env python3
"""Fix 3b, the dowel rule: a soft dowel shell (a geom named *_dowel_* with its own soft solref, priority 1) keeps its
soft contact only with the panels that carry its holes; against everything else a stiff twin of the shell answers,
with the scene's default contact. Collision bits only; no geom is moved, no other contact changes.

  - root <default> geom: conaffinity 1 -> 5 (bit 4 added). Between ordinary geoms (contype 1) nothing changes.
  - hole panels (the bodies whose geoms the dowels touch in the reference; eket: side_a, side_b): their collision
    geoms get conaffinity 3 (bit 2 for the soft shells, no bit 4).
  - each soft shell: contype 2 conaffinity 2 -> it meets only geoms with bit 2: the hole panels.
  - each soft shell gets a stiff twin right after it: the same cylinder, contype 4 conaffinity 4, mass 0 (the bodies
    have explicit inertials), the part class's default solref / solimp, no priority -> it meets every geom with bit 4
    (everything but the hole panels and the soft shells).

usage: dowel_rule.py <scene.xml in> <scene.xml out> --hole-bodies side_a side_b
"""
import argparse, re, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--hole-bodies", nargs="+", required=True)
    a = ap.parse_args()
    lines = open(a.src).read().split("\n")
    out, body, n_root, n_hole, n_shell = [], [], 0, 0, 0
    in_default = 0
    for ln in lines:
        s = ln.strip()
        if s.startswith("<default>") and not body:
            in_default += 1
        m = re.match(r'\s*<body name="([^"]+)"', ln)
        if m:
            body.append(m.group(1))
        if in_default == 1 and n_root == 0 and s.startswith("<geom ") and "solref" in s and "class=" not in s:
            assert "conaffinity" not in s
            ln = ln.replace("<geom ", '<geom conaffinity="5" ', 1); n_root += 1; in_default = 2
        elif s.startswith("<geom ") and body and body[-1] in a.hole_bodies and 'class="part"' in s:
            assert "contype" not in s and "conaffinity" not in s, ln
            ln = ln.replace("<geom ", '<geom conaffinity="3" ', 1); n_hole += 1
        elif s.startswith("<geom ") and re.search(r'name="[^"]*_dowel_\d+"', s):
            assert "contype" not in s and "conaffinity" not in s and "priority=" in s, ln
            soft = ln.replace("<geom ", '<geom contype="2" conaffinity="2" ', 1)
            stiff = ln
            stiff = re.sub(r'name="([^"]+)"', r'name="\1_stiff"', stiff)
            for attr in ("priority", "solref", "solimp"):
                stiff = re.sub(rf' {attr}="[^"]*"', "", stiff)
            stiff = stiff.replace("<geom ", '<geom contype="4" conaffinity="4" mass="0" ', 1)
            out.append(soft); out.append(stiff); n_shell += 1
            continue
        if s.startswith("</body>") and body:
            body.pop()
        out.append(ln)
    assert n_root == 1 and n_hole > 0 and n_shell > 0, (n_root, n_hole, n_shell)
    open(a.dst, "w").write("\n".join(out))
    print(f"root default +bit4: {n_root}; hole-panel geoms: {n_hole}; dowel shells split: {n_shell}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
