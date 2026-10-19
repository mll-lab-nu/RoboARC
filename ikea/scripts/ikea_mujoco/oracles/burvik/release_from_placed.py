"""probe: release_home from a p6_placed snapshot (argv: code scenes base dir); BURVIK_UP4_* knobs as in upright4.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build
from upright4 import release_home
from plan_up import PARTS
B = build.Builder(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
B.load("p6_placed")
release_home(B, {B.o.m.body(n).id for n in PARTS})
fc = B.o.s.final_check()
B.note(phase="end", success=fc["success"], report=fc["end"]["report"].get("failures"), terminal=fc["end"]["terminal"], settled=fc["settled"]["success"])
B.snap("end", len(B.base)); B.o.write(os.path.join(sys.argv[4], "ref.npz"))
