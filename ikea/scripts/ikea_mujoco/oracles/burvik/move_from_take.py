"""probe: upright_move from a p6_take snapshot (argv: snap dir); BURVIK_UP_* knobs as in upright.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build
import upright3, upright4
upright_move = upright4.upright_move if os.environ.get("BURVIK_UP", "4") == "4" else upright3.upright_move
B = build.Builder(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
B.load("p6_take")
upright_move(B)
fc = B.o.s.final_check()
B.note(phase="end", success=fc["success"], report=fc["end"]["report"].get("failures"), terminal=fc["end"]["terminal"], settled=fc["settled"]["success"])
B.snap("end", len(B.base)); B.o.write(os.path.join(sys.argv[4], "ref.npz"))
