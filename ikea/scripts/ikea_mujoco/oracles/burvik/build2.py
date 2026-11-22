"""build.py with two phases replaced by closed-loop ones: P1 seats the top in the jig (seat.py) and P6 turns the table
upright with the demo's rigid two-hand motion on our grasps (upright4.py; BURVIK_UP=3: upright3.py). Same arguments as build.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build
from seat import seat_p1
import upright3, upright4
upright_p6 = upright4.upright_p6 if os.environ.get("BURVIK_UP", "4") == "4" else upright3.upright_p6


def p1(self):
    seat_p1(self); self.snap("p1", 70000)


_follow = build.Builder.follow


def follow(self, k0, k1, hold=True):
    if k1 == len(self.base) and not hold:          # build.main's p6: the base rows to the end
        upright_p6(self)
        return k1
    return _follow(self, k0, k1, hold)


build.Builder.p1 = p1
build.Builder.follow = follow

if __name__ == "__main__":
    build.main()
