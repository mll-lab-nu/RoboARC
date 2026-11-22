"""write a reference npz from a builder snapshot (its executed rows): snap2ref.py <code> <scenes> <snap.pkl> <out.npz>"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orc import Orc, setup
code, scenes, snap, out = sys.argv[1:5]
o = Orc(setup(code, scenes)); o.rows = list(pickle.load(open(snap, "rb"))["o"]["rows"]); o.write(out)
print(out, len(o.rows))
