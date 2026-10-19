"""Run a fiber tool (any script that builds a Session from an instance) with the redesign tiers' and labs' variations
registered in-process first, so that a frozen redesign instance resolves to its task class.

usage: run_registered.py <script.py> [script arguments ...]
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import CLASSES, register  # noqa: E402

for name in CLASSES:
    register(name)
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
