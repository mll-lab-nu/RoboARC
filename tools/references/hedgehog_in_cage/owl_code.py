# hedgehog_in_cage owl (Bird in Cage) reference: run-oracle code that steps the rows of the -i01 reference until the
# episode ends. The rows were built on this Session by the design's builder (runs/bird_in_cage_design_20261010/scripts/
# v7_build.py with v8_session.py; not in the repository): rod taken, two strokes turn the wings from 270 to 221 deg with
# the owl standing free, rod back; the tail held and the owl laid on its side on the cage floor; rod again, three strokes
# turn the wings to 264 deg, rod back; the tail held and the owl pulled out under the rung, carried into the goal, let go.
# Its instance is made in the native gripper space (the rows command the Robotiq drive):
#   python scripts/roboarc/shared/rcb_topogym.py make-instance --task hedgehog_in_cage \
#       --variation hedgehog_in_cage_owl --scene-seed 0 --gripper-space native --out <dir>/instance.json
#   python scripts/roboarc/shared/rcb_topogym.py run-oracle --task hedgehog_in_cage --instance <dir>/instance.json \
#       --code tools/references/hedgehog_in_cage/owl_code.py --out <dir>/privileged.npz
from pathlib import Path
import topogym

ref = np.load(Path(topogym.__file__).resolve().parents[2] / "scripts/roboarc/oracles/hedgehog-in-cage-07-i01/privileged.npz")
assert ref["action_layout"].tolist() == session.action_layout, (ref["action_layout"].tolist(), session.action_layout)
for row in ref["actions"]:
    if session.step(row):
        break
