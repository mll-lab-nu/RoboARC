#!/bin/bash
# kyrre reference (the scene with native cylinders, the one exception to the prism rule): the demo -> Franka Hand slew with grip waits -> screw 3-1's
# second grasp re-aimed at the screw's actual pose -> the hex key set down 12 mm along its rest (off the knife-edge pose).
# Run in the IKEA image (0.3.2+: kyrre's scene.py has NATIVE_CYLINDERS = True) with scripts/ikea_mujoco/shared on
# PYTHONPATH; RCB = that shared dir. DEMO = the author's demo rows (kyrre_vlim, 659,752 rows, sha256 465163457683...: the
# former WIP tasks' scripts/ikea_mujoco/wip/tasks/ikea-kyrre-i00-privileged/solution/trajectory.npz, git dd8d36fef6).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); DEMO=${DEMO:?}; RCB=${RCB:?}; OUT=${OUT:-$H/out}; mkdir -p "$OUT"
python "$RCB/../standard/std_reference.py" make --task kyrre --priv "$DEMO" --out "$OUT/A.npz" --method clamp --despike 0 \
  --grip wait --rcb-ikea "$RCB"                                                         # 733,558 rows (73,806 grip waits)
PG_MATH=v1 python "$H/patch_grasp.py" "$OUT/A.npz" "$OUT/P1.npz" right \
  '299.24,198.23,106.08,-0.4541,0.392,-0.5053,-0.6203' '299.89,192.31,106.08,-0.4848,0.3661,-0.5244,-0.5967' \
  501000 502800 504000 506500                                                           # screw 3-1: demo pose -> actual
python "$H/patch_grasp.py" "$OUT/P1.npz" "$OUT/kyrre_cyl.npz" right \
  '-230.0,-63.2,152.2,-0.500015,0.499985,-0.499985,-0.500015' '-230.0,-75.2,155.2,-0.500015,0.499985,-0.499985,-0.500015' \
  729800 730800 732700 733400                                                           # the key's set-down
sha256sum "$OUT/kyrre_cyl.npz"   # = kyrre_ref.npz, 0808ce7829703cda7ae2077f7fd160d0b2334e7d53fdaa8857a40eb60ebf5ec7 (host C, Zen 4)
