# RoboTangle: the vendored Franka Emika Panda

Source: [MuJoCo Menagerie / franka_emika_panda](https://github.com/google-deepmind/mujoco_menagerie/tree/c5d2e83f49a2a1201153a686de0d26cc02b3bd98/franka_emika_panda)
at revision `c5d2e83f49a2a1201153a686de0d26cc02b3bd98` (2023-12-06), Apache-2.0 ([LICENSE](LICENSE)). The files were copied
from genesis-world 1.2.1 (`genesis/assets/xml/franka_emika_panda/`), which packages the same bytes: the upstream
`panda.xml`, `README.md` (the URDF -> MJCF derivation) and the 67 meshes under `assets/` (visual OBJ, collision STL /
OBJ). Genesis does not ship the LICENSE; it is the upstream file at the revision. [source.json](source.json) holds every
file's sha256. `.gitattributes` keeps the bytes (no line-ending conversion).

Byte identity, checked 2026-10-09: every file's sha256 equals the genesis package's in the image (69 of 69), and its git
blob id equals Menagerie's at the revision (69 of 69, and the LICENSE).

Use: frozen MuJoCo scenes name the Panda's meshes `@genesis-assets@/xml/franka_emika_panda/...`
(`topogym/world/mujoco/scene.py`). At load those paths resolve here (`robots/franka_emika_panda/` in the robot asset
root, TOPOGYM_ASSET_ROOT), and to the installed genesis package only where this copy is absent. The scene files, and so
the instance and state hashes, do not change: 14 frozen Panda scenes (fiber 4, hedgehog 5, buzz 5) compile to the same
model from either copy (only the stored file paths differ) and 500 steps from their initial states end in the same
qpos / qvel bytes. The scene builder and the Genesis families still read the Panda from the genesis package
(`topogym.assets.robot`).
