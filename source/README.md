# source/: the release image's build context

`images/roboarc/build.sh` builds `images/robotangle/Dockerfile.mujoco` with this directory as the Docker build context;
`.dockerignore` keeps the context to what the recipe copies.

| path | in the image |
|---|---|
| `topogym/` | `/opt/topogym/topogym/`: the simulator (see `topogym/README.md`), pruned at build by `images/topogym/prune.py`, which writes `/opt/topogym/SOURCE.json` |
| `assets/robots/` | `/opt/topogym-assets/robots/`: ALOHA, Panda and Robotiq 2F-85 (MuJoCo Menagerie, with their licences and `source.json`) |
| `images/robotangle/Dockerfile.mujoco` | the recipe |
| `images/robotangle/vbd-requirements.txt`, `warp_precompile.py` | build steps: the VBD cloth wheels and their precompiled Warp kernels ([docs/simulator/VBD_CLOTH.md](../docs/simulator/VBD_CLOTH.md)) |
| `images/robotangle/mujoco-3.14.0-epa/bytepatch_libmujoco.py` | build step: the EPA fix of `libmujoco.so` ([docs/simulator/MUJOCO_EPA_PATCH.md](../docs/simulator/MUJOCO_EPA_PATCH.md)) |
| `images/topogym/prune.py` | build step: drops what could hold a solution, hashes the rest |

Not in the image: `topogym/planning/` (the oracle sources the references were recorded with), every `*.md`, and the
authoring-only parts prune.py removes.

**Do not change what the image takes without a new image version.** SOURCE.json hashes the 443 simulator files that
remain after pruning; every task's `runtime_binding.json` names that hash, and each Session refuses another runtime.
`python3 tools/release/source_hash.py --check` recomputes it from the checkout (no Docker) and compares it with
`scripts/roboarc/release.json`. Keep the bytes and paths of the other build inputs too: the build then reuses the
released layers, and the image is the released one (a cache-cold build gives the same files except the source-mtime
field of the `.pyc` files). Comments in the recipe may change.
