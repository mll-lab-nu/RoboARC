# source/: the release image's build context

`images/roboarc/build.sh` builds `images/roboarc/Dockerfile` with this directory as the Docker build context;
`.dockerignore` keeps the context to what the recipe copies.

| path | in the image |
|---|---|
| `topogym/` | `/opt/topogym/topogym/`: the simulator (see `topogym/README.md`), pruned at build by `images/topogym/prune.py`, which writes `/opt/topogym/SOURCE.json` |
| `assets/robots/` | `/opt/topogym-assets/robots/`: ALOHA, Panda and Robotiq 2F-85 (MuJoCo Menagerie, with their licences and `source.json`) |
| `assets/textures/genesis/` | `/opt/topogym-visuals/genesis/`: the ground plane's checker texture (genesis-world's file, Apache-2.0; its README) |
| `images/roboarc/Dockerfile` | the recipe (images/roboarc/README.md: what it installs and how each input is pinned) |
| `images/roboarc/runtime.lock`, `vbd-requirements.txt` | the Python packages, by version and wheel sha256 |
| `images/roboarc/prepare_nyx_studio.py` | build step: Nyx's studio HDRI, wood maps and the appearance seeds' texture maps (sha256-checked downloads, CREDITS.txt) |
| `images/roboarc/warp_precompile.py` | build step: the VBD cloth's precompiled Warp kernels ([docs/simulator/VBD_CLOTH.md](../docs/simulator/VBD_CLOTH.md)) |
| `images/roboarc/mujoco-3.14.0-epa/bytepatch_libmujoco.py` | build step: the EPA fix of `libmujoco.so` ([docs/simulator/MUJOCO_EPA_PATCH.md](../docs/simulator/MUJOCO_EPA_PATCH.md)) |
| `images/roboarc/stage_instance.py` | `/opt/robotangle/stage_instance.py`: stages a task's frozen instance and checks it against its binding |
| `images/topogym/prune.py` | build step: drops what could hold a solution, hashes the rest |

Not in the image: every `*.md` and the authoring-only parts prune.py removes. `.dockerignore` and prune.py also name
`topogym/planning/`, the planners that recorded the references: they are not part of the public release.

**Do not change what the image takes without a new image version.** SOURCE.json hashes the simulator files that remain
after pruning (426 in 1.1.0); every task's `runtime_binding.json` names that hash, and each Session refuses another
runtime. `python3 tools/release/source_hash.py --check` recomputes it from the checkout (no Docker) and compares it with
`scripts/roboarc/release.json`.
