# Third-party material in this repository

RoboARC's own code and documents are under the MIT licence in [`../LICENSE`](../LICENSE). The material below comes
from others and keeps its own licence; the licence texts that must travel with it are in this directory and next to
the material itself. Items whose use rests on a permission the RoboARC owners confirmed, rather than on a licence file, say so; the MIT
licence does not extend to them.

The task images also contain third-party packages installed at build time (MuJoCo, gs-nyx, Newton, NVIDIA Warp,
PyTorch, Debian packages and their dependencies, pinned by the recipes under `source/images/` and `ikea/images/`; until
1.0.x the puzzle image also held Genesis and libuipc). They are not part of this repository and keep their own
licences, shipped inside the packages.

## Robot models

The robot descriptions under `source/assets/robots/` are MuJoCo Menagerie's
(https://github.com/google-deepmind/mujoco_menagerie), copied unchanged. Each directory keeps the upstream `LICENSE`
and README and records every file's sha256 (`source.json` or `source_hashes.json`). The IKEA scenes use the same
Panda and ALOHA 2 models from their asset dataset (`ikea/scripts/ikea_mujoco/assets.json`, the Panda's `LICENSE` with
the same sha256).

| model | path | upstream revision | licence | copyright | text |
|---|---|---|---|---|---|
| Franka Emika Panda | `source/assets/robots/franka_emika_panda/` | `c5d2e83f49a2a1201153a686de0d26cc02b3bd98` (bytes as packaged by genesis-world 1.2.1) | Apache-2.0 | none named in the upstream LICENSE | [Apache-2.0.txt](Apache-2.0.txt) |
| ALOHA 2 (ViperX 300 s arms, D405 mount) | `source/assets/robots/aloha/` | `8161bba264d7fa7c99ca301e91e7fb44737676ad` | BSD-3-Clause | Copyright (c) 2023, Trossen Robotics | [BSD-3-Clause-Trossen-Robotics.txt](BSD-3-Clause-Trossen-Robotics.txt) |
| Robotiq 2F-85 | `source/assets/robots/robotiq_2f85/` | Menagerie `main` as of 2026-09-14 (no revision recorded; bytes in `source_hashes.json`) | BSD-2-Clause | Copyright (c) 2013, ROS-Industrial | [BSD-2-Clause-ROS-Industrial.txt](BSD-2-Clause-ROS-Industrial.txt) |

## Code

| what | where | origin | licence |
|---|---|---|---|
| The shared task layer (task conventions, egress tool, `eai-standard` service layer, resource monitor, CI checks, Harbor runner) | `scripts/`, `ci_checks/`, copies in every task directory | robot_coding_bench | MIT, Copyright (c) 2026 Kangrui Wang and robot_coding_bench contributors: [robot_coding_bench-MIT.txt](robot_coding_bench-MIT.txt) |
| VBD cloth kernels adapted from Newton (`newton/_src/solvers/vbd/particle_vbd_kernels.py`, newton 1.6.1) and a kernel that evaluates Newton's contact and elasticity functions | `source/topogym/world/mujoco/vbd_fast.py`, `vbd_gpu.py`, `cloth_coupling.py` (each carries an SPDX header) | Newton, https://github.com/newton-physics/newton | Apache-2.0, Copyright (c) 2025 The Newton Developers: [Apache-2.0.txt](Apache-2.0.txt) |
| Two camera kernels of Genesis (`_tc_R_to_quat`, `_tc_z_up_to_R` from `genesis/utils/geom.py`), copied unchanged | `scripts/roboarc/shared/nyx_direct.py`, `ikea/scripts/ikea_mujoco/standard/nyx_direct.py` (and their copies in the task directories) | Genesis, https://github.com/Genesis-Embodied-AI/Genesis | Apache-2.0: [Apache-2.0.txt](Apache-2.0.txt) |
| MuJoCo EPA fix: a byte patch applied to the PyPI `mujoco==3.14.0` wheel's `libmujoco.so.3.14.0` at image build, backporting upstream commit `6d662aedda693a1225a03a26c6b3ddd5e3cc4fbe` (MuJoCo 3.15+) | `source/images/roboarc/mujoco-3.14.0-epa/bytepatch_libmujoco.py` | MuJoCo, https://github.com/google-deepmind/mujoco | MuJoCo is Apache-2.0 ([Apache-2.0.txt](Apache-2.0.txt)). The patch script is RoboARC's; the library it modifies is not in this repository. Images built with it ship a modified `libmujoco.so.3.14.0`: this entry and the script's docstring state the change (14 bytes at 5 sites, input and output sha256 asserted) |

## Scenes, textures and lighting

| what | where | origin | licence |
|---|---|---|---|
| IKEA furniture-assembly scenes: scene models, mechanisms, success checks and the robot demos the references start from | `ikea/images/ikea-mujoco/scenes/`, `ikea/scripts/ikea_mujoco/`; meshes in a pinned asset dataset (`ikea/scripts/ikea_mujoco/assets.json`) | authored by Xiaoyu and Zhen Wu's team, co-authors of RoboARC; parts ported from their assembly code base (`ikea/scripts/ikea_mujoco/provenance.json`) | contributed under this repository's MIT licence ([LICENSE](../LICENSE)), scene code and meshes alike |
| Studio lighting HDRI "Studio Small 09" by Sergej Majboroda | not in this repository: the IKEA sim image downloads it at build (`ikea/images/ikea-mujoco/sim/Dockerfile`, sha256 asserted, with a CREDITS.txt); RoboARC's renderer reads it from its base image (`source/topogym/config/profiles/presentation_configs/compact_lab.yaml`, `env_map`) | Poly Haven, https://polyhaven.com/a/studio_small_09 | CC0 |
| Table-top, table-frame and floor textures of the appearance seeds: Poly Haven `ash_veneer`, `cherry_veneer`, `coated_pine`, `terrazzo_tiles`, `brushed_concrete`, `diagonal_parquet`, `interior_tiles`, `terracotta_floor_tiles`, `poly_wool_herringbone` (1k colour, OpenGL normal and ARM maps, JPG, unchanged) | not in this repository: the puzzle image downloads them at build into `/opt/topogym-visuals/` (`source/images/roboarc/prepare_nyx_studio.py`, sha256 asserted, authors in its CREDITS.txt); `source/topogym/world/look.py` names them | Poly Haven, https://polyhaven.com/a/ followed by the name | CC0 |
| Ground plane checker texture (`genesis/assets/textures/checker.png` of genesis-world 1.2.1, unchanged) | `source/assets/textures/genesis/textures/checker.png` (installed in the puzzle image at `/opt/topogym-visuals/genesis/`) | Genesis, https://github.com/Genesis-Embodied-AI/Genesis | Apache-2.0: [Apache-2.0.txt](Apache-2.0.txt) |
| Dark wood texture (hedgehog-in-cage) | `source/topogym/assets/resources/textures/hedgehog_in_cage_materials_dark_wood.png` | Poly Haven `wood_table_001`, https://polyhaven.com/a/wood_table_001 (recorded in `source/topogym/assets/resources/templates/hedgehog_in_cage_materials_sources.json`) | CC0 |
| Light wood texture (hedgehog-in-cage) | `source/topogym/assets/resources/textures/hedgehog_in_cage_materials_light_wood.png` | LIBERO's `table_light_wood.png` (same record) | permission confirmed by the RoboARC owners (2026-10-20) |
| Interlocked claws reference meshes | `source/topogym/assets/resources/meshes/interlocked_claws_piece-{0,1}.obj`, `templates/interlocked_claws_reference_scene.json` | an upstream claw-pair geometry (`asset_id` `cspace_claw_wang_v1`, retrieved 2026-09-17) | permission confirmed by the RoboARC owners (2026-10-20); the asset record notes that these meshes do not fall under the code licence |

## Names and designs of commercial products

| what | where | note |
|---|---|---|
| Hanayama's Huzzle Cast puzzles: Cast Keyhole, Cast Coil, Cast ABC, Cast Rotor, Cast Cage | the cast-keyhole and cast-coil families, interlocked-claws bases 11 and 12, the hedgehog-in-cage Cast Cage base; sources in `source/topogym/assets/resources/templates/` | geometry reconstructed from the manufacturer's public product pages and photos, not from CAD. Use of the names and designs: permission confirmed by the RoboARC owners (2026-10-20). |
| IKEA products (LACK, SVALLET, GLADOM, VARIERA, EKET, BURVIK, VESKEN, KYRRE, STENERIK, STEGESUND, UPPSTA, IKEA 365+) | `ikea/` | modelled after the products and their public assembly instructions. Use of the names and designs: permission confirmed by the RoboARC owners (2026-10-20). |

## Runtime components without a licence file

| what | note |
|---|---|
| gs-nyx 0.1.3 (the Nyx renderer, installed in the images; gs-nyx-plugin 0.1.4 too until 1.0.x) | the wheels carry no licence file or metadata; their documentation repository (Genesis-Embodied-AI/nyx-for-genesis) is Apache-2.0. No plugin code is copied into this repository (`scripts/roboarc/shared/nyx_direct.py` calls its public API). Use in the images: permission confirmed by the RoboARC owners (2026-10-20) |
