# Third-party material in RoboARC: the images and the repository

RoboARC's own code and documents are under the MIT licence in [`../LICENSE`](../LICENSE). Everything on this page comes
from others and keeps its own licence. The page names every third-party component, where it is, its licence, and the
file or link that holds the licence text: first what the images contain, then what is in the repository's own files.
An item that is used by permission and not under a licence says so. The MIT licence does not extend to such items.

## What you accept when you pull or build the images

RoboARC's images (`images/pins.json`) contain everything this page lists. Pulling an image copies all of it to your
machine. Building one downloads it from its publishers. So `bash images/pull_all.sh`, the image build scripts and
`roboarc images` ask for your acceptance before they pull or build: the option `--accept-licences`, or
`ROBOARC_ACCEPT_LICENCES=1` in the environment. Giving it states that:

1. You have read this page.
2. You accept, for your use of the images, the licence of every component in the table [In the images](#in-the-images),
   as its publisher states it. Some of them are not permissive open-source licences. Read these first:
   - **gs-nyx**, the Nyx renderer, has no published licence. It is in the images with its authors' permission and on
     their terms ([gs-nyx-NOTICE.md](gs-nyx-NOTICE.md)).
   - **NVIDIA's CUDA libraries** are proprietary software under NVIDIA's licence agreements.
   - **TetGen** is under the GNU Affero General Public License, version 3 or later.
   - **ffmpeg** is under the GNU General Public License, and **GEOS** under the GNU Lesser General Public License,
     version 2.1.
3. RoboARC's MIT licence covers RoboARC's own code and documents only. It gives no right in these components, nor in
   the items of this page that are used by permission.

The acceptance is an option you give to a command on your machine. Nothing is sent anywhere.

## In the images

The four images are `roboarc` (the puzzles), `roboarc-ikea` and `roboarc-ikea-sim` (IKEA) and `roboarc-monitor` (a
Python base image and nothing else). Their recipes pin every input: `source/images/roboarc/runtime.lock` and
`vbd-requirements.txt`, `ikea/images/ikea-mujoco/requirements.lock` and `sim/nyx-requirements.txt` give each Python
package's version and sha256. The licences below are those of each package's metadata. The paths were read in the
released images: `<sp>` is `/usr/local/lib/python3.11/site-packages` in `roboarc`,
`/usr/local/lib/python3.12/site-packages` in the IKEA images, and `/opt/nyx/lib/python3.12/site-packages` for the Nyx
environment of `roboarc-ikea-sim`. Every image also has the texts of the GPL, the LGPL and Apache-2.0 in
`/usr/share/common-licenses/`.

| component | image | licence | text |
|---|---|---|---|
| gs-nyx 0.1.3, the Nyx renderer (Genesis AI) | `roboarc`, `roboarc-ikea-sim` | none published: redistributed with its authors' permission, on their terms | [gs-nyx-NOTICE.md](gs-nyx-NOTICE.md). The wheel has no licence file |
| NVIDIA's CUDA 12.8 libraries, which PyTorch loads: the wheels `nvidia-cublas-cu12`, `-cuda-cupti-`, `-cuda-nvrtc-`, `-cuda-runtime-`, `-cufft-`, `-cufile-`, `-curand-`, `-cusolver-`, `-cusparse-`, `-nvjitlink-`, `-nvshmem-`, and `nvidia-cudnn-cu12`, `nvidia-cusparselt-cu12` | `roboarc`, `roboarc-ikea-sim` | proprietary. NVIDIA's End User License Agreement for the CUDA libraries, and its License Agreement for NVIDIA Software Development Kits for cuDNN and cuSPARSELt | in the image: `<sp>/nvidia_<library>_cu12-<version>.dist-info/License.txt` (cuDNN and NVSHMEM: `licenses/License.txt`; cuSPARSELt: `<sp>/nvidia/cusparselt/LICENSE.txt`). Online: https://docs.nvidia.com/cuda/eula/index.html and https://docs.nvidia.com/deeplearning/cudnn/latest/reference/eula.html. The text inside the wheel is the one that applies |
| `nvidia-nccl-cu12`, `nvidia-nvtx-cu12` | `roboarc`, `roboarc-ikea-sim` | BSD-3-Clause (NCCL), Apache-2.0 (NVTX) | the `License.txt` of each wheel, as above |
| TetGen, in the Python package `tetgen` 0.8.2, unchanged. The simulator calls it to mesh rope volumes | `roboarc` | AGPL-3.0-or-later (TetGen's author also sells it under a commercial licence) | in the image: `<sp>/tetgen-0.8.2.dist-info/licenses/LICENSE`. Online: https://www.gnu.org/licenses/agpl-3.0.html. Source: https://github.com/pyvista/tetgen/tree/v0.8.2 and the source archive at https://pypi.org/project/tetgen/0.8.2/#files; TetGen itself: https://wias-berlin.de/software/tetgen/ |
| GEOS 3.13.1, bundled in the `shapely` 2.1.2 wheel (shapely itself: BSD-3-Clause) | `roboarc` | LGPL-2.1 | in the image: `<sp>/shapely-2.1.2.dist-info/licenses/LICENSE_GEOS`. Online: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html. Source: https://github.com/libgeos/geos/tree/3.13.1 |
| ffmpeg 5.1.9 with libx264 and libx265: the Debian 12 package `ffmpeg` 7:5.1.9-0+deb12u1 and its libraries | `roboarc` | GPL, version 2 or later (built with `--enable-gpl`); parts under the LGPL and other licences | in the image: `/usr/share/doc/ffmpeg/copyright`, `/usr/share/doc/libx264-164/copyright`, `/usr/share/doc/libx265-199/copyright`. Online: https://www.gnu.org/licenses/old-licenses/gpl-2.0.html. Source: https://snapshot.debian.org/package/ffmpeg/7%3A5.1.9-0%2Bdeb12u1/ |
| The ffmpeg 7.0.2 executable bundled in the `imageio-ffmpeg` 0.6.0 wheel (the wrapper itself: BSD-2-Clause) | `roboarc-ikea`, `roboarc-ikea-sim` | GPL, version 3 or later: a static build with `--enable-gpl --enable-version3`, libx264 and libx265 (https://ffmpeg.org/legal.html) | in the image: `/usr/share/common-licenses/GPL-3`; the executable, `<sp>/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`, prints its terms with `-L`. Online: https://www.gnu.org/licenses/gpl-3.0.html. Source: https://ffmpeg.org/releases/ffmpeg-7.0.2.tar.xz; the wheel: https://github.com/imageio/imageio-ffmpeg |
| GCC's runtime libraries inside the `numpy`, `scipy` and `torch` wheels (libgfortran, libquadmath, libgomp), with OpenBLAS | `roboarc`, `roboarc-ikea`, `roboarc-ikea-sim` | libgfortran and libgomp: GPL-3.0-or-later with the GCC Runtime Library Exception; libquadmath: LGPL-2.1-or-later; OpenBLAS: BSD-3-Clause | in the image: the `LICENSE.txt` of each wheel's `.dist-info` |
| MuJoCo 3.14.0, with RoboARC's EPA fix patched into `libmujoco.so.3.14.0` ([Code](#code)) | `roboarc`, `roboarc-ikea`, `roboarc-ikea-sim` | Apache-2.0 | [Apache-2.0.txt](Apache-2.0.txt) |
| PyTorch 2.11.0 (`torch`, the cu128 build) | `roboarc`, `roboarc-ikea-sim` | BSD-3-Clause | in the image: `<sp>/torch-2.11.0+cu128.dist-info/licenses/` (`LICENSE`, `NOTICE`) |
| Newton 1.6.1, NVIDIA Warp 1.18.0 (`warp-lang`), `cuda-bindings`, Cython, `manifold3d` | `roboarc` (`cuda-bindings` also `roboarc-ikea-sim`) | Apache-2.0 | [Apache-2.0.txt](Apache-2.0.txt), and each package's `.dist-info` |
| `numpy`, `scipy`, `networkx`, `dill`, `pycparser`, `omegaconf`, `PyOpenGL`, `python-fcl`, `antlr4-python3-runtime`, `shapely`, `imageio`, `imageio-ffmpeg` | as the lock files say | BSD, 2- or 3-clause | each package's `.dist-info` in the image, and its page https://pypi.org/project/<name>/ |
| `coacd`, `glfw`, `hydra-core`, `PyYAML`, `rtree`, `setuptools`, `trimesh`, `cffi` (MIT-0) | as the lock files say | MIT | as above |
| `pillow` (MIT-CMU), `packaging` (Apache-2.0 or BSD-2-Clause), `mapbox-earcut` (ISC), `tqdm` (MPL-2.0 and MIT), `typing-extensions` (PSF-2.0) | as the lock files say | as named | as above |
| The Python base images (Python 3.11 on Debian 12 in `roboarc`, Python 3.12 on Debian 13 in the IKEA images, Python 3.12 on Debian 12 in `roboarc-monitor`) and the Debian packages the recipes add: ffmpeg (above), the GL, EGL, OpenGL and Vulkan loaders, glib, X11 client libraries, libgomp, liblzma, curl, ca-certificates and ripgrep in `roboarc`; OSMesa and the same loaders in the IKEA images | all four | Python: PSF-2.0. Each Debian package has its own licence, the GPL and the LGPL among them | in the image: `/usr/share/doc/<package>/copyright` (the recipes keep these files) and `/usr/share/common-licenses/` |
| Poly Haven's studio lighting and textures ([Scenes, textures and lighting](#scenes-textures-and-lighting)) | `roboarc`, `roboarc-ikea-sim` | CC0 | none needed. Credits in the image: `/opt/topogym-visuals/CREDITS.txt`, `/opt/nyx-assets/CREDITS.txt` |
| What comes from this repository: the robot models, the checker texture, the simulator source with the adapted Newton kernels, the frozen instances, the IKEA scenes and meshes | `roboarc`, `roboarc-ikea`, `roboarc-ikea-sim` | the sections below | the sections below |

Until 1.0.x the puzzle image also held Genesis, libuipc and gs-nyx-plugin 0.1.4. No image of `images/pins.json` does.

The sections below list the third-party material in the repository's own files.

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
| The same fix as a diff of MuJoCo's source, kept as the record of a source build that the images do not use | `records/mujoco-3.14.0-epa-source-build/epa_horizon.patch` | MuJoCo, https://github.com/google-deepmind/mujoco | Apache-2.0: [Apache-2.0.txt](Apache-2.0.txt) |

## Scenes, textures and lighting

| what | where | origin | licence |
|---|---|---|---|
| IKEA furniture-assembly scenes: scene models, mechanisms, success checks and the robot demos the references start from | `ikea/images/ikea-mujoco/scenes/`, `ikea/scripts/ikea_mujoco/`; meshes in a pinned asset dataset (`ikea/scripts/ikea_mujoco/assets.json`) | authored by Xiaoyu and Zhen Wu's team, co-authors of RoboARC; parts ported from their assembly code base (`ikea/scripts/ikea_mujoco/provenance.json`) | contributed under this repository's MIT licence ([LICENSE](../LICENSE)), scene code and meshes alike |
| Studio lighting HDRI "Studio Small 09" by Sergej Majboroda | not in this repository: the IKEA sim image downloads it at build (`ikea/images/ikea-mujoco/sim/Dockerfile`, sha256 asserted, with a CREDITS.txt); RoboARC's renderer reads it from its base image (`source/topogym/config/profiles/presentation_configs/compact_lab.yaml`, `env_map`) | Poly Haven, https://polyhaven.com/a/studio_small_09 | CC0 |
| Table-top, table-frame and floor textures of the appearance seeds: Poly Haven `ash_veneer`, `cherry_veneer`, `coated_pine`, `terrazzo_tiles`, `brushed_concrete`, `diagonal_parquet`, `interior_tiles`, `terracotta_floor_tiles`, `poly_wool_herringbone` (1k colour, OpenGL normal and ARM maps, JPG, unchanged) | not in this repository: the puzzle image downloads them at build into `/opt/topogym-visuals/` (`source/images/roboarc/prepare_nyx_studio.py`, sha256 asserted, authors in its CREDITS.txt); `source/topogym/world/look.py` names them | Poly Haven, https://polyhaven.com/a/ followed by the name | CC0 |
| Ground plane checker texture (`genesis/assets/textures/checker.png` of genesis-world 1.2.1, unchanged) | `source/assets/textures/genesis/textures/checker.png` (installed in the puzzle image at `/opt/topogym-visuals/genesis/`) | Genesis, https://github.com/Genesis-Embodied-AI/Genesis | Apache-2.0: [Apache-2.0.txt](Apache-2.0.txt) |
| Dark wood texture (hedgehog-in-cage) | `source/topogym/assets/resources/textures/hedgehog_in_cage_materials_dark_wood.png` | Poly Haven `wood_table_001`, https://polyhaven.com/a/wood_table_001 (recorded in `source/topogym/assets/resources/templates/hedgehog_in_cage_materials_sources.json`) | CC0 |
| Light wood texture (hedgehog-in-cage) | `source/topogym/assets/resources/textures/hedgehog_in_cage_materials_light_wood.png` | LIBERO's `table_light_wood.png`, https://github.com/Lifelong-Robot-Learning/LIBERO (same record) | MIT, Copyright (c) 2023 Lifelong Robot Learning: [LIBERO-MIT.txt](LIBERO-MIT.txt) |
| Interlocked claws reference meshes | `source/topogym/assets/resources/meshes/interlocked_claws_piece-{0,1}.obj`, `templates/interlocked_claws_reference_scene.json` | designed by one of RoboARC's authors, who published the design before this project (`asset_id` `cspace_claw_wang_v1`) | released with RoboARC under this repository's licence ([LICENSE](../LICENSE)). The asset record in the released image still calls them an upstream geometry without a licence: that text is corrected with the next image |

## Names and designs of commercial products

| what | where | note |
|---|---|---|
| Hanayama's Huzzle Cast puzzles: Cast Keyhole, Cast Coil, Cast ABC, Cast Rotor, Cast Cage | the cast-keyhole and cast-coil families, interlocked-claws bases 11 and 12, the hedgehog-in-cage Cast Cage base; sources in `source/topogym/assets/resources/templates/` | geometry reconstructed from the manufacturer's public product pages and photos, not from CAD. Use of the names and designs: permission confirmed by the RoboARC owners (2026-10-20). |
| IKEA products (LACK, SVALLET, GLADOM, VARIERA, EKET, BURVIK, VESKEN, KYRRE, STENERIK, STEGESUND, UPPSTA, IKEA 365+) | `ikea/` | modelled after the products and their public assembly instructions. Use of the names and designs: permission confirmed by the RoboARC owners (2026-10-20). |
