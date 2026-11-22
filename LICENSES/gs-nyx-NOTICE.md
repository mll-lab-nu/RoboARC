# gs-nyx (the Nyx renderer) in RoboARC's images

- **What.** `gs-nyx` 0.1.3, the Nyx renderer by Genesis AI, as published at https://pypi.org/project/gs-nyx/0.1.3/.
  The images `roboarc` and `roboarc-ikea-sim` install that wheel unchanged, by its sha256
  (`source/images/roboarc/runtime.lock`, `ikea/images/ikea-mujoco/sim/nyx-requirements.txt`). This repository holds
  none of its code: `nyx_direct.py` calls its public interface.
- **Licence.** The package is published without a licence text: none in the wheel, none in its metadata. Its
  documentation repository, https://github.com/Genesis-Embodied-AI/nyx-for-genesis, is under Apache-2.0, which does
  not name the package.
- **Basis.** gs-nyx is redistributed inside RoboARC's images with the permission of its authors.
- **Terms.** gs-nyx stays its authors', and its terms are theirs. RoboARC's MIT licence does not extend to it.
