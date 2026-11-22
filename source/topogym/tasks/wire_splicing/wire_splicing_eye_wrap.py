"""wire_splicing, the first example of BV1xT411m7AN ("两根硬线如何并线"): an eye bent into wire A's bare end round the
bending post's rod, wire B threaded through it, B's bare tail wound one and a half turns round A's doubled end (at least
one round both of its strands together), A's end folded back over the coil (runs/wire_splice_design_20261014, README_v5 / README_v6)."""
from .common import WireSplicingTask


class WireSplicingEyeWrapTask(WireSplicingTask):
    variation_id = "wire_splicing_eye_wrap"
    family_id = "wire_eye_wrap"
    model_revision = "wire_splicing_v5_eye_wrap"
    difficulty = None
