"""bed_bolt's drives (user 2026-10-18; runs/robotangle_mujoco/bed_bolt/20261018_variants/DESIGN.md): bed_bolt_crossbar's
scene, bolt, thread and rules with another recess in the bolt's head and the matching tips on the L-shaped tool
(topogym.assets.geometry.bed_bolt DRIVES); nothing else differs."""
from .bed_bolt_crossbar import BedBoltCrossbarTask


class BedBoltTorxTask(BedBoltCrossbarTask):
    """A six-lobe (Torx) recess and an L-shaped Torx key: like the hex, the recess matches itself every 60 degrees, so the
    key goes in, swings, comes out and goes back in as with the hex key."""
    variation_id = "bed_bolt_torx"
    model_revision = "bed_bolt_v2_torx"
    drive = "torx"


class BedBoltSquareTask(BedBoltCrossbarTask):
    """A square recess and a ratcheting L-shaped key with a square tip (DRIVES square, RATCHET): the recess matches itself
    every 90 degrees, more than the bay lets the lever sweep; the ratchet lets the key swing 60 degrees and back without
    leaving the recess."""
    variation_id = "bed_bolt_square"
    model_revision = "bed_bolt_v2_square"
    drive = "square"


class BedBoltPhillipsTask(BedBoltCrossbarTask):
    """A cross (Phillips) recess and a ratcheting L-shaped key with a cross tip (DRIVES phillips, RATCHET): the recess
    matches itself every 90 degrees."""
    variation_id = "bed_bolt_phillips"
    model_revision = "bed_bolt_v2_phillips"
    drive = "phillips"


class BedBoltSlottedTask(BedBoltCrossbarTask):
    """A straight slot across the head and a ratcheting L-shaped key with a flat blade (DRIVES slotted, RATCHET): the slot
    matches itself only every 180 degrees; the blade can slide out along the open slot."""
    variation_id = "bed_bolt_slotted"
    model_revision = "bed_bolt_v2_slotted"
    drive = "slotted"
