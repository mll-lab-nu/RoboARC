"""The lock_picking tiers: a chain lock (model chain_lock_v1). It replaced the original five tiers on 2026-10-10
.

A two-tab key with a long tip tab and a short second tab turns rotor 1 alone or
one adjacent rotor pair. Deep slots pass both tabs, shallow slots only the short
one; every slot is a through-cut. Physical stops sit 2 degrees beyond the rotor angles.
"""
from topogym.assets.geometry.lock_picking import DISC_COLORS, ring_travel_clear
from topogym.assets.geometry.lock_picking_chain import ChainLockGeometry
from topogym.tasks.base import merge_parameters

from .common import LockPickingTask

CHAIN_COLORS = (*DISC_COLORS, (.62, .24, .42))


def chain_parameters(*, deep, shallow, initial, stops, slot_half_deg=8., shaft_radius=.0065, endpoint_deg=0.):
    """Variant parameters for an N-rotor chain lock on the 120 mm pitch."""
    n = len(deep)
    return {
        "geometry": {
            "pitch": .12, "grip_offset": round(n*.12+.045, 6), "disc_radius": .08, "disc_thickness": .01,
            "bore_radius": .0075, "slot_radius": .045, "shallow_radius": .028, "slot_half_deg": float(slot_half_deg),
            "tab_radius": .036, "second_tab_radius": .022, "tab_half_deg": 4., "tab_thickness": .005,
            "shaft_radius": float(shaft_radius), "handle_radius": .018,
            "slots": [[float(s) for s in row] for row in deep],
            "shallow_slots": [[float(s) for s in row] for row in shallow],
            "initial_disc_deg": [float(a) for a in initial], "initial_key_deg": 0., "endpoint_deg": float(endpoint_deg),
        },
        # Keep the usual ear phase unless the travel would swing the ear into a frame
        # leg; then centre the travel on the top of the ring.
        "rotor_stops": [{"style": "single_ring", "bounds_deg": [float(lo), float(hi)],
                         "ear_phase_deg": -35. if ring_travel_clear(lo, hi, -35.) else -(lo+hi)/2}
                        for lo, hi in stops],
        "disc_colors": [list(c) for c in CHAIN_COLORS[:n]],
        # Wrist clearance at the tray: its rear cradle's robot-facing side is cut to 7 mm
        # and the tray dropped 5 mm; a deeper drop brings the wrist onto the table at pickup.
        "key_rest_drop_m": .105, "key_rest_rear_robot_side_m": .007,
    }


class ChainLockPickingTask(LockPickingTask):
    geometry_type = ChainLockGeometry
    model_revision = "chain_lock_v1"
    terminal_revision = "terminal_v6_chain_lock"
    fixed_parameters = ("geometry", "rotor_stops")
    instruction = ("Insert the key through every rotor plate and the final END plate until its leading tab "
                   "clears END. The key's leading tab is longer than its second tab.")

    def identity_config(self):
        """Tiers set their own tray values, so compare those with the original lock's
        defaults: a tray that differs from it must enter the instance identity."""
        tray = ("key_rest_drop_m", "key_rest_rear_robot_side_m")
        return {key: value for key, value in self.config.items()
                if not (key in self.hash_when_changed
                        and value == (LockPickingTask.defaults if key in tray else self.defaults)[key])}


def _tier(name, difficulty, family, parameters):
    defaults = merge_parameters(LockPickingTask.defaults, parameters)
    return type(name, (ChainLockPickingTask,), dict(
        variation_id=f"lock_picking_{difficulty}", difficulty=difficulty, family_id=family,
        variant_parameters=parameters, defaults=defaults, __module__=__name__))


LockPickingVeryEasyTask = _tier("LockPickingVeryEasyTask", "very_easy", "lock_design_1", chain_parameters(
    deep=[[0.], [0.]], shallow=[[], []], initial=[20., -25.], stops=[(-29., 24.), (-29., 4.)]))

LockPickingEasierTask = _tier("LockPickingEasierTask", "easier", "lock_design_2", chain_parameters(
    deep=[[-5.], [-5.], [0.]], shallow=[[-35.], [-35.], []], initial=[15., -15., -50.],
    stops=[(-17., 17.), (-17., 17.), (-57., 7.)]))

LockPickingEasyTask = _tier("LockPickingEasyTask", "easy", "lock_design_3", chain_parameters(
    deep=[[-5.], [-5.], [0.]], shallow=[[45., 16.], [-105., -55., -155., 30.], []],
    initial=[-25., -25., -130.], stops=[(-27., 27.), (-27., 27.), (-137., 7.)]))

LockPickingMediumTask = _tier("LockPickingMediumTask", "medium", "lock_design_4", chain_parameters(
    deep=[[-30.], [-30.], [0.]], shallow=[[6., -12.], [-102., -66., 6., -138., 24.], []],
    initial=[-18., -18., -120.], stops=[(-20., 20.), (-20., 20.), (-127., 7.)], slot_half_deg=6.))


LockPickingHardTask = _tier("LockPickingHardTask", "hard", "lock_design_5", chain_parameters(
    deep=[[5.], [5.], [5.], [20.]], shallow=[[-25.], [35.], [-115., -85., -55., -25., 35., -145.], []],
    initial=[15., -15., 15., -150.], stops=[(-17., 17.), (-17., 17.), (-17., 17.), (-152., 2.)],
    slot_half_deg=5.5, shaft_radius=.00675, endpoint_deg=20.))


class LockPickingDesignTask(ChainLockPickingTask):
    """Any chain lock through task parameters, so planning/instance_optimizer can lift a
    route to IK before the lock becomes a tier. Defaults to the very_easy lock."""
    variation_id = "lock_picking_design"
    family_id = "chain_design"
    fixed_parameters = ()
    variant_parameters = LockPickingVeryEasyTask.variant_parameters
    defaults = LockPickingVeryEasyTask.defaults
