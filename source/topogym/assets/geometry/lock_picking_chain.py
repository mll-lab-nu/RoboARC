"""Chain lock: N passive rotors and a two-tab key whose tip tab is longer.

Level n (key depth n) puts the long tip tab in rotor n and the short second tab
in rotor n-1, so the key turns at most one adjacent pair of rotors at a time.
Deep slots (to slot_radius) pass both tabs; shallow slots (to shallow_radius)
pass only the short tab. Every slot is a through-cut, visible on both faces.
"""
from dataclasses import dataclass
from typing import ClassVar

from .lock_picking import LockGeometry, angle_error


@dataclass(frozen=True)
class ChainLockGeometry(LockGeometry):
    tab_count: ClassVar[int] = 2
    pitch: float = .120
    grip_offset: float = .405
    slots: tuple = ((0.,), (0.,), (0.,))
    shallow_slots: tuple = ((), (), ())
    initial_disc_deg: tuple = (0., 0., 0.)
    shallow_radius: float = .028
    second_tab_radius: float = .022

    @property
    def disc_count(self):
        return len(self.slots)

    def tab_radius_at(self, tab):
        return self.tab_radius if tab == 0 else self.second_tab_radius

    def shallow_of(self, i):
        return tuple(self.shallow_slots[i])

    def tab_slots(self, i, tab):
        """The long tip tab fits only deep slots; the short tab fits both kinds."""
        return tuple(self.slots[i]) + (() if tab == 0 else self.shallow_of(i))

    def validate(self):
        super().validate()
        if len(self.shallow_slots) != self.disc_count:
            raise ValueError(f'{self.disc_count} shallow slot rows are required')
        if not self.bore_radius < self.second_tab_radius < self.shallow_radius < self.tab_radius:
            raise ValueError('Shallow slots must pass the short tab and block the long tab')
        for deep, shallow in zip(self.slots, self.shallow_slots):
            row = (*deep, *shallow)
            if any(angle_error(a, b) <= 2*self.slot_half_deg for i, a in enumerate(row) for b in row[i+1:]):
                raise ValueError('Slots must be distinct and non-overlapping')

    def slot_margin(self, angle, disc_angle, i, tab=None):
        slots = self.tab_slots(i, 0 if tab is None else tab)
        return self.slot_half_deg-self.tab_half_deg-min(angle_error(angle, disc_angle+s) for s in slots)

    def clearance(self, depth, angle, disc_angles, tolerance=0.):
        issues = []
        for tab in range(self.tab_count):
            x = self.tip_x(depth)-tab*self.pitch
            for i in range(self.disc_count):
                if abs(x-self.disc_x(i)) < (self.tab_thickness+self.disc_thickness)/2:
                    if self.slot_margin(angle, disc_angles[i], i, tab) < -tolerance:
                        issues.append(f'tab{tab}/disc{i}')
            if abs(x-self.disc_x(self.disc_count)) < (self.tab_thickness+self.disc_thickness)/2:
                if self.slot_half_deg-self.tab_half_deg-angle_error(angle, self.endpoint_deg) < -tolerance:
                    issues.append(f'tab{tab}/endpoint')
        return issues
