"""The scene numbers the planner modules use (oracle4.py, plan3.py, scenario3.py): taken from the task's own geometry
module (topogym.assets.geometry.caliper_sorting) where it has them, the rest (the reading camera geometry, the layout
helpers) as the design defined them (runs/caliper_sorting_design_20261014/scripts: caliper_scene2.py, caliper_scene3.py,
sleeves2.py). Lengths in mm unless noted; positions in the ALOHA mount frame (m)."""
import math

import numpy as np

from topogym.assets.geometry import caliper_sorting as G

MM = 1e-3
# caliper (caliper_scene2 / caliper_scene3)
BEAM, OUT_JAW, IN_JAW, EDGE_R, SLIDER, Z0 = G.BEAM, G.OUT_JAW, G.IN_JAW, G.EDGE_R, G.SLIDER, G.Z0
ROLLER, KNOB = G.ROLLER, G.KNOB
GRIP_R = G.GRIP_SLIDER                   # the slider's grip (slider frame, mm)
S_OPEN = 80.0                   # jaw opening before the outside jaws go around a sleeve (OD up to 74 mm + 2 x 3)
RACK, BINS = G.RACK, G.BINS
SLEEVE_L = G.SLEEVE_L
# the reading camera (caliper_scene2): 70 mm from the scale, view tilted 55 deg from the face normal, axis on slider
# (35.5, 1) = the middle of the 0.1 mm long vernier, 1 mm into the main scale
READ_D, READ_AT, READ_GAMMA = 70.0, (35.5, 1.0), -55.0
# (2026-10-18, the caliper on its stand) the view from the inside jaws' side (-55: the hand-held design looked from the
# outside jaws' side, +55; on the stand the left hand's fingers there came within 1 mm of the slider, and from this side,
# the hand turned half a turn about the line of sight (oracle4.READ_ROLL), 14 mm: stand_check/read_design.py)
# sleeves (the task's scenario: four sleeves in rack slots, the class table as caliper readings)
DEV = G.SIZE_DEV


def make_sleeves(scen):
    """The scenario's sleeves as the planner keeps them (sizes in mm, their rack slot)."""
    return [dict(name=f'sleeve{k}', cls=s['cls'], slot=s['slot'], id_mm=s['id_mm'], od_mm=s['od_mm'], ri=s['id_mm'] / 2,
                 ro=s['od_mm'] / 2) for k, s in enumerate(scen['sleeves'])]


def vernier_reading(s_mm, scen):
    """What a reader who never misreads sees on the vernier at jaw opening s (mm): the caliper's reading in its own unit
    (the task's own vernier_reading with the scenario's unit and zero)."""
    return G.vernier_reading(s_mm, scen['unit_mm'], scen['zero_reading'])


def read_cam_geometry(read_d=READ_D, xa=None, gamma=READ_GAMMA):
    """Reading camera in the slider frame (mm): looks at (xa, READ_AT[1], beam face) from distance read_d along
    v = (0, sin g, -cos g); right = +x, up = (0, cos g, sin g).  Returns (pos, right, up, fwd)."""
    g = math.radians(gamma)
    xa = READ_AT[0] if xa is None else xa
    fwd = np.array([0, math.sin(g), -math.cos(g)])
    up = np.array([0, math.cos(g), math.sin(g)])
    pos = np.array([xa, READ_AT[1], BEAM['half_t']]) - read_d * fwd
    return pos, np.array([1.0, 0, 0]), up, fwd


def layout(scen):
    """The scene's places (mount frame, m) as the task lays them out (topogym.assets.geometry.caliper_sorting.layout):
    the caliper's C origin on its stand, the rack's six places, the four bins."""
    L = G.layout(scen)
    return dict(caliper=np.asarray(L["caliper"], float), rack=[np.asarray(p, float) for p in L["rack"]],
                bins=[np.asarray(p, float) for p in L["bins"]])


def sleeve_in_rack(L, k, ro):
    p = L['rack'][k]
    return np.array([p[0], p[1], (RACK['floor'] + ro) * MM])


def sleeve_in_bin(L, k):
    p = L['bins'][k]
    return np.array([p[0], p[1], (BINS['wall'] + SLEEVE_L / 2) * MM])


def quat_axis_y():
    """Sleeve local z (its axis) along world y: rotate -90 deg about x."""
    a = -math.pi / 2
    return (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)
