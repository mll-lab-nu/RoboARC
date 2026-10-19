"""The released interlocked_claws families' separation routes (keyframes of B relative to A) and their negative
controls: authoring and oracle data, not part of the simulator image (topogym.assets.geometry.interlocked_claws.FamilyModel
builds the pieces and their start only)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from topogym.assets.geometry.interlocked_claws import Segment, pose, rot


def targets(family):
    """[(name, duration_s, relative pose 4x4)] of the family's designed separation."""
    if family == "helical_crown":
        targets=[("screw_100deg_40mm",8.,pose([0,0,.04],rot("z",100))),
                 ("clear_crown",3.,pose([0,0,.085],rot("z",100))),
                 ("side_exit",4.,pose([.16,0,.085],rot("z",100)))]
        return targets
    if family == "orthogonal_gate":
        first = rot("x", 90)
        second = rot("z", 90) * first
        targets=[("first_axis_roll",5.,pose([-.04,0,0],first)),
                 ("first_window_transfer",3.,pose([.18,0,0],first)),
                 ("offset_clearance_lane",2.,pose([.18,-.12,0],first)),
                 ("transfer_between_windows",5.,pose([.26,-.12,0],first)),
                 ("second_axis_yaw",5.,pose([.26,-.12,0],second)),
                 ("align_second_window",2.,pose([.32,-.12,0],second)),
                 ("second_window_exit",6.,pose([.32,.32,0],second))]
        return targets
    if family == "helical_claws_balanced_v2":
        initial_rotation = rot("z", 180) * rot("x", 180)
        targets=[("coupled_screw_first_100deg",6.,pose([0,0,.040],rot("z",100)*initial_rotation)),
                 ("coupled_screw_second_100deg",6.,pose([0,0,.080],rot("z",200)*initial_rotation)),
                 ("separate_tips",2.,pose([0,0,.115],rot("z",200)*initial_rotation))]
        return targets
    if family == "orthogonal_gate_balanced_v2":
        first = rot("x", 90)
        second = rot("z", 90) * first
        targets=[("first_axis_roll",4.,pose([.04,0,0],first)),
                 ("first_window_pull",4.,pose([-.15,0,0],first)),
                 ("turn_clearance",2.,pose([-.15,-.04,0],first)),
                 ("second_axis_yaw",8.,pose([-.15,-.04,0],second)),
                 ("align_second_window",2.,pose([-.12,0,-.015],second)),
                 ("second_window_exit",6.,pose([-.12,.28,-.015],second))]
        return targets
    return []


def segments(model):
    """The route as FamilyModel segments, from the model's start pose of B."""
    out, previous = [], model.initial[1]
    for name, duration, target in targets(model.family):
        out.append(Segment(name, duration, previous, target))
        previous = target
    return out


def relative_pose(model, seconds):
    t=max(0.,float(seconds))
    for segment in segments(model):
        if t <= segment.duration:
            return segment.at(t/segment.duration),segment.name
        t-=segment.duration
    return segments(model)[-1].end.copy(),"separated"

def negative_paths(model):
    if model.family=="helical_claws_balanced_v2":
        initial=model.initial[1];r=Rotation.from_matrix(initial[:3,:3])
        return [Segment("pull_without_rotation",4.,initial,pose([0,0,.115],r)),
                Segment("turn_without_advance",4.,initial,pose(rotation=rot("z",100)*r)),
                Segment("reverse_screw",4.,initial,pose([0,0,.04],rot("z",-100)*r)),
                *[Segment(f"premature_side_exit_{axis}_{sign}",4.,initial,
                          pose(np.eye(3)[axis]*sign*.20,r)) for axis in (0,1) for sign in (-1,1)]]
    if model.family=="helical_crown":
        return [Segment("push_without_rotation",4.,model.initial[1],pose([0,0,.04])),
                Segment("turn_without_advance",4.,model.initial[1],pose(rotation=rot("z",100))),
                Segment("reverse_screw",4.,model.initial[1],pose([0,0,.04],rot("z",-100))),
                Segment("premature_side_exit",4.,model.initial[1],pose([.16,0,0]))]
    if model.family=="orthogonal_gate_balanced_v2":
        return [Segment("omit_first_roll",4.,model.initial[1],pose([-.15,0,0])),
                Segment("omit_second_yaw",4.,pose([-.12,0,-.015],rot("x",90)),
                        pose([-.12,.28,-.015],rot("x",90)))]
    first=rot("x",90)
    return [Segment("omit_first_roll",4.,model.initial[1],pose([.32,0,0])),
            Segment("omit_second_yaw",4.,pose([.32,0,0],first),pose([.32,.32,0],first))]


def geometry_audit(model):
    from types import SimpleNamespace
    from topogym.solvers.collision.triangle_meshes import TriangleMeshPair
    collision=TriangleMeshPair(part.mesh for part in model.parts)
    minimum=float("inf");leaves=0;queries=0;failures=[]
    radius=float(np.linalg.norm(model.parts[1].mesh.vertices,axis=1).max())
    for segment in segments(model):
        cache={}
        def gap(u):
            nonlocal queries,minimum
            if u not in cache:
                cache[u]=collision.clearance([np.eye(4),segment.at(u)])
                queries+=1;minimum=min(minimum,cache[u])
            return cache[u]
        def bound(a,b,depth=0):
            nonlocal leaves
            mid=(a+b)/2
            g=min(gap(a),gap(mid),gap(b))
            # Smoothstep has max derivative 1.875. This bounds all material
            # point motion on this original rigid interpolation segment.
            travel=1.875*(b-a)/2*(np.linalg.norm(segment.end[:3,3]-segment.start[:3,3])+radius*segment.rotation_angle)
            if g<0:
                failures.append(dict(phase=segment.name,start=a,end=b,reason="collision"));return
            if gap(mid)>travel+1e-7:
                leaves+=1;return
            if depth>=18:
                failures.append(dict(phase=segment.name,start=a,end=b,reason="unresolved_bound"));return
            bound(a,mid,depth+1);bound(mid,b,depth+1)
        bound(0.,1.)
    controls=[]
    for segment in negative_paths(model):
        witness=next((float(u) for u in np.linspace(0,1,201)
                      if collision.clearance([np.eye(4),segment.at(u)])<0),None)
        controls.append(dict(name=segment.name,blocked=witness is not None,witness_fraction=witness))
    return dict(passed=not failures and all(c["blocked"] for c in controls),
        continuous_pair_path_certified=not failures,minimum_sampled_gap_m=minimum,
        certified_intervals=leaves,queries=queries,failures=failures,negative_controls=controls,
        robot_collision_certified=False,physics_verified=False)
