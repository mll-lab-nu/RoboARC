"""The chain of the FEM settings on MuJoCo (task parameter cord_model "chain"; common.py): the team's 10-07 rigid link
chain (topogym/assets/geometry/slotted_post_chain.py: 16.5 mm capsule links on ball joints along the setting's authored
cord, the discs on ball joints at their eyes, the beads sliding on it, the ring as capsules) and its goal measurement.

This is the construction the below-easy settings used until they moved them onto #108's chain file
(slotted_chain.py). The FEM settings keep it unchanged: their MuJoCo scenes, settling and
acceptance stay as built. Functions of a SlottedPostRingTask instance (formerly SlottedBelowEasyTask's methods).
"""
from pathlib import Path

import numpy as np
import trimesh

from topogym.assets.geometry.slotted_post_chain import (
    DISC_EYE_M, chain_roots, disc_mesh, write_cord_assembly, write_ring)
from topogym.assets.geometry.slotted_post_ring import generate_assets, make_meshes
from topogym.assets.scene_visuals import visual_collision_urdf
from topogym.evaluation.task_metrics import body_points, quiet_body, rigid_asset_vertices
from topogym.tasks.base import EntitySpec, TaskSpec
from topogym.tasks.scene_presets import record_preset
from topogym.utils.geometry.regions import placement_measurements


def chain_parts(task):
    """The authored layout, meshes and cord, the link joints, the disc centres and the beads, without writing assets."""
    layout = task.make_layout().validate()
    parts, cord = make_meshes(layout)
    discs = [np.r_[parts[f"disc_{side}"].bounds.mean(axis=0)[:2], layout.floor_clearance_m] for side in ("left", "right")]
    beads = []
    for name, index in zip(("bead_left", "bead_right"), layout.bead_controls()):
        k = int(np.argmin(np.linalg.norm(cord-layout.cord_controls()[index], axis=1)))
        beads.append((name, cord[k], cord[k+1]-cord[k-1]))
    return layout, parts, cord, chain_roots(layout, cord, discs), discs, beads


def declare(task, asset_dir, table_height_m):
    task.table_height_m = float(table_height_m)
    preset = record_preset(task, asset_dir)
    xy = np.asarray(task.config["center_xy_m"], dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Invalid Slotted Post Ring center_xy_m")
    origin = tuple(np.r_[xy, table_height_m])
    task.layout, report = generate_assets(asset_dir, task.make_layout())
    layout, parts, cord, roots, discs, beads = chain_parts(task)
    folder = Path(asset_dir)
    xml, task.link_names, task.link_lengths, task.link_reversed = write_cord_assembly(
        folder, layout, roots, discs, beads, young_modulus_pa=task.chain_young_modulus_pa)
    ring_xml, ring_mass = write_ring(folder, layout, layout.ring_z)
    entities = [EntitySpec(name, "urdf", visual_collision_urdf(folder, name), origin, fixed=True,
                           morph_options={"convexify": False, "decimate": False, "file_meshes_are_zup": True,
                                          "align": False, "merge_fixed_links": False})
                for name in ("base", "post")]
    entities += [EntitySpec("target_ring", "mjcf", ring_xml, origin), EntitySpec("cord", "mjcf", xml, origin)]
    # Authored geometry in each entity's frame (the goal region's separator) and in each body's frame (evaluation).
    task.placement_vertices = {e.name: rigid_asset_vertices(e) for e in entities[:2]}
    task.placement_vertices["target_ring"] = np.asarray(parts["target_ring"].vertices)
    task.placement_vertices["cord"] = np.vstack([cord, *(parts[n].vertices for n in (
        "disc_left", "disc_right", "bead_left", "bead_right"))])
    disc = np.asarray(disc_mesh(layout).vertices)+[0., -DISC_EYE_M, -layout.disc_thickness_m]
    bead = np.asarray(trimesh.creation.icosphere(subdivisions=2, radius=layout.bead_radius_m).vertices)
    task.body_vertices = dict(target_ring=task.placement_vertices["target_ring"]-[0., 0., layout.ring_z],
                              disc_left=disc, disc_right=disc, bead_left=bead, bead_right=bead)
    task.placement_targets = ("target_ring",)
    task.placement_remainder = ("base", "post", "cord")
    return TaskSpec(
        tuple(entities),
        {"rigid": {"enable_self_collision": True, "constraint_solver": "CG"}},
        dict(
            **preset,
            assembly_origin_m=list(origin),
            task_id=task.task_id,
            cord_chain=dict(links=len(task.link_names), radius_m=layout.rope_radius_m,
                            link_lengths_m=[float(v) for v in task.link_lengths],
                            length_m=float(task.link_lengths.sum()), ring_mass_kg=float(ring_mass)),
            geometry_report=report,
            permanent_connections=["cord ends to the disc eyes (ball joints)", "links (ball joints)"],
            goal_evaluation_complete=bool(task.goal_complete),
        ),
    )


def link_cover(task, length):
    """Capsule axis ends and their radial extremes, in the link frame."""
    r = task.layout.rope_radius_m
    return np.array([[x, *offset] for x in (0., length) for offset in
                     ((0., 0.), (r, 0.), (-r, 0.), (0., r), (0., -r))])


def evaluate_goal_region(task, state, region):
    """The whole ring inside the goal and still; the fixture, chain, discs and beads on the system side."""
    links = state["articulations"]["cord"]["links"]
    ring = state["bodies"]["target_ring"]
    cord = [body_points(links[name], link_cover(task, length))
            for name, length in zip(task.link_names, task.link_lengths)]
    cord += [body_points(links[name], task.body_vertices[name])
             for name in ("disc_left", "disc_right", "bead_left", "bead_right")]
    remainder = {name: body_points(state["bodies"][name], task.placement_vertices[name]) for name in ("base", "post")}
    remainder["cord"] = np.vstack(cord)
    metrics = placement_measurements(region, {"target_ring": body_points(ring, task.body_vertices["target_ring"])},
                                     remainder)
    quiet = quiet_body(ring)
    metrics.update(quiet=bool(quiet), connections_intact=True, evaluation_status="available", holding_allowed=True)
    reached = metrics["all_target_inside"] and metrics["remainder_on_system_side"] and quiet
    return bool(reached), None, metrics
