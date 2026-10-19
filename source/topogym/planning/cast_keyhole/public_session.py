"""Private reference adapter: passive policy -> checked IK -> public Session.step.

This module belongs to the oracle's private solution, never the public image.
It does not create/reset a scene, move objects, attach grasps, or change physics.
The caller's run-oracle recorder owns the complete public action trajectory.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def public_row(session, native):
    """Use the shared adapter for either native or width public grippers."""
    from topogym.planning.harbor_bridge.actions import public_row as convert
    return convert(session, native)


def run_feedback_policy(session, output):
    from topogym.assets.geometry.cast_keyhole import plate_assembly_transforms
    from topogym.assets.variation_resources import read_json
    from topogym.robot.control import action_spec
    from topogym.utils.io import to_jsonable, write_json_atomic
    from .policy import PARTS
    from .simplified_policy import SimplifiedKeyholePolicy
    from .kinematics import preflight_simplified_pickup, preflight_simplified_goal

    if session.n_steps != 0 or session._resets != 0:
        raise ValueError('Feedback reference requires a fresh frozen Session')
    output = Path(output)
    task, robot = session.env._task, session.env._robot
    geometry = read_json(task.geometry_resource)
    built = json.loads((session.env._asset_build.root / 'geometry_report.json').read_text())
    origins = {body['name']: np.asarray(body['origin_m']) for body in built['bodies']}
    transforms = plate_assembly_transforms(geometry)
    # Task placement vertices already include the complete collision/visible
    # geometry in each link frame. Transform those existing vertices for planning.
    vertices = {side: Rotation.from_matrix(transforms[metal][:3, :3]).inv().apply(
        np.asarray(task.placement_vertices[PARTS[side]])+origins[PARTS[side]]-transforms[metal][:3, 3])
        for side, metal in [('left', 'gold'), ('right', 'silver')]}
    closed = {side: session.robot.width_to_native(side, 0.) for side in PARTS}
    if not np.isclose(closed['left'], closed['right'], atol=1e-12, rtol=0):
        raise ValueError('This reference requires matching ALOHA gripper adapters')
    class TableClearancePolicy(SimplifiedKeyholePolicy):
        # Small, bounded TCP planning offsets accommodate naturally settled
        # plate tilt. They never move a body or relax the collision audit.
        grasp_raise_m = 0.

        def grasp_targets(self, bodies):
            return {side: (position + [0., 0., self.grasp_raise_m], rotation)
                    for side, (position, rotation) in super().grasp_targets(bodies).items()}

    policy = TableClearancePolicy(geometry=geometry, variation=task.variation_id,
        body_origins=origins, goal_region=task.terminal_metadata()['goal_region'],
        mount_position=[0., 0., 0.], canonical_vertices=vertices,
        part='all', speed=1., gripper_closed_native_m=closed['left'])
    policy.initialize(dict(action_spec=action_spec(robot.description, 'EE_ABS'),
                           timing_spec=session.env.timing_spec))

    def observe():
        return dict(state=dict(sim_time_s=session.env._world.sim_time_s),
                    robot=robot.observe(), privileged=dict(world=session.env._world.read_state()))

    def submit(action):
        from topogym.planning.oracle_ik import compile_oracle_action
        native = compile_oracle_action(session.env, action, 'EE_ABS')
        return session.step(public_row(session, native))

    report = dict(status='running', public_actions_only=True, from_frozen_home=True,
        initial_state_sha=session.instance_sha, gripper_closed_native_m=closed['left'], phases=[],
        scope='feedback through public Session; independent open-loop grading still required')
    write_json_atomic(output/'feedback_report.json', report)
    try:
        for index in range(30):
            if submit({s: dict(gripper_position_m=.024) for s in PARTS} if index == 0 else {}):
                raise RuntimeError('Episode ended during initial settling')
        policy.obs = observe()
        # Pre-grasp opening x TCP raise. The first candidate that passes the preflight without any planned finger
        # contact before closing wins; else the first that passes (the released choice: .024, raise 0-2 mm). On MuJoCo
        # (2026-10-09) the released .024 approach cut 1 mm into the resting gold plate's edge (a permitted own-finger
        # contact); the brush shoved and turned the plate 5-10 mm / 15 deg and the pregrasp never settled. A wider opening
        # lowers the lower finger toward the table, which the raise gives back.
        attempts, chosen = [], None
        for opening in (.024, .026, .028, .030):
            for raise_m in (0., .001, .002, .003, .004, .006):
                policy.open_native_m, policy.grasp_raise_m = opening, raise_m
                plan = preflight_simplified_pickup(robot, session.env._world.entities, origins,
                                                  geometry, policy, task.table_height_m)
                audit = plan['collision_audit'] or {}
                clean = plan['passed'] and not audit.get('allowed_own_finger_contacts')
                attempts.append(dict(open_native_m=opening, grasp_raise_m=raise_m, passed=plan['passed'], clean=clean,
                                     failure=plan['failure'], collision_audit=audit))
                if plan['passed'] and (chosen is None or clean):
                    chosen = (opening, raise_m, plan)
                if clean:
                    break
            if chosen is not None and attempts[-1]['clean']:
                break
        report['pickup_candidates'] = attempts
        if chosen is None:
            write_json_atomic(output/'pickup_plan.json', plan)
            raise RuntimeError('Frozen-session pickup preflight failed')
        policy.open_native_m, policy.grasp_raise_m, plan = chosen
        policy.grip = {s: policy.open_native_m for s in PARTS}
        report.update(grasp_raise_m=policy.grasp_raise_m, open_native_m=policy.open_native_m)
        write_json_atomic(output/'pickup_plan.json', plan)
        policy.pickup_transit = plan['tcp_path']
        last_phase = None
        with (output/'feedback_states.jsonl').open('w', buffering=1) as states:
            while session.n_steps < session.max_steps:
                obs = observe()
                action = policy.act(obs)
                if policy.phase != last_phase:
                    report['phases'].append(dict(phase=policy.phase, sim_time_s=session.sim_time_s))
                    last_phase = policy.phase
                    if policy.phase == 'gold_raise_clear':
                        goal = preflight_simplified_goal(robot, session.env._world.entities, origins,
                                                        geometry, policy, task.table_height_m)
                        write_json_atomic(output/'goal_plan.json', goal)
                        if not goal['passed']:
                            raise RuntimeError('Frozen-session carried-piece goal preflight failed')
                    print(json.dumps(dict(oracle_phase=policy.phase, sim_time_s=session.sim_time_s)), flush=True)
                if session.n_steps % 20 == 0:
                    world = obs['privileged']['world']
                    states.write(json.dumps(to_jsonable(dict(row=session.n_steps, state=obs['state'],
                        robot=obs['robot'], bodies={n: world['bodies'][n] for n in PARTS.values()},
                        grasp=world['grasp'], diagnostics=policy.diagnostics)))+'\n')
                if action is None or submit(action):
                    break
                if session.n_steps % 200 == 0:
                    print(json.dumps(dict(oracle_row=session.n_steps, phase=policy.phase,
                                          sim_time_s=session.sim_time_s)), flush=True)
            report.update(status='passed' if session.success() else 'failed',
                          task_success=session.success(), session_status=session.status())
            if not session.success():
                raise RuntimeError('Feedback reference did not complete the Task')
    except BaseException as error:
        report.update(status='failed', error=repr(error), last_phase=policy.phase,
                      diagnostics=to_jsonable(policy.diagnostics), task_success=session.success())
        raise
    finally:
        report['task_result'] = to_jsonable(task.result)
        write_json_atomic(output/'feedback_report.json', report)
