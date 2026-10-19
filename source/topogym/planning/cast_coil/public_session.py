"""Run the existing Coil policy against the exact manager-created Harbor Session."""
from topogym.planning.harbor_bridge.actions import SessionPolicyRunner
from .policy import CastCoilPolicy, variant_reference
from .simplified_policy import SimplifiedCoilPolicy


def run_feedback_policy(session, output):
    if session.task != "cast_coil" or session.n_steps or session._resets:
        raise ValueError("Coil reference requires a fresh frozen Coil Session")
    task = session.env._task
    if task.variation_id in ("cast_coil_v2_very_easy", "cast_coil_v2_easier", "cast_coil_v2_easy", "cast_coil_v2_design"):
        # The v2 tiers and the v2 design tier (MuJoCo): its own planner on the task's solution record and its own
        # speed-limited joint targets (official_policy.py). A tier ships its pieces only; its route is here (a design
        # given as a task parameter carries its own).
        from .official_policy import OfficialCoilPolicy, solution_record
        record = task.solution()
        if "route" not in record:
            record = solution_record(task.geometry_resource)
        policy = OfficialCoilPolicy(seed=session.config["seed"], task_parameters=task.config, record=record)
        return SessionPolicyRunner(session, control_mode="JOINT_ABS").run_policy(
            request_id="coil-frozen-instance", policy=policy, submission_hash=session.instance_sha,
            state_fields=("world",))
    arguments = dict(variant=task.variation_id, seed=session.config["seed"], task_parameters=task.config)
    assembly, _ = variant_reference(**arguments)
    policy_class = SimplifiedCoilPolicy if hasattr(assembly, "record") else CastCoilPolicy
    policy = policy_class(body_origins=task.body_origins, part="all", **arguments)
    return SessionPolicyRunner(session, control_mode="EE_ABS").run_policy(
        request_id="coil-frozen-instance", policy=policy, submission_hash=session.instance_sha,
        state_fields=("world",))
