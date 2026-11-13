#!/usr/bin/env python3
"""write_task_results.classify(): which trial is a result (RUNNING.md "Infrastructure failures"). The agent's own end
is a result: its deliverable, the cutoff (AgentTimeoutError), a model response over the harness's output-token maximum
(OutputTokenExceededError) or a request over the model's context window (ContextWindowExceededError), and the kernel
killing the agent at its own container's memory cap in the limits' scope (NonZeroAgentExitCodeError with
resource_oom_kills, no host OOM kill, no emergency kill). Any other Harbor exception, no reward.json, the verifier's
infra_error, or the gateway's failed last calls is not a result. Synthetic trial directories; standard library only."""
import json
import pathlib
import sys
import tempfile

sys.path[:0] = [str(pathlib.Path(__file__).resolve().parents[1]), str(pathlib.Path(__file__).resolve().parents[2])]
import write_task_results as wtr  # noqa: E402

fails = []
root = pathlib.Path(tempfile.mkdtemp(prefix="verdict-"))


def trial(name, ex=None, reward=None, gateway=None):
    t = root / "job" / name
    (t / "verifier").mkdir(parents=True)
    r = {"task_name": "roboarc-x-01-i01-standard"}
    if ex:
        r["exception_info"] = {"exception_type": ex, "exception_message": "..."}
    (t / "result.json").write_text(json.dumps(r))
    if reward is not None:
        (t / "verifier" / "reward.json").write_text(json.dumps(reward))
    if gateway is not None:
        (t / "gateway").mkdir()
        (t / "gateway" / "totals.json").write_text(json.dumps(gateway))
    return t, r, wtr.load_reward(t)


def expect(name, want, **kw):
    t, r, rw = trial(name, **kw)
    status, why, _ = wtr.classify(t, r, rw)
    if status != want:
        fails.append(f"{name}: {status} ({why}), want {want}")


F = {"success": 0, "infra_error": 0, "grader_error": 0}
OOM = dict(F, resource_oom_kills=2, resource_host_oom_kills=0, resource_emergency_kills=0)

# the agent's own end: a result
expect("cutoff", "fail", ex="AgentTimeoutError", reward=F)
expect("output_token_maximum", "fail", ex="OutputTokenExceededError", reward=F)
expect("context_window", "fail", ex="ContextWindowExceededError", reward=F)
expect("own_cap_oom", "fail", ex="NonZeroAgentExitCodeError", reward=OOM)
expect("own_cap_oom_success", "success", ex="NonZeroAgentExitCodeError", reward=dict(OOM, success=1))
expect("no_exception", "fail", reward=F)

# infrastructure: not a result
expect("exit_without_oom", "not_a_result", ex="NonZeroAgentExitCodeError", reward=F)
expect("host_oom_too", "not_a_result", ex="NonZeroAgentExitCodeError", reward=dict(OOM, resource_host_oom_kills=1))
expect("emergency_kill_too", "not_a_result", ex="NonZeroAgentExitCodeError", reward=dict(OOM, resource_emergency_kills=1))
expect("other_exception_with_oom", "not_a_result", ex="NetworkConnectionError", reward=OOM)
expect("api_error", "not_a_result", ex="UnknownApiError", reward=F)
expect("output_maximum_no_reward", "not_a_result", ex="OutputTokenExceededError")
expect("output_maximum_infra_error", "not_a_result", ex="OutputTokenExceededError", reward=dict(F, infra_error=1))
expect("own_cap_oom_infra_error", "not_a_result", ex="NonZeroAgentExitCodeError", reward=dict(OOM, infra_error=1))
expect("output_maximum_gateway_failed", "not_a_result", ex="OutputTokenExceededError", reward=F,
       gateway={"requests": 9, "input": 9, "output": 9, "upstream_failures": 1, "failures_in_a_row": 1,
                "last_upstream_failure": "HTTP 502"})

# over the token budget is the agent's failure whatever ended the trial
expect("over_budget", "fail", ex="UnknownApiError", reward=F,
       gateway={"requests": 9, "input": 100000001, "output": 9, "over_token_budget": True})

if fails:
    print("FAIL test_result_verdict:\n  " + "\n  ".join(fails))
    sys.exit(1)
print("ok test_result_verdict")
