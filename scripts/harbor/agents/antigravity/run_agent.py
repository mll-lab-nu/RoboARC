#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#   "google-antigravity==0.1.23",
# ]
# ///
"""The Antigravity SDK's agent on one task, inside the trial's container (antigravity_sdk.py uploads this file and its
uv lock, run_agent.py.lock, and starts it with `uv run --script --locked --offline`):

    run_agent.py --instruction-file F --logs-dir /logs/agent --trajectory-path /logs/agent/trajectory.json
    run_agent.py --version

Env: MODEL_NAME (a provider prefix, google/..., is dropped), GEMINI_API_KEY, GOOGLE_GEMINI_BASE_URL (optional: the
usage gateway's route), REASONING_EFFORT (optional: minimal, low, medium, high, xhigh; another value stops the run).

The configuration is the SDK's evaluation preset, LocalAgentConfig(...).eval() ("a standardized, product-agnostic
default intended to represent Gemini's core coding ability on benchmarks": no subagents, daemon commands allowed,
policy.allow_all(), RetryConfig.benchmark()), as Harbor 0.25's antigravity-sdk runner uses it, with the thinking level
of REASONING_EFFORT on the model target and three tools off (the preset turns off generate_image only):
- search_web: the SDK answers it with a second model call that declares Google Search, which the provider runs, past
  any egress allowlist (2026-11-01, google-antigravity 0.1.9 and 0.1.23 offline against a recording server);
- read_url_content: an HTTP fetch from the container;
- generate_image: an image model at the provider.
The trajectory (ATIF) follows Harbor's runner: the conversation's finished steps, and the SDK's session token totals
(conversation.total_usage) as final_metrics.
"""
import argparse
import asyncio
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

LEVELS = {"minimal": "MINIMAL", "low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "EXTRA_HIGH",
          "extra_high": "EXTRA_HIGH"}
OFF = ("SEARCH_WEB", "READ_URL_CONTENT", "GENERATE_IMAGE")
NOTES = ("final_metrics token totals are the SDK session aggregate (conversation.total_usage), which includes model calls "
         "not represented by a step (compaction, system steps) and can exceed the sum of per-step metrics. Completion "
         "tokens include thinking tokens; final_metrics.extra keeps the candidates/thoughts split. "
         "final_metrics.extra.accounting.complete = false: the run did not finish and the totals cover what was reported "
         "before it stopped.")


def _count(usage: Any, field: str) -> int | None:
    return getattr(usage, field, None)


def _generated(usage: Any) -> int | None:
    candidates, thoughts = _count(usage, "candidates_token_count"), _count(usage, "thoughts_token_count")
    return None if candidates is None and thoughts is None else (candidates or 0) + (thoughts or 0)


def step_metrics(usage: Any) -> dict:
    return {"prompt_tokens": _count(usage, "prompt_token_count"), "completion_tokens": _generated(usage),
            "cached_tokens": _count(usage, "cached_content_token_count"),
            "extra": {"candidates_tokens": _count(usage, "candidates_token_count"),
                      "thoughts_tokens": _count(usage, "thoughts_token_count")}}


def final_metrics(usage: Any, complete: bool) -> dict:
    accounting = {"source": "sdk_session_total_usage", "complete": complete}
    if usage is None:
        return {"total_prompt_tokens": None, "total_completion_tokens": None, "total_cached_tokens": None,
                "extra": {"accounting": accounting}}
    return {"total_prompt_tokens": _count(usage, "prompt_token_count"), "total_completion_tokens": _generated(usage),
            "total_cached_tokens": _count(usage, "cached_content_token_count"),
            "extra": {"candidates_tokens": _count(usage, "candidates_token_count"),
                      "thoughts_tokens": _count(usage, "thoughts_token_count"),
                      "sdk_total_tokens": _count(usage, "total_token_count"), "accounting": accounting}}


def step_record(step: Any, model: str, user, model_source) -> dict | None:
    """One finished step as an ATIF step (Harbor's runner), or None for the user's own."""
    if step.source == user:
        return None
    rec: dict[str, Any] = {"timestamp": None, "source": "agent" if step.source == model_source else "system",
                           "message": step.content or ""}
    if step.source == model_source:            # ATIF: model_name, metrics, reasoning and tool calls on agent steps only
        rec["model_name"] = model
        if getattr(step, "usage_metadata", None) is not None:
            rec["metrics"] = step_metrics(step.usage_metadata)
        if step.thinking:
            rec["reasoning_content"] = step.thinking
        if step.tool_calls:
            rec["tool_calls"] = [{"tool_call_id": tc.id, "function_name": tc.name, "arguments": tc.args or {}}
                                 for tc in step.tool_calls]
            results = [{"source_call_id": tc.id, "content": tc.output} for tc in step.tool_calls
                       if getattr(tc, "output", None) is not None]
            if results:
                rec["observation"] = {"results": results}
    return rec


def write_trajectory(path: Path, steps: list[dict], usage: Any, complete: bool) -> None:
    for i, s in enumerate(steps):
        s["step_id"] = i + 1
    doc = {"schema_version": "ATIF-v1.7", "session_id": os.environ.get("SESSION_ID", "harbor-session"),
           "agent": {"name": "antigravity", "version": version("google-antigravity")}, "steps": steps,
           "final_metrics": {"total_steps": len(steps), "total_cost_usd": None, **final_metrics(usage, complete)},
           "notes": NOTES}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


async def run(args) -> None:
    from google.antigravity import (Agent, BuiltinTools, CapabilitiesConfig, GeminiAPIEndpoint, GeminiModelOptions,
                                    LocalAgentConfig)
    from google.antigravity.hooks import policy
    from google.antigravity.models import ModelTarget, ThinkingLevel
    from google.antigravity.types import StepSource, StepStatus

    model = os.environ.get("MODEL_NAME", "").split("/")[-1]
    key = os.environ.get("GEMINI_API_KEY")
    if not model or not key:
        sys.exit("MODEL_NAME and GEMINI_API_KEY must be set")
    effort = (os.environ.get("REASONING_EFFORT") or "").lower()
    if effort and effort not in LEVELS:
        sys.exit(f"REASONING_EFFORT={effort}: not one of {', '.join(k for k in LEVELS if k != 'extra_high')}")
    options = GeminiModelOptions(thinking_level=getattr(ThinkingLevel, LEVELS[effort])) if effort else GeminiModelOptions()
    target = ModelTarget(name=model, endpoint=GeminiAPIEndpoint(
        api_key=key, base_url=os.environ.get("GOOGLE_GEMINI_BASE_URL") or None, options=options))
    logs = Path(args.logs_dir).resolve()
    (logs / "session").mkdir(parents=True, exist_ok=True)
    (logs / "app_data").mkdir(parents=True, exist_ok=True)
    workspace = Path.cwd().resolve()
    config = LocalAgentConfig(
        model=target, api_key=key, workspaces=[str(workspace)], save_dir=str(logs / "session"),
        app_data_dir=str(logs / "app_data"), policies=[policy.allow_all()],
        capabilities=CapabilitiesConfig(disabled_tools=[getattr(BuiltinTools, t) for t in OFF]),
    ).eval(thinking_level=None)               # the preset; the thinking level stays the model target's (None: the model's own)
    disabled = {str(getattr(t, "value", t)) for t in config.capabilities.disabled_tools or []}
    missing = {getattr(BuiltinTools, t).value for t in OFF} - disabled
    if missing:                               # the preset replaced our list: never run with a web tool on
        sys.exit(f"the SDK's eval() preset left these tools on: {sorted(missing)}")
    instruction = Path(args.instruction_file).read_text()
    print(f"Antigravity SDK {version('google-antigravity')}, model {model}, thinking level {effort or 'the model default'}, "
          f"tools off: {sorted(disabled)}", flush=True)
    steps: list[dict] = [{"timestamp": None, "source": "user", "message": instruction}]
    usage, complete = None, False
    try:
        async with Agent(config) as agent:
            try:
                response = await agent.chat(instruction)
                (logs / "output.txt").write_text(await response.text() or "")
                await response.resolve()
                complete = True
            finally:
                for step in getattr(agent.conversation, "history", []) or []:
                    if step.status == StepStatus.DONE:
                        rec = step_record(step, model, StepSource.USER, StepSource.MODEL)
                        if rec:
                            steps.append(rec)
                try:
                    usage = agent.conversation.total_usage
                except Exception as e:  # noqa: BLE001 -- a cleanup path must not hide the run's own error
                    print(f"the SDK's session usage could not be read: {e}", file=sys.stderr)
    finally:
        write_trajectory(Path(args.trajectory_path).resolve(), steps, usage, complete)
    print(f"done; trajectory {args.trajectory_path}", flush=True)


def main() -> None:
    if "--version" in sys.argv:
        print(version("google-antigravity"))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--instruction-file", required=True)
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--trajectory-path", required=True)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
