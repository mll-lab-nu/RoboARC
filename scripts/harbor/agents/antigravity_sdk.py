"""The Antigravity SDK (PyPI google-antigravity: Google's Antigravity agent as a library) as a Harbor agent: Antigravity +
Gemini 4 Argon, the model served by an OpenAI-compatible Chat Completions server (GitHub Copilot's proxy) and reached
through the usage gateway's --gemini-chat translation (scripts/usage_gateway/gemini_chat.py). These runs are
Antigravity + Gemini's official runs (decided 2026-11-01).

    PYTHONPATH=scripts/harbor/agents harbor run ... -a antigravity_sdk:AntigravitySdk -m google/gemini-4-argon \\
        --ak reasoning_effort=high --ae GOOGLE_GEMINI_BASE_URL=<the gateway's route to the Chat Completions server>

scripts/harbor/run_agent_batch.sh runs it with AGENT=antigravity (docs/RUNNING.md). Harbor's own antigravity-sdk agent
pins SDK 0.1.9 (Harbor 0.23, 0.24; its OpenAI-compatible mode drops the images tools return) or takes >= 0.1.20 (0.25),
changes shape between those releases, and leaves the SDK's web tools on; this class, on Harbor's BaseInstalledAgent:
- install, during agent setup (the environment's network: a task's egress allowlist applies to the agent's run only):
  uv 0.11.30 (astral's installer, into /installed-agent/uv/bin), the runner antigravity/run_agent.py with its uv lock
  (google-antigravity 0.1.23 and every dependency pinned) and `uv sync --locked --script`; the run then starts from
  /installed-agent/uv with UV_OFFLINE=1, without the network;
- the run: the SDK's evaluation preset with search_web, read_url_content and generate_image off (run_agent.py: why);
- the instruction travels in a file, never on a command line (dsh_acp.py: an agent that runs `pkill -f <a word of the
  task>` would kill its own harness, the exam's issue #228);
- reasoning_effort -> the SDK's thinking level (minimal, low, medium, high, xhigh = the SDK's extra_high); another value
  is refused when the agent is made, before the trial starts; the gateway sends the level on as reasoning_effort;
- token counts: the SDK's session totals in agent/trajectory.json (ATIF), read back here; with the usage gateway its
  totals.json counts the same requests on the wire.
Credentials: GEMINI_API_KEY (the key the model server expects: the gateway sends it upstream as a Bearer token) and
GOOGLE_GEMINI_BASE_URL, from Harbor's environment (its --env-file) or --ae (which wins).
"""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

HERE = Path(__file__).resolve().parent / "antigravity"
SDK_VERSION = "0.1.23"                   # run_agent.py's dependency, pinned with the rest in run_agent.py.lock
UV_VERSION = "0.11.30"
UV_DIR = "/installed-agent/uv"
UV = f"{UV_DIR}/bin/uv"
UV_ENV = {"UV_CACHE_DIR": f"{UV_DIR}/cache", "UV_PYTHON_INSTALL_DIR": f"{UV_DIR}/python", "UV_PYTHON_PREFERENCE": "system",
          "UV_NO_CONFIG": "1"}
SCRIPT = "/installed-agent/run_agent.py"
PROMPT = "/tmp/antigravity-prompt/instruction.txt"
LEVELS = ("minimal", "low", "medium", "high", "xhigh")
OUTPUT = "antigravity.txt"
TRAJECTORY = "trajectory.json"


class AntigravitySdkOptions(InstalledAgentOptions):
    reasoning_effort: str | None = Field(default=None, description="The thinking level: minimal, low, medium, high or xhigh.")


class AntigravitySdk(BaseInstalledAgent):
    """The Antigravity SDK's agent, pinned, its web tools off, its instruction in a file."""

    # Harbor 0.25 takes a reasoning_effort option only from an agent that declares it; 0.23 and 0.24 have no such field
    capabilities = AgentCapabilities(**{k: True for k in ("atif", "reasoning_effort") if k in AgentCapabilities.model_fields})
    options_model = AntigravitySdkOptions
    options: AntigravitySdkOptions

    def __init__(self, version: str | None = None, **kwargs: Any):
        super().__init__(version=version or SDK_VERSION, **kwargs)
        effort = self.options.reasoning_effort
        if effort is not None and effort.lower() not in LEVELS:
            raise ValueError(f"reasoning_effort={effort}: the Antigravity SDK's levels are {', '.join(LEVELS)}")

    @staticmethod
    def name() -> str:
        return "antigravity"

    def _env(self, name: str) -> str | None:
        return self._extra_env.get(name) or os.environ.get(name)     # --ae wins over the environment's own

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl", "ca_certificates"))
        await self.exec_as_root(
            environment,
            command=f"mkdir -p {UV_DIR}/bin && chmod 777 {UV_DIR} && curl -LsSf "
                    f"https://astral.sh/uv/{UV_VERSION}/install.sh | env UV_UNMANAGED_INSTALL={UV_DIR}/bin sh "
                    f"&& {UV} --version")
        for name in ("run_agent.py", "run_agent.py.lock"):
            local = self.logs_dir / name
            local.write_text((HERE / name).read_text())
            await environment.upload_file(source_path=local, target_path=f"/installed-agent/{name}")
        await environment.exec(command=f"chmod 644 {SCRIPT} {SCRIPT}.lock", user="root")
        # the SDK and its dependencies (a Python too, where the image's is older than 3.11), now, while there is network
        await self.exec_as_agent(environment, env=UV_ENV,
                                 command=f"{UV} sync --locked --script {SCRIPT} && {UV} run --script --locked --offline "
                                         f"{SCRIPT} --version | grep -qx {shlex.quote(SDK_VERSION)}")

    def populate_context_post_run(self, context: AgentContext) -> None:
        path = self.logs_dir / TRAJECTORY
        try:
            metrics = json.loads(path.read_text()).get("final_metrics") or {}
        except (OSError, ValueError) as e:
            self.logger.warning(f"{path.name} not read: {e}")
            return
        context.n_input_tokens = metrics.get("total_prompt_tokens")
        context.n_output_tokens = metrics.get("total_completion_tokens")
        context.n_cache_tokens = metrics.get("total_cached_tokens")

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        key = self._env("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set: put the model server's key in the env file Harbor loads "
                               "(run_agent_batch.sh ENV_FILE)")
        if not self.model_name:
            raise RuntimeError("no model: harbor run -m <model>")
        local = self.logs_dir / "instruction.txt"
        local.write_text(instruction)
        prompt_dir = shlex.quote(str(PurePosixPath(PROMPT).parent))
        await environment.exec(command=f"mkdir -p {prompt_dir} && chmod 755 {prompt_dir}", user="root")
        await environment.upload_file(source_path=local, target_path=PROMPT)
        await environment.exec(command=f"chmod 644 {PROMPT}", user="root")
        env = {**UV_ENV, "UV_OFFLINE": "1", "GEMINI_API_KEY": key, "MODEL_NAME": self.model_name}
        base_url = self._env("GOOGLE_GEMINI_BASE_URL")
        if base_url:
            env["GOOGLE_GEMINI_BASE_URL"] = base_url
        if self.options.reasoning_effort:
            env["REASONING_EFFORT"] = self.options.reasoning_effort.lower()
        logs = EnvironmentPaths.agent_dir.as_posix()
        command = (f"set -o pipefail; {UV} run --script --locked --offline {SCRIPT} --instruction-file={PROMPT} "
                   f"--logs-dir={logs} --trajectory-path={logs}/{TRAJECTORY} 2>&1 | stdbuf -oL tee {logs}/{OUTPUT}")
        await self.exec_as_agent(environment, command=command, env=env)
