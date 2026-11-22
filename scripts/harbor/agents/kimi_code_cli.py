"""Kimi Code CLI (npm @moonshot-ai/kimi-code, github.com/MoonshotAI/kimi-code) as a Harbor agent: Kimi Code + a Kimi model
on Moonshot's API (Kimi K3.1 Pro: kimi-k3.1-pro), reached through the usage gateway (its `moonshot` provider).

    PYTHONPATH=scripts/harbor/agents harbor run ... -a kimi_code_cli:KimiCodeCli -m moonshot/kimi-k3.1-pro \\
        --ak reasoning_effort=max --ae KIMI_BASE_URL=<the gateway's route to Moonshot's API>

scripts/harbor/run_agent_batch.sh runs it with AGENT=kimi-code (docs/RUNNING.md); whether its results are official is
the benchmark owner's call. Harbor's own kimi-code agent (Harbor 0.23 to 0.25) installs whatever version is latest,
leaves the CLI's web tools on, takes no reasoning effort, and writes trajectory.json from 0.25 on only; this class, on
Harbor's BaseInstalledAgent, runs the CLI as Harbor's does (`kimi --prompt ... --output-format stream-json`, its output
in agent/kimi-code.txt, its data in agent/.kimi-code/) with these differences:
- the version is pinned: 2.5.2 (npm `latest` on 2026-11-06), installed during agent setup (Node 22 through Harbor's nvm
  snippet, then npm), while the environment still has its network (a task's egress allowlist applies to the agent's
  run only);
- its web tools off: WebSearch and FetchURL are disabled in its config.toml ([tools] disabled), which the CLI applies
  to every agent and subagent and checks again before running a tool (FetchURL would otherwise fetch from the
  container itself);
- the model from the CLI's KIMI_MODEL_* variables: the model id (MODEL's last segment), the key (KIMI_API_KEY), the base
  URL (KIMI_BASE_URL, default https://api.moonshot.ai/v1), the context window (KIMI_MODEL_MAX_CONTEXT_SIZE, default
  1048576, Kimi K3.1 Pro's) and reasoning_effort as KIMI_MODEL_THINKING_EFFORT (low, medium, high, xhigh or max; Kimi
  K3.1 Pro takes low, high and max, max being its default); another effort is refused when the agent is made, before
  the trial starts. The key stays in the process's environment: nothing under agent/.kimi-code/ holds it;
- the instruction is uploaded as a file and handed to the CLI from it. The CLI takes a prompt on its command line only
  (no stdin, no file), so the instruction is in its argv, as it is in Harbor's own Codex and Kimi Code agents: an agent
  that runs `pkill -f <a word of the task>` can end its own harness;
- token counts: from Harbor's own converter of the CLI's session journal (harbor.agents.installed.kimi_code_trajectory,
  Harbor 0.25), which also writes agent/trajectory.json (ATIF); with an older Harbor neither, and the journal stays in
  agent/.kimi-code/. With the usage gateway its totals.json counts the same requests on the wire.
Credentials: KIMI_API_KEY (a Moonshot API key) and KIMI_BASE_URL, from Harbor's environment (its --env-file) or --ae
(which wins).
"""
from __future__ import annotations

import os
import shlex
from pathlib import PurePosixPath
from typing import Any

from pydantic import Field

from harbor.agents.capabilities import AgentCapabilities
from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.agents.options import InstalledAgentOptions
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

PACKAGE = "@moonshot-ai/kimi-code"
VERSION = "2.5.2"
LEVELS = ("low", "medium", "high", "xhigh", "max")      # the CLI's thinking efforts (KIMI_MODEL_THINKING_EFFORT)
BASE_URL = "https://api.moonshot.ai/v1"
CONTEXT = "1048576"                                     # Kimi K3.1 Pro's context window, in tokens
PROMPT = "/tmp/kimi-prompt/instruction.txt"
HOME = (EnvironmentPaths.agent_dir / ".kimi-code").as_posix()      # KIMI_CODE_HOME: kept with the trial's agent logs
OUTPUT = "kimi-code.txt"
CONFIG = '# RoboARC (scripts/harbor/agents/kimi_code_cli.py): no web tools; the model from KIMI_MODEL_*\n' \
         'telemetry = false\n\n[tools]\ndisabled = ["WebSearch", "FetchURL"]\n'
PATH_SETUP = 'if [ -s "$HOME/.nvm/nvm.sh" ]; then . "$HOME/.nvm/nvm.sh"; fi; export PATH="$HOME/.local/bin:$PATH"; '


class KimiCodeCliOptions(InstalledAgentOptions):
    reasoning_effort: str | None = Field(default=None, description="The thinking effort: low, medium, high, xhigh or max.")


class KimiCodeCli(BaseInstalledAgent):
    """Kimi Code CLI, pinned, its web tools off, its model and effort from KIMI_MODEL_*."""

    # Harbor 0.25 takes a reasoning_effort option only from an agent that declares it; 0.23 and 0.24 have no such field
    capabilities = AgentCapabilities(**{k: True for k in ("atif", "reasoning_effort") if k in AgentCapabilities.model_fields})
    options_model = KimiCodeCliOptions
    options: KimiCodeCliOptions

    def __init__(self, version: str | None = None, **kwargs: Any):
        super().__init__(version=version or VERSION, **kwargs)
        effort = self.options.reasoning_effort
        if effort is not None and effort.lower() not in LEVELS:
            raise ValueError(f"reasoning_effort={effort}: Kimi Code's thinking efforts are {', '.join(LEVELS)}")

    @staticmethod
    def name() -> str:
        return "kimi-code"

    def _env(self, name: str) -> str | None:
        return self._extra_env.get(name) or os.environ.get(name)     # --ae wins over the environment's own

    async def install(self, environment: BaseEnvironment) -> None:
        await self.ensure_system_dependencies(environment, ("curl", "ca_certificates"))
        version = shlex.quote(self._version)
        await self.exec_as_agent(
            environment,
            command=f'{nvm_node_install_snippet()} && mkdir -p "$HOME/.local" && npm install --global --prefix "$HOME/.local" '
                    f'{shlex.quote(f"{PACKAGE}@{self._version}")} && {PATH_SETUP}kimi --version | grep -qwF {version}')

    def populate_context_post_run(self, context: AgentContext) -> None:
        try:
            from harbor.agents.installed.kimi_code_trajectory import build_trajectory     # Harbor 0.25
            from harbor.utils.trajectory_utils import format_trajectory_json
        except ImportError:
            self.logger.warning("this Harbor has no Kimi Code trajectory converter (0.25 has): no trajectory.json; the "
                                "session journal stays in agent/.kimi-code/")
            return
        try:
            trajectory = build_trajectory(self.logs_dir, agent_name=self.name(), agent_version=self.version(),
                                          model_name=self.model_name)
            if trajectory is None:
                return
            (self.logs_dir / "trajectory.json").write_text(format_trajectory_json(trajectory.to_json_dict()))
        except Exception as e:  # noqa: BLE001 -- a trial is not lost for its trajectory
            self.logger.warning(f"trajectory.json not written: {type(e).__name__}: {e}")
            return
        fm = trajectory.final_metrics
        if fm and (fm.total_prompt_tokens is not None or fm.total_completion_tokens is not None):
            context.n_input_tokens = fm.total_prompt_tokens or 0
            context.n_output_tokens = fm.total_completion_tokens or 0
            context.n_cache_tokens = fm.total_cached_tokens or 0

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        key = self._env("KIMI_API_KEY")
        if not key:
            raise RuntimeError("KIMI_API_KEY is not set: put the Moonshot API key in the env file Harbor loads "
                               "(run_agent_batch.sh ENV_FILE)")
        if not self.model_name:
            raise RuntimeError("no model: harbor run -m <model>, e.g. moonshot/kimi-k3.1-pro")
        local = self.logs_dir / "instruction.txt"
        local.write_text(instruction)
        prompt_dir = shlex.quote(str(PurePosixPath(PROMPT).parent))
        await environment.exec(command=f"mkdir -p {prompt_dir} && chmod 755 {prompt_dir}", user="root")
        await environment.upload_file(source_path=local, target_path=PROMPT)
        await environment.exec(command=f"chmod 644 {PROMPT}", user="root")
        env = {"KIMI_CODE_HOME": HOME, "KIMI_DISABLE_TELEMETRY": "1", "KIMI_CODE_NO_AUTO_UPDATE": "1", "NO_COLOR": "1",
               "KIMI_MODEL_NAME": self.model_name.rsplit("/", 1)[-1], "KIMI_MODEL_API_KEY": key,
               "KIMI_MODEL_BASE_URL": self._env("KIMI_BASE_URL") or BASE_URL,
               "KIMI_MODEL_MAX_CONTEXT_SIZE": self._env("KIMI_MODEL_MAX_CONTEXT_SIZE") or CONTEXT}
        if self.options.reasoning_effort:
            env["KIMI_MODEL_THINKING_EFFORT"] = self.options.reasoning_effort.lower()
        await self.exec_as_agent(environment, env={"KIMI_CODE_HOME": HOME},
                                 command=f'mkdir -p "$KIMI_CODE_HOME" && printf %s {shlex.quote(CONFIG)} > "$KIMI_CODE_HOME/config.toml"')
        logs = EnvironmentPaths.agent_dir.as_posix()
        command = (f'{PATH_SETUP}kimi --prompt="$(cat {PROMPT})" --output-format stream-json </dev/null 2>&1 '
                   f"| stdbuf -oL tee {logs}/{OUTPUT}")
        await self.exec_as_agent(environment, command=command, env=env)
