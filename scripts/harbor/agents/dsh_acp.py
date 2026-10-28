"""DSH, the DeepSeek Harness (npm `@deepseek-ai/dsh`, github.com/deepseek-ai/deepseek-harness), as a Harbor agent over the
Agent Client Protocol (ACP).

    PYTHONPATH=scripts/harbor/agents harbor run ... -a dsh_acp:DshAcp -m deepseek/deepseek-v4.1-flash \\
        --ak reasoning_effort=high --allow-agent-host api.deepseek.com

scripts/harbor/run_agent_batch.sh runs it with AGENT=dsh (docs/RUNNING.md). Harbor's generic ACP agent
(harbor.agents.installed.acp.AcpAgent) drives the session, answers the agent's permission requests, logs the ACP events
and turns them into agent/trajectory.json; this class adds what DSH needs:
- install: Node (Harbor's nvm snippet) and `npm install -g @deepseek-ai/dsh@<version>` during agent setup, which still has
  the environment's network (a task's egress allowlist applies to the agent's run only), and a launcher for
  `dsh --profile acp`, the harness's ACP stdio profile;
- the version is pinned: 0.2.1-rc.2 (npm `latest`), the release line of the exam's DSH runs (2026-10-15 to 10-18). It
  picks models from its built-in catalog: its `deepseek-flash` is DeepSeek-V4.1-Flash, so the model deepseek-v4.1-flash is
  asked for as `deepseek-flash` (the trial's records keep the model as given). Releases from 0.2.1-rc.3 on also list
  models at the API origin's /models, which the usage gateway's /u/<host>/ routes cannot carry;
- no web tools unless web=true: a dsh patch file turns web_search and web_fetch off (PROTOCOL D11: DeepSeek would run the
  search, past the egress allowlist);
- the instruction travels in a file, never on a command line: Harbor's ACP runner takes it as an argument, and an agent
  that runs `pkill -f <a word of the task>` would kill its own harness (the exam's issue #228). dsh/acp_prompt_file.py
  reads the file and starts Harbor's runner in its own process;
- reasoning_effort: Harbor's runner sets only the model; acp_prompt_file.py sets the session's `reasoning_effort` option
  (dsh advertises it per model) after the session is created and after each model change, recorded in
  agent/dsh-effort.json.
Credentials: DEEPSEEK_API_KEY from Harbor's environment (its --env-file); DEEPSEEK_BASE_URL, when set there (the usage
gateway), points dsh at another Messages root.
"""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field

from harbor.agents.installed.acp import AcpAgent, AcpOptions
from harbor.agents.installed.base import with_prompt_template
from harbor.agents.installed.node_install import nvm_node_install_snippet
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

HERE = Path(__file__).resolve().parent / "dsh"
PACKAGE = "@deepseek-ai/dsh"
VERSION = "0.2.1-rc.2"
LAUNCHER = "/installed-agent/dsh-acp.sh"
PATCH = "/installed-agent/dsh-roboarc.patch.yml"
SHIM = "/installed-agent/acp_prompt_file.py"
PROMPT = "/tmp/acp-prompt/instruction.txt"
# the model as RoboARC names it (the exam's run id) -> the id of dsh's built-in catalog entry (0.2.1-rc.2 models.ts:
# id 'deepseek-flash', name 'DeepSeek-V4.1-Flash')
MODEL_IDS = {"deepseek-v4.1-flash": "deepseek-flash"}


class DshAcpOptions(AcpOptions):
    reasoning_effort: str | None = Field(default=None, description="The session's reasoning_effort option, e.g. high.")
    web: bool = Field(default=False, description="Keep dsh's web_search and web_fetch tools (not for official runs).")


class DshAcp(AcpAgent):
    """DSH over ACP: Harbor's ACP agent with dsh installed, its instruction in a file and its reasoning effort set."""

    options_model = DshAcpOptions
    options: DshAcpOptions

    def __init__(self, version: str | None = None, **kwargs: Any):
        version = version or VERSION
        entry = {
            "id": "dsh", "name": "DeepSeek Harness", "version": version,
            "description": "DeepSeek Harness, dsh --profile acp",
            "repository": "https://github.com/deepseek-ai/deepseek-harness", "license": "MIT",
            "distribution": {"local": {"cmd": LAUNCHER, "env": {"DSH_TELEMETRY_DISABLED": "1"}}},
        }
        super().__init__(registry_entry=entry, version=version, **kwargs)

    @staticmethod
    def name() -> str:
        return "dsh"

    def _launcher(self) -> str:
        patch = f" --patch {PATCH}" if not self.options.web else ""
        return ("#!/usr/bin/env sh\nset -eu\n"
                'nvm_node_bin="$(ls -d "${NVM_DIR:-$HOME/.nvm}"/versions/node/*/bin 2>/dev/null | sort -V | tail -n 1)"\n'
                'if [ -n "$nvm_node_bin" ]; then PATH="$nvm_node_bin:$PATH"; export PATH; fi\n'
                f'exec dsh --profile acp{patch} "$@"\n')

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)       # python, uv, the runner's venv, Harbor's launcher (-> LAUNCHER), acp_runner.py
        await self.exec_as_agent(
            environment,
            command=f"{nvm_node_install_snippet()} && npm install -g {shlex.quote(f'{PACKAGE}@{self._version}')} "
                    "&& command -v dsh",
        )
        files = {LAUNCHER: self._launcher(), SHIM: (HERE / "acp_prompt_file.py").read_text()}
        if not self.options.web:
            files[PATCH] = (HERE / "roboarc.patch.yml").read_text()
        for remote, text in files.items():
            local = self.logs_dir / PurePosixPath(remote).name
            local.write_text(text)
            await environment.upload_file(source_path=local, target_path=remote)
            await environment.exec(command=f"chmod 755 {shlex.quote(remote)}", user="root")

    def _requested_model(self) -> str | None:
        if not self.model_name:
            return None
        provider, _, model = self.model_name.rpartition("/")
        model = MODEL_IDS.get(model.lower(), model)
        return f"{provider}/{model}" if provider else model

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        key = os.environ.get("DEEPSEEK_API_KEY") or self._extra_env.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set: put it in the env file Harbor loads (run_agent_batch.sh ENV_FILE)")
        agent_id, agent_version = self._agent_identity()
        self._last_instruction = instruction
        local = self.logs_dir / "instruction.txt"
        local.write_text(instruction)
        prompt_dir = shlex.quote(str(PurePosixPath(PROMPT).parent))
        await environment.exec(command=f"mkdir -p {prompt_dir} && chmod 755 {prompt_dir}", user="root")
        await environment.upload_file(source_path=local, target_path=PROMPT)
        await environment.exec(command=f"chmod 644 {PROMPT}", user="root")
        env = {
            "HARBOR_ACP_MCP_SERVERS_JSON": json.dumps(self._build_mcp_servers_payload()),
            "HARBOR_ACP_PERMISSION_MODE": self._permission_mode,
            "HARBOR_ACP_AUTH_POLICY": self._auth_policy,
            "HARBOR_ACP_AGENT_ID": agent_id,
            "HARBOR_ACP_AGENT_VERSION": agent_version,
            "DEEPSEEK_API_KEY": key,
            "DSH_TELEMETRY_DISABLED": "1",
        }
        # --ae (the usage gateway's route) wins over the environment's own
        base_url = self._extra_env.get("DEEPSEEK_BASE_URL") or os.environ.get("DEEPSEEK_BASE_URL")
        if base_url:
            env["DEEPSEEK_BASE_URL"] = base_url
        if self._authenticate_method_id:
            env["HARBOR_ACP_AUTHENTICATE_METHOD_ID"] = self._authenticate_method_id
        if self._requested_model():
            env["HARBOR_ACP_REQUESTED_MODEL"] = self._requested_model()
        effort = f" --reasoning-effort={shlex.quote(self.options.reasoning_effort)}" if self.options.reasoning_effort else ""
        command = (
            f"{self._RUNNER_VENV_PATH}/bin/python {SHIM} --instruction-file={PROMPT} --runner={self._RUNNER_REMOTE_PATH}"
            f"{effort} -- --logs-dir={EnvironmentPaths.agent_dir.as_posix()} --launcher={self._LAUNCHER_REMOTE_PATH} "
            f"2>&1 | stdbuf -oL tee {EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME}"
        )
        await self.exec_as_agent(environment, command=command, env=env)
