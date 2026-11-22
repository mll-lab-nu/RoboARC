"""Tests for AGENT=kimi-code's Harbor agent (scripts/harbor/agents/kimi_code_cli.py): no docker, no model, no network. The
runner's side (arguments, the gateway's moonshot provider, refusals) is in test_usage_gateway_option.py; the gateway's in
scripts/usage_gateway/tests/test_paths.py.

    python3 scripts/harbor/tests/test_kimi_code.py     # stdlib only (Python 3.11+); the class checks need Harbor

With Harbor importable (HARBOR_PYTHON, or the interpreter of `harbor` on PATH; Harbor 0.24 and 0.25 both), against a
stand-in environment that records every command: the version is 2.5.2 and an effort the CLI has no level for is refused
when the agent is made; install puts Node (Harbor's nvm snippet) and that version of the CLI in place while there is
network and checks the version; the run writes the CLI's config.toml (WebSearch and FetchURL off, no key in it), uploads
the instruction as a file, and starts the CLI from it with the model (MODEL's last segment), the key, the base URL
(--ae over the environment's own), the context window and the effort in the CLI's KIMI_MODEL_* variables; with Harbor
0.25 trajectory.json is written from the CLI's output, with 0.24 none.
"""
import json
import os
import pathlib
import shutil
import subprocess
import tomllib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
AGENTS = REPO / "scripts" / "harbor" / "agents"


def harbor_python():
    if os.environ.get("HARBOR_PYTHON"):
        return os.environ["HARBOR_PYTHON"]
    exe = shutil.which("harbor")
    if exe:
        first = pathlib.Path(exe).read_text(errors="replace").splitlines()[0]
        if first.startswith("#!"):
            return first[2:].strip()
    return None


CLASS_CHECK = '''
import asyncio, json, pathlib, sys, tempfile, types
sys.path.insert(0, {agents!r})
from kimi_code_cli import KimiCodeCli, CONFIG
from harbor.models.agent.context import AgentContext
try:
    import harbor.agents.installed.kimi_code_trajectory
    converter = True
except ImportError:
    converter = False

class Env:                                   # the trial's container, as far as the agent uses it
    def __init__(self):
        self.calls = []
    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.calls.append({{"exec": command, "user": user, "env": env or {{}}}})
        out = "apt-get" if "for manager in" in command else "2.5.2\\n"     # Harbor's package-manager probe; the rest
        return types.SimpleNamespace(return_code=0, stdout=out, stderr="")
    async def upload_file(self, source_path, target_path):
        self.calls.append({{"upload": str(target_path), "text": pathlib.Path(source_path).read_text()}})

out = {{"converter": converter, "config": CONFIG}}
with tempfile.TemporaryDirectory() as d:
    logs = pathlib.Path(d)
    try:
        KimiCodeCli(logs_dir=logs, model_name="moonshot/kimi-k3.1-pro", reasoning_effort="minimal")
        out["bad_effort"] = "accepted"
    except Exception as e:
        out["bad_effort"] = type(e).__name__ + ": " + str(e)
    a = KimiCodeCli(logs_dir=logs, model_name="moonshot/kimi-k3.1-pro", reasoning_effort="Max",
                    extra_env={{"KIMI_BASE_URL": "http://gw:1/u/api.moonshot.ai/v1"}})
    out["name"], out["version"] = a.name(), a.version()
    env = Env()
    asyncio.run(a.install(env))
    out["install"] = env.calls
    env = Env()
    task = "Thread the rope through the TASKWORD ring.\\n" * 40
    asyncio.run(a.run(task, env, AgentContext()))
    out["run"] = env.calls
    out["task"] = task
    (logs / "kimi-code.txt").write_text("\\n".join(json.dumps(e) for e in (
        {{"role": "assistant", "content": "", "tool_calls": [{{"id": "c1", "function": {{"name": "Bash", "arguments": "{{}}"}}}}]}},
        {{"role": "tool", "tool_call_id": "c1", "content": "ok"}},
        {{"role": "assistant", "content": "done"}})) + "\\n")
    ctx = AgentContext()
    a.populate_context_post_run(ctx)
    t = logs / "trajectory.json"
    out["trajectory_steps"] = len(json.loads(t.read_text())["steps"]) if t.exists() else None
print(json.dumps(out))
'''


class HarborClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        py = harbor_python()
        if not py or subprocess.run([py, "-c", "import harbor"], capture_output=True).returncode != 0:
            raise unittest.SkipTest("Harbor is not importable (set HARBOR_PYTHON to its interpreter)")
        p = subprocess.run([py, "-c", CLASS_CHECK.format(agents=str(AGENTS))], capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, KIMI_API_KEY="sk-from-the-env-file-0123456789",
                                    KIMI_BASE_URL="https://not-this-one/v1"))
        if p.returncode != 0:
            raise AssertionError(p.stderr[-3000:])
        cls.out = json.loads(p.stdout.strip().splitlines()[-1])

    def test_identity_and_effort(self):
        self.assertEqual((self.out["name"], self.out["version"]), ("kimi-code", "2.5.2"))
        self.assertIn("ValueError", self.out["bad_effort"])
        self.assertIn("low, medium, high, xhigh, max", self.out["bad_effort"])

    def test_install(self):
        cmds = [c["exec"] for c in self.out["install"] if "exec" in c]
        (npm,) = [c for c in cmds if "npm install" in c]
        self.assertIn("nvm-sh/nvm", npm)                                   # Node through Harbor's nvm snippet
        self.assertIn('npm install --global --prefix "$HOME/.local" @moonshot-ai/kimi-code@2.5.2', npm)
        self.assertIn("kimi --version | grep -qwF 2.5.2", npm)
        self.assertFalse([c for c in self.out["install"] if "upload" in c])

    def test_config_turns_the_web_tools_off(self):
        cfg = tomllib.loads(self.out["config"])
        self.assertEqual(cfg["tools"]["disabled"], ["WebSearch", "FetchURL"])
        self.assertIs(cfg["telemetry"], False)
        (write,) = [c for c in self.out["run"] if "exec" in c and "config.toml" in c["exec"]]
        self.assertIn("WebSearch", write["exec"])
        self.assertEqual(write["env"], {"KIMI_CODE_HOME": "/logs/agent/.kimi-code"})          # no key on that command
        self.assertNotIn("sk-from-the-env-file", json.dumps(self.out["run"][:-1]))

    def test_run(self):
        calls = self.out["run"]
        uploads = [c for c in calls if "upload" in c]
        self.assertEqual([(u["upload"], u["text"]) for u in uploads], [("/tmp/kimi-prompt/instruction.txt", self.out["task"])])
        run = calls[-1]
        self.assertIn('kimi --prompt="$(cat /tmp/kimi-prompt/instruction.txt)" --output-format stream-json', run["exec"])
        self.assertNotIn("TASKWORD", run["exec"])                   # the command names the file; the CLI's argv holds the text
        self.assertIn("| stdbuf -oL tee /logs/agent/kimi-code.txt", run["exec"])
        env = run["env"]
        self.assertEqual({k: env[k] for k in ("KIMI_MODEL_NAME", "KIMI_MODEL_THINKING_EFFORT", "KIMI_MODEL_MAX_CONTEXT_SIZE",
                                              "KIMI_CODE_HOME", "KIMI_DISABLE_TELEMETRY", "KIMI_CODE_NO_AUTO_UPDATE")},
                         {"KIMI_MODEL_NAME": "kimi-k3.1-pro", "KIMI_MODEL_THINKING_EFFORT": "max",
                          "KIMI_MODEL_MAX_CONTEXT_SIZE": "1048576", "KIMI_CODE_HOME": "/logs/agent/.kimi-code",
                          "KIMI_DISABLE_TELEMETRY": "1", "KIMI_CODE_NO_AUTO_UPDATE": "1"})
        self.assertEqual(env["KIMI_MODEL_API_KEY"], "sk-from-the-env-file-0123456789")
        self.assertEqual(env["KIMI_MODEL_BASE_URL"], "http://gw:1/u/api.moonshot.ai/v1")          # --ae wins

    def test_trajectory(self):
        self.assertEqual(self.out["trajectory_steps"], 2 if self.out["converter"] else None)


if __name__ == "__main__":
    unittest.main()
