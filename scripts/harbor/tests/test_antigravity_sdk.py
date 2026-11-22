"""Tests for AGENT=antigravity's Harbor agent (scripts/harbor/agents/antigravity_sdk.py, antigravity/run_agent.py): no
docker, no model, no network. The runner's side (arguments, the gateway's --gemini-chat, refusals) is in
test_usage_gateway_option.py; the translation in scripts/usage_gateway/tests/test_gemini_chat.py.

    python3 scripts/harbor/tests/test_antigravity_sdk.py     # stdlib only (Python 3.11+); the class checks need Harbor

- run_agent.py pins google-antigravity 0.1.23, and its uv lock was made from exactly these requirements and holds that
  version; it turns search_web, read_url_content and generate_image off and maps the efforts to the SDK's levels;
- with Harbor importable (HARBOR_PYTHON, or the interpreter of `harbor` on PATH), against a stand-in environment that
  records every command: the version is the SDK's; an effort without a level is refused when the agent is made; install
  puts uv 0.11.30 and the two files in place and syncs the locked environment while there is network; the run takes
  the instruction from a file (none of it on the command line), starts offline, with the key, the model, the effort
  and GOOGLE_GEMINI_BASE_URL (--ae over the environment's own); the token counts come from the trajectory.
"""
import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import textwrap
import tomllib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
AGENTS = REPO / "scripts" / "harbor" / "agents"
SCRIPT = AGENTS / "antigravity" / "run_agent.py"


class RunnerScript(unittest.TestCase):
    def test_pinned_and_locked(self):
        text = SCRIPT.read_text()
        block = re.search(r"(?ms)^# /// script\n(.*?)^# ///$", text).group(1)
        meta = tomllib.loads("".join(line[2:] + "\n" for line in block.splitlines()))
        self.assertEqual(meta["dependencies"], ["google-antigravity==0.1.23"])
        lock = tomllib.loads((SCRIPT.parent / "run_agent.py.lock").read_text())
        self.assertEqual(lock["manifest"]["requirements"], [{"name": "google-antigravity", "specifier": "==0.1.23"}])
        self.assertEqual(lock["requires-python"].replace(" ", ""), meta["requires-python"].replace(" ", ""))
        (sdk,) = [p for p in lock["package"] if p["name"] == "google-antigravity"]
        self.assertEqual(sdk["version"], "0.1.23")

    def test_tools_off_and_levels(self):
        tree = ast.parse(SCRIPT.read_text())
        consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  for t in n.targets if isinstance(t, ast.Name) and t.id in ("OFF", "LEVELS")}
        self.assertEqual(set(consts["OFF"]), {"SEARCH_WEB", "READ_URL_CONTENT", "GENERATE_IMAGE"})
        self.assertEqual(consts["LEVELS"]["xhigh"], "EXTRA_HIGH")
        self.assertEqual({consts["LEVELS"][k] for k in ("minimal", "low", "medium", "high")}, {"MINIMAL", "LOW", "MEDIUM", "HIGH"})


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
import antigravity_sdk as m
from antigravity_sdk import AntigravitySdk
from harbor.models.agent.context import AgentContext

class Env:                                   # the trial's container, as far as the agent uses it
    def __init__(self):
        self.calls = []
    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.calls.append({{"exec": command, "user": user, "env": env or {{}}}})
        out = "apt-get" if "for manager in" in command else "0.1.23\\n"     # Harbor's package-manager probe; the rest
        return types.SimpleNamespace(return_code=0, stdout=out, stderr="")
    async def upload_file(self, source_path, target_path):
        self.calls.append({{"upload": str(target_path), "text": pathlib.Path(source_path).read_text()}})

out = {{}}
with tempfile.TemporaryDirectory() as d:
    logs = pathlib.Path(d)
    try:
        AntigravitySdk(logs_dir=logs, model_name="google/gemini-4-argon", reasoning_effort="max")
        out["bad_effort"] = "accepted"
    except Exception as e:
        out["bad_effort"] = type(e).__name__ + ": " + str(e)
    a = AntigravitySdk(logs_dir=logs, model_name="google/gemini-4-argon", reasoning_effort="High",
                       extra_env={{"GOOGLE_GEMINI_BASE_URL": "http://gw:1/h/127.0.0.1:4141/v1"}})
    out["name"], out["version"] = a.name(), a.version()
    env = Env()
    asyncio.run(a.install(env))
    out["install"] = env.calls
    env = Env()
    task = "Thread the rope through the TASKWORD ring.\\n" * 40
    asyncio.run(a.run(task, env, AgentContext()))
    out["run"] = env.calls
    out["task"] = task
    (logs / "trajectory.json").write_text(json.dumps({{"final_metrics": {{"total_prompt_tokens": 1200,
        "total_completion_tokens": 80, "total_cached_tokens": 1000}}}}))
    ctx = AgentContext()
    a.populate_context_post_run(ctx)
    out["tokens"] = [ctx.n_input_tokens, ctx.n_output_tokens, ctx.n_cache_tokens]
print(json.dumps(out))
'''


class HarborClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        py = harbor_python()
        if not py or subprocess.run([py, "-c", "import harbor"], capture_output=True).returncode != 0:
            raise unittest.SkipTest("Harbor is not importable (set HARBOR_PYTHON to its interpreter)")
        p = subprocess.run([py, "-c", CLASS_CHECK.format(agents=str(AGENTS))], capture_output=True, text=True, timeout=120,
                           env=dict(os.environ, GEMINI_API_KEY="from-the-env-file-0123456789",
                                    GOOGLE_GEMINI_BASE_URL="http://not-this-one:1"))
        if p.returncode != 0:
            raise AssertionError(p.stderr[-3000:])
        cls.out = json.loads(p.stdout.strip().splitlines()[-1])

    def test_identity_and_effort(self):
        self.assertEqual((self.out["name"], self.out["version"]), ("antigravity", "0.1.23"))
        self.assertIn("ValueError", self.out["bad_effort"])
        self.assertIn("minimal, low, medium, high, xhigh", self.out["bad_effort"])

    def test_install(self):
        calls = self.out["install"]
        cmds = [c["exec"] for c in calls if "exec" in c]
        self.assertTrue(any("https://astral.sh/uv/0.11.30/install.sh" in c and "UV_UNMANAGED_INSTALL=/installed-agent/uv/bin" in c
                            for c in cmds), cmds)
        uploads = {c["upload"]: c["text"] for c in calls if "upload" in c}
        self.assertEqual(uploads["/installed-agent/run_agent.py"], SCRIPT.read_text())
        self.assertEqual(uploads["/installed-agent/run_agent.py.lock"], (SCRIPT.parent / "run_agent.py.lock").read_text())
        (sync,) = [c for c in calls if "exec" in c and " sync --locked --script " in c["exec"]]
        self.assertEqual(sync["env"]["UV_CACHE_DIR"], "/installed-agent/uv/cache")
        self.assertNotIn("UV_OFFLINE", sync["env"])                       # the network is still there during setup
        self.assertIn("--version | grep -qx 0.1.23", sync["exec"])

    def test_run(self):
        calls = self.out["run"]
        uploads = [c for c in calls if "upload" in c]
        self.assertEqual([(u["upload"], u["text"]) for u in uploads], [("/tmp/antigravity-prompt/instruction.txt", self.out["task"])])
        (run,) = [c for c in calls if "exec" in c and " run --script " in c["exec"]]
        self.assertNotIn("TASKWORD", run["exec"])                           # pkill -f <a word of the task> cannot match it
        self.assertIn("--instruction-file=/tmp/antigravity-prompt/instruction.txt", run["exec"])
        self.assertIn("--locked --offline", run["exec"])
        env = run["env"]
        self.assertEqual((env["UV_OFFLINE"], env["MODEL_NAME"], env["REASONING_EFFORT"]), ("1", "google/gemini-4-argon", "high"))
        self.assertEqual(env["GEMINI_API_KEY"], "from-the-env-file-0123456789")
        self.assertEqual(env["GOOGLE_GEMINI_BASE_URL"], "http://gw:1/h/127.0.0.1:4141/v1")   # --ae wins

    def test_tokens_from_the_trajectory(self):
        self.assertEqual(self.out["tokens"], [1200, 80, 1000])


if __name__ == "__main__":
    unittest.main()
