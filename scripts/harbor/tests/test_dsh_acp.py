"""Tests for DSH, the DeepSeek Harness, in the batch runner (scripts/harbor/agents/dsh_acp.py, dsh/acp_prompt_file.py): no
docker, no model, no network.

    python3 scripts/harbor/tests/test_dsh_acp.py     # stdlib only (Python 3.11+); the DshAcp class check needs Harbor

- run_agent_batch.sh with AGENT=dsh: Harbor gets `-a dsh_acp:DshAcp` with scripts/harbor/agents on its PYTHONPATH, the
  effort, no web tools (web=false; WEB_SEARCH=allowed: web=true) and DeepSeek's API host on the job's egress; the run id
  is the exam's, dsh-deepseek_v4_1_flash-high; AGENT_CONFIG is refused;
- acp_prompt_file.py: the runner gets the whole instruction while the shim's own command line carries none of it, and
  the session's reasoning_effort is set after session/new and after a model change (skipped, and recorded, when the
  session does not offer the value);
- with Harbor importable (HARBOR_PYTHON, or the interpreter of `harbor` on PATH): DshAcp's registry entry, pinned version,
  model id and launcher.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHIM = REPO / "scripts" / "harbor" / "agents" / "dsh" / "acp_prompt_file.py"
sys.path.insert(0, str(HERE))
import test_runner_seeds as trs  # noqa: E402

FAKE_ACP = '''
import contextlib, types
calls = []
VALUES = {values!r}
def opts(effort):
    return [types.SimpleNamespace(id="model", options=[types.SimpleNamespace(value="deepseek-official/deepseek-flash")]),
            types.SimpleNamespace(id="reasoning_effort", options=[types.SimpleNamespace(value=v) for v in effort])]
class Conn:
    async def new_session(self, cwd=None, mcp_servers=None):
        calls.append(["new_session"])
        return types.SimpleNamespace(session_id="s1", config_options=opts(VALUES))
    async def set_config_option(self, session_id, config_id, value):
        calls.append(["set", config_id, value])
        return types.SimpleNamespace(config_options=opts(VALUES))
@contextlib.asynccontextmanager
async def spawn_agent_process(client, launcher, **kw):
    yield Conn(), None
'''

FAKE_RUNNER = '''
import asyncio, json, os, sys
from acp import spawn_agent_process
import acp
async def main():
    async with spawn_agent_process(None, "launcher") as (conn, _):
        s = await conn.new_session(cwd="/app", mcp_servers=[])
        await conn.set_config_option(session_id=s.session_id, config_id="model", value="deepseek-official/deepseek-flash")
    cmdline = open(f"/proc/{os.getpid()}/cmdline", "rb").read().replace(b"\\0", b" ").decode()
    print(json.dumps({"instruction": sys.argv[1], "rest": sys.argv[2:], "cmdline": cmdline, "calls": acp.calls}))
if __name__ == "__main__":
    asyncio.run(main())
'''


class Shim(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_shim(self, values, effort="high"):
        (self.root / "acp").mkdir(exist_ok=True)
        (self.root / "acp" / "__init__.py").write_text(FAKE_ACP.format(values=values))
        (self.root / "runner.py").write_text(FAKE_RUNNER)
        (self.root / "logs").mkdir(exist_ok=True)
        task = "Thread the rope through the TASKWORD ring.\n" * 40
        (self.root / "instruction.txt").write_text(task)
        p = subprocess.run([sys.executable, str(SHIM), f"--instruction-file={self.root / 'instruction.txt'}",
                            f"--runner={self.root / 'runner.py'}", f"--reasoning-effort={effort}", "--",
                            f"--logs-dir={self.root / 'logs'}", "--launcher=/installed-agent/acp-launch.sh"],
                           capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(self.root)), timeout=60)
        self.assertEqual(p.returncode, 0, p.stderr)
        return json.loads(p.stdout.strip().splitlines()[-1]), task

    def test_instruction_off_the_command_line(self):
        out, task = self.run_shim(["low", "medium", "high"])
        self.assertEqual(out["instruction"], "--instruction=" + task)
        self.assertNotIn("TASKWORD", out["cmdline"])                  # pkill -f <a word of the task> cannot match it
        self.assertEqual(out["rest"][1], "--launcher=/installed-agent/acp-launch.sh")

    def test_effort_after_session_and_model(self):
        out, _ = self.run_shim(["low", "medium", "high"])
        self.assertEqual(out["calls"], [["new_session"], ["set", "reasoning_effort", "high"],
                                        ["set", "model", "deepseek-official/deepseek-flash"],
                                        ["set", "reasoning_effort", "high"]])
        log = json.loads((self.root / "logs" / "dsh-effort.json").read_text())
        self.assertEqual([a.get("set") for a in log["attempts"]], ["high", "high"])

    def test_effort_not_offered(self):
        out, _ = self.run_shim(["low", "medium"])
        self.assertNotIn(["set", "reasoning_effort", "high"], out["calls"])
        log = json.loads((self.root / "logs" / "dsh-effort.json").read_text())
        self.assertTrue(all("skipped" in a for a in log["attempts"]))


class Runner(unittest.TestCase):
    """run_agent_batch.sh with AGENT=dsh, through test_runner_seeds' fixture repository and fake harbor."""

    setUp = trs.RunnerSeeds.setUp
    tearDown = trs.RunnerSeeds.tearDown
    calls = trs.RunnerSeeds.calls

    @classmethod
    def setUpClass(cls):
        cls.multi, cls.single = trs.pair_bases()

    def run_batch(self, *bases, ok=True, **env):
        p = subprocess.run(["bash", str(self.repo / "scripts/harbor/run_agent_batch.sh"), "b", *bases],
                           env=dict(self.env, **env), capture_output=True, text=True, timeout=900, cwd=self.root)
        if ok:
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def test_dsh_arguments(self):
        self.run_batch(self.single, MODES="privileged", AGENT="dsh", MODEL="deepseek/deepseek-v4.1-flash", EFFORT="high")
        (c,) = self.calls()
        a = c["argv"]
        self.assertEqual(a[a.index("-a") + 1], "dsh_acp:DshAcp")
        self.assertEqual(a[a.index("-m") + 1], "deepseek/deepseek-v4.1-flash")
        aks = [a[i + 1] for i, x in enumerate(a) if x == "--ak"]
        self.assertIn("reasoning_effort=high", aks)
        self.assertIn("web=false", aks)
        self.assertEqual(a[a.index("--allow-agent-host") + 1], "api.deepseek.com")
        self.assertIn(str(self.repo / "scripts" / "harbor" / "agents"), c["pythonpath"].split(":"))
        self.assertTrue((self.root / "results" / "b" / f"{self.single}-privileged" / "dsh-deepseek_v4_1_flash-high").is_dir())

    def test_dsh_web_allowed_and_config_refused(self):
        self.run_batch(self.single, MODES="privileged", AGENT="dsh", MODEL="deepseek/deepseek-v4.1-flash",
                       WEB_SEARCH="allowed")
        (c,) = self.calls()
        self.assertIn("web=true", [c["argv"][i + 1] for i, x in enumerate(c["argv"]) if x == "--ak"])
        p = self.run_batch(self.single, ok=False, AGENT="dsh", MODEL="deepseek/deepseek-v4.1-flash",
                           AGENT_CONFIG=str(self.root / "c.toml"))
        self.assertEqual(p.returncode, 2)
        self.assertIn("dsh takes none", p.stderr)


def harbor_python():
    if os.environ.get("HARBOR_PYTHON"):
        return os.environ["HARBOR_PYTHON"]
    exe = shutil.which("harbor")
    if exe:
        first = pathlib.Path(exe).read_text(errors="replace").splitlines()[0]
        if first.startswith("#!"):
            return first[2:].strip()
    return None


class HarborClass(unittest.TestCase):
    def test_dsh_acp_class(self):
        py = harbor_python()
        if not py or subprocess.run([py, "-c", "import harbor"], capture_output=True).returncode != 0:
            self.skipTest("Harbor is not importable (set HARBOR_PYTHON to its interpreter)")
        code = textwrap.dedent(f"""
            import json, sys, tempfile, pathlib
            sys.path.insert(0, {str(REPO / 'scripts' / 'harbor' / 'agents')!r})
            from dsh_acp import DshAcp
            d = pathlib.Path(tempfile.mkdtemp())
            a = DshAcp(logs_dir=d, model_name="deepseek/deepseek-v4.1-flash", reasoning_effort="high")
            w = DshAcp(logs_dir=d, model_name="deepseek-v4-pro", web="true", version="0.2.1-rc.4")
            try:
                DshAcp(logs_dir=d, bogus=1); refused = False
            except ValueError:
                refused = True
            print(json.dumps({{"name": a.name(), "version": a._version, "cmd": a._registry_entry.distribution.local.cmd,
                               "model": a._requested_model(), "launcher": a._launcher(), "web": w.options.web,
                               "w_version": w._version, "w_launcher": w._launcher(), "refused": refused}}))
        """)
        p = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr)
        out = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual((out["name"], out["version"], out["cmd"]), ("dsh", "0.2.1-rc.2", "/installed-agent/dsh-acp.sh"))
        self.assertEqual(out["model"], "deepseek/deepseek-flash")          # dsh's catalog id for DeepSeek-V4.1-Flash
        self.assertIn("dsh --profile acp --patch /installed-agent/dsh-roboarc.patch.yml", out["launcher"])
        self.assertEqual((out["web"], out["w_version"]), (True, "0.2.1-rc.4"))
        self.assertNotIn("--patch", out["w_launcher"])
        self.assertTrue(out["refused"])
        patch = (REPO / "scripts" / "harbor" / "agents" / "dsh" / "roboarc.patch.yml").read_text()
        self.assertIn("- id: tool-web", patch)
        self.assertIn("search: false", patch)
        self.assertIn("fetch: false", patch)


if __name__ == "__main__":
    unittest.main()
