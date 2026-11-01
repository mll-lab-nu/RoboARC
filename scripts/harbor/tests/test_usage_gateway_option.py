"""Tests for run_agent_batch.sh's optional usage recording (USAGE_GATEWAY, docs/RUNNING.md "Usage recording"), with a
fake `harbor` and a fake gateway: no docker, no model, no network.

    python3 scripts/harbor/tests/test_usage_gateway_option.py     # stdlib only (Python 3.11+, as the runner's helpers)

- off (the default): the agent's base URL and the task's egress are left alone, no gateway directory is made, the job's
  usage_recording.json and the result's config.json say that no gateway ran, the token counts are the harness's;
- on: the agent's base URL points at the gateway, the job's task copy allows the gateway and the package registries only,
  and the records say so (with the token budget and the image cap, 50 by default, which the gateway is given);
- on with AGENT=dsh: dsh's Messages root (DEEPSEEK_BASE_URL) points at the gateway, which forwards to DeepSeek's API
  (--extra-host api.deepseek.com), and the job's egress has the gateway, not DeepSeek's host;
- on with AGENT=antigravity: the SDK's Gemini base URL (GOOGLE_GEMINI_BASE_URL) is the gateway's route to the Chat
  Completions server of ANTIGRAVITY_BASE_URL, the gateway translates (--gemini-chat) and may forward to exactly that
  plain-HTTP server, the records name the translation (api_translation); AGENT=antigravity without the gateway, with an
  EFFORT the SDK has no level for, with WEB_SEARCH=allowed or AGENT_CONFIG, or without a plain-HTTP ANTIGRAVITY_BASE_URL
  stops the runner at start;
- USAGE_GATEWAY naming a directory without the gateway, a GATEWAY_IMAGE_CAP that is not a whole number, and the switch
  of earlier versions (GATEWAY=1), stop the runner at start, before any job.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/harbor/run_agent_batch.sh"

FAKE_HARBOR = '''#!{py}
"""A stand-in for `harbor run`: records its arguments and the task copy's egress, writes one finished trial."""
import json, os, pathlib, re, sys
a = sys.argv[1:]
val = lambda k: a[a.index(k) + 1]
task, jobs, job = val("-i"), pathlib.Path(val("-o")), val("--job-name")
toml = (pathlib.Path(val("-p")) / task / "task.toml").read_text()
hosts = re.search(r"(?m)^allowed_hosts = (.*)$", toml)
with open(os.environ["FAKE_HARBOR_LOG"], "a") as f:
    f.write(json.dumps({{"argv": a, "allowed_hosts": json.loads(hosts.group(1)) if hosts else None}}) + "\\n")
t = jobs / job / f"{{task}}__fake01"
(t / "verifier").mkdir(parents=True)
(t / "result.json").write_text(json.dumps({{
    "task_name": f"robot-coding-bench/{{task}}", "started_at": "2026-10-19T00:00:00Z",
    "agent_execution": {{"started_at": "2026-10-19T00:00:10Z", "finished_at": "2026-10-19T00:01:10Z"}},
    "agent_result": {{"n_input_tokens": 1000, "n_cache_tokens": 0, "n_output_tokens": 50, "cost_usd": None}}}}))
(t / "verifier" / "reward.json").write_text(json.dumps({{"success": 0}}))
if os.environ.get("FAKE_GATEWAY_TOTALS") == "1":
    (t / "gateway").mkdir()
    (t / "gateway" / "totals.json").write_text(json.dumps({{"input": 900, "output": 40, "requests": 3}}))
'''

FAKE_GATEWAY = '''
"""A stand-in for the external usage gateway: listens, writes its port to --ready-file, answers every GET with 200."""
import argparse, http.server
p = argparse.ArgumentParser()
p.add_argument("--bind"); p.add_argument("--ready-file"); p.add_argument("--port", type=int, default=0)
args, _ = p.parse_known_args()
import sys; open(args.ready_file + ".argv", "w").write(" ".join(sys.argv[1:]))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a):
        pass
s = http.server.ThreadingHTTPServer((args.bind, args.port), H)
open(args.ready_file, "w").write(str(s.server_address[1]))
s.serve_forever()
'''


def a_task():
    """A privileged task of the checkout and its base name."""
    for d in sorted((REPO / "tasks").glob("roboarc-*-privileged")):
        if (d / "task.toml").is_file():
            return d.name, d.name.removesuffix("-privileged")
    raise unittest.SkipTest("no privileged task in tasks/")


class UsageGatewayOption(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        harbor = bin_dir / "harbor"
        harbor.write_text(FAKE_HARBOR.format(py=sys.executable))
        harbor.chmod(0o755)
        self.log = self.root / "harbor.jsonl"
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("GATEWAY", "USAGE_GATEWAY", "TOKEN_BUDGET", "OPENAI_", "ANTHROPIC_", "CODEX_", "DEEPSEEK_",
                                            "ANTIGRAVITY_", "GEMINI_", "GOOGLE_GEMINI_"))}
        self.env.update(PATH=f"{bin_dir}:{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}",
                        FAKE_HARBOR_LOG=str(self.log), MODEL="openai/test-model", EFFORT="high", MODES="privileged",
                        JOBS=str(self.root / "jobs"), RESULTS=str(self.root / "results"),
                        SLOTS_DIR=str(self.root / "slots"), GPUS="0")
        self.task, self.base = a_task()

    def tearDown(self):
        self.tmp.cleanup()

    def run_batch(self, **env):
        e = dict(self.env, **env)
        return subprocess.run(["bash", str(RUNNER), "b", self.base], env=e, capture_output=True, text=True, timeout=600,
                              cwd=self.root)

    def calls(self):
        return [json.loads(x) for x in self.log.read_text().splitlines()] if self.log.exists() else []

    def results_config(self, run_id="codex-test_model-high"):
        return json.loads((self.root / "results" / "b" / self.task / run_id / "t1" / "config.json").read_text())

    def test_off_by_default(self):
        p = self.run_batch()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        (call,) = self.calls()
        argv = " ".join(call["argv"])
        self.assertNotIn("OPENAI_BASE_URL", argv)
        self.assertNotIn("--allow-agent-host", argv)
        self.assertNotIn("roboarc_docker", argv)         # Harbor's own Docker environment without the gateway
        own = re.search(r"(?m)^allowed_hosts = (.*)$", (REPO / "tasks" / self.task / "task.toml").read_text())
        self.assertEqual(call["allowed_hosts"], json.loads(own.group(1)) if own else None)
        jobs = self.root / "jobs"
        self.assertFalse((jobs / ".gateway").exists())
        job = jobs / f"b-privileged-{self.base}"
        self.assertEqual(json.loads((job / "usage_recording.json").read_text()),
                         {"usage_gateway": False, "token_budget": None, "image_cap": None})
        cfg = self.results_config()
        self.assertEqual((cfg["usage_gateway"], cfg["usage_source"], cfg["usage_recorded"], cfg["token_budget"],
                          cfg["image_cap"]), (False, "harness", True, None, None))
        summary = (jobs / "b.summary.md").read_text()
        self.assertNotIn("usage gateway", summary)
        self.assertIn("usage gateway off", (jobs / "b.log").read_text())

    def test_on(self):
        gw = self.root / "gw"
        gw.mkdir()
        (gw / "gateway.py").write_text(textwrap.dedent(FAKE_GATEWAY))
        (gw / "rawlog.py").write_text("")
        p = self.run_batch(USAGE_GATEWAY=str(gw), GATEWAY_BIND="127.0.0.1", FAKE_GATEWAY_TOTALS="1")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        (call,) = self.calls()
        argv = call["argv"]
        base_urls = [x for x in argv if x.startswith("OPENAI_BASE_URL=")]
        self.assertEqual(len(base_urls), 1, argv)
        self.assertRegex(base_urls[0], r"^OPENAI_BASE_URL=http://127\.0\.0\.1:\d+/u/api\.openai\.com/v1$")
        self.assertIn("127.0.0.1", argv[argv.index("--allow-agent-host") + 1])
        # the environment whose egress sidecar waits as long as the gateway (roboarc_docker.py)
        self.assertEqual(argv[argv.index("-e") + 1], "roboarc_docker:RoboARCDocker", argv)
        self.assertEqual(call["allowed_hosts"][0], "127.0.0.1")
        self.assertNotIn("api.openai.com", call["allowed_hosts"])
        job = self.root / "jobs" / f"b-privileged-{self.base}"
        rec = json.loads((job / "usage_recording.json").read_text())
        self.assertEqual(rec, {"usage_gateway": True, "raw_bodies": True,
                               "token_budget": {"input": 100000000, "output": 10000000}, "image_cap": 50})
        (ready,) = (self.root / "jobs" / ".gateway").glob("port.*.argv")
        self.assertIn("--image-cap 50", ready.read_text())          # the default cap reaches the gateway
        cfg = self.results_config()
        self.assertEqual((cfg["usage_gateway"], cfg["usage_source"], cfg["usage_recorded"]), (True, "usage gateway", True))
        self.assertEqual(cfg["token_budget"], {"input": 100000000, "output": 10000000})
        self.assertEqual(cfg["image_cap"], 50)
        summary = (self.root / "jobs" / "b.summary.md").read_text()
        self.assertIn("usage gateway in / out", summary)
        self.assertIn("900 / 40", summary)

    def test_on_dsh(self):
        gw = self.root / "gw"
        gw.mkdir()
        (gw / "gateway.py").write_text(textwrap.dedent(FAKE_GATEWAY))
        (gw / "rawlog.py").write_text("")
        p = self.run_batch(USAGE_GATEWAY=str(gw), GATEWAY_BIND="127.0.0.1", AGENT="dsh", MODEL="deepseek/deepseek-v4.1-flash")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        (call,) = self.calls()
        argv = call["argv"]
        self.assertEqual(argv[argv.index("-a") + 1], "dsh_acp:DshAcp")
        base_urls = [x for x in argv if x.startswith("DEEPSEEK_BASE_URL=")]
        self.assertEqual(len(base_urls), 1, argv)
        self.assertRegex(base_urls[0], r"^DEEPSEEK_BASE_URL=http://127\.0\.0\.1:\d+/u/api\.deepseek\.com/anthropic$")
        hosts = [argv[i + 1] for i, x in enumerate(argv) if x == "--allow-agent-host"]
        self.assertEqual(hosts, ["127.0.0.1"])                      # through the gateway, never DeepSeek directly
        self.assertNotIn("api.deepseek.com", call["allowed_hosts"])
        (ready,) = (self.root / "jobs" / ".gateway").glob("port.*.argv")
        self.assertIn("--extra-host api.deepseek.com", ready.read_text())
        self.assertTrue((self.root / "results" / "b" / self.task / "dsh-deepseek_v4_1_flash-high").is_dir())

    def gateway_dir(self) -> pathlib.Path:
        gw = self.root / "gw"
        gw.mkdir(exist_ok=True)
        (gw / "gateway.py").write_text(textwrap.dedent(FAKE_GATEWAY))
        (gw / "rawlog.py").write_text("")
        return gw

    def test_on_antigravity(self):
        p = self.run_batch(USAGE_GATEWAY=str(self.gateway_dir()), GATEWAY_BIND="127.0.0.1", AGENT="antigravity",
                           MODEL="google/gemini-4-argon", ANTIGRAVITY_BASE_URL="http://127.0.0.1:4141/v1",
                           GEMINI_API_KEY="k" * 20, FAKE_GATEWAY_TOTALS="1")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        (call,) = self.calls()
        argv = call["argv"]
        self.assertEqual((argv[argv.index("-a") + 1], argv[argv.index("-m") + 1]), ("antigravity_sdk:AntigravitySdk", "google/gemini-4-argon"))
        aks = [argv[i + 1] for i, x in enumerate(argv) if x == "--ak"]
        self.assertEqual(aks, ["reasoning_effort=high"])            # no web option: the tools are off in run_agent.py
        base_urls = [x for x in argv if "BASE_URL=" in x]
        self.assertEqual(len(base_urls), 1, argv)
        self.assertRegex(base_urls[0], r"^GOOGLE_GEMINI_BASE_URL=http://127\.0\.0\.1:\d+/h/127\.0\.0\.1:4141/v1$")
        hosts = [argv[i + 1] for i, x in enumerate(argv) if x == "--allow-agent-host"]
        self.assertEqual(hosts, ["127.0.0.1"])                      # the gateway, never the model server directly
        (ready,) = (self.root / "jobs" / ".gateway").glob("port.*.argv")
        self.assertIn("--gemini-chat", ready.read_text().split())
        self.assertIn("--http-host 127.0.0.1:4141", ready.read_text())
        job = self.root / "jobs" / f"b-privileged-{self.base}"
        self.assertEqual(json.loads((job / "usage_recording.json").read_text())["api_translation"], "gemini-chat")
        cfg = self.results_config("antigravity-gemini_4_argon-high")
        self.assertEqual((cfg["usage_gateway"], cfg["api_translation"]), (True, "gemini-chat"))

    def test_antigravity_refusals(self):
        gw, url = str(self.gateway_dir()), "http://127.0.0.1:4141/v1"
        for env, msg in ((dict(ANTIGRAVITY_BASE_URL=url), "needs the usage gateway"),
                         (dict(USAGE_GATEWAY=gw, ANTIGRAVITY_BASE_URL=url, EFFORT="max"), "EFFORT for AGENT=antigravity"),
                         (dict(USAGE_GATEWAY=gw, ANTIGRAVITY_BASE_URL=url, WEB_SEARCH="allowed"), "WEB_SEARCH=allowed is not available"),
                         (dict(USAGE_GATEWAY=gw, ANTIGRAVITY_BASE_URL=url, AGENT_CONFIG=str(self.root / "c.toml")), "antigravity takes none"),
                         (dict(USAGE_GATEWAY=gw, ANTIGRAVITY_BASE_URL="https://proxy.example/v1"), "plain-HTTP base"),
                         (dict(USAGE_GATEWAY=gw), "plain-HTTP base")):
            p = self.run_batch(AGENT="antigravity", MODEL="google/gemini-4-argon", GATEWAY_BIND="127.0.0.1", **env)
            self.assertEqual(p.returncode, 2, (env, p.stdout + p.stderr))
            self.assertIn(msg, p.stdout + p.stderr, env)
            self.assertEqual(self.calls(), [], env)

    def test_missing_gateway_stops_at_start(self):
        empty = self.root / "empty"
        empty.mkdir()
        p = self.run_batch(USAGE_GATEWAY=str(empty))
        self.assertEqual(p.returncode, 2)
        self.assertIn("no gateway.py", p.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "jobs").exists())

    def test_bad_image_cap_stops_at_start(self):
        gw = self.root / "gw"
        gw.mkdir()
        (gw / "gateway.py").write_text(textwrap.dedent(FAKE_GATEWAY))
        p = self.run_batch(USAGE_GATEWAY=str(gw), GATEWAY_BIND="127.0.0.1", GATEWAY_IMAGE_CAP="fifty")
        self.assertEqual(p.returncode, 2)
        self.assertIn("GATEWAY_IMAGE_CAP", p.stderr)
        self.assertEqual(self.calls(), [])

    def test_old_switch_is_refused(self):
        p = self.run_batch(GATEWAY="1")
        self.assertEqual(p.returncode, 2)
        self.assertIn("USAGE_GATEWAY=", p.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
