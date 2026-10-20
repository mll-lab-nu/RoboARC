"""Tests for run_agent_batch.sh's optional usage recording (USAGE_GATEWAY, docs/RUNNING.md "Usage recording"), with a
fake `harbor` and a fake gateway: no docker, no model, no network.

    python3 scripts/harbor/tests/test_usage_gateway_option.py     # stdlib only (Python 3.11+, as the runner's helpers)

- off (the default): the agent's base URL and the task's egress are left alone, no gateway directory is made, the job's
  usage_recording.json and the result's config.json say that no gateway ran, the token counts are the harness's;
- on: the agent's base URL points at the gateway, the job's task copy allows the gateway and the package registries only,
  and the records say so (with the token budget);
- USAGE_GATEWAY naming a directory without the gateway, and the switch of earlier versions (GATEWAY=1), stop the
  runner at start, before any job.
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
                    if not k.startswith(("GATEWAY", "USAGE_GATEWAY", "TOKEN_BUDGET", "OPENAI_", "ANTHROPIC_", "CODEX_"))}
        self.env.update(PATH=f"{bin_dir}:{os.path.dirname(sys.executable)}:{os.environ.get('PATH', '')}",
                        FAKE_HARBOR_LOG=str(self.log), MODEL="openai/test-model", EFFORT="high", MODES="privileged",
                        JOBS=str(self.root / "jobs"), RESULTS=str(self.root / "results"))
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
        return json.loads((self.root / "results" / "b" / self.task / run_id / "config.json").read_text())

    def test_off_by_default(self):
        p = self.run_batch()
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        (call,) = self.calls()
        argv = " ".join(call["argv"])
        self.assertNotIn("OPENAI_BASE_URL", argv)
        self.assertNotIn("--allow-agent-host", argv)
        own = re.search(r"(?m)^allowed_hosts = (.*)$", (REPO / "tasks" / self.task / "task.toml").read_text())
        self.assertEqual(call["allowed_hosts"], json.loads(own.group(1)) if own else None)
        jobs = self.root / "jobs"
        self.assertFalse((jobs / ".gateway").exists())
        job = jobs / f"b-privileged-{self.base}"
        self.assertEqual(json.loads((job / "usage_recording.json").read_text()), {"usage_gateway": False, "token_budget": None})
        cfg = self.results_config()
        self.assertEqual((cfg["usage_gateway"], cfg["usage_source"], cfg["usage_recorded"], cfg["token_budget"]),
                         (False, "harness", True, None))
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
        self.assertEqual(call["allowed_hosts"][0], "127.0.0.1")
        self.assertNotIn("api.openai.com", call["allowed_hosts"])
        job = self.root / "jobs" / f"b-privileged-{self.base}"
        rec = json.loads((job / "usage_recording.json").read_text())
        self.assertEqual(rec, {"usage_gateway": True, "raw_bodies": True,
                               "token_budget": {"input": 100000000, "output": 10000000}})
        cfg = self.results_config()
        self.assertEqual((cfg["usage_gateway"], cfg["usage_source"], cfg["usage_recorded"]), (True, "usage gateway", True))
        self.assertEqual(cfg["token_budget"], {"input": 100000000, "output": 10000000})
        summary = (self.root / "jobs" / "b.summary.md").read_text()
        self.assertIn("usage gateway in / out", summary)
        self.assertIn("900 / 40", summary)

    def test_missing_gateway_stops_at_start(self):
        empty = self.root / "empty"
        empty.mkdir()
        p = self.run_batch(USAGE_GATEWAY=str(empty))
        self.assertEqual(p.returncode, 2)
        self.assertIn("no gateway.py", p.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / "jobs").exists())

    def test_old_switch_is_refused(self):
        p = self.run_batch(GATEWAY="1")
        self.assertEqual(p.returncode, 2)
        self.assertIn("USAGE_GATEWAY=", p.stderr)
        self.assertEqual(self.calls(), [])


if __name__ == "__main__":
    unittest.main()
