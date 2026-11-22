"""The usage gateway this repository ships (scripts/usage_gateway/) against the runner that starts it
(scripts/harbor/run_agent_batch.sh, USAGE_GATEWAY=scripts/usage_gateway): no docker, no model, no network beyond the
loopback.

    python3 scripts/harbor/tests/test_usage_gateway_shipped.py     # stdlib only

- every option the runner passes to gateway.py is one the gateway takes;
- started as the runner starts it, the gateway writes its port to --ready-file and answers /_gateway/health;
- the gateway forwards to the model-API hosts of scripts/set_agent_egress.py (HOSTS without PACKAGE_HOSTS), and
  importing set_agent_egress.py runs nothing (its command line is under `if __name__ == "__main__"`).
"""
import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/harbor/run_agent_batch.sh"
GATEWAY_DIR = REPO / "scripts/usage_gateway"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ShippedGateway(unittest.TestCase):
    def test_runner_options_are_known(self):
        src = RUNNER.read_text()
        passed = set()
        for m in re.finditer(r"gw_args\+?=\(([^)]*)\)", src):
            passed |= set(re.findall(r"(--[a-z][a-z-]*)", m.group(1)))
        passed |= set(re.findall(r'\$\{1:\+(--[a-z-]+) "\$1"\}', src))           # the restart's --port
        self.assertTrue({"--jobs", "--bind", "--ready-file", "--budget-input", "--budget-output", "--port"} <= passed,
                        f"the runner's gateway arguments were not found: {sorted(passed)}")
        out = subprocess.run([sys.executable, str(GATEWAY_DIR / "gateway.py"), "--help"], capture_output=True,
                             text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        missing = sorted(o for o in passed if not re.search(rf"(?<![\w-]){re.escape(o)}(?![\w-])", out.stdout))
        self.assertEqual(missing, [], "options the runner passes but scripts/usage_gateway/gateway.py does not take")

    def test_starts_as_the_runner_starts_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            ready = tmp / "port"
            env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}
            p = subprocess.Popen([sys.executable, str(GATEWAY_DIR / "gateway.py"), "--jobs", str(tmp / "jobs"),
                                  "--bind", "127.0.0.1", "--ready-file", str(ready), "--budget-input", "100000000",
                                  "--budget-output", "10000000", "--raw"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
            try:
                for _ in range(150):
                    if ready.is_file() and ready.read_text().strip():
                        break
                    self.assertIsNone(p.poll(), p.stderr.read().decode() if p.poll() is not None else "")
                    time.sleep(0.1)
                self.assertTrue(ready.is_file(), "no port in --ready-file within 15 s")
                url = f"http://127.0.0.1:{int(ready.read_text())}/_gateway/health"
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=15) as r:
                    self.assertEqual(r.status, 200)
            finally:
                p.terminate()
                p.wait(timeout=30)
                p.stderr.close()

    def test_model_hosts_are_the_egress_lists(self):
        argv = sys.argv[:]
        try:
            sys.argv = ["importer", "not-a-mode"]          # a command line run on import would exit on this
            egress = load("set_agent_egress", REPO / "scripts/set_agent_egress.py")
        finally:
            sys.argv = argv
        sys.path.insert(0, str(GATEWAY_DIR))
        try:
            gateway = load("usage_gateway_under_test", GATEWAY_DIR / "gateway.py")
        finally:
            sys.path.remove(str(GATEWAY_DIR))
        self.assertEqual(gateway.MODEL_HOSTS, [h for h in egress.HOSTS if h not in egress.PACKAGE_HOSTS])


if __name__ == "__main__":
    unittest.main(verbosity=1)
