"""Tests for the gateway's model-API path filter (gateway.py, api_paths.json): default deny, per provider.

    python3 scripts/usage_gateway/tests/test_paths.py     # stdlib only; the end-to-end part starts gateway.py on loopback
"""
from __future__ import annotations

import http.client
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import gateway  # noqa: E402

gateway.PROVIDERS[:] = gateway.load_paths()


def ok(host: str, method: str, path: str, plain: bool = False) -> bool:
    return gateway.path_allowed(gateway.provider_of(host, plain), method, path)


class Rules(unittest.TestCase):
    def test_model_calls_of_recorded_runs(self):
        """Every (host, method, path) the gateway recorded in official runs up to 2026-10-10."""
        for host, method, path in [
                ("chatgpt.com", "POST", "/backend-api/codex/responses"),
                ("chatgpt.com", "GET", "/backend-api/codex/models?client_version=0.160.1"),
                ("api.openai.com", "POST", "/v1/responses"),
                ("res-1.cognitiveservices.azure.com", "POST", "/openai/v1/responses"),
                ("res-1.openai.azure.com", "POST", "/openai/v1/responses?api-version=preview"),
                ("res-1.services.ai.azure.com", "POST", "/openai/v1/responses"),
                ("api.anthropic.com", "POST", "/v1/messages?beta=true"),
                ("api.anthropic.com", "POST", "/v1/messages/count_tokens?beta=true"),
                ("api.deepseek.com", "POST", "/anthropic/v1/messages")]:              # DSH (dsh_acp.py)
            self.assertTrue(ok(host, method, path), (host, method, path))
        for path in ("/v1/messages", "/v1/messages/count_tokens", "/v1/chat/completions"):
            self.assertTrue(ok("10.0.0.5:8080", "POST", path, plain=True), path)

    def test_chat_completions_only_on_a_self_hosted_upstream(self):
        """At the providers that API also serves models that search without a tool declaration: not forwarded there."""
        for host, path in (("api.openai.com", "/v1/chat/completions"), ("openrouter.ai", "/api/v1/chat/completions"),
                           ("res-1.openai.azure.com", "/openai/deployments/gpt-6/chat/completions"),
                           ("res-1.openai.azure.com", "/openai/v1/chat/completions")):
            self.assertFalse(ok(host, "POST", path), (host, path))
        self.assertTrue(ok("10.0.0.5:8080", "POST", "/v1/chat/completions", plain=True))

    def test_moonshot(self):
        """Kimi Code (AGENT=kimi-code): Moonshot's chat completions, model list and image or video uploads; nothing else."""
        for host in ("api.moonshot.ai", "api.moonshot.cn"):
            for method, path in (("POST", "/v1/chat/completions"), ("GET", "/v1/models"), ("POST", "/v1/files")):
                self.assertTrue(ok(host, method, path), (host, method, path))
            for method, path in (("GET", "/v1/files"), ("GET", "/v1/files/f-1/content"), ("POST", "/v1/files/f-1"),
                                 ("POST", "/v1/batches"), ("GET", "/v1/users/me/balance"), ("POST", "/v1/responses"),
                                 ("POST", "/v1/tokenizers/estimate-token-count"), ("POST", "/anthropic/v1/messages")):
                self.assertFalse(ok(host, method, path), (host, method, path))
        self.assertFalse(ok("api.kimi.com", "POST", "/coding/v1/chat/completions"))     # another host: no list
        moon = gateway.provider_of("api.moonshot.ai", False)
        self.assertEqual((moon["uploads"], moon["record_fields"]), ({"/v1/files": {"image", "video"}}, ["thinking", "reasoning_effort"]))
        self.assertTrue(all(p["uploads"] == {} and p["record_fields"] == [] for p in gateway.PROVIDERS if p["name"] != "moonshot"))
        doc = json.loads(gateway.API_PATHS.read_text())     # record_fields are recorded, never provider-side fields
        self.assertEqual(gateway.PROVIDER_FIELDS, set(doc["provider_side_fields"]))

    def test_upload_purpose(self):
        def form(*parts, boundary="----formdata-undici-0123"):
            body = b"".join(b"--" + boundary.encode() + b"\r\n" + h + b"\r\n\r\n" + v + b"\r\n" for h, v in parts)
            return body + b"--" + boundary.encode() + b"--\r\n", f"multipart/form-data; boundary={boundary}"
        png = (b'Content-Disposition: form-data; name="file"; filename="cam.png"\r\nContent-Type: image/png', b"\x89PNG\r\n\x1a\n\x00")
        up = gateway.upload_purpose
        self.assertEqual(up(*form(png, (b'Content-Disposition: form-data; name="purpose"', b"image"))), "image")
        self.assertEqual(up(*form((b'content-disposition: form-data; name="purpose"', b"video"), png)), "video")
        self.assertEqual(up(*form(png, (b'Content-Disposition: form-data; name="purpose"', b"file-extract"))), "file-extract")
        self.assertIsNone(up(*form(png)))                                                       # no purpose
        self.assertIsNone(up(*form((b'Content-Disposition: form-data; name="purpose"', b"image"),
                                   (b'Content-Disposition: form-data; name="purpose"', b"file-extract"), png)))   # two
        self.assertIsNone(up(*form((b'Content-Disposition: form-data; name="purposes"', b"image"), png)))
        body, _ = form(png, (b'Content-Disposition: form-data; name="purpose"', b"image"))
        self.assertIsNone(up(body, "application/json"))
        self.assertIsNone(up(body, "multipart/form-data"))                                     # no boundary
        self.assertEqual(up(body, 'multipart/form-data; charset=utf-8; boundary="----formdata-undici-0123"'), "image")

    def test_a_star_is_one_segment(self):
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "p.json"
            f.write_text(json.dumps({"client_tool_types": ["function", "custom", "namespace"], "provider_side_fields": ["mcp_servers"],
                                     "providers": [{"name": "x", "hosts": ["h.example"], "allow": [["POST", "/d/*/chat"]]}]}))
            saved = (list(gateway.PROVIDERS), set(gateway.CLIENT_TOOLS), set(gateway.PROVIDER_FIELDS))
            try:
                gateway.PROVIDERS[:] = gateway.load_paths(f)
                self.assertTrue(ok("h.example", "POST", "/d/gpt-6/chat"))
                for path in ("/d/a/b/chat", "/d/../chat", "/d//chat", "/d/chat", "/d/*/chat/x"):
                    self.assertFalse(ok("h.example", "POST", path), path)
            finally:
                gateway.PROVIDERS[:] = saved[0]
                gateway.CLIENT_TOOLS.clear(); gateway.CLIENT_TOOLS.update(saved[1])
                gateway.PROVIDER_FIELDS.clear(); gateway.PROVIDER_FIELDS.update(saved[2])

    def test_everything_else_is_refused(self):
        for host, method, path in [
                ("chatgpt.com", "POST", "/backend-api/wham/apps"),                  # account connectors (apps)
                ("chatgpt.com", "GET", "/backend-api/wham/usage"),
                ("chatgpt.com", "GET", "/backend-api/wham/settings/user"),
                ("chatgpt.com", "GET", "/backend-api/wham/accounts/check"),
                ("chatgpt.com", "POST", "/backend-api/codex/analytics-events/events"),
                ("chatgpt.com", "POST", "/backend-api/wham/tasks"),
                ("chatgpt.com", "GET", "/backend-api/plugins/list"),
                ("chatgpt.com", "POST", "/backend-api/codex/alpha/search"),         # Codex's own web search
                ("chatgpt.com", "POST", "/backend-api/conversation"),
                ("chatgpt.com", "POST", "/backend-api/files"),
                ("chatgpt.com", "GET", "/backend-api/codex/responses"),             # a listed path, another method
                ("chatgpt.com", "POST", "/backend-api/codex/models"),
                ("api.openai.com", "POST", "/v1/files"),
                ("api.openai.com", "POST", "/v1/images/generations"),
                ("api.openai.com", "GET", "/v1/responses/resp_123"),
                ("api.openai.com", "POST", "/v1/conversations"),
                ("api.openai.com", "POST", "/v1/vector_stores"),
                ("api.anthropic.com", "POST", "/v1/files"),
                ("api.anthropic.com", "POST", "/v1/messages/batches"),
                ("api.anthropic.com", "GET", "/api/oauth/profile"),
                ("api.anthropic.com", "POST", "/v1/mcp_servers"),
                ("auth.openai.com", "POST", "/oauth/token"),                        # an allowlisted host without a list
                ("ab.chatgpt.com", "POST", "/backend-api/codex/responses"),
                ("mcp.anthropic.com", "POST", "/v1/messages"),
                ("api.deepseek.com", "POST", "/anthropic/v1/files"),                # DSH's image uploads: sent inline
                ("api.deepseek.com", "GET", "/models"),
                ("api.deepseek.com", "POST", "/chat/completions"),
                ("unknown.example.com", "POST", "/v1/responses")]:
            self.assertFalse(ok(host, method, path), (host, method, path))

    def test_no_way_around_a_listed_path(self):
        for path in ("/backend-api/codex/responses/../../wham/apps", "/backend-api/codex/responses/%2e%2e/apps",
                     "/backend-api/codex//responses", "/backend-api/codex/responses/", "/backend-api/codex/./responses",
                     "/backend-api/codex/responses;x=1", "/backend-api/codex/responses\\..\\apps",
                     "/BACKEND-API/codex/responses", "/backend-api/codex/responses/compact/more", "", "/",
                     "/backend-api/codex/responses%3Fx", "/backend-api/codex/*"):
            self.assertFalse(ok("chatgpt.com", "POST", path), path)
        self.assertFalse(ok("res-1.openai.azure.com", "POST", "/openai/v1/../responses"))
        self.assertFalse(ok("10.0.0.5:8080", "POST", "/v1/messages"))        # an HTTPS host without a provider
        self.assertFalse(ok("chatgpt.com", "POST", "/backend-api/codex/responses", plain=True))   # http: its own list only

    def test_bad_rule_files_do_not_load(self):
        with tempfile.TemporaryDirectory() as td:
            f = pathlib.Path(td) / "p.json"
            base = {"client_tool_types": ["function"], "provider_side_fields": ["mcp_servers"]}
            for allow in ([["PUT", "/v1/x"]], [["POST", "v1/x"]], [["POST", "/v1/x?y"]], [["POST", "/v1//x"]], [["POST"]]):
                f.write_text(json.dumps({**base, "providers": [{"name": "x", "hosts": ["h"], "allow": allow}]}))
                with self.assertRaises((ValueError, TypeError)):
                    gateway.load_paths(f)
            for doc in ({}, {"providers": []}, {**base, "client_tool_types": [1], "providers": []}):
                f.write_text(json.dumps(doc))
                with self.assertRaises((KeyError, ValueError)):
                    gateway.load_paths(f)
            for probe in (["POST", "/v1/files"], ["GET", "/backend-api/wham/apps"], ["HEAD", "/a?b"]):   # a probe carries nothing
                f.write_text(json.dumps({**base, "harness_probes": [probe], "providers": []}))
                with self.assertRaises(ValueError):
                    gateway.load_paths(f)
            self.assertEqual(gateway.CLIENT_TOOLS, {"function", "custom", "namespace"})      # a bad file changes nothing
            files = [["POST", "/v1/files"]]
            for extra in ({"uploads": {"/v1/upload": ["image"]}}, {"uploads": {"/v1/files": []}}, {"uploads": {"/v1/files": "image"}},
                          {"uploads": {"/v1/files": [""]}}, {"uploads": [["/v1/files", ["image"]]]}, {"record_fields": "thinking"},
                          {"record_fields": [1]}):
                f.write_text(json.dumps({**base, "providers": [{"name": "x", "hosts": ["h"], "allow": files, **extra}]}))
                with self.assertRaises(ValueError, msg=extra):
                    gateway.load_paths(f)

    def test_server_tools(self):
        st = gateway.server_tools
        # what the harnesses declare: Claude Code's tools carry no type; Codex 0.160 declares its tools inside `input`
        self.assertEqual(st(json.dumps({"tools": [{"name": "Bash", "input_schema": {}}, {"type": "function", "name": "x"},
                                                  {"type": "custom", "name": "apply_patch"}]})), [])
        codex = {"model": "m", "input": [{"type": "message", "content": [{"type": "input_text", "text": "the word \"tools\""}]},
                                         {"type": "additional_tools", "tools": [{"type": "namespace", "name": "functions", "tools": [
                                             {"type": "function", "name": "exec_command"}, {"type": "custom", "name": "apply_patch"}]}]}]}
        self.assertEqual(st(json.dumps(codex)), [])
        codex["input"][1]["tools"][0]["tools"].append({"type": "web_search"})
        self.assertEqual(st(json.dumps(codex)), ["web_search"])
        self.assertEqual(st(json.dumps({"tools": [{"type": "web_search"}, {"type": "function"}, {"type": "mcp"},
                                                  {"type": "web_search_20250305", "name": "web_search"},
                                                  {"type": "shell", "environment": {"type": "container_auto"}}]})),
                         ["mcp", "shell", "web_search", "web_search_20250305"])
        self.assertEqual(st('{"\\u0074ools": [{"type": "image_generation"}]}'), ["image_generation"])       # an escaped key
        self.assertEqual(st(json.dumps({"tools": [], "mcp_servers": [{"url": "https://x"}]})), ["mcp_servers"])
        self.assertEqual(st(json.dumps({"model": "gpt-search", "messages": [], "web_search_options": {}})), ["web_search_options"])
        self.assertEqual(st(json.dumps({"model": "openai/gpt-x:online", "messages": [], "plugins": [{"id": "web"}]})),
                         ["model:online", "plugins"])
        self.assertEqual(st(json.dumps({"model": "m", "mcp_servers": [], "plugins": None})), [])
        self.assertEqual(st(json.dumps({"model": "a", "models": ["b", "c:ONLINE "]})), ["model:online"])
        self.assertEqual(st(json.dumps({"model": "m", "input": "hi", "prompt": {"id": "pmpt_1"}})), ["prompt"])
        self.assertEqual(st(json.dumps({"tools": [{"type": ["web_search"]}, {"type": {"a": 1}}]})), ["['web_search']", "{'a': 1}"])
        self.assertEqual(st(json.dumps({"input": [{"type": "tool_search_output", "tools": [{"type": "mcp", "server_url": "https://x"}]}]})), ["mcp"])
        # data is not a declaration: a tool call's arguments, a tool's schema, a message that talks about tools
        data = {"model": "m", "tools": [{"name": "Agent", "input_schema": {"type": "object", "properties": {
                    "tools": {"type": "array", "items": {"type": "string"}}}}},
                    {"type": "function", "name": "f", "parameters": {"properties": {"tools": {"type": "web_search"}}}}],
                "messages": [{"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Agent", "input": {"tools": ["Bash", "Read"], "x": {"tools": "all"}}},
                    {"type": "tool_use", "id": "t2", "name": "Plan", "input": {"tools": [{"type": "hammer"}], "mcp_servers": [1],
                                                                               "source": {"type": "url", "url": "https://x"}}}]},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                                 {"type": "text", "text": "{\"tools\": [{\"type\": \"web_search\"}]}"},
                                 {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}]}]}],
                "input": [{"type": "function_call", "name": "x", "arguments": "{\"tools\": [{\"type\": \"mcp\"}]}"},
                          {"type": "message", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]}]}
        self.assertEqual(st(json.dumps(data)), [])
        # content the provider would fetch by address
        for block in ({"type": "image", "source": {"type": "url", "url": "https://x/a.png"}},
                      {"type": "document", "source": {"type": "url", "url": "https://x/a.pdf"}},
                      {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "image", "source": {"type": "url", "url": "http://x"}}]},
                      {"type": "image_url", "image_url": {"url": " HTTPS://x/a.png"}},
                      {"type": "input_image", "image_url": "https://x/a.png"}, {"type": "input_file", "file_url": "https://x/a.pdf"}):
            for key in ("messages", "input"):
                self.assertEqual(st(json.dumps({"model": "m", key: [{"role": "user", "content": [block]}]})), ["url_source"], (key, block))
        for item in ({"type": "function_call_output", "call_id": "c", "output": [{"type": "input_file", "file_url": "https://x/a.pdf"}]},
                     {"type": "custom_tool_call_output", "call_id": "c", "output": [{"type": "input_image", "image_url": "https://x/a.png"}]}):
            self.assertEqual(st(json.dumps({"model": "m", "input": [item]})), ["url_source"], item)
        nested = {"type": "document", "source": {"type": "content", "content": [{"type": "image", "source": {"type": "url", "url": "https://x"}}]}}
        self.assertEqual(st(json.dumps({"model": "m", "messages": [{"role": "user", "content": [nested]}]})), ["url_source"])
        ok_out = {"type": "function_call_output", "call_id": "c", "output": [{"type": "input_image", "image_url": "data:image/png;base64,AA"},
                                                                             {"type": "input_text", "text": "https://x is a string"}]}
        self.assertEqual(st(json.dumps({"model": "m", "input": [ok_out, {"type": "function_call_output", "call_id": "d", "output": "https://x"}]})), [])
        for body in ("", "not json", "[1]", '"x"', '{"tools": 5}', '{"tools": [1]}', '{"tools": [], "tools": [{"type": "mcp"}]}',
                     "\ufeff{}", "[" * 100000):
            with self.assertRaises(gateway.Unreadable, msg=body[:40]):
                st(body)

    def test_moonshot_tools(self):
        """Moonshot's own web search is a declared tool of type builtin_function: refused, on the request or on a message
        (Kimi Code can put its tools on the system message); its function tools pass."""
        st = gateway.server_tools
        fn = {"type": "function", "function": {"name": "Bash", "parameters": {"type": "object"}}}
        search = {"type": "builtin_function", "function": {"name": "$web_search"}}
        self.assertEqual(st(json.dumps({"model": "kimi-k3.1-pro", "messages": [], "tools": [fn],
                                        "thinking": {"type": "enabled", "effort": "max"}})), [])
        self.assertEqual(st(json.dumps({"model": "k", "messages": [], "tools": [fn, search]})), ["builtin_function"])
        self.assertEqual(st(json.dumps({"model": "k", "messages": [{"role": "system", "content": "s", "tools": [fn]}]})), [])
        self.assertEqual(st(json.dumps({"model": "k", "messages": [{"role": "system", "content": "s", "tools": [search]}]})),
                         ["builtin_function"])
        self.assertEqual(st(json.dumps({"model": "k", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "ms://file-1", "id": "file-1"}}]}]})), [])   # an uploaded image: the agent's

    def test_decode_request(self):
        import gzip
        dec = gateway.decode_request
        self.assertEqual(dec(b'{"a": 1}', ""), ('{"a": 1}', "text"))
        self.assertEqual(dec(gzip.compress(b'{"a": 1}'), "gzip"), ('{"a": 1}', "text"))
        self.assertIsNone(dec(gzip.compress(b"{}") + b"trailing", "gzip")[0])       # more than one gzip member's worth
        self.assertIsNone(dec(b'{"a": 1}', "gzip")[0])                               # the header lies
        for enc in ("br", "deflate", "x-gzip", "gzip, identity"):
            self.assertIsNone(dec(b"{}", enc)[0], enc)
        self.assertIsNone(dec(b"\xff\xfe{\x00}\x00", "")[0])                        # not UTF-8
        try:
            import zstandard
        except ModuleNotFoundError:
            return
        c = zstandard.ZstdCompressor()
        two = c.compress(b'{"tools": [') + c.compress(b'{"type": "mcp"}]}')         # two frames: all of them are read
        self.assertEqual(dec(two, "zstd"), ('{"tools": [{"type": "mcp"}]}', "text"))
        import shutil
        if shutil.which("zstd"):                                                    # without the module: the zstd program
            saved = sys.modules.get("zstandard")
            sys.modules["zstandard"] = None
            try:
                self.assertEqual(dec(two, "zstd"), ('{"tools": [{"type": "mcp"}]}', "text"))
            finally:
                sys.modules["zstandard"] = saved


class Stub(BaseHTTPRequestHandler):
    seen: list[tuple[str, str]] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        Stub.seen.append((self.command, self.path))
        body = json.dumps({"type": "message", "model": "stub", "usage": {"input_tokens": 3, "output_tokens": 2,
                                                                          "cache_read_input_tokens": 1}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    do_GET = do_POST


class EndToEnd(unittest.TestCase):
    """gateway.py as the runner starts it, in front of a stub model server on loopback (an --http-host upstream)."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.jobs = pathlib.Path(cls.td.name)
        cls.stub = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.up = f"127.0.0.1:{cls.stub.server_address[1]}"
        ready = cls.jobs / "ready"
        cls.proc = subprocess.Popen([sys.executable, str(HERE / "gateway.py"), "--jobs", str(cls.jobs), "--bind", "127.0.0.1",
                                     "--ready-file", str(ready), "--http-host", cls.up],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(100):
            if ready.exists() and ready.read_text():
                break
            if cls.proc.poll() is not None:
                raise RuntimeError(cls.proc.stderr.read().decode()[-2000:])
            time.sleep(0.1)
        cls.port = int(ready.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate(); cls.proc.wait(10); cls.proc.stderr.close()
        cls.stub.shutdown(); cls.stub.server_close(); cls.td.cleanup()

    def call(self, method: str, path: str, body: dict | None = None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        data = json.dumps(body).encode() if body is not None else None
        c.request(method, path, body=data, headers={"Content-Type": "application/json", "Authorization": "Bearer SECRET-KEY"})
        r = c.getresponse(); out = r.read(); c.close()
        return r.status, out

    def records(self) -> list[dict]:
        f = self.jobs / ".gateway" / "unattributed.jsonl"       # no trial owns loopback
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    def test_forwards_model_calls_and_refuses_the_rest(self):
        Stub.seen.clear()
        st, out = self.call("POST", f"/h/{self.up}/v1/messages?beta=true",
                            {"model": "m", "messages": [], "tools": [{"name": "Bash", "input_schema": {}},
                                                                     {"type": "custom", "name": "Edit"}]})
        self.assertEqual(st, 200, out)
        self.assertEqual(Stub.seen, [("POST", "/v1/messages?beta=true")])
        # a tool the provider would run for the agent: refused in the provider's error format, nothing forwarded
        for tools in ([{"type": "web_search_20250305", "name": "web_search"}], [{"type": "function", "name": "f"}, {"type": "mcp"}],
                      [{"type": "image_generation"}], [{"type": "web_search"}]):
            st, out = self.call("POST", f"/h/{self.up}/v1/messages", {"model": "m", "messages": [], "tools": tools})
            self.assertEqual(st, 403, out)
            self.assertIn("asks the model provider to run", json.loads(out)["error"]["message"])
        # a body the gateway cannot read is not forwarded either (one keep-alive connection for all of them)
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        for raw, enc in ((b"", None), (b"not json", None), (b'{"tools": [], "tools": [{"type": "mcp"}]}', None),
                         (b'{"model": "m"}', "br"), (b'{"model": "m"}', "gzip")):
            hdr = {"Content-Type": "application/json", **({"Content-Encoding": enc} if enc else {})}
            c.request("POST", f"/h/{self.up}/v1/messages", body=raw, headers=hdr)
            r = c.getresponse(); out = r.read()
            self.assertEqual(r.status, 403, (raw, enc, out))
            self.assertIn("not one JSON object", json.loads(out)["error"]["message"])
        c.request("POST", f"/h/{self.up}/v1/messages", body=b"{}", headers={"Content-Length": "abc"})   # answered, not dropped
        self.assertEqual(c.getresponse().status, 400)
        c.close()
        # two Content-Encoding headers (which one would the upstream believe?), a chunked body, a zstd label on other bytes
        import socket
        for head, body in ((b"Content-Encoding: identity\r\nContent-Encoding: gzip\r\nContent-Length: 2\r\n", b"{}"),
                           (b"Transfer-Encoding: chunked\r\n", b"2\r\n{}\r\n0\r\n\r\n"),
                           (b"Content-Encoding: zstd\r\nContent-Length: 14\r\n", b'{"model": "m"}')):
            with socket.create_connection(("127.0.0.1", self.port), timeout=30) as sk:
                sk.sendall(b"POST /h/" + self.up.encode() + b"/v1/messages HTTP/1.1\r\nHost: x\r\n" + head + b"\r\n" + body)
                self.assertTrue(sk.recv(65536).startswith(b"HTTP/1.1 403"), head)
        before = len(self.records())                 # Claude Code's own start-up probe: refused, but no record of a refusal
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("HEAD", f"/h/{self.up}/api/hello")
        r = c.getresponse(); r.read(); c.close()
        self.assertEqual((r.status, len(self.records())), (403, before))
        self.assertEqual(self.call("GET", f"/h/{self.up}/api/hello")[0], 403)          # only that method: this one is recorded
        self.assertEqual(len(self.records()), before + 1)
        for method in ("PUT", "DELETE", "PATCH"):
            self.assertEqual(self.call(method, f"/h/{self.up}/v1/messages", {} if method != "DELETE" else None)[0], 403)
        self.assertEqual(len(Stub.seen), 1, Stub.seen)
        st, _ = self.call("POST", f"/h/{self.up}/v1/chat/completions", {"model": "m", "messages": []})     # the self-hosted upstream
        self.assertEqual((st, Stub.seen[-1]), (200, ("POST", "/v1/chat/completions")))
        Stub.seen.pop()
        try:                                       # a zstd body, as Codex sends with a ChatGPT login: read, checked, forwarded as sent
            import zstandard
            z = zstandard.ZstdCompressor()
            for payload, want in (({"model": "m", "messages": []}, 200), ({"model": "m", "messages": [], "tools": [{"type": "mcp"}]}, 403)):
                c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
                c.request("POST", f"/h/{self.up}/v1/messages", body=z.compress(json.dumps(payload).encode()),
                          headers={"Content-Type": "application/json", "Content-Encoding": "zstd"})
                r = c.getresponse(); r.read(); c.close()
                self.assertEqual(r.status, want, payload)
            Stub.seen.pop()
        except ModuleNotFoundError:
            pass
        self.assertEqual(len(Stub.seen), 1, Stub.seen)
        for method, path in (("POST", "/v1/files"), ("GET", "/v1/messages"), ("POST", "/v1/messages/../files"),
                             ("POST", "/backend-api/wham/apps"), ("POST", "/v1/messages/batches")):
            st, out = self.call(method, f"/h/{self.up}{path}", {"secret": "BODY-NOT-LOGGED"} if method == "POST" else None)
            self.assertEqual(st, 403, (method, path, out))
            self.assertIn("not a model API call", json.loads(out)["error"]["message"])
        self.assertEqual(len(Stub.seen), 1, Stub.seen)            # nothing refused reached the upstream
        st, _ = self.call("POST", "/u/chatgpt.com/backend-api/wham/apps", {})      # refused before any connection
        self.assertEqual(st, 403)
        st, _ = self.call("POST", "/u/github.com/v1/responses", {})                # not a model API host at all
        self.assertEqual(st, 403)
        recs = self.records()
        refused = [r for r in recs if r.get("refused") == "path_not_allowed"]
        self.assertEqual(sorted((r["method"], r["path"]) for r in refused if r["method"] in ("GET", "POST")),
                         sorted([("POST", "/v1/files"), ("GET", "/v1/messages"), ("POST", "/v1/messages/../files"),
                                 ("POST", "/backend-api/wham/apps"), ("POST", "/v1/messages/batches"), ("GET", "/api/hello"),
                                 ("POST", "/backend-api/wham/apps")]))
        self.assertFalse([r for r in recs if r["method"] == "HEAD"])
        self.assertTrue(all(r["status"] == 403 for r in refused))
        self.assertEqual([(r["host"], r["path"]) for r in recs if r.get("refused") == "host_not_allowed"], [("github.com", "/v1/responses")])
        tool = [r for r in recs if r.get("refused") == "server_tool"]
        self.assertEqual([r["server_tools"] for r in tool][:4], [["web_search_20250305"], ["mcp"], ["image_generation"], ["web_search"]])
        self.assertEqual(len([r for r in recs if r.get("refused") == "body_unreadable"]), 8)
        self.assertEqual(sorted(r["method"] for r in refused if r["method"] not in ("GET", "POST")), ["DELETE", "PATCH", "PUT"])
        fwd = [r for r in recs if not r.get("refused")]
        self.assertTrue(all("server_tools" not in r and "body_unchecked" not in r for r in fwd), fwd)
        fwd = [r for r in fwd if r["path"] == "/v1/messages" and r["request_bytes"] > 60]
        self.assertEqual(len(fwd), 1)
        self.assertEqual(fwd[0]["usage"], {"input": 4, "cached": 1, "output": 2})
        everything = "".join(p.read_text() for p in (self.jobs / ".gateway").iterdir() if p.is_file())
        self.assertNotIn("SECRET-KEY", everything)
        self.assertNotIn("BODY-NOT-LOGGED", everything)


class Broken(BaseHTTPRequestHandler):
    """An upstream that fails: /v1/messages answers 503; /v1/responses starts a stream and breaks it off mid-way."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path.startswith("/v1/messages"):
            body = b'{"type": "error", "error": {"type": "overloaded_error"}}'
            self.send_response(503); self.send_header("Content-Length", str(len(body))); self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", "100000"); self.end_headers()
        self.wfile.write(b'data: {"type": "response.created"}\n\n'); self.wfile.flush()
        self.close_connection = True               # far short of the announced length: the stream is broken off


class UpstreamFailures(unittest.TestCase):
    """Board 0121 M4: a model call that fails on the way (the upstream unreachable: 502; a stream broken off; a provider
    5xx) is recorded as an upstream_failure, and totals.json counts the failures in a row up to the latest call."""

    def test_tally(self):
        gateway.ARGS = type("A", (), {"budget_input": 0, "budget_output": 0})()
        tot = {"requests": 0, "input": 0, "cached": 0, "output": 0, "no_usage": 0}
        ok = {"ts": 1, "usage": {"input": 5, "cached": 0, "output": 1}}
        for rec in (ok, {"ts": 2, "forwarded": False, "upstream_failure": "unreachable: ConnectionRefusedError"},
                    {"ts": 3, "usage": {"input": 2, "cached": 0, "output": 0}, "upstream_failure": "stream broken off: X"},
                    {"ts": 4, "refused": "path_not_allowed"}):
            gateway.tally(tot, rec)
        self.assertEqual((tot["requests"], tot["input"], tot["upstream_failures"], tot["failures_in_a_row"]), (2, 7, 2, 2))
        self.assertEqual(tot["last_upstream_failure"], "stream broken off: X")
        gateway.tally(tot, {**ok, "ts": 5})
        self.assertEqual((tot["requests"], tot["upstream_failures"], tot["failures_in_a_row"]), (3, 2, 0))
        gateway.tally(tot, {"ts": 6, "usage": {}, "upload": {"purpose": "image", "bytes": 10}})   # Moonshot: no model call
        gateway.tally(tot, {"ts": 7, "usage": {}})
        self.assertEqual((tot["requests"], tot["uploads"], tot["no_usage"]), (5, 1, 1))

    def test_end_to_end(self):
        import socket
        with tempfile.TemporaryDirectory() as td:
            jobs = pathlib.Path(td)
            stub = ThreadingHTTPServer(("127.0.0.1", 0), Broken)
            threading.Thread(target=stub.serve_forever, daemon=True).start()
            up = f"127.0.0.1:{stub.server_address[1]}"
            with socket.socket() as sk:                                 # a port nothing listens on
                sk.bind(("127.0.0.1", 0)); dead = f"127.0.0.1:{sk.getsockname()[1]}"
            ready = jobs / "ready"
            proc = subprocess.Popen([sys.executable, str(HERE / "gateway.py"), "--jobs", str(jobs), "--bind", "127.0.0.1",
                                     "--ready-file", str(ready), "--http-host", up, "--http-host", dead],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                for _ in range(100):
                    if ready.exists() and ready.read_text():
                        break
                    time.sleep(0.1)
                port = int(ready.read_text())

                def call(path):
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
                    c.request("POST", path, body=b'{"model": "m", "input": []}', headers={"Content-Type": "application/json"})
                    r = c.getresponse()
                    try:
                        r.read()
                        whole = True
                    except http.client.HTTPException:
                        whole = False
                    c.close()
                    return r.status, whole

                self.assertEqual(call(f"/h/{dead}/v1/responses")[0], 502)
                self.assertEqual(call(f"/h/{up}/v1/messages"), (503, True))
                self.assertEqual(call(f"/h/{up}/v1/responses"), (200, False))      # the agent sees the stream end broken
                recs = [json.loads(x) for x in (jobs / ".gateway" / "unattributed.jsonl").read_text().splitlines()]
                fails = [r.get("upstream_failure", "") for r in recs]
                self.assertEqual(len(recs), 3, recs)
                self.assertTrue(fails[0].startswith("unreachable: ") and recs[0]["forwarded"] is False, recs[0])
                self.assertEqual(fails[1], "HTTP 503")
                self.assertTrue(fails[2].startswith("stream broken off: "), recs[2])
                self.assertEqual(recs[2]["status"], 200)
            finally:
                proc.terminate(); proc.wait(10)
                stub.shutdown(); stub.server_close()


class MoonshotStub(BaseHTTPRequestHandler):
    """Moonshot as Kimi Code meets it: an upload answered with a file id; a streamed answer whose usage is in the last
    chunk's choices[0], the cache hits as cached_tokens."""
    seen: list[tuple[str, str, str, bytes]] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        MoonshotStub.seen.append((self.command, self.path, self.headers.get("Content-Type") or "", body))
        if self.path == "/v1/files":
            out, ctype = json.dumps({"id": "file-1", "object": "file", "purpose": "image"}).encode(), "application/json"
        else:
            usage = {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107, "cached_tokens": 90}
            out = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in (
                {"model": "kimi-k3.1-pro", "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
                {"model": "kimi-k3.1-pro", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "usage": usage}]})) \
                + b"data: [DONE]\n\n"
            ctype = "text/event-stream"
        self.send_response(200); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


class MoonshotEndToEnd(unittest.TestCase):
    """gateway.py with the moonshot provider's rules (uploads, record_fields) given to a stub on loopback (an
    --http-host upstream gets the `http` provider's list: here a copy of api_paths.json whose `http` has moonshot's)."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.jobs = pathlib.Path(cls.td.name)
        doc = json.loads(gateway.API_PATHS.read_text())
        moon = next(p for p in doc["providers"] if p["name"] == "moonshot")
        for p in doc["providers"]:
            if p["name"] == "http":
                p.update(allow=moon["allow"], uploads=moon["uploads"], record_fields=moon["record_fields"])
        paths = cls.jobs / "api_paths.json"
        paths.write_text(json.dumps(doc))
        cls.stub = ThreadingHTTPServer(("127.0.0.1", 0), MoonshotStub)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.up = f"127.0.0.1:{cls.stub.server_address[1]}"
        ready = cls.jobs / "ready"
        cls.proc = subprocess.Popen([sys.executable, str(HERE / "gateway.py"), "--jobs", str(cls.jobs), "--bind", "127.0.0.1",
                                     "--ready-file", str(ready), "--http-host", cls.up, "--api-paths", str(paths),
                                     "--image-cap", "2"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(100):
            if ready.exists() and ready.read_text():
                break
            if cls.proc.poll() is not None:
                raise RuntimeError(cls.proc.stderr.read().decode()[-2000:])
            time.sleep(0.1)
        cls.port = int(ready.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate(); cls.proc.wait(10); cls.proc.stderr.close()
        cls.stub.shutdown(); cls.stub.server_close(); cls.td.cleanup()

    def post(self, path: str, body: bytes, ctype: str):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("POST", f"/h/{self.up}{path}", body=body, headers={"Content-Type": ctype, "Authorization": "Bearer SECRET-KEY"})
        r = c.getresponse(); out = r.read(); c.close()
        return r.status, out

    @staticmethod
    def upload(purpose: str) -> tuple[bytes, str]:
        b = "----formdata-undici-042"
        body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="image"\r\nContent-Type: image/png\r\n\r\n'
                ).encode() + b'\x89PNG\r\n\x1a\n"image_url"' + (
                f'\r\n--{b}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\n{purpose}\r\n--{b}--\r\n').encode()
        return body, f"multipart/form-data; boundary={b}"

    def records(self) -> list[dict]:
        f = self.jobs / ".gateway" / "unattributed.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    def test_uploads_and_model_calls(self):
        MoonshotStub.seen.clear()
        body, ctype = self.upload("image")
        st, out = self.post("/v1/files", body, ctype)
        self.assertEqual((st, json.loads(out)["id"]), (200, "file-1"))
        self.assertEqual(MoonshotStub.seen, [("POST", "/v1/files", ctype, body)])          # forwarded as sent
        for purpose, why in (("file-extract", "server_tool"), ("batch", "server_tool")):
            st, out = self.post("/v1/files", *self.upload(purpose))
            self.assertEqual(st, 403, out)
        for raw, ct in ((b'{"purpose": "image"}', "application/json"), (self.upload("image")[0], "multipart/form-data")):
            self.assertEqual(self.post("/v1/files", raw, ct)[0], 403)
        self.assertEqual(len(MoonshotStub.seen), 1)
        chat = {"model": "kimi-k3.1-pro", "stream": True, "thinking": {"type": "enabled", "effort": "max"},
                "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "ms://file-1"}}]}],
                "tools": [{"type": "function", "function": {"name": "Bash", "parameters": {}}}]}
        st, out = self.post("/v1/chat/completions", json.dumps(chat).encode(), "application/json")
        self.assertEqual(st, 200, out)
        self.assertIn(b'"usage"', out)                                                    # the stream as it came
        chat["tools"].append({"type": "builtin_function", "function": {"name": "$web_search"}})
        self.assertEqual(self.post("/v1/chat/completions", json.dumps(chat).encode(), "application/json")[0], 403)
        recs = self.records()
        (up,) = [r for r in recs if r.get("upload")]
        self.assertEqual((up["upload"], up["usage"], up["status"]), ({"purpose": "image", "bytes": len(body)}, {}, 200))
        self.assertNotIn("request_fields", up)
        self.assertEqual(sorted(tuple(r["server_tools"]) for r in recs if r.get("refused") == "server_tool"),
                         [("builtin_function",), ("files:batch",), ("files:file-extract",)])
        self.assertEqual(len([r for r in recs if r.get("refused") == "body_unreadable"]), 2)
        (call,) = [r for r in recs if r["path"] == "/v1/chat/completions" and not r.get("refused")]
        self.assertEqual(call["request_fields"], {"thinking": {"type": "enabled", "effort": "max"}})
        self.assertEqual((call["model"], call["usage"]), ("kimi-k3.1-pro", {"input": 100, "cached": 90, "output": 7}))
        self.assertEqual(call["images"], 1)                     # an uploaded image still counts toward the cap
        everything = "".join(p.read_text() for p in (self.jobs / ".gateway").iterdir() if p.is_file())
        self.assertNotIn("SECRET-KEY", everything)


if __name__ == "__main__":
    unittest.main()
