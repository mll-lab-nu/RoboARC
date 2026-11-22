"""Tests for the Gemini -> Chat Completions translation (gemini_chat.py) and the gateway's --gemini-chat route (gateway.py).

    python3 scripts/usage_gateway/tests/test_gemini_chat.py     # stdlib only; the end-to-end part starts gateway.py on loopback

The requests below have the shape google-antigravity 0.1.23 sends (recorded 2026-11-01 from the SDK, run offline against a
recording server): systemInstruction, a model content with the functionCall, a model content with the functionResponse
(images in its parts), functionDeclarations with parametersJsonSchema, functionCallingConfig AUTO, thinkingLevel.
"""
from __future__ import annotations

import base64
import http.client
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import gemini_chat as gc  # noqa: E402

PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
RUN = {"name": "run_command", "description": "Run a command.",
       "parametersJsonSchema": {"$schema": "https://json-schema.org/draft/2020-12/schema", "additionalProperties": False,
                                "properties": {"CommandLine": {"type": "string"}}, "required": ["CommandLine"],
                                "type": "object"}}
VIEW = {"name": "view_file", "parametersJsonSchema": {"type": "object", "properties": {"AbsolutePath": {"type": "string"}}}}


def sdk_request(**over) -> dict:
    req = {
        "systemInstruction": {"role": "user", "parts": [{"text": "<identity>You are Antigravity.</identity>"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "<USER_REQUEST>\nLook at the camera.\n</USER_REQUEST>"}]},
            {"role": "model", "parts": [{"functionCall": {"id": "u-1", "name": "view_file", "args": {"AbsolutePath": "/a.png"}}}]},
            {"role": "model", "parts": [{"functionResponse": {"id": "u-1", "name": "view_file", "parts": [
                {"inlineData": {"mimeType": "image/png", "data": PNG}}], "response": {"output": "The file."}}}]},
        ],
        "tools": [{"functionDeclarations": [RUN, VIEW]}],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig": {"maxOutputTokens": 65535, "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "low"}},
        "sessionId": "1664173275559259932",
    }
    req.update(over)
    return req


class Request(unittest.TestCase):
    def test_route(self):
        self.assertEqual(gc.match("/v1beta/models/gemini-4-argon:streamGenerateContent"), ("", "gemini-4-argon", True))
        self.assertEqual(gc.match("/v1/v1beta/models/g:generateContent"), ("/v1", "g", False))
        for p in ("/v1/chat/completions", "/v1beta/models/g:countTokens", "/v1beta/models/g", "/x/../v1beta/models/g:generateContent"):
            self.assertIsNone(gc.match(p), p)

    def test_the_sdks_request(self):
        chat, provider = gc.to_chat(sdk_request(), "gemini-4-argon", True)
        self.assertEqual(provider, [])
        self.assertEqual({k: v for k, v in chat.items() if k not in ("messages", "tools")},
                         {"model": "gemini-4-argon", "tool_choice": "auto", "max_tokens": 65535, "reasoning_effort": "low",
                          "stream": True, "stream_options": {"include_usage": True}})
        self.assertEqual([t["function"]["name"] for t in chat["tools"]], ["run_command", "view_file"])
        self.assertEqual(chat["tools"][0]["function"]["parameters"], RUN["parametersJsonSchema"])   # as the SDK wrote it
        self.assertEqual(chat["tools"][0]["function"]["description"], "Run a command.")
        system, user, call, tool, images = chat["messages"]
        self.assertEqual(system, {"role": "system", "content": "<identity>You are Antigravity.</identity>"})
        self.assertEqual(user["role"], "user")
        self.assertEqual(call, {"role": "assistant", "content": None, "tool_calls": [
            {"id": "u-1", "type": "function", "function": {"name": "view_file", "arguments": '{"AbsolutePath": "/a.png"}'}}]})
        self.assertEqual(tool, {"role": "tool", "tool_call_id": "u-1", "content": "The file."})
        self.assertEqual(images["role"], "user")            # a tool message carries text only: the image follows the turn
        self.assertEqual(images["content"][1], {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}})

    def test_images_after_all_tool_messages_of_a_turn(self):
        turn = {"role": "model", "parts": [
            {"functionResponse": {"id": f"u-{i}", "name": "view_file", "response": {"output": str(i)},
                                  "parts": [{"inlineData": {"mimeType": "image/png", "data": PNG}}]}} for i in (1, 2)]}
        calls = {"role": "model", "parts": [{"functionCall": {"id": f"u-{i}", "name": "view_file", "args": {}}} for i in (1, 2)]}
        chat, _ = gc.to_chat(sdk_request(contents=[{"role": "user", "parts": [{"text": "go"}]}, calls, turn]), "m", False)
        self.assertEqual([m["role"] for m in chat["messages"]], ["system", "user", "assistant", "tool", "tool", "user"])
        self.assertEqual(sum(b["type"] == "image_url" for b in chat["messages"][-1]["content"]), 2)
        self.assertNotIn("stream", chat)

    def test_tool_results_text(self):
        for response, text in (({"output": "ok"}, "ok"), ({"error": "Encountered error"}, "Encountered error"),
                               ({"output": "a", "exit": 1}, '{"output": "a", "exit": 1}'), ({"output": 3}, '{"output": 3}')):
            req = sdk_request(contents=[{"role": "model", "parts": [{"functionCall": {"id": "c", "name": "f", "args": {}}}]},
                                        {"role": "model", "parts": [{"functionResponse": {"id": "c", "name": "f", "response": response}}]}])
            self.assertEqual(gc.to_chat(req, "m", False)[0]["messages"][-1]["content"], text)

    def test_calls_without_ids(self):
        req = sdk_request(contents=[
            {"role": "model", "parts": [{"functionCall": {"name": "a", "args": {}}}, {"functionCall": {"name": "b", "args": {}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "b", "response": {"output": "B"}}},
                                       {"functionResponse": {"name": "a", "response": {"output": "A"}}}]}])
        msgs = gc.to_chat(req, "m", False)[0]["messages"]
        ids = {tc["function"]["name"]: tc["id"] for tc in msgs[1]["tool_calls"]}
        self.assertEqual([(m["tool_call_id"], m["content"]) for m in msgs[2:]], [(ids["b"], "B"), (ids["a"], "A")])

    def test_provider_tools_are_reported(self):
        for tool in ({"googleSearch": {}}, {"urlContext": {}}, {"codeExecution": {}}, {"googleSearchRetrieval": {}}):
            _, provider = gc.to_chat(sdk_request(tools=[{"functionDeclarations": [RUN]}, tool]), "m", True)
            self.assertEqual(provider, list(tool))

    def test_gemini_schema_and_choices(self):
        old = {"name": "f", "parameters": {"type": "OBJECT", "propertyOrdering": ["x"],
                                           "properties": {"x": {"type": "STRING"}, "type": {"type": "ARRAY", "items": {"type": "INTEGER"}}}}}
        chat, _ = gc.to_chat(sdk_request(tools=[{"functionDeclarations": [old]}]), "m", False)
        self.assertEqual(chat["tools"][0]["function"]["parameters"], {
            "type": "object", "properties": {"x": {"type": "string"}, "type": {"type": "array", "items": {"type": "integer"}}}})
        for mode, names, choice in (("NONE", [], "none"), ("ANY", [], "required"), ("VALIDATED", [], "auto"),
                                    ("ANY", ["f"], {"type": "function", "function": {"name": "f"}})):
            cfg = {"functionCallingConfig": {"mode": mode, "allowedFunctionNames": names}}
            self.assertEqual(gc.to_chat(sdk_request(toolConfig=cfg), "m", False)[0]["tool_choice"], choice)
        self.assertNotIn("tool_choice", gc.to_chat(sdk_request(tools=[]), "m", False)[0])
        for level, effort in (("minimal", "minimal"), ("HIGH", "high"), ("extra_high", "xhigh")):
            gen = {"thinkingConfig": {"thinkingLevel": level}, "temperature": 1, "topP": 0.9, "stopSequences": ["x"]}
            chat = gc.to_chat(sdk_request(generationConfig=gen), "m", False)[0]
            self.assertEqual((chat["reasoning_effort"], chat["temperature"], chat["top_p"], chat["stop"]), (effort, 1, 0.9, ["x"]))
        self.assertNotIn("reasoning_effort", gc.to_chat(sdk_request(generationConfig={}), "m", False)[0])

    def test_thoughts_and_signatures(self):
        mine = gc._signature({"reasoning_opaque": "OPQ", "reasoning_text": "t"})
        foreign = base64.b64encode(b"a Gemini signature").decode()
        req = sdk_request(contents=[
            {"role": "user", "parts": [{"text": "go"}]},
            {"role": "model", "parts": [{"text": "t", "thought": True},
                                        {"functionCall": {"id": "c", "name": "f", "args": {}}, "thoughtSignature": mine}]},
            {"role": "model", "parts": [{"functionResponse": {"id": "c", "name": "f", "response": {"output": "o"}}}]},
            {"role": "model", "parts": [{"text": "done"}, {"text": "", "thoughtSignature": foreign}]}])
        msgs = gc.to_chat(req, "m", False)[0]["messages"]
        self.assertEqual(msgs[2], {"role": "assistant", "content": None, "reasoning_opaque": "OPQ", "reasoning_text": "t",
                                   "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]})
        self.assertEqual(msgs[4], {"role": "assistant", "content": "done"})      # not ours: nothing restored

    def test_untranslatable(self):
        for req in ({"contents": ["x"]}, {"contents": [{"parts": ["x"]}]}, {"tools": {"a": 1}},
                    {"contents": [{"role": "model", "parts": [{"inlineData": {"mimeType": "image/png", "data": PNG}}]}]}):
            with self.assertRaises(gc.Untranslatable, msg=req):
                gc.to_chat(req, "m", False)


def sse(*objs) -> bytes:
    return b"".join(b"data: " + json.dumps(o).encode() + b"\n\n" for o in objs) + b"data: [DONE]\n\n"


def events(raw: bytes) -> list[dict]:
    return [json.loads(e[len("data: "):]) for e in raw.decode().split("\r\n\r\n") if e.startswith("data: ")]


USAGE = {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
         "completion_tokens_details": {"reasoning_tokens": 20}, "prompt_tokens_details": {"cached_tokens": 64}}


class Answer(unittest.TestCase):
    def test_stream_with_a_tool_call_in_pieces(self):
        raw = sse({"model": "g4", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Let me look. ",
                                                                      "reasoning_text": "hmm"}}]},
                  {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                                       "function": {"name": "run_command", "arguments": '{"Comm'}}]}}]},
                  {"choices": [{"index": 0, "delta": {"reasoning_opaque": "OPQ", "tool_calls": [
                      {"index": 0, "function": {"arguments": 'andLine": "ls"}'}}]}, "finish_reason": "tool_calls"}]},
                  {"choices": [], "usage": USAGE})
        s, out = gc.Stream(), b""
        for i in range(0, len(raw), 7):                   # any split of the upstream's bytes
            out += s.feed(raw[i:i + 7])
        evs = events(out + s.close())
        self.assertEqual(evs[0]["candidates"][0]["content"]["parts"], [{"text": "hmm", "thought": True}, {"text": "Let me look. "}])
        last = evs[-1]
        self.assertEqual(last["candidates"][0]["finishReason"], "STOP")
        part = last["candidates"][0]["content"]["parts"][0]
        self.assertEqual(part["functionCall"], {"name": "run_command", "args": {"CommandLine": "ls"}, "id": "c1"})
        self.assertEqual(gc._signature_fields(part), {"reasoning_text": "hmm", "reasoning_opaque": "OPQ"})
        self.assertEqual(last["usageMetadata"], {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 130,
                                                 "thoughtsTokenCount": 20, "cachedContentTokenCount": 64})
        self.assertEqual(last["modelVersion"], "g4")
        self.assertEqual(s.close(), b"")

    def test_stream_text_only_and_finish_reasons(self):
        for finish, reason in (("stop", "STOP"), ("length", "MAX_TOKENS"), ("content_filter", "SAFETY"), ("odd", "OTHER")):
            s = gc.Stream()
            evs = events(s.feed(sse({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": finish}]})) + s.close())
            self.assertEqual(evs[-1]["candidates"][0], {"content": {"role": "model", "parts": []}, "index": 0, "finishReason": reason})
        s = gc.Stream()                                   # an error event from the upstream: a Gemini error event
        self.assertEqual(events(s.feed(sse({"error": {"message": "boom"}})))[0]["error"]["message"], "boom")

    def test_whole_and_errors(self):
        w = gc.Whole()
        w.feed(json.dumps({"model": "g4", "usage": USAGE, "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function",
                                                                  "function": {"name": "f", "arguments": "not json"}}]}}]}).encode())
        doc = json.loads(w.close())
        self.assertEqual(doc["candidates"][0]["content"]["parts"], [{"functionCall": {"name": "f", "args": {"arguments": "not json"}, "id": "c"}}])
        self.assertEqual(doc["usageMetadata"]["promptTokenCount"], 100)
        e = gc.Error(429)
        e.feed(b'{"error": {"message": "Rate limit exceeded", "code": "rate_limited"}}')
        self.assertEqual(json.loads(e.close()), {"error": {"code": 429, "message": "Rate limit exceeded", "status": "RESOURCE_EXHAUSTED"}})
        e = gc.Error(503)
        e.feed(b"upstream down")
        self.assertEqual(json.loads(e.close())["error"], {"code": 503, "message": "upstream down", "status": "UNAVAILABLE"})

    def test_server_errors_are_given_as_503(self):
        # the SDK retries 429 and 503 only: a 500 of a proxy whose fetch failed ended a trial (2026-11-02)
        self.assertEqual([gc.agent_status(s) for s in (400, 429, 500, 501, 502, 503, 504)], [400, 429, 503, 501, 503, 503, 503])
        e = gc.Error(500)
        e.feed(b'{"error":{"message":"fetch failed","type":"error"}}')
        self.assertEqual((e.status, json.loads(e.close())["error"]),
                         (503, {"code": 503, "message": "HTTP 500 from the model server: fetch failed", "status": "UNAVAILABLE"}))


class ChatStub(BaseHTTPRequestHandler):
    seen: list[dict] = []
    status = 200

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        ChatStub.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if ChatStub.status != 200:
            data = json.dumps({"error": {"message": "slow down"}}).encode()
            self.send_response(ChatStub.status); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)
            return
        if body.get("stream"):
            data = sse({"model": "g4", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c9", "type": "function",
                        "function": {"name": "run_command", "arguments": '{"CommandLine": "ls"}'}}]}, "finish_reason": "tool_calls"}]},
                       {"model": "g4", "choices": [], "usage": USAGE})     # the usage chunk names the model, as OpenAI's do
            ctype = "text/event-stream"
        else:
            data = json.dumps({"model": "g4", "usage": USAGE, "choices": [{"index": 0, "finish_reason": "stop",
                                                                           "message": {"role": "assistant", "content": "ok"}}]}).encode()
            ctype = "application/json"
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)


class EndToEnd(unittest.TestCase):
    """gateway.py as the runner starts it for AGENT=antigravity (--gemini-chat, the image cap), in front of a stub Chat
    Completions server on loopback; and the same gateway without the flag, as every other run starts it."""

    @classmethod
    def start(cls, *extra):
        ready = cls.jobs / f"ready{len(cls.procs)}"
        proc = subprocess.Popen([sys.executable, str(HERE / "gateway.py"), "--jobs", str(cls.jobs), "--bind", "127.0.0.1",
                                 "--ready-file", str(ready), "--http-host", cls.up, *extra],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        cls.procs.append(proc)
        for _ in range(100):
            if ready.exists() and ready.read_text():
                return int(ready.read_text())
            if proc.poll() is not None:
                raise RuntimeError(proc.stderr.read().decode()[-2000:])
            time.sleep(0.1)
        raise RuntimeError("the gateway did not start")

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        cls.jobs = pathlib.Path(cls.td.name)
        cls.stub = ThreadingHTTPServer(("127.0.0.1", 0), ChatStub)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.up = f"127.0.0.1:{cls.stub.server_address[1]}"
        cls.procs = []
        cls.port = cls.start("--gemini-chat", "--image-cap", "2")
        cls.plain_port = cls.start()
        with socket.socket() as s:                       # a port nothing listens on: an unreachable model server
            s.bind(("127.0.0.1", 0))
            cls.dead = f"127.0.0.1:{s.getsockname()[1]}"
        cls.up, up = cls.dead, cls.up
        cls.dead_port = cls.start("--gemini-chat")
        cls.up = up

    @classmethod
    def tearDownClass(cls):
        for p in cls.procs:
            p.terminate(); p.wait(10); p.stderr.close()
        cls.stub.shutdown(); cls.stub.server_close(); cls.td.cleanup()

    def setUp(self):
        ChatStub.seen.clear()
        ChatStub.status = 200

    def call(self, path: str, body: dict, port: int | None = None):
        c = http.client.HTTPConnection("127.0.0.1", port or self.port, timeout=30)
        c.request("POST", path, body=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json", "x-goog-api-key": "PROXY-KEY", "User-Agent": "google-antigravity-sdk/0.1.23"})
        r = c.getresponse(); out = r.read(); c.close()
        return r.status, r.getheader("Content-Type"), out

    def records(self, more_than: int = -1) -> list[dict]:
        """The records so far; with more_than, once there are more than that many (the gateway writes a request's record
        after its answer has gone out)."""
        f = self.jobs / ".gateway" / "unattributed.jsonl"       # no trial owns loopback
        for _ in range(100):
            # whole lines only: the gateway may be half-way through writing the last one
            recs = [json.loads(x) for x in f.read_text().split("\n")[:-1]] if f.exists() else []
            if len(recs) > more_than:
                return recs
            time.sleep(0.05)
        return recs

    def last(self, before: int) -> dict:
        return self.records(before)[-1]

    def test_streamed_call(self):
        n = len(self.records())
        st, ctype, out = self.call(f"/h/{self.up}/v1/v1beta/models/gemini-4-argon:streamGenerateContent?alt=sse", sdk_request())
        self.assertEqual((st, ctype), (200, "text/event-stream"), out)
        self.assertEqual(len(ChatStub.seen), 1)
        sent = ChatStub.seen[0]
        self.assertEqual((sent["path"], sent["auth"]), ("/v1/chat/completions", "Bearer PROXY-KEY"))
        self.assertEqual((sent["body"]["model"], sent["body"]["reasoning_effort"], sent["body"]["tool_choice"]), ("gemini-4-argon", "low", "auto"))
        last = events(out)[-1]
        self.assertEqual(last["candidates"][0]["content"]["parts"][0]["functionCall"]["args"], {"CommandLine": "ls"})
        rec = self.records(n)[n]
        self.assertEqual((rec["path"], rec["translated_from"], rec["status"], rec["model"]),
                         ("/v1/chat/completions", "/v1/v1beta/models/gemini-4-argon:streamGenerateContent", 200, "g4"))
        self.assertEqual(rec["usage"], {"input": 100, "cached": 64, "output": 30})
        self.assertEqual(rec["images"], 1)
        self.assertNotIn("PROXY-KEY", (self.jobs / ".gateway" / "unattributed.jsonl").read_text())

    def test_image_cap_applies_to_the_translated_request(self):
        resp = {"role": "model", "parts": [{"functionResponse": {"id": f"u-{i}", "name": "view_file", "response": {"output": "f"},
                                                                 "parts": [{"inlineData": {"mimeType": "image/png", "data": PNG}}]}} for i in range(3)]}
        calls = {"role": "model", "parts": [{"functionCall": {"id": f"u-{i}", "name": "view_file", "args": {}}} for i in range(3)]}
        n = len(self.records())
        st, _, out = self.call(f"/h/{self.up}/v1/v1beta/models/g:streamGenerateContent?alt=sse",
                               sdk_request(contents=[{"role": "user", "parts": [{"text": "go"}]}, calls, resp]))
        self.assertEqual(st, 200, out)
        blocks = ChatStub.seen[0]["body"]["messages"][-1]["content"]
        self.assertEqual([b["type"] for b in blocks], ["text", "text", "image_url", "image_url"])   # 3 images, cap 2: the oldest goes
        self.assertEqual(self.last(n)["images_omitted"], 1)

    def test_whole_call_and_upstream_error(self):
        st, ctype, out = self.call(f"/h/{self.up}/v1/v1beta/models/g:generateContent", sdk_request())
        self.assertEqual((st, ctype), (200, "application/json"))
        self.assertEqual(json.loads(out)["candidates"][0]["content"]["parts"], [{"text": "ok"}])
        self.assertNotIn("stream", ChatStub.seen[0]["body"])
        ChatStub.status = 429
        st, _, out = self.call(f"/h/{self.up}/v1/v1beta/models/g:streamGenerateContent?alt=sse", sdk_request())
        self.assertEqual((st, json.loads(out)["error"]["status"]), (429, "RESOURCE_EXHAUSTED"))

    def test_server_errors_reach_the_sdk_as_503(self):
        n = len(self.records())
        for status, given in ((500, 503), (502, 503), (504, 503), (400, 400)):
            ChatStub.status = status
            st, _, out = self.call(f"/h/{self.up}/v1/v1beta/models/g:streamGenerateContent?alt=sse", sdk_request())
            self.assertEqual((st, json.loads(out)["error"]["code"]), (given, given), status)
            rec = self.last(n)
            n += 1
            self.assertEqual(rec["status"], status)                  # the record keeps the server's
            self.assertEqual(rec.get("agent_status"), given if given != status else None)
            self.assertEqual(rec.get("upstream_failure"), f"HTTP {status}" if status >= 500 else None)
        # the gateway's own 502 (the server unreachable) too
        st, _, out = self.call(f"/h/{self.dead}/v1/v1beta/models/g:streamGenerateContent?alt=sse", sdk_request(), self.dead_port)
        self.assertEqual((st, json.loads(out)["error"]["status"]), (503, "UNAVAILABLE"))
        rec = self.last(n)
        self.assertEqual((rec["status"], rec["agent_status"], rec["upstream_failure"]), (502, 503, "unreachable: ConnectionRefusedError"))

    def test_refusals(self):
        # a provider-run Gemini tool: refused in Gemini's error format, nothing forwarded
        n = len(self.records())
        st, _, out = self.call(f"/h/{self.up}/v1/v1beta/models/g:generateContent",
                               {"contents": [{"role": "user", "parts": [{"text": "q"}]}], "tools": [{"googleSearch": {}}]})
        self.assertEqual((st, json.loads(out)["error"]["status"]), (403, "PERMISSION_DENIED"))
        self.assertEqual(ChatStub.seen, [])
        rec = self.last(n)
        self.assertEqual((rec["refused"], rec["server_tools"]), ("server_tool", ["googleSearch"]))
        # a body that cannot be translated, and a prefix whose chat path is on no list
        st, _, out = self.call(f"/h/{self.up}/v1/v1beta/models/g:generateContent", {"contents": ["x"]})
        self.assertEqual((st, self.last(n + 1)["refused"]), (403, "body_unreadable"))
        st, _, out = self.call(f"/h/{self.up}/other/v1beta/models/g:generateContent", sdk_request())
        self.assertEqual((st, self.last(n + 2)["refused"]), (403, "path_not_allowed"))
        self.assertEqual(ChatStub.seen, [])

    def test_without_the_flag_nothing_changes(self):
        n = len(self.records())
        st, _, out = self.call(f"/h/{self.up}/v1/v1beta/models/g:streamGenerateContent?alt=sse", sdk_request(), self.plain_port)
        self.assertEqual(st, 403)
        self.assertEqual(json.loads(out)["error"]["code"], "path_not_allowed")      # the OpenAI-format refusal, as before
        self.assertEqual(self.last(n)["path"], "/v1/v1beta/models/g:streamGenerateContent")
        self.assertEqual(ChatStub.seen, [])
        ChatStub.status = 500                            # a server error of a Chat Completions call: as it came
        st, _, out = self.call(f"/h/{self.up}/v1/chat/completions", {"model": "g4", "messages": []}, self.plain_port)
        self.assertEqual((st, json.loads(out)), (500, {"error": {"message": "slow down"}}))
        self.assertNotIn("agent_status", self.last(n + 1))


if __name__ == "__main__":
    unittest.main()
