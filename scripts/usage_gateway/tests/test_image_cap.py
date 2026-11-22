"""Tests for the gateway's image cap (gateway.py --image-cap, the module doc's "Image cap"; owner 2026-10-31).

    python3 scripts/usage_gateway/tests/test_image_cap.py     # stdlib only; the end-to-end part starts gateway.py on loopback

- the rule: a request with more than N images has its oldest step x ceil((n - N) / step) replaced, step = N // 2;
- the image blocks of the Anthropic, Responses and Chat Completions shapes, also inside a tool result (Codex sends the
  images it looks at in custom_tool_call_output.output), each replaced by a text block of its own API's shape (an
  Anthropic block keeps its cache_control); a tool call's arguments are not looked into;
- everything else in the request is left as it was, and the replaced images of a conversation stay the same from one
  call to the next until the following step (the provider's prompt cache is broken once per step);
- end to end: a capped request reaches the upstream re-serialized and uncompressed, one within the cap as it was sent
  (gzip and all); usage.jsonl records images / images_omitted, totals.json counts image_cap_requests;
- the size cap (--image-bytes, decided 2026-11-04): off, the count cap alone, byte for byte; on, a request still over
  it has more of its oldest images replaced, in the same steps, until it fits, the newest image always kept; the
  replaced images stay the same between calls until the next step; the six requests GitHub Copilot refused with 413 in
  a collaborator's Gemini 4 Argon run (5.0-5.4 MiB, 8-10 images) fit under 4.5 MiB; end to end, a request that still does not
  fit is forwarded as sent and recorded with over_image_bytes.
"""
from __future__ import annotations

import gzip
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

NOTE = "[earlier image removed: this conversation keeps at most {} images, the oldest are dropped]"


def responses_body(n: int) -> dict:
    """A Codex-like Responses request whose tool outputs carry n images, numbered in their data URLs."""
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]}]
    for i in range(n):
        items.append({"type": "custom_tool_call", "call_id": f"c{i}", "name": "exec", "input": "show()"})
        items.append({"type": "custom_tool_call_output", "call_id": f"c{i}", "output": [
            {"type": "input_text", "text": f"frame {i}"},
            {"type": "input_image", "image_url": f"data:image/png;base64,IMG{i}", "detail": "high"}]})
    return {"model": "m", "input": items, "stream": True}


def kept(body: dict) -> list[str]:
    """The image_url of every image block still in a Responses body, in order."""
    return [b["image_url"] for it in body["input"] for b in it.get("output", []) if isinstance(it.get("output"), list)
            and b.get("type") == "input_image"]


class Rule(unittest.TestCase):
    def test_steps_of_half_the_cap(self):
        for n, omitted in ((0, 0), (1, 0), (50, 0), (51, 25), (75, 25), (76, 50), (100, 50), (101, 75)):
            text, images, d = gateway.cap_images(json.dumps(responses_body(n)), 50)
            self.assertEqual((images, d), (n, omitted), n)
            if omitted:
                body = json.loads(text)
                self.assertEqual(kept(body), [f"data:image/png;base64,IMG{i}" for i in range(omitted, n)])   # the newest stay
                self.assertTrue(25 <= len(kept(body)) <= 50)
            else:
                self.assertIsNone(text)                  # forwarded as sent

    def test_small_caps(self):
        self.assertEqual(gateway.cap_images(json.dumps(responses_body(3)), 1)[1:], (3, 2))     # step 1: one stays
        self.assertEqual(gateway.cap_images(json.dumps(responses_body(5)), 4)[1:], (5, 2))     # step 2: 3 stay

    def test_the_replaced_images_stay_the_same_between_calls(self):
        """A conversation only grows: the call with 70 images replaces the same 25 as the call with 60, so both start
        with the same items (the provider's cache holds) up to the 60th image."""
        a = json.loads(gateway.cap_images(json.dumps(responses_body(60)), 50)[0])
        b = json.loads(gateway.cap_images(json.dumps(responses_body(70)), 50)[0])
        self.assertEqual(b["input"][:len(a["input"])], a["input"])
        c = json.loads(gateway.cap_images(json.dumps(responses_body(76)), 50)[0])     # the next step: 25 more go
        self.assertNotEqual(c["input"][:len(a["input"])], a["input"])


class Shapes(unittest.TestCase):
    def test_responses(self):
        body = responses_body(3)
        body["input"].append({"type": "function_call_output", "call_id": "f", "output": [
            {"type": "input_image", "image_url": "data:image/png;base64,F"}]})
        body["input"].append({"type": "message", "role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,M"}]})
        text, n, d = gateway.cap_images(json.dumps(body), 2)
        out = json.loads(text)
        self.assertEqual((n, d), (5, 3))
        self.assertEqual(out["input"][2]["output"], [{"type": "input_text", "text": "frame 0"},
                                                     {"type": "input_text", "text": NOTE.format(2)}])
        self.assertEqual(out["input"][-2]["output"], [{"type": "input_image", "image_url": "data:image/png;base64,F"}])
        self.assertEqual(out["input"][-1]["content"], [{"type": "input_image", "image_url": "data:image/png;base64,M"}])
        out["input"][2]["output"][1] = body["input"][2]["output"][1]       # put the replaced ones back: nothing else changed
        out["input"][4]["output"][1] = body["input"][4]["output"][1]
        out["input"][6]["output"][1] = body["input"][6]["output"][1]
        self.assertEqual(out, body)

    def test_anthropic(self):
        img = lambda d, **k: {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": d}, **k}
        body = {"model": "m", "system": "s", "messages": [
            {"role": "user", "content": [img("A"), {"type": "text", "text": "look"}]},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                                               "input": {"content": [img("NOT-AN-IMAGE-BLOCK")]}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                          "content": [img("B", cache_control={"type": "ephemeral"})]}]},
            {"role": "user", "content": [img("C")]}]}
        text, n, d = gateway.cap_images(json.dumps(body), 1)
        out = json.loads(text)
        self.assertEqual((n, d), (3, 2))                 # the tool call's input is not looked into
        self.assertEqual(out["messages"][0]["content"][0], {"type": "text", "text": NOTE.format(1)})
        self.assertEqual(out["messages"][2]["content"][0]["content"][0],
                         {"type": "text", "text": NOTE.format(1), "cache_control": {"type": "ephemeral"}})
        self.assertEqual(out["messages"][3]["content"][0], img("C"))
        self.assertEqual(out["messages"][1], body["messages"][1])

    def test_chat_completions(self):
        url = lambda d: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{d}"}}
        body = {"model": "m", "messages": [{"role": "user", "content": [url("A"), url("B"), url("C")]}]}
        text, n, d = gateway.cap_images(json.dumps(body), 2)
        self.assertEqual((n, d), (3, 1))
        self.assertEqual(json.loads(text)["messages"][0]["content"], [{"type": "text", "text": NOTE.format(2)}, url("B"), url("C")])

    def test_no_images(self):
        self.assertEqual(gateway.cap_images(json.dumps({"model": "m", "input": "hi"}), 50), (None, 0, 0))
        args = {"type": "function_call", "call_id": "x", "name": "f", "arguments": json.dumps({"type": "input_image"})}
        self.assertEqual(gateway.cap_images(json.dumps({"model": "m", "input": [args]}), 1), (None, 0, 0))

    def test_tally(self):
        tot = {"requests": 0, "input": 0, "cached": 0, "output": 0, "no_usage": 0}
        saved = getattr(gateway, "ARGS", None)
        gateway.ARGS = type("A", (), {"budget_input": 0, "budget_output": 0})()
        try:
            gateway.tally(tot, {"ts": 1, "usage": {}, "images": 6, "images_omitted": 2})
            gateway.tally(tot, {"ts": 2, "usage": {}, "images": 3})
        finally:
            gateway.ARGS = saved
        self.assertEqual((tot["requests"], tot["image_cap_requests"]), (2, 1))


MIB = 2 ** 20
BOTH = "[earlier image removed: this conversation keeps at most 10 images and requests of at most 4.5 MiB, the oldest are dropped]"


def chat_body(n: int, size: int) -> dict:
    """A Chat Completions request (what the gateway sends for Gemini 4 Argon) with n images of about `size` bytes each,
    one per message, numbered in their data URLs."""
    url = lambda i: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,IMG{i}:" + "A" * size}}
    return {"model": "gemini-4-argon", "messages": [{"role": "user", "content": [{"type": "text", "text": "task"}]}]
            + [{"role": "user", "content": [{"type": "text", "text": f"frame {i}"}, url(i)]} for i in range(n)]}


def chat_kept(body: dict) -> list[int]:
    """The numbers of the images still in a Chat Completions body, in order."""
    return [int(b["image_url"]["url"].split("IMG")[1].split(":")[0]) for m in body["messages"] for b in m["content"]
            if b["type"] == "image_url"]


class Bytes(unittest.TestCase):
    def test_off_is_the_count_cap_alone(self):
        """Off, the size of a request does not matter: 1 MB images, up to 25 MB, get the count cap's rule and note."""
        for n, omitted in ((3, 0), (10, 0), (11, 5), (16, 10), (25, 15)):
            text, images, d = gateway.cap_images(json.dumps(chat_body(n, MIB)), 10, 0)
            self.assertEqual((images, d), (n, omitted), n)
            if omitted:
                out = json.loads(text)
                self.assertEqual(chat_kept(out), list(range(omitted, n)), n)
                self.assertEqual(out["messages"][1]["content"][1], {"type": "text", "text": NOTE.format(10)})
            else:
                self.assertIsNone(text)                  # forwarded as sent, 10 MB and all

    def test_under_the_size_is_forwarded_as_sent(self):
        self.assertEqual(gateway.cap_images(json.dumps(chat_body(6, 100_000)), 10, int(4.5 * MIB)), (None, 6, 0))

    def test_over_the_size_the_oldest_go_in_steps(self):
        text, n, d = gateway.cap_images(json.dumps(chat_body(8, 600_000)), 10, int(4.5 * MIB))   # 4.8 MB: over
        out = json.loads(text)
        self.assertEqual((n, d, chat_kept(out)), (8, 5, [5, 6, 7]))           # one step of 5; the newest stay
        self.assertLessEqual(len(text.encode()), 4.5 * MIB)
        self.assertEqual(out["messages"][1]["content"][1], {"type": "text", "text": BOTH})
        self.assertEqual(out["messages"][1]["content"][0], {"type": "text", "text": "frame 0"})   # the rest as it was

    def test_the_requests_copilot_refused_fit(self):
        """A collaborator's Gemini 4 Argon run, 2026-11-04: the six requests answered 413 (MiB, images)."""
        for mib, n in ((5.01, 10), (5.09, 10), (5.15, 8), (5.17, 9), (5.32, 10), (5.40, 10)):
            size = int(mib * MIB / n)
            text, images, d = gateway.cap_images(json.dumps(chat_body(n, size)), 10, int(4.5 * MIB))
            self.assertGreater(len(json.dumps(chat_body(n, size)).encode()), 5 * MIB, mib)
            self.assertLessEqual(len(text.encode()), 4.5 * MIB, mib)
            self.assertEqual((images, d % 5), (n, 0), mib)
            self.assertEqual(chat_kept(json.loads(text))[-1], n - 1, mib)    # the newest image is there

    def test_the_replaced_images_stay_the_same_between_calls(self):
        """0.6 MB images, 10 and 4.5 MiB: 8 to 12 images replace the same 5 (the count cap's step at 11 and 12 too), so
        those calls start alike up to the 8th image; at 13 the next 5 go."""
        outs = {n: json.loads(gateway.cap_images(json.dumps(chat_body(n, 600_000)), 10, int(4.5 * MIB))[0])
                for n in (8, 9, 10, 11, 12, 13)}
        head = outs[8]["messages"]
        for n in (9, 10, 11, 12):
            self.assertEqual(outs[n]["messages"][:len(head)], head, n)
            self.assertEqual(chat_kept(outs[n])[0], 5, n)
        self.assertEqual(chat_kept(outs[13])[0], 10)
        self.assertNotEqual(outs[13]["messages"][:len(head)], head)

    def test_the_newest_image_stays(self):
        self.assertEqual(gateway.cap_images(json.dumps(chat_body(1, 5 * MIB)), 10, int(4.5 * MIB)), (None, 1, 0))
        text, n, d = gateway.cap_images(json.dumps(chat_body(3, 5 * MIB)), 10, int(4.5 * MIB))
        self.assertEqual((n, d, chat_kept(json.loads(text))), (3, 2, [2]))     # still over: forwarded so, and recorded
        self.assertGreater(len(text.encode()), 4.5 * MIB)

    def test_size_cap_without_a_count_cap_goes_one_at_a_time(self):
        text, n, d = gateway.cap_images(json.dumps(chat_body(5, 1000)), 0, 3500)
        self.assertEqual((n, d, chat_kept(json.loads(text))), (5, 3, [3, 4]))
        self.assertLessEqual(len(text.encode()), 3500)
        self.assertIn("keeps requests of at most", text)


class Upstream(BaseHTTPRequestHandler):
    got: list[tuple[dict, bytes]] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        Upstream.got.append((dict(self.headers.items()), self.rfile.read(int(self.headers.get("Content-Length") or 0))))
        body = json.dumps({"model": "stub", "usage": {"input_tokens": 3, "output_tokens": 2}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


class EndToEnd(unittest.TestCase):
    def test_capped_and_uncapped_requests(self):
        with tempfile.TemporaryDirectory() as td:
            jobs = pathlib.Path(td)
            stub = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
            threading.Thread(target=stub.serve_forever, daemon=True).start()
            up = f"127.0.0.1:{stub.server_address[1]}"
            ready = jobs / "ready"
            proc = subprocess.Popen([sys.executable, str(HERE / "gateway.py"), "--jobs", str(jobs), "--bind", "127.0.0.1",
                                     "--ready-file", str(ready), "--http-host", up, "--image-cap", "4"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                for _ in range(100):
                    if ready.exists() and ready.read_text():
                        break
                    if proc.poll() is not None:
                        raise RuntimeError(proc.stderr.read().decode()[-2000:])
                    time.sleep(0.1)
                port = int(ready.read_text())

                def call(n: int) -> bytes:
                    sent = gzip.compress(json.dumps(responses_body(n)).encode())
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
                    c.request("POST", f"/h/{up}/v1/responses", body=sent,
                              headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
                    r = c.getresponse(); r.read(); c.close()
                    self.assertEqual(r.status, 200)
                    return sent

                Upstream.got.clear()
                call(6)
                sent = call(3)
                (h1, b1), (h2, b2) = Upstream.got
                self.assertNotIn("content-encoding", {k.lower() for k in h1})          # capped: plain JSON
                out = json.loads(b1)
                self.assertEqual(kept(out), [f"data:image/png;base64,IMG{i}" for i in (2, 3, 4, 5)])
                self.assertEqual(int(h1["Content-Length"]), len(b1))
                self.assertEqual(({k.lower(): v for k, v in h2.items()}["content-encoding"], b2), ("gzip", sent))   # as sent
                f = jobs / ".gateway" / "unattributed.jsonl"
                for _ in range(100):       # a request is recorded after its answer is sent: wait for both, in any order
                    recs = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
                    if len(recs) == 2:
                        break
                    time.sleep(0.1)
                self.assertEqual(sorted((r.get("images"), r.get("images_omitted", 0)) for r in recs), [(3, 0), (6, 2)])
                self.assertIn("image cap: 4", "".join(p.read_text() for p in (jobs / ".gateway").glob("gateway-*.log")))
            finally:
                proc.terminate(); proc.wait(10); proc.stderr.close()
                stub.shutdown(); stub.server_close()

    def test_size_cap_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            jobs = pathlib.Path(td)
            stub = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
            threading.Thread(target=stub.serve_forever, daemon=True).start()
            up = f"127.0.0.1:{stub.server_address[1]}"
            ready = jobs / "ready"
            proc = subprocess.Popen([sys.executable, str(HERE / "gateway.py"), "--jobs", str(jobs), "--bind", "127.0.0.1",
                                     "--ready-file", str(ready), "--http-host", up, "--image-cap", "10",
                                     "--image-bytes", "4000"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                for _ in range(100):
                    if ready.exists() and ready.read_text():
                        break
                    if proc.poll() is not None:
                        raise RuntimeError(proc.stderr.read().decode()[-2000:])
                    time.sleep(0.1)
                port = int(ready.read_text())

                def call(body: dict) -> bytes:
                    sent = gzip.compress(json.dumps(body).encode())
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
                    c.request("POST", f"/h/{up}/v1/chat/completions", body=sent,
                              headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
                    r = c.getresponse(); r.read(); c.close()
                    self.assertEqual(r.status, 200)
                    return sent

                Upstream.got.clear()
                call(chat_body(6, 1000))                 # 6 images, about 6 KB: under the count cap, over the size
                sent = call(chat_body(1, 5000))          # one image over the size: nothing older to replace
                (h1, b1), (h2, b2) = Upstream.got
                self.assertNotIn("content-encoding", {k.lower() for k in h1})          # capped: plain JSON
                self.assertLessEqual(len(b1), 4000)
                self.assertEqual(chat_kept(json.loads(b1)), [5])
                self.assertEqual(({k.lower(): v for k, v in h2.items()}["content-encoding"], b2), ("gzip", sent))   # as sent
                f = jobs / ".gateway" / "unattributed.jsonl"
                for _ in range(100):
                    recs = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
                    if len(recs) == 2:
                        break
                    time.sleep(0.1)
                self.assertEqual(sorted((r.get("images"), r.get("images_omitted", 0), r.get("over_image_bytes", False))
                                        for r in recs), [(1, 0, True), (6, 5, False)])
                self.assertIn("image bytes: 4000", "".join(p.read_text() for p in (jobs / ".gateway").glob("gateway-*.log")))
            finally:
                proc.terminate(); proc.wait(10); proc.stderr.close()
                stub.shutdown(); stub.server_close()

    def test_negative_cap_does_not_start(self):
        for flag in ("--image-cap", "--image-bytes"):
            p = subprocess.run([sys.executable, str(HERE / "gateway.py"), "--jobs", "/nonexistent", flag, "-1"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 2, flag)
            self.assertIn(flag, p.stderr)


if __name__ == "__main__":
    unittest.main()
