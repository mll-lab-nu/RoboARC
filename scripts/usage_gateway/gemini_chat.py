"""Gemini API requests carried to an OpenAI-compatible Chat Completions server, and the answers carried back: the usage
gateway's --gemini-chat (gateway.py), for AGENT=antigravity (scripts/harbor/agents/antigravity_sdk.py).

The Antigravity SDK speaks the Gemini API only (POST <base>/v1beta/models/<model>:streamGenerateContent?alt=sse, and
:generateContent); its OpenAI-compatible mode, meant for local servers, drops every image a tool returns and never sends a
reasoning effort, so it cannot run a RoboARC task. A model served only through Chat Completions (GitHub Copilot's proxy for
Gemini 4 Argon) is reached by translating each request here, in the gateway, and each answer back; the user decided
2026-11-01 that runs through it are Antigravity + Gemini's official runs. What is mapped, and nothing else:

  request (Gemini -> Chat Completions)
    systemInstruction                 -> a system message
    model content: text               -> assistant content; thought parts (the model's thinking) are not sent back
                   functionCall       -> assistant tool_calls (id kept; one is made, call_<n>, where the SDK gave none)
                   thoughtSignature   -> the assistant message's extra fields, when it is one this module wrote (below)
    functionResponse (any role)       -> a tool message: response.output (or .error) when that is all it holds, else the
                                         response as JSON; its images (parts[].inlineData image/*) follow the turn's tool
                                         messages as one user message (a tool message carries text only)
    user content: text / inlineData   -> user content: text / image_url data: URL (another type: a note that it was left out)
    functionDeclarations              -> tools [{type: function}]; parametersJsonSchema as it is, a Gemini `parameters`
                                         schema with its types in lower case and propertyOrdering dropped
    any other tool (googleSearch, urlContext, codeExecution, ...): the request is refused (gateway.py: server_tool), never
                                         translated: the provider would run it
    toolConfig.functionCallingConfig  -> tool_choice: AUTO / VALIDATED auto, ANY required (one allowed name: that
                                         function), NONE none
    generationConfig                  -> maxOutputTokens max_tokens, temperature, topP top_p, stopSequences stop,
                                         thinkingConfig.thinkingLevel reasoning_effort (EFFORT below; the level the SDK
                                         was given, EFFORT of the runner, so the run's label is on the wire)
    :streamGenerateContent            -> stream: true, stream_options.include_usage
    the URL's model                   -> model
  answer (Chat Completions -> Gemini)
    content                           -> text parts (streamed as they come)
    reasoning_content / reasoning_text / reasoning (text)
                                      -> thought parts (thought: true)
    tool_calls                        -> functionCall parts (arguments parsed; text that is not a JSON object is given as
                                         {"arguments": <text>}), in the last event, once complete
    other fields of the message       -> kept, with the reasoning text, in a thoughtSignature (base64 of SIG + JSON) on
                                         the first functionCall part, or on an empty text part; the SDK sends it back
                                         with the history and the next request restores the fields: whatever the
                                         server needs to see again of its own earlier turns (an opaque reasoning token)
    finish_reason                     -> finishReason (FINISH); usage -> usageMetadata (usage_metadata)
    an HTTP error                     -> the same status, a Gemini error body
"""
from __future__ import annotations

import base64
import json
import re
from typing import NamedTuple

ROUTE = re.compile(r"^(?P<prefix>(?:/[A-Za-z0-9._-]+)*)/v1beta/models/(?P<model>[A-Za-z0-9._-]+)"
                   r":(?P<method>generateContent|streamGenerateContent)$")
SIG = b"rcb-gw1:"            # the start of a thoughtSignature this module wrote
EFFORT = {"minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "extra_high": "xhigh"}
FINISH = {"stop": "STOP", "tool_calls": "STOP", "function_call": "STOP", "length": "MAX_TOKENS",
          "content_filter": "SAFETY"}
TOOL_CHOICE = {"AUTO": "auto", "VALIDATED": "auto", "ANY": "required", "NONE": "none"}
REASONING = ("reasoning_content", "reasoning_text", "reasoning")   # message fields holding the model's thinking as text
STANDARD = {"role", "content", "tool_calls", "refusal", "function_call", "audio", "annotations"}
STATUS = {400: "INVALID_ARGUMENT", 401: "UNAUTHENTICATED", 403: "PERMISSION_DENIED", 404: "NOT_FOUND",
          408: "DEADLINE_EXCEEDED", 409: "ABORTED", 429: "RESOURCE_EXHAUSTED", 499: "CANCELLED", 500: "INTERNAL",
          501: "UNIMPLEMENTED", 502: "UNAVAILABLE", 503: "UNAVAILABLE", 504: "DEADLINE_EXCEEDED"}


class Route(NamedTuple):
    prefix: str      # the base URL's path, before /v1beta: the Chat Completions server's is <prefix>/chat/completions
    model: str
    stream: bool


class Untranslatable(ValueError):
    """A Gemini request this module cannot carry faithfully."""


def match(path: str) -> Route | None:
    """The Gemini model call a request path (without its query) names, or None."""
    m = ROUTE.match(path)
    if m and any(s in (".", "..") for s in m["prefix"].split("/")):
        return None
    return Route(m["prefix"], m["model"], m["method"] == "streamGenerateContent") if m else None


# --------------------------------------------------------------------------- request
def _schema(s):
    """A Gemini `parameters` schema as JSON Schema: types in lower case, propertyOrdering dropped."""
    if isinstance(s, list):
        return [_schema(x) for x in s]
    if not isinstance(s, dict):
        return s
    out = {}
    for k, v in s.items():
        if k == "propertyOrdering":
            continue
        if k == "type" and isinstance(v, str):
            out[k] = v.lower()
        elif k == "properties" and isinstance(v, dict):
            out[k] = {name: _schema(p) for name, p in v.items()}
        else:
            out[k] = _schema(v)
    return out


def _tools(req: dict) -> tuple[list[dict], list[str]]:
    tools, provider = [], set()
    entries = req.get("tools") or []
    if not isinstance(entries, list):
        raise Untranslatable("tools is not a list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise Untranslatable("a tool is not an object")
        for key, val in entry.items():
            if key not in ("functionDeclarations", "function_declarations"):
                provider.add(key[:64])
                continue
            for fd in val or []:
                if not isinstance(fd, dict) or not isinstance(fd.get("name"), str):
                    raise Untranslatable("a function declaration without a name")
                params = fd.get("parametersJsonSchema", fd.get("parameters_json_schema"))
                if params is None:
                    params = _schema(fd.get("parameters")) if fd.get("parameters") is not None else {"type": "object",
                                                                                                    "properties": {}}
                fn = {"name": fd["name"], "parameters": params}
                if fd.get("description"):
                    fn["description"] = fd["description"]
                tools.append({"type": "function", "function": fn})
    return tools, sorted(provider)


def _signature_fields(part: dict) -> dict:
    """The assistant fields a thoughtSignature of this module carries, else {}."""
    sig = part.get("thoughtSignature", part.get("thought_signature"))
    if not isinstance(sig, str):
        return {}
    try:
        raw = base64.b64decode(sig, validate=True)
    except ValueError:
        return {}
    if not raw.startswith(SIG):
        return {}
    try:
        fields = json.loads(raw[len(SIG):])
    except ValueError:
        return {}
    return fields if isinstance(fields, dict) else {}


def _image(part: dict) -> dict:
    data = part.get("inlineData", part.get("inline_data")) or {}
    mime = str(data.get("mimeType", data.get("mime_type")) or "")
    if mime.startswith("image/") and isinstance(data.get("data"), str):
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data['data']}"}}
    return {"type": "text", "text": f"[a {mime or 'file'} attachment was left out: this model API takes images only]"}


def _tool_text(response) -> str:
    if isinstance(response, dict) and len(response) == 1:
        (k, v), = response.items()
        if k in ("output", "error") and isinstance(v, str):
            return v
    return json.dumps(response, ensure_ascii=False)


def _content(texts: list[str], extra: list[dict]):
    """A message's content: a string, or text and image blocks when there are images."""
    if not extra:
        return "\n".join(texts)
    return [*({"type": "text", "text": t} for t in texts if t), *extra]


def to_chat(req: dict, model: str, stream: bool) -> tuple[dict, list[str]]:
    """(the Chat Completions request, the provider-run tools the Gemini request declares: if any, the caller refuses it
    and the request is not to be sent)."""
    tools, provider = _tools(req)
    messages: list[dict] = []
    system = req.get("systemInstruction", req.get("system_instruction"))
    if isinstance(system, dict):
        text = "\n".join(p["text"] for p in system.get("parts") or [] if isinstance(p, dict) and isinstance(p.get("text"), str))
        if text:
            messages.append({"role": "system", "content": text})
    elif isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    open_calls: list[tuple[str, str]] = []           # (id, name) of calls not yet answered, oldest first
    images: list[dict] = []                          # the images of the tool results since the last non-tool message
    n_calls = 0

    def flush_images() -> None:
        if images:
            messages.append({"role": "user", "content": [{"type": "text", "text": "[the images the tool calls above returned]"},
                                                         *images]})
            images.clear()

    for content in req.get("contents") or []:
        if not isinstance(content, dict):
            raise Untranslatable("a content is not an object")
        role = content.get("role") or "user"
        texts, blocks, calls, fields = [], [], [], {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                raise Untranslatable("a part is not an object")
            fields.update(_signature_fields(part))
            if "functionResponse" in part or "function_response" in part:
                fr = part.get("functionResponse", part.get("function_response")) or {}
                cid = fr.get("id")
                name = fr.get("name") or ""
                if not cid:
                    i = next((i for i, (_, n) in enumerate(open_calls) if n == name), 0 if open_calls else None)
                    cid = open_calls[i][0] if i is not None else f"call_{n_calls}"
                open_calls[:] = [c for c in open_calls if c[0] != cid]
                if not messages or messages[-1]["role"] != "tool":
                    flush_images()
                messages.append({"role": "tool", "tool_call_id": cid, "content": _tool_text(fr.get("response", {}))})
                images.extend(_image(p) for p in fr.get("parts") or []
                              if isinstance(p, dict) and ("inlineData" in p or "inline_data" in p))
            elif "functionCall" in part or "function_call" in part:
                fc = part.get("functionCall", part.get("function_call")) or {}
                n_calls += 1
                cid = fc.get("id") or f"call_{n_calls}"
                calls.append({"id": cid, "type": "function", "function": {
                    "name": fc.get("name") or "", "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False)}})
                open_calls.append((cid, fc.get("name") or ""))
            elif part.get("thought"):
                continue                              # the model's own thinking: not sent back (its fields ride the signature)
            elif isinstance(part.get("text"), str):
                if part["text"]:
                    texts.append(part["text"])
            elif "inlineData" in part or "inline_data" in part:
                blocks.append(_image(part))
            elif "fileData" in part or "file_data" in part:
                blocks.append({"type": "text", "text": "[a file given by URL was left out]"})
        if not (texts or blocks or calls or fields):
            continue                                  # a content of tool results only (done above) or of thoughts only
        flush_images()
        if role == "model":
            if blocks:
                raise Untranslatable("an image in the model's own turn")
            # with tool calls and no text the content is null (as the SDK's own OpenAI mode sends it), else a string
            msg = {"role": "assistant", "content": "\n".join(texts) if texts or not calls else None}
            if calls:
                msg["tool_calls"] = calls
            msg.update({k: v for k, v in fields.items() if k not in STANDARD})
            messages.append(msg)
        else:
            messages.append({"role": "user", "content": _content(texts, blocks)})
    flush_images()
    out = {"model": model, "messages": messages}
    if tools:
        out["tools"] = tools
    fcc = ((req.get("toolConfig") or req.get("tool_config") or {}).get("functionCallingConfig")
           or (req.get("toolConfig") or req.get("tool_config") or {}).get("function_calling_config") or {})
    mode = str(fcc.get("mode") or "").upper()
    if mode in TOOL_CHOICE and tools:
        names = fcc.get("allowedFunctionNames", fcc.get("allowed_function_names")) or []
        out["tool_choice"] = ({"type": "function", "function": {"name": names[0]}} if mode == "ANY" and len(names) == 1
                              else TOOL_CHOICE[mode])
    gen = req.get("generationConfig", req.get("generation_config")) or {}
    for src, dst in (("maxOutputTokens", "max_tokens"), ("temperature", "temperature"), ("topP", "top_p"),
                     ("stopSequences", "stop")):
        if gen.get(src) is not None:
            out[dst] = gen[src]
    level = (gen.get("thinkingConfig") or gen.get("thinking_config") or {}).get("thinkingLevel")
    if isinstance(level, str) and level:
        out["reasoning_effort"] = EFFORT.get(level.lower(), level.lower())
    if stream:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out, provider


# --------------------------------------------------------------------------- answer
def usage_metadata(u: dict) -> dict:
    """Chat Completions usage as Gemini's usageMetadata: candidates exclude the thinking, which thoughts count."""
    if not isinstance(u, dict):
        return {}
    prompt, completion = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
    thoughts = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    out = {"promptTokenCount": prompt, "candidatesTokenCount": max(0, completion - thoughts),
           "totalTokenCount": int(u.get("total_tokens") or prompt + completion)}
    if thoughts:
        out["thoughtsTokenCount"] = thoughts
    if cached:
        out["cachedContentTokenCount"] = cached
    return out


def _args(text: str) -> dict:
    try:
        v = json.loads(text) if text.strip() else {}
    except ValueError:
        v = None
    return v if isinstance(v, dict) else {"arguments": text}


def _signature(fields: dict) -> str:
    return base64.b64encode(SIG + json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode()).decode()


class Answer:
    """One answer, whole or streamed, as it accumulates: the parts not yet sent and what the last event needs."""

    def __init__(self):
        self.calls: dict[int, dict] = {}       # tool_calls by index: id, name, arguments text
        self.fields: dict = {}                 # the message's other fields, for the signature
        self.finish = None
        self.usage: dict = {}
        self.model = None

    def take(self, msg: dict) -> list[dict]:
        """The parts a message (or a stream's delta) gives now: text and thoughts; tool calls are kept for the end."""
        parts = []
        for k in REASONING:
            if isinstance(msg.get(k), str) and msg[k]:
                parts.append({"text": msg[k], "thought": True})
                self.fields[k] = self.fields.get(k, "") + msg[k]
        if isinstance(msg.get("content"), str) and msg["content"]:
            parts.append({"text": msg["content"]})
        for i, tc in enumerate(msg.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            c = self.calls.setdefault(tc.get("index", i) if isinstance(tc.get("index", i), int) else i,
                                      {"id": None, "name": "", "arguments": ""})
            c["id"] = c["id"] or tc.get("id")
            fn = tc.get("function") or {}
            c["name"] = c["name"] or fn.get("name") or ""
            if isinstance(fn.get("arguments"), str):
                c["arguments"] += fn["arguments"]
        for k, v in msg.items():
            if k in STANDARD or k in REASONING or v is None:
                continue
            self.fields[k] = self.fields[k] + v if isinstance(v, str) and isinstance(self.fields.get(k), str) else v
        return parts

    def last_parts(self) -> list[dict]:
        parts = []
        for i in sorted(self.calls):
            c = self.calls[i]
            fc = {"name": c["name"], "args": _args(c["arguments"])}
            if c["id"]:
                fc["id"] = c["id"]
            parts.append({"functionCall": fc})
        if self.fields:
            if parts:
                parts[0]["thoughtSignature"] = _signature(self.fields)
            else:
                parts.append({"text": "", "thoughtSignature": _signature(self.fields)})
        return parts

    def event(self, parts: list[dict], last: bool = False) -> dict:
        cand = {"content": {"role": "model", "parts": parts}, "index": 0}
        out = {"candidates": [cand]}
        if last:
            cand["finishReason"] = FINISH.get(self.finish, "STOP" if self.finish is None else "OTHER")
            if self.usage:
                out["usageMetadata"] = usage_metadata(self.usage)
        if self.model:
            out["modelVersion"] = self.model
        return out


def from_chat(resp: dict) -> dict:
    """A whole (non-streamed) Chat Completions answer as a Gemini generateContent answer."""
    a = Answer()
    a.model, a.usage = resp.get("model"), resp.get("usage") or {}
    choice = (resp.get("choices") or [{}])[0] or {}
    a.finish = choice.get("finish_reason")
    parts = a.take(choice.get("message") or {})
    return a.event(parts + a.last_parts(), last=True)


class Stream:
    """A Chat Completions SSE stream as a Gemini SSE stream (alt=sse): feed() the upstream's bytes as they come and send
    what it returns; close() gives the last event (tool calls, finishReason, usage)."""

    def __init__(self):
        self.a, self.buf, self.closed = Answer(), b"", False

    @staticmethod
    def _sse(obj: dict) -> bytes:
        return b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\r\n\r\n"

    def feed(self, data: bytes) -> bytes:
        self.buf += data
        out = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            if obj.get("error"):
                out.append(self._sse(error_body(500, obj)))
                continue
            self.a.model = self.a.model or obj.get("model")
            if isinstance(obj.get("usage"), dict):
                self.a.usage = obj["usage"]
            for ch in obj.get("choices") or []:
                if not isinstance(ch, dict) or ch.get("index", 0) != 0:
                    continue
                parts = self.a.take(ch.get("delta") or ch.get("message") or {})
                if ch.get("finish_reason"):
                    self.a.finish = ch["finish_reason"]
                if parts:
                    out.append(self._sse(self.a.event(parts)))
        return b"".join(out)

    def close(self) -> bytes:
        if self.closed:
            return b""
        self.closed = True
        tail = self.feed(b"\n") if self.buf else b""
        return tail + self._sse(self.a.event(self.a.last_parts(), last=True))


class Whole:
    """A non-streamed answer: feed() keeps the bytes, close() gives the translated document."""

    def __init__(self):
        self.buf = b""

    def feed(self, data: bytes) -> bytes:
        self.buf += data
        return b""

    def close(self) -> bytes:
        try:
            resp = json.loads(self.buf)
        except ValueError:
            return json.dumps(error_body(502, {"error": {"message": "the model server's answer is not JSON"}})).encode()
        return json.dumps(from_chat(resp if isinstance(resp, dict) else {}), ensure_ascii=False).encode()


class Error:
    """An HTTP error of the Chat Completions server: feed() keeps its body, close() gives the Gemini error body."""

    def __init__(self, status: int):
        self.status, self.buf = status, b""

    def feed(self, data: bytes) -> bytes:
        self.buf += data
        return b""

    def close(self) -> bytes:
        return json.dumps(error_body(self.status, self.buf), ensure_ascii=False).encode()


def error_body(status: int, upstream) -> dict:
    """A Gemini error body for an HTTP error (or an error event) of the Chat Completions server; `upstream` is its body
    (bytes or a parsed object) or a message."""
    msg = upstream
    if isinstance(upstream, bytes):
        try:
            upstream = json.loads(upstream)
        except ValueError:
            msg = upstream.decode("utf-8", "replace")[:2000]
    if isinstance(upstream, dict):
        e = upstream.get("error", upstream)
        msg = e.get("message") if isinstance(e, dict) and e.get("message") else json.dumps(e)[:2000]
    return {"error": {"code": status, "message": str(msg), "status": STATUS.get(status, "UNKNOWN")}}
