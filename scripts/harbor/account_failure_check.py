#!/usr/bin/env python3
"""account_failure_check.py <trial dir> [...] | --selftest: did a finished trial hit a model-account or config failure (out of
credits, rate-limited, logged out, a missing key), so that every further job of the lane would fail within minutes?
run_agent_batch.sh stops the lane on exit 4 (a runner-side tool, not task code).

Structured evidence only, from what Harbor's native agents write (2026-09-28: a grep of codex.txt once stopped a lane on
Codex's "Reconnecting... 1/5" notice after a transient server error, whose request ID happened to contain "429", in a trial
whose model calls had all answered 200 and that ended at its time budget):
  quota    Codex's own rate_limits (payload.rate_limits in the last session rollout, parsed): rate_limit_reached_type set,
           or a window at >= 100 % used;
  codex    the LAST root-level Codex event of <trial>/agent/codex.txt (the parsed object's own "type"; a failure Codex
           recovered from is followed by other events) is turn.failed / error and its message names 401 / 402 / 429 /
           usage limit / Too Many Requests / Missing environment variable;
  claude   the LAST root-level event of <trial>/agent/claude-code.txt is Claude Code's result with is_error set and a text
           that names the same failures (or an invalid key, a low credit balance, a needed login).
  gateway  (only with the optional usage gateway, run_agent_batch.sh USAGE_GATEWAY) its <trial>/gateway/totals.json:
           the trial's last model calls failed on the way (failures_in_a_row: the upstream unreachable, the gateway's
           502; a stream broken off; a provider 5xx). One or more: the trial is an infrastructure failure ("infra"; write_task_results.py: not a result, not
           the agent's 0); STOP_AFTER or more in a row: the upstream is down for the lane too ("stop"). Also "infra": the
           runner found the gateway dead after the job (<job>/gateway_down.json, run_agent_batch.sh).
  dsh      (AGENT=dsh, scripts/harbor/agents/dsh_acp.py) writes no structured account record of its own (agent/acp.txt is
           the ACP runner's output): only the gateway rule applies; a failing account shows as Harbor exceptions, so
           its trials are not results (write_task_results.py), but the lane does not stop by itself.
  kimi     (AGENT=kimi-code, scripts/harbor/agents/kimi_code_cli.py: agent/kimi-code.txt; the gateway is required) the
           LAST model call of <trial>/gateway/usage.jsonl (a forwarded /chat/completions request) was answered 401, 402
           or 429 by the provider: a key Moonshot does not accept, or a 429 Kimi Code gave up on (its retries ran out,
           or exceeded_current_quota_error, which it does not retry: the account's balance is spent). A 429 the CLI
           retried past is followed by other calls.
Prints one JSON line per trial {"trial", "stop", "why", "infra"} (long tokens in messages masked); exit 4 if any trial
should stop its lane, 0 otherwise. --selftest checks the rules on synthetic trials (exit 0 = all as expected)."""
import glob, json, os, re, sys, tempfile

PAT = re.compile(r"\b(401|402|429)\b|usage limit|Too Many Requests|Missing environment variable|Payment Required|Unauthorized|"
                 r"invalid api key|credit balance|please run /login|authentication_error|rate_limit_error", re.I)


def mask(s: str) -> str:
    return re.sub(r"[A-Za-z0-9_\-+/=.]{24,}", "<masked>", s)[:200]


STOP_AFTER = 3   # model calls in a row that failed on the way at the end of a trial: the harness gave up retrying


def check(t: str) -> dict:
    why, infra = [], []
    name = os.path.basename(t.rstrip("/"))
    gw = os.path.join(t, "gateway", "totals.json")             # the usage gateway refused it: the trial's own budget, not
    try:                                                        # the account (PROTOCOL §5); never stop the lane for it
        tot = json.load(open(gw))
        if not isinstance(tot, dict):
            raise ValueError
        if tot.get("over_token_budget"):
            return {"trial": name, "stop": False, "why": ["over its token budget (gateway)"], "infra": []}
        n = int(tot.get("failures_in_a_row") or 0)
        if n > 0:
            msg = f"gateway: the trial's last {n} model call(s) failed on the way ({mask(str(tot.get('last_upstream_failure')))})"
            infra.append(msg)
            if n >= STOP_AFTER:
                why.append(msg)
    except (OSError, ValueError, TypeError):
        pass
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(t.rstrip("/"))), "gateway_down.json")):
        infra.append("gateway: the usage gateway was down when the job ended (run_agent_batch.sh health check)")
    for f in sorted(glob.glob(os.path.join(t, "agent", "sessions", "**", "rollout-*.jsonl"), recursive=True),
                    key=os.path.getmtime)[-1:]:
        last = None
        for line in open(f, errors="replace"):
            if '"rate_limits"' not in line:
                continue
            try:
                rl = (json.loads(line).get("payload") or {}).get("rate_limits")
            except (ValueError, AttributeError):
                continue
            if isinstance(rl, dict):
                last = rl
        if last:
            used = [float((last.get(w) or {}).get("used_percent") or 0) for w in ("primary", "secondary")]
            if last.get("rate_limit_reached_type") or max(used) >= 100:
                why.append(f"quota: rate_limit_reached_type={last.get('rate_limit_reached_type')} used_percent={used}")
    for f in glob.glob(os.path.join(t, "agent", "codex.txt")):
        last = None
        for line in open(f, errors="replace"):
            if line.startswith("{"):
                try:
                    last = json.loads(line)
                except ValueError:
                    continue
        if isinstance(last, dict) and last.get("type") in ("turn.failed", "error"):
            err = last.get("error")
            msg = str(err.get("message") if isinstance(err, dict) else last.get("message") or "")
            if PAT.search(msg):
                why.append(f"codex: {last['type']}: {mask(msg)}")
    for f in glob.glob(os.path.join(t, "agent", "claude-code.txt")):
        last = None
        for line in open(f, errors="replace"):
            if line.startswith("{"):
                try:
                    last = json.loads(line)
                except ValueError:
                    continue
        if isinstance(last, dict) and last.get("type") == "result" and last.get("is_error"):
            msg = str(last.get("result") or last.get("subtype") or "")
            if PAT.search(msg):
                why.append(f"claude: result: {mask(msg)}")
    usage = os.path.join(t, "gateway", "usage.jsonl")
    if os.path.exists(os.path.join(t, "agent", "kimi-code.txt")) and os.path.exists(usage):
        last = None
        for line in open(usage, errors="replace"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and str(rec.get("path", "")).endswith("/chat/completions") and not rec.get("refused") \
                    and rec.get("forwarded") is not False:
                last = rec
        if last and last.get("status") in (401, 402, 429):
            why.append(f"kimi: the trial's last model call was answered HTTP {last['status']} by the provider")
    return {"trial": name, "stop": bool(why), "why": why, "infra": infra + [w for w in why if w not in infra]}


def selftest() -> int:
    def trial(root, name, rate_limits=None, codex=(), claude=None, totals=None, down=False):
        t = os.path.join(root, name, f"{name}__x")
        os.makedirs(os.path.join(t, "agent", "sessions", "2026", "09", "28"))
        if totals is not None:
            os.makedirs(os.path.join(t, "gateway"))
            with open(os.path.join(t, "gateway", "totals.json"), "w") as f:
                json.dump(totals, f)
        if down:
            open(os.path.join(root, name, "gateway_down.json"), "w").write("{}")
        if claude is not None:
            with open(os.path.join(t, "agent", "claude-code.txt"), "w") as f:
                f.writelines(json.dumps(e) + "\n" for e in claude)
        if rate_limits is not None:
            with open(os.path.join(t, "agent", "sessions", "2026", "09", "28", "rollout-x.jsonl"), "w") as f:
                f.write(json.dumps({"type": "event_msg", "payload": {"type": "token_count", "rate_limits": rate_limits}}) + "\n")
        with open(os.path.join(t, "agent", "codex.txt"), "w") as f:
            f.write("WARNING: a plain text line\n")
            f.writelines(json.dumps(e) + "\n" for e in codex)
        return t

    ok_rl = {"primary": {"used_percent": 19.0, "window_minutes": 10080}, "secondary": None}
    done = [{"type": "turn.completed"}]
    # the false positive of 2026-09-28: Codex's reconnect notice after a transient 500, its request ID containing "429",
    # then the run went on (every other call 200); and "429" / usage limit inside a tool result
    fp = [{"type": "error", "message": "Reconnecting... 1/5 (The server had an error while processing your request. Please "
           "include the request ID 7c1e0429f3ab in your message.)"}, {"type": "item.completed", "item": {
           "type": "command_execution", "aggregated_output": 'x = 429 ... {"type":"error","code":429} usage limit'}}] + done
    tot = lambda n, last="unreachable: ConnectionRefusedError": {"requests": 4, "input": 9, "output": 3, "upstream_failures": n + 1,
                                                                  "failures_in_a_row": n, "last_upstream_failure": last}
    cases = {  # name: (kwargs, expected stop[, expected infra])
        "rate_limit_reached_type": (dict(rate_limits={**ok_rl, "rate_limit_reached_type": "primary"}, codex=done), True),
        "window_100_percent": (dict(rate_limits={"primary": {"used_percent": 100.0}}, codex=done), True),
        "codex_turn_failed_usage_limit": (dict(rate_limits=ok_rl, codex=[{"type": "turn.failed", "error": {
            "message": "You've hit your usage limit. Upgrade to Pro or try again later."}}]), True),
        "codex_error_missing_env": (dict(rate_limits=None, codex=[{"type": "error", "message":
            "Missing environment variable: `OPENAI_API_KEY`."}]), True),
        "codex_error_401": (dict(rate_limits=None, codex=[{"type": "turn.failed", "error": {"message": "unexpected status 401 Unauthorized"}}]), True),
        "false_positive_2026_09_28": (dict(rate_limits=ok_rl, codex=fp), False),
        "recovered_429": (dict(rate_limits=ok_rl, codex=[
            {"type": "error", "message": "stream error: 429 Too Many Requests; retrying"}, {"type": "item.completed"}] + done), False),
        "backend_400": (dict(rate_limits=ok_rl, codex=[{"type": "turn.failed", "error": {
            "message": "prompt_cache_retention is not supported on this model"}}]), False),
        "timeout": (dict(rate_limits=ok_rl, codex=[{"type": "item.started"}]), False),
        "claude_invalid_key": (dict(claude=[{"type": "system"}, {"type": "result", "subtype": "success", "is_error": True,
                                                                  "result": "Invalid API key · Please run /login"}]), True),
        "claude_credit": (dict(claude=[{"type": "result", "is_error": True, "result": "Credit balance is too low"}]), True),
        "claude_ok": (dict(claude=[{"type": "assistant"}, {"type": "result", "subtype": "success", "is_error": False, "result": "done"}]), False),
        "claude_tool_429_then_ok": (dict(claude=[{"type": "user", "message": "HTTP 429 from a tool"}, {"type": "result", "is_error": False,
                                                                                                         "result": "finished"}]), False),
        # the gateway's upstream unreachable / a stream broken off / a provider 5xx at the end of the trial
        "gateway_502_x5": (dict(codex=[{"type": "turn.failed", "error": {"message": "stream disconnected before completion: "
                                "error sending request for url (http://172.17.0.1:4000/u/api.openai.com/v1/responses)"}}],
                                totals=tot(5)), True, True),
        "gateway_stream_broken_once": (dict(claude=[{"type": "result", "is_error": True, "result": "API Error: Connection error."}],
                                            totals=tot(1, "stream broken off: TimeoutError")), False, True),
        "gateway_failure_then_ok": (dict(codex=done, totals=tot(0)), False, False),
        "gateway_down_after_job": (dict(claude=[{"type": "result", "is_error": True, "result": "API Error: Connection error."}],
                                        down=True), False, True),
        "gateway_bad_totals": (dict(codex=done, totals=[1, 2]), False, False),
    }
    bad = 0
    with tempfile.TemporaryDirectory() as root:
        for name, (kw, want, *infra) in cases.items():
            r = check(trial(root, name, **kw))
            want_infra = infra[0] if infra else want
            good = r["stop"] == want and bool(r["infra"]) == want_infra
            bad += not good
            print(f"{'ok  ' if good else 'FAIL'} {name}: stop={r['stop']} (want {want}) infra={bool(r['infra'])} "
                  f"(want {want_infra}) {r['why'] or r['infra']}")
        for name, totals, want in (("dsh_429_in_output", None, False), ("dsh_gateway_502_x5", tot(5), True)):
            t = os.path.join(root, name, f"{name}__x")   # a dsh trial: Harbor's ACP output, no codex.txt / claude-code.txt
            os.makedirs(os.path.join(t, "agent"))
            open(os.path.join(t, "agent", "acp.txt"), "w").write("HTTP 429 Too Many Requests; Unauthorized\n")
            if totals is not None:
                os.makedirs(os.path.join(t, "gateway"))
                json.dump(totals, open(os.path.join(t, "gateway", "totals.json"), "w"))
            r = check(t)
            good = r["stop"] == want
            bad += not good
            print(f"{'ok  ' if good else 'FAIL'} {name}: stop={r['stop']} (want {want})")
        chat = lambda status, **k: {"path": "/v1/chat/completions", "status": status, **k}
        kimi = {  # a Kimi Code trial (agent/kimi-code.txt): its model calls as the gateway recorded them
            "kimi_quota_429": ([chat(200), chat(429)], True),
            "kimi_bad_key_401": ([chat(401)], True),
            "kimi_429_retried": ([chat(429), chat(429), chat(200)], False),
            "kimi_refused_last": ([chat(200), chat(403, refused="server_tool")], False),
            "kimi_upload_after": ([chat(429), {"path": "/v1/files", "status": 200}], True),
            "kimi_unreachable_last": ([chat(200), chat(502, forwarded=False)], False),
            "kimi_no_usage": (None, False)}
        for name, (recs, want) in kimi.items():
            t = os.path.join(root, name, f"{name}__x")
            os.makedirs(os.path.join(t, "agent"))
            open(os.path.join(t, "agent", "kimi-code.txt"), "w").write('{"role": "assistant", "content": "429"}\n')
            if recs is not None:
                os.makedirs(os.path.join(t, "gateway"))
                with open(os.path.join(t, "gateway", "usage.jsonl"), "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in recs)
            r = check(t)
            good = r["stop"] == want
            bad += not good
            print(f"{'ok  ' if good else 'FAIL'} {name}: stop={r['stop']} (want {want}) {r['why']}")
        r = check(os.path.join(root, "no_such_trial"))
        bad += r["stop"]
        print(f"{'ok  ' if not r['stop'] else 'FAIL'} missing trial dir: stop={r['stop']} (want False)")
    print(f"selftest: {len(cases) + 3 + len(kimi) - bad} / {len(cases) + 3 + len(kimi)} as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    rows = [check(t) for t in sys.argv[1:]]
    for r in rows:
        print(json.dumps(r))
    sys.exit(4 if any(r["stop"] for r in rows) else 0)
