#!/usr/bin/env python3
"""Write per-task agent results for the results site (not into the repository, PROTOCOL D3): one folder per trial,
<out>/<task>/<run id>/t<k>/ (README.md, config.json, attempts.json, + replay.mp4, replay.last.png), and an index of the
task's trials, <out>/<task>/<run id>/README.md and trials.json. One run id per agent+model; the privileged and standard
twins each get their own.
usage: write_task_results.py <jobs_dir> <job_name> <run-id> "<agent label>" --out <dir> [--analysis <a Python file with ANALYSIS = {family: {"U": text, "L": text}}>]

Trials and seeds (RUNNING.md "Seeds", "Results"): a run gives each task K trials (k = 1..K), each from its own seed. The
batch runner records a job's trial and the seed of each of its tasks in <job>/seed.json (else $JOBS/.seeds/<job>.json);
a job without that record (a plain `harbor run`, a runner before seeds) numbers the trials of each task in it by their
start, k = 1, 2, ..., on the task's default seed.
config.json (RUNNING.md "Results", PROTOCOL v2.0): the trial's effective configuration (scripts/task_conventions.py
effective_config: protocol version, mode, cutoff, reset budget, slowdown S, seed, network, validated flag, digest), its
trial (`trial`, `trials`) and how its seed is known (`seed_source`: "runner" from the seed record, "default"), its result
(`success`, `fail` or `not_a_result` with the reason) and whether it is counted. Results are ranked only within one
comparable configuration (task_conventions.config_key), validated and unvalidated tasks together.
Usage recording is optional (PROTOCOL H.6, RUNNING.md "Usage recording"); config.json says what the result has:
The machine the trial ran on, from the batch runner's <job>/host.json (scripts/harbor/host_info.py): `host` (cpu_model,
cpus, memory_gb, gpus [{index, name, driver, memory_mib}], docker, harbor; null for a job without it) and, when the runner
placed the job on a GPU (`gpu`, its index), that GPU's name and driver (`gpu_name`, `gpu_driver`).
`usage_gateway` (true / false: whether the runner routed the trial through the usage gateway, from
<job>/usage_recording.json; null when the job has no such record, e.g. a plain `harbor run`), `usage_source` (where the
token counts come from: "usage gateway" (<trial>/gateway/totals.json), "harness" (Harbor's result.json agent_result), or
null: none recorded), `usage_recorded` (usage_source is not null), `token_budget` (the budget the gateway enforced,
{"input", "output"}; null without the gateway) and `image_cap` (the most images the gateway let a model request carry,
0 = no cap; null without the gateway, or for a job recorded before the cap existed). An AGENT=antigravity result also
has `api_translation` ("gemini-chat": the gateway translated the agent's Gemini requests for a Chat Completions server,
scripts/usage_gateway/gemini_chat.py); no other result has the key. A result whose gateway also capped the size of a
request (GATEWAY_IMAGE_BYTES) has `image_bytes`, that size; no other result has the key.
attempts.json: every attempt of this task x run id x trial seen so far. RUNNING.md "Infrastructure failures": an
infrastructure failure (not a result) allows at most one full retry with the same instance, seed and configuration; the
voided attempt is never counted, an agent's own failure is never retried, and more than 2 attempts of one trial is
flagged (`attempts_over_cap`). The K trials of a task are not attempts of one another."""
import json, pathlib, shutil, sys, re, glob, datetime as dt, ast
sys.path[:0] = [str(pathlib.Path(__file__).resolve().parent), str(pathlib.Path(__file__).resolve().parents[1])]; from task_names import base, canonical, mode as mode_of  # noqa: E402
import account_failure_check  # noqa: E402
import task_conventions as tc  # noqa: E402
import retry_policy as rp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_ATTEMPTS = 1 + rp.MAX_INFRA_RETRIES   # PROTOCOL Appendix H.5: the first attempt and at most one retry after an infrastructure failure
OUTCOME = {"not_a_result": "infra_failure", "success": "success"}   # anything else: the agent's failure (a result)
OVERRIDES = ("RCB_RESET_BUDGET", "RCB_SLOWDOWN")   # run-time overrides of a task's declared settings (PROTOCOL §2)


def span(x):
    try:
        a = dt.datetime.fromisoformat(x["started_at"].replace("Z", "+00:00")); b = dt.datetime.fromisoformat(x["finished_at"].replace("Z", "+00:00")); return round((b - a).total_seconds())
    except Exception: return None


def last_msgs(trial, k=3):
    try:
        d = json.load(open(glob.glob(str(trial / "agent/trajectory.json"))[0]))
        return [s["message"].strip().replace("\n", " ") for s in d.get("steps", []) if s.get("source") == "agent" and (s.get("message") or "").strip()][-k:]
    except Exception: return []


NOT_A_RESULT = "**not a result** — infrastructure error ({}): run it again; it is not the agent's 0"


def not_a_result(trial, r, rw):
    """Why a trial that is not a success is no result either (PROTOCOL §2.6, §2.12: an infrastructure failure is neither a
    success nor the agent's 0; only the agent's own end, its deliverable or the cutoff, AgentTimeoutError, is a result),
    or None. A trial whose verifier never ran (no reward.json: `docker compose up` failed, e.g. a CPU
    cap above the host's; the verifier timed out) was published as the agent's 0."""
    if rw.get("infra_error") or rw.get("grader_error"):
        return "`infra_error` / `grader_error` in the verifier line"
    ex = (r.get("exception_info") or {}).get("exception_type") or ""
    if not rw:
        return "the verifier wrote no reward.json" + (f"; Harbor: `{ex}`" if ex else "")
    if ex and ex != "AgentTimeoutError":
        return f"Harbor ended the trial with `{ex}`"
    try:
        chk = account_failure_check.check(str(trial))
        if chk.get("stop") and not any(w.startswith("gateway:") for w in chk.get("why", [])):
            return "a model-account or configuration failure (scripts/harbor/account_failure_check.py)"
        # the usage gateway died, or the trial's last model calls failed on the way (upstream unreachable,
        # a stream broken off, a provider 5xx): the agent could not reach its model
        gw = [w for w in chk.get("infra", []) if w.startswith("gateway:")]
        if gw:
            return "the model API could not be reached through the usage gateway (" + gw[0].removeprefix("gateway: ") + ")"
    except Exception:
        pass
    return None


def load_json(p):
    try:
        d = json.load(open(p))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def load_reward(trial):
    return load_json(trial / "verifier/reward.json")


def classify(trial, r, rw):
    """(status, reason, over the token budget): status `success`, `fail` (the agent's own result, counted) or
    `not_a_result` (infrastructure: voided, never counted)."""
    gw = load_json(trial / "gateway/totals.json")
    over = bool(gw.get("over_token_budget"))               # PROTOCOL §5: over the token budget = failed
    # PROTOCOL §2.6: infra_error (or a grader error) = not a result, neither a success nor the agent's 0; so is a trial
    # the verifier never graded, or that Harbor or the model account ended (not_a_result). A verdict of success with no
    # infrastructure flag stands; over the token budget is the agent's failure unless the verifier flagged infrastructure.
    infra = bool(rw.get("infra_error") or rw.get("grader_error"))
    ok = rw.get("success") == 1 and not over and not infra
    why = None if ok or (over and not infra) else not_a_result(trial, r, rw)
    return ("success" if ok else "not_a_result" if why else "fail"), why, over


def seed_record(trial):
    """The runner's record of the trial's job (<job>/seed.json, else <jobs>/.seeds/<job>.json): its trial and the seed of
    each task; {} when there is none."""
    job = trial.parent
    rec = load_json(job / "seed.json")
    return rec or load_json(job.parent / ".seeds" / f"{job.name}.json")


def trial_index(trial, task, r, rec=None):
    """(k, K) of the trial: the seed record's, else its place among the trials of the same task in its job, by start
    (K = their number)."""
    rec = seed_record(trial) if rec is None else rec
    if isinstance(rec.get("trial"), int) and task in (rec.get("tasks") or {}):
        return rec["trial"], rec.get("trials") if isinstance(rec.get("trials"), int) else None
    mine = []
    for t in trial.parent.glob("*__*"):
        rr = load_json(t / "result.json")
        if rr.get("task_name") and canonical(rr["task_name"].split("/")[-1]) == task:
            mine.append((started(rr), t.name))
    mine.sort()
    key = (started(r), trial.name)
    return (mine.index(key) + 1 if key in mine else 1), len(mine) or 1


def trial_config(trial, task, r):
    """The trial's effective configuration (task_conventions.effective_config) with its trial and how its seed is known.
    The seed: the runner's seed record (`seed_source = "runner"`), else the runner's copy of the task the trial ran from
    if it still exists ("runner copy"), else the task's default seed ("default"). Run-time overrides: the trial's recorded
    environment. The task version: the digest the seed record names (the committed task the runner copied), else Harbor's
    `task_checksum`; a trial of another version of the task (not the task's DIGEST now) keeps only what its name and its
    seed record say; the rest is "legacy"."""
    tdir = ROOT / "tasks" / task
    hc = load_json(trial / "config.json")
    env = {}
    for src in ((hc.get("environment") or {}).get("env"), (hc.get("agent") or {}).get("env"), hc.get("env")):
        if isinstance(src, dict):
            env.update({k: str(v) for k, v in src.items() if k in OVERRIDES})
    rec = seed_record(trial)
    ent = (rec.get("tasks") or {}).get(task) if isinstance(rec.get("tasks"), dict) else None
    seed, source = None, "default"
    if isinstance(ent, dict) and isinstance(ent.get("seed"), int):
        seed, source = ent["seed"], "runner"
    else:
        p = ((hc.get("task") or {}).get("path") or "")
        if p:
            copy = pathlib.Path(p) if pathlib.Path(p).is_absolute() else ROOT / p
            if copy.resolve() != tdir.resolve() and (copy / "task.toml").is_file():
                try:
                    seed, source = tc.seeds_of(tc.metadata_of(copy))["default"], "runner copy"
                except (ValueError, KeyError, TypeError):
                    pass
    cfg = tc.effective_config(tdir, seed=seed, env=env)
    cfg["seed_source"] = source
    k, n = trial_index(trial, task, r, rec)
    cfg["trial"], cfg["trials"] = k, n
    if rec.get("batch"):
        cfg["batch"] = rec["batch"]
    copy_sum = (r.get("task_checksum") or "").removeprefix("sha256:")
    want = ((ent or {}).get("digest") or "").removeprefix("sha256:") if source == "runner" else copy_sum
    have = cfg.get("digest", "").removeprefix("sha256:")
    cfg["digest_match"] = (want == have) if want and have else None
    if source == "runner" and copy_sum:
        cfg["copy_checksum"] = "sha256:" + copy_sum      # Harbor's digest of the runner's copy (seed, egress, CPU caps, GPU pin)
    place = load_json(trial.parent / "placement.json")    # the runner's GPU and reservation for the job (GPU_SCHED=1)
    if place:
        cfg["gpu"] = place.get("gpu")
    host = load_json(trial.parent / "host.json")          # the machine the runner ran the job on (host_info.py)
    cfg["host"] = host or None
    g = next((g for g in (host.get("gpus") or []) if g.get("index") == cfg.get("gpu")), None) if host else None
    if g:
        cfg["gpu_name"], cfg["gpu_driver"] = g.get("name"), g.get("driver")
    if cfg["digest_match"] is False and source in ("default", "runner"):   # another version of the task
        for key in ("cutoff_s", "reset_budget", "slowdown", "network", "validated", "oracle", "service_protocol"):
            cfg[key] = "legacy"
        if source == "default":
            cfg["seed"] = "legacy"
        cfg["protocol_version"] = "legacy"
        cfg["digest"] = "sha256:" + want
    return cfg


def usage_of(trial, r):
    """The trial's usage recording (module docstring): usage_gateway, usage_source, usage_recorded, token_budget,
    image_cap."""
    rec = load_json(trial.parent / "usage_recording.json")
    on = rec.get("usage_gateway") if isinstance(rec.get("usage_gateway"), bool) else None
    ar = r.get("agent_result") or {}
    if (trial / "gateway/totals.json").is_file():
        source = "usage gateway"
    elif ar.get("n_input_tokens") is not None or ar.get("n_output_tokens") is not None:
        source = "harness"
    else:
        source = None
    budget = rec.get("token_budget") if on and isinstance(rec.get("token_budget"), dict) else None
    cap = rec.get("image_cap") if on and type(rec.get("image_cap")) is int else None
    out = {"usage_gateway": on, "usage_source": source, "usage_recorded": source is not None, "token_budget": budget,
           "image_cap": cap}
    if on and isinstance(rec.get("api_translation"), str):     # AGENT=antigravity only (module docstring)
        out["api_translation"] = rec["api_translation"]
    if on and type(rec.get("image_bytes")) is int:              # GATEWAY_IMAGE_BYTES on only (module docstring)
        out["image_bytes"] = rec["image_bytes"]
    return out


def started(r):
    return r.get("started_at") or ""


def record_attempt(out, trial, r, status):
    """attempts.json in the trial's result folder: every attempt of this task x run id x trial, oldest first."""
    f = out / "attempts.json"
    led = load_json(f).get("attempts", [])
    key = r.get("id") or trial.name
    led = [a for a in led if a.get("id") != key]
    led.append({"id": key, "trial": trial.name, "job": trial.parent.name, "started_at": started(r), "status": status,
                "counted": status != "not_a_result"})
    led.sort(key=lambda a: a.get("started_at") or "")
    f.write_text(json.dumps({"attempts": led}, indent=1) + "\n")
    return led


def main(argv):
    jobs, job, slug, label = pathlib.Path(argv[1]), argv[2], argv[3], argv[4]
    if "--out" not in argv: sys.exit("--out <dir> is required: results are published on the results site, not committed (PROTOCOL D3)")
    out_root = pathlib.Path(argv[argv.index("--out") + 1]).resolve()
    if out_root == ROOT or ROOT in out_root.parents: sys.exit("--out must be outside the repository (PROTOCOL D3)")
    analysis = {}
    if "--analysis" in argv:
        src = pathlib.Path(argv[argv.index("--analysis") + 1]).read_text()
        m = re.search(r"ANALYSIS = (\{.*?\n\})\n", src, re.S)
        if m: analysis = ast.literal_eval(m.group(1))
    n = 0
    touched = set()
    trials = []
    for trial in (jobs / job).glob("*__*"):
        if (trial / "result.json").exists():
            trials.append((trial, json.load(open(trial / "result.json"))))
    for trial, r in sorted(trials, key=lambda tr: (started(tr[1]), tr[0].name)):
        task = canonical(r["task_name"].split("/")[-1])   # also pre-v1.0 jobs
        tdir = ROOT / "tasks" / task
        if not tdir.is_dir(): continue
        top = out_root / task / slug
        if (top / "config.json").is_file():
            sys.exit(f"{top} holds a result in the layout before trials (config.json directly in it): write into a new "
                     "--out directory (RUNNING.md \"Results\")")
        rw = load_reward(trial)
        ar = r.get("agent_result") or {}; mode = mode_of(task); limited = mode != "privileged"; fam = base(task)
        cfg = trial_config(trial, task, r)
        k = cfg["trial"]
        out = top / f"t{k}"; out.mkdir(parents=True, exist_ok=True)
        touched.add(top)
        status, why, over = classify(trial, r, rw)
        led = record_attempt(out, trial, r, status)
        rule = rp.check_attempts([{"outcome": OUTCOME.get(a["status"], "agent_failure")} for a in led])
        over_cap = bool(rule)
        if over_cap:
            print(f"WARN {task} / {slug} / trial {k}: {len(led)} attempts break the retry rule (PROTOCOL Appendix H.5, "
                  f"at most {MAX_ATTEMPTS}: one retry after an infrastructure failure): {'; '.join(rule)}")
        counted = [a for a in led if a["counted"]]
        if status == "not_a_result" and counted:
            # a voided attempt never replaces a counted one: it is recorded in attempts.json only
            print(f"{task} / {slug} / trial {k}: {trial.name} is not a result (voided); the counted attempt stays")
            continue
        cfg.update({"run_id": slug, "job": job, "trial_dir": trial.name, "result": status, "not_a_result_reason": why,
                    "counted": status != "not_a_result", "attempts": len(led), "attempts_over_cap": over_cap})
        usage = usage_of(trial, r)
        cfg.update(usage)
        gw = load_json(trial / "gateway/totals.json")
        (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
        for f in ("replay.mp4", "replay.last.png"):
            if (trial / "verifier" / f).exists(): shutil.copy(trial / "verifier" / f, out / f)
        ok = status == "success"
        an = analysis.get(fam, {}).get("L" if limited else "U", "")
        lines = [f"# {label} — {task}\n",
                 f"Run: Harbor job `{job}`, trial `{trial.name}`, {(r.get('started_at') or '')[:10]}. Mode: **{ {'standard': 'standard (robot as a service)', 'standard-realtime': 'standard-realtime (robot as a service, time does not stop)'}.get(mode, 'privileged')}**.\n",
                 "| | |", "|---|---|",
                 f"| result | {'**1 (success)**' if ok else NOT_A_RESULT.format(why) if why else '**0**' + (' — over the token budget' if over else ' — no trajectory handed in' if rw.get('missing_trajectory') else '')} |",
                 f"| trial | {k}" + (f" of {cfg['trials']}" if cfg.get("trials") else "") + f"; seed {cfg.get('seed')} ({cfg.get('seed_source')}) |",
                 f"| configuration | {tc.config_label(cfg)}; "
                 f"{'validated' if cfg.get('validated') is True else 'unvalidated' if cfg.get('validated') is False else 'legacy'}"
                 f"{'; easy set (docs/easy_set.toml)' if cfg.get('set') == 'easy' else ''} (`config.json`) |",
                 f"| attempts | {len(led)}" + (f" — **over the cap of {MAX_ATTEMPTS}**" if over_cap else "") + " |",
                 f"| agent time | {span(r.get('agent_execution') or {})} s |",
                 f"| actions / physics steps | {rw.get('steps') or rw.get('n_actions') or '—'} / {rw.get('sim_steps') or '—'} |",
                 "| tokens in / out | " + "; ".join(
                     ([f"{ar.get('n_input_tokens')} / {ar.get('n_output_tokens')} (harness)"]
                      if ar.get("n_input_tokens") is not None or ar.get("n_output_tokens") is not None else [])
                     + ([f"{gw.get('input')} / {gw.get('output')} (usage gateway)"] if gw else [])) + " |"
                 if usage["usage_recorded"] else "| tokens in / out | not recorded |",
                 f"| verifier | `{json.dumps({k: v for k, v in rw.items() if k not in ('agent_sessions','agent_records','agent_replays')})}` |",
                 f"| verifier sandbox | {r.get('verifier_environment_mode')} |\n"]
        if not limited and rw.get("agent_resets") is not None:
            lines.append(f"Debugging footprint (self-reported by the agent's tooling log): {rw.get('agent_sessions')} sessions, {rw.get('agent_resets')} resets, {rw.get('agent_sim_steps')} physics steps.\n")
        if (out / "replay.mp4").exists():
            lines += ["## Replay (verifier's six-camera video)\n", f"![last frame](replay.last.png)\n", "[replay.mp4](replay.mp4) — top: world / observer / head; bottom: front / left wrist / right wrist.\n"]
        if an:
            lines += ["## Analysis\n", an + "\n"]
        msgs = last_msgs(trial)
        if msgs:
            lines += ["## Agent's last messages\n"] + [f"- {m[:600]}" for m in msgs] + [""]
        (out / "README.md").write_text("\n".join(lines)); n += 1
    for top in sorted(touched):
        write_index(top, label)
    print(f"{job}: wrote {n} result folders for {slug}")


def write_index(top, label):
    """<task>/<run id>/trials.json and README.md: the task's trials in this run, their seeds and results (RUNNING.md
    "Results": a task's success over its trials counts results only)."""
    rows = []
    for d in sorted(top.glob("t*"), key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0):
        cfg = load_json(d / "config.json")
        if not d.name[1:].isdigit() or not cfg:
            continue
        rows.append({"trial": int(d.name[1:]), "dir": d.name, "seed": cfg.get("seed"), "seed_source": cfg.get("seed_source"),
                     "result": cfg.get("result"), "counted": cfg.get("counted"), "attempts": cfg.get("attempts"),
                     "attempts_over_cap": cfg.get("attempts_over_cap"), "job": cfg.get("job")})
    res = [x for x in rows if x["counted"]]
    ok = sum(x["result"] == "success" for x in res)
    (top / "trials.json").write_text(json.dumps({"task": top.parent.name, "run_id": top.name, "trials": rows,
                                                 "results": len(res), "successes": ok}, indent=1) + "\n")
    lines = [f"# {label} — {top.parent.name}\n",
             f"{len(rows)} trial(s); success over its trials (results only): **{ok} / {len(res)}**"
             + (f"; {len(rows) - len(res)} not a result (voided)" if len(rows) != len(res) else "") + ".\n",
             "| trial | seed | result | attempts |", "|---|---|---|---|"]
    for x in rows:
        res_txt = {"success": "**1 (success)**", "fail": "0"}.get(x["result"], "not a result")
        lines.append(f"| [{x['trial']}]({x['dir']}/README.md) | {x['seed']} | {res_txt} | {x['attempts']}"
                     + (" (over the cap)" if x["attempts_over_cap"] else "") + " |")
    (top / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv)
