"""A stress acceptance round, or a quick self-check, on frozen MuJoCo instances (docs/rcb/STRESS_VALIDATION.md).

usage: round.py run --instances <instance.json> [...] --out <dir> [--refs <dir> [--routes-only]] [--corpus <manifest.json>]
                    [--reproducers <dir>] [--seed0 S] [--episodes N | --replicas R --budget B] [--strategies ...]
                    [--procs P] [--replay-workers W] [--self-check]
       round.py report <dir>
run, in <dir>:
  plan.json            the declared round: instances (with sha256), seeds, budgets, inputs, criteria, provenance
  results/reproducers  every reproducer of these instances, twice (two fresh processes: verdict and determinism)
  results/corpus       the corpus streams of these instances (+40 hold controls)
  results/refs, refs2  each reference (<refs>/<base>/privileged.npz) twice, in fresh processes (success, sha, wall time);
                       with --routes-only none: the references only route the prefix strategy (blanket's CPU twin runs
                       the GPU references' rows, judged on the GPU profile)
  results/nop          400 hold controls per instance
  search/              the self-play search, one worker per strategy (random, directed, mutate, cem, prefix, periodic)
                       and replica, seed = seed0 + k in the declared order (instances by name, replicas, strategies).
                       --episodes N: N episodes per instance, an equal share per strategy; a strategy named n times
                       splits its share over n workers (for slow ones, such as prefix on a long reference).
                       --budget B: B controls per worker, so a strategy named n times gets n budgets.
                       Prefix starts from mid-task states along the instance's reference, periodic drives joints and
                       grippers with sustained sinusoids (search protocol 3, 2026-10-10; --strategies without periodic
                       gives protocol 2, without prefix and periodic protocol 1). An extension of a round is a new
                       round from --seed0 = this seed0 + its number of workers: the seed range continues.
  results/confirm      every search candidate (BREAKDOWN, or VIOLENT above 10 m/s or 5 mm deep, or a finger-finger force
                       above 2 kN, a finger-finger or finger-joint row with A_ii below 0.1, or a degenerate-row signature)
                       in a fresh process
  report.md, .json     the verdict per instance (criteria v2.11 decide), with the v2 / v2.2 / v2.3 / v2.4 / v2.5 / v2.6 /
                       v2.7 / v2.8 / v2.9 / v2.10 / v2.11 readings side by side and the detection power
--self-check: a quick setup check (2,000 controls per worker, one replica); it is never an acceptance round.
Acceptance (docs/rcb/STRESS_VALIDATION.md): about 1,000 episodes per instance, all strategies, fresh seeds.
A rerun resumes: finished replays and search workers are kept."""
import _env  # noqa: F401  (before numpy: SIMD masking, threads, sys.path)
import argparse, hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path

import numpy as np

from pool import run_jobs
import verdict

HERE = Path(__file__).resolve().parent
STRATEGIES = ("random", "directed", "mutate", "cem", "prefix", "periodic")     # search protocol 3 (2026-10-10)
CRITERIA = "2.13b"
READ_ALL = ("v2", "v2.2", "v2.3", "v2.4", "v2.5", "v2.6", "v2.7", "v2.8", "v2.9", "v2.10", "v2.11", "v2.12", "v2.13b")
NOP_CONTROLS = 400
PRESS_WARN_MAX = 0.01        # v2.12 pressed-pile warnings (VBD cloth under a guarded press) at most 1 % of a tier's episodes
WALL_LIMIT_MIN = 180         # a reference execution: within the 3 h agent budget (user 2026-10-13; 2 h from 10-12, 72 before)
# STRESS_WALL_LIMIT_MIN: a loaded stress host replays a reference with the observer (~24 % slower) far slower than the
# trial host runs it; there the limit is raised so that a slow host never fails a reference (the speed acceptance is
# judged on the trial host). The report says when it is.
WALL_LIMIT_MIN = float(os.environ.get("STRESS_WALL_LIMIT_MIN", WALL_LIMIT_MIN))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def base_of(instance):
    return Path(instance).resolve().parent.name


# ------------------------------------------------------------------ plan
def plan_round(a):
    insts = sorted({str(Path(i).resolve()) for i in a.instances}, key=base_of)
    bases = [base_of(i) for i in insts]
    replicas, budget = (1, 2000) if a.self_check else (a.replicas, a.budget)
    episodes = None if a.self_check else a.episodes
    if episodes:
        budget = None
    strategies = tuple(a.strategies.split(",")) if a.strategies else STRATEGIES
    jobs, k = [], 0
    for inst in insts:                      # the declared order: instances by name, then replicas, then strategies
        for r in range(replicas):
            seen = {}
            for st in strategies:           # a strategy named n times gets n workers: st, st2, st3 ... (a larger share)
                seen[st] = seen.get(st, 0) + 1
                name = st if seen[st] == 1 else f"{st}{seen[st]}"
                k += 1
                jobs.append(dict(instance=inst, base=base_of(inst), strategy=st, replica=r, seed=a.seed0 + k,
                                 tag=f"{base_of(inst)}__{name}{'' if replicas == 1 else f'_r{r}'}"))
    if episodes:                            # N per instance: equal shares per strategy, each split over its workers
        names = list(dict.fromkeys(strategies))
        q, rem = divmod(episodes, len(names))
        share = {st: q + (i < rem) for i, st in enumerate(names)}
        for inst in insts:
            for st in names:
                mine = [j for j in jobs if j["instance"] == inst and j["strategy"] == st]
                q2, rem2 = divmod(share[st], len(mine))
                for i, j in enumerate(mine):
                    j["episodes"] = q2 + (i < rem2)
    reps = []
    for rdir in (a.reproducers or []):
        for npz in sorted(Path(rdir).glob("*.npz")):
            meta = npz.with_suffix(".json")
            b = json.loads(meta.read_text()).get("base") if meta.exists() else None
            if b in bases:
                reps.append(dict(name=npz.stem, base=b, actions=str(npz.resolve()), sha256=sha256(npz)))
    corpus = []
    if a.corpus:
        root = Path(a.corpus).resolve().parent
        for e in json.loads(Path(a.corpus).read_text()).get("cases", []):
            if e["base"] in bases:
                corpus.append(dict(id=f"corpus_{e['actions_sha256'][:16]}", base=e["base"], actions=str((root / e["input"]).resolve()),
                                   sha256=e["actions_sha256"]))
    refs = []
    if a.refs:
        for inst in insts:
            p = Path(a.refs) / base_of(inst) / "privileged.npz"
            if p.exists():
                refs.append(dict(base=base_of(inst), actions=str(p.resolve()), sha256=sha256(p)))
    return dict(criteria=CRITERIA, self_check=bool(a.self_check), created=time.time(), routes_only=bool(a.routes_only),
                instances=[dict(path=i, base=base_of(i), sha256=sha256(i)) for i in insts],
                search=dict(seed0=a.seed0, replicas=replicas, budget_per_worker=budget, strategies=list(strategies),
                            protocol=3 if "periodic" in strategies else 2 if "prefix" in strategies else 1,
                            episodes_per_instance=episodes,
                            budget_per_instance=None if episodes else len(strategies) * replicas * budget,
                            next_seed0=a.seed0 + len(jobs), jobs=jobs),
                corpus_manifest=str(Path(a.corpus).resolve()) if a.corpus else None,
                corpus_manifest_sha256=sha256(a.corpus) if a.corpus else None,
                corpus=corpus, reproducers=reps, refs=refs, provenance=_env.provenance())


# ------------------------------------------------------------------ run
def run_round(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else plan_round(a)
    plan_path.write_text(json.dumps(plan, indent=1))
    inst_of = {i["base"]: i["path"] for i in plan["instances"]}
    res = out / "results"
    replay_sets = {
        "reproducers": [dict(id=f"{r['name']}__run{n}", instance=inst_of[r["base"]], actions=r["actions"], hold=0)
                        for r in plan["reproducers"] for n in (1, 2)],
        "corpus": [dict(id=c["id"], instance=inst_of[c["base"]], actions=c["actions"], hold=40) for c in plan["corpus"]],
        "refs": [dict(id=f"ref_{r['base']}", instance=inst_of[r["base"]], actions=r["actions"], hold=0, trace=True)
                 for r in plan["refs"] if not plan.get("routes_only")],
        "refs2": [dict(id=f"ref_{r['base']}", instance=inst_of[r["base"]], actions=r["actions"], hold=0)
                  for r in plan["refs"] if not plan.get("routes_only")],
        "nop": [dict(id=f"nop_{b}", instance=p, actions=f"nop:{NOP_CONTROLS}", hold=0) for b, p in inst_of.items()],
    }
    # the search workers run as their own processes, next to the replay pool
    sdir = out / "search"
    sdir.mkdir(parents=True, exist_ok=True)
    pending = [j for j in plan["search"]["jobs"] if not (sdir / f"{j['tag']}.summary.json").exists()]
    for j in pending:                       # a restarted worker starts over (its episodes must stay one sequence)
        (sdir / f"{j['tag']}.jsonl").unlink(missing_ok=True)
    budget = plan["search"]["budget_per_worker"]

    def search_all():
        running = []
        while pending or running:
            running = [p for p in running if p.poll() is None]
            while pending and len(running) < a.procs:
                j = pending.pop(0)
                cmd = [sys.executable, str(HERE / "search.py"), "--instance", j["instance"], "--out", str(sdir), "--tag", j["tag"],
                       "--seed", str(j["seed"]), "--strategy", j["strategy"]]
                cmd += ["--episodes", str(j["episodes"])] if j.get("episodes") is not None else ["--budget", str(budget)]
                if plan["corpus_manifest"]:
                    cmd += ["--corpus", plan["corpus_manifest"]]
                ref = next((r["actions"] for r in plan["refs"] if r["base"] == j["base"]), None)
                if j["strategy"] == "prefix" and ref:
                    cmd += ["--reference", ref]
                running.append(subprocess.Popen(cmd, stdout=open(sdir / f"{j['tag']}.log", "w"), stderr=subprocess.STDOUT))
            time.sleep(5)

    t = threading.Thread(target=search_all)
    t.start()
    for name, jobs in replay_sets.items():
        if jobs:
            print(f"{name}: {len(jobs)} replays", flush=True)
            run_jobs(jobs, res / name, a.replay_workers, echo=False)
    t.join()
    # confirm every candidate in a fresh process
    cands = []
    for jf in sorted((sdir / "cand").glob("*.json")):
        rec = json.loads(jf.read_text())
        if (rec["verdict"] == "BREAKDOWN" or rec.get("speed", 0) > 10 or rec.get("task_depth_mm", 0) > 5
                or rec.get("ff_force", 0) > 2000 or min(rec.get("ff_min_A", 1e9), rec.get("finger_eqlim_min_A", 1e9)) < .1
                or rec.get("degenerate_rows", 0) > 0):
            cands.append(dict(id=jf.stem, instance=rec["instance"], actions=str(jf.with_suffix(".npz")), hold=0))
    print(f"confirm: {len(cands)} candidates", flush=True)
    run_jobs(cands, res / "confirm", a.replay_workers, echo=False)
    report(out)


# ------------------------------------------------------------------ report
def detection(n):
    """Detection power of n independent episodes: P(a mechanism with per-episode rate p is never hit) = (1 - p)^n."""
    miss = {f"{p:.1%}": (1 - p) ** n for p in (.001, .002, .005, .01)}
    return dict(episodes=n, p_miss=miss, floor_95=1 - .05 ** (1 / n) if n else 1.)


def report(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    res = out / "results"
    load = lambda d: {p.stem: json.loads(p.read_text()) for p in sorted((res / d).glob("*.json")) if not p.name.startswith("_")}
    rows, lines = {}, []
    sets = {d: load(d) for d in ("reproducers", "corpus", "refs", "refs2", "nop", "confirm")}
    for inst in plan["instances"]:
        b = inst["base"]
        mine = lambda d: {k: r for k, r in sets[d].items() if base_of(r["instance"]) == b}
        rd = dict(base=b, checks={}, readings={})
        ok = True
        # completeness: every declared replay has a result and every search worker its summary
        missing = []
        declared = dict(reproducers=[f"{r['name']}__run{n}" for r in plan["reproducers"] if r["base"] == b for n in (1, 2)],
                        corpus=[c["id"] for c in plan["corpus"] if c["base"] == b],
                        refs=[f"ref_{r['base']}" for r in plan["refs"] if r["base"] == b and not plan.get("routes_only")],
                        refs2=[f"ref_{r['base']}" for r in plan["refs"] if r["base"] == b and not plan.get("routes_only")],
                        nop=[f"nop_{b}"])
        for d, ids in declared.items():
            missing += [f"{d}/{i}" for i in ids if i not in sets[d]]
        missing += [f"search/{j['tag']}" for j in plan["search"]["jobs"]
                    if j["base"] == b and not (out / "search" / f"{j['tag']}.summary.json").exists()]
        rd["missing"] = missing
        # corpus and confirmed candidates: the three readings side by side
        for d in ("corpus", "confirm", "reproducers"):
            for reading in READ_ALL:
                n = sum(verdict.classify(r, reading)["verdict"] == "BREAKDOWN" for r in mine(d).values())
                rd["readings"].setdefault(d, {})[reading] = n
        c = mine("corpus")
        bad = [k for k, r in c.items() if verdict.classify(r)["verdict"] == "BREAKDOWN"]
        rd["checks"]["corpus"] = dict(streams=len(c), breakdown=bad)
        ok &= not bad
        r1, r2 = mine("refs"), mine("refs2")
        if r1:
            k = next(iter(r1))
            a1, a2 = r1[k], r2.get(k)
            v = verdict.classify(a1)["verdict"]
            det = None if a2 is None else a1.get("state_sha") == a2.get("state_sha")   # None: the second run is missing
            wall = max(x["wall_s"] for x in (a1, a2) if x) / 60
            good = bool(a1.get("success")) and v == "CLEAN" and det is not False and wall <= WALL_LIMIT_MIN
            rd["checks"]["reference"] = dict(success=a1.get("success"), verdict=v, deterministic=det, state_sha=a1.get("state_sha"),
                                             wall_min=round(wall, 1), steps=a1.get("control_steps"), ok=good)
            ok &= good
        else:
            rd["checks"]["reference"] = None
            if not declared["refs"] and not plan.get("routes_only"):   # acceptance needs a reference (routes only: judged elsewhere)
                ok = False
        nop = mine("nop")
        if nop:
            n1 = next(iter(nop.values()))
            good = not n1.get("success") and verdict.classify(n1)["verdict"] == "CLEAN"
            rd["checks"]["nop"] = dict(success=n1.get("success"), verdict=verdict.classify(n1)["verdict"], ok=good)
            ok &= good
        reps = mine("reproducers")
        rep_rows = {}
        for k, r in reps.items():
            name = k.rsplit("__run", 1)[0]
            rep_rows.setdefault(name, []).append((verdict.classify(r)["verdict"], r.get("state_sha")))
        rd["checks"]["reproducers"] = {n: dict(verdict=v[0][0], deterministic=len({x[1] for x in v}) == 1) for n, v in rep_rows.items()}
        ok &= all(x["verdict"] != "BREAKDOWN" and x["deterministic"] for x in rd["checks"]["reproducers"].values())
        # the search
        eps, controls, pops, slips, per, pw, pt = 0, 0, [], 0, {}, 0, 0
        for f in sorted((out / "search").glob(f"{b}__*.jsonl")):
            recs = [json.loads(l) for l in open(f) if l.strip()]
            recs = [x for x in recs if "verdict" in x]
            eps += len(recs)
            controls += recs[-1]["used"] if recs else 0
            st = f.stem.split("__", 1)[1].split("_r")[0].rstrip("0123456789")
            d = per.setdefault(st, dict(episodes=0, controls=0, BREAKDOWN=0, VIOLENT=0, CLEAN=0))
            d["episodes"] += len(recs)
            d["controls"] += recs[-1]["used"] if recs else 0
            for x in recs:
                pops += x.get("pops") or []
                slips += x.get("grasp_slips") or 0
                d[x["verdict"]] += 1
                pw += bool(x.get("warnings"))
                pt += any("pressed pile thrown" in t for t in x.get("triggers") or [])
        conf = mine("confirm")
        cbad = [k for k, r in conf.items() if verdict.classify(r)["verdict"] == "BREAKDOWN"]
        rd["checks"]["search"] = dict(controls=controls, episodes=eps, candidates=len(conf), confirmed_breakdown=cbad,
                                      grasp_slips=slips, detection=detection(eps), per_strategy=per,
                                      pressed_pile=dict(warnings=pw, thrown=pt, rate=round(pw / eps, 4) if eps else 0.,
                                                        ceiling=PRESS_WARN_MAX))
        ok &= not eps or pw / eps <= PRESS_WARN_MAX         # v2.12 pressed pile: a warning must stay rare
        pa = np.asarray(pops, float)
        rd["pops"] = dict(count=int(len(pa)), max=float(pa.max()) if len(pa) else 0., p95=float(np.percentile(pa, 95)) if len(pa) else 0.)
        ok &= not cbad
        rd["verdict"] = "FAIL" if not ok else "INCOMPLETE" if missing else "PASS"
        rows[b] = rd
    title = "Self-check (not an acceptance round)" if plan["self_check"] else "Stress acceptance round"
    ps = plan["search"]
    size = (f"{ps['episodes_per_instance']:,} episodes per instance, an equal share per strategy"
            if ps.get("episodes_per_instance") else f"{ps['budget_per_instance']:,} controls per instance")
    crit = f"Criteria v{CRITERIA}" + (f" (the round was planned under v{plan['criteria']})" if plan["criteria"] != CRITERIA else "")
    if WALL_LIMIT_MIN != 180:
        crit += f"; reference wall limit {WALL_LIMIT_MIN:g} min on this loaded host (the speed acceptance is the trial host's)"
    else:
        crit += "; reference wall limit 180 min (a reference finishes within the 3 h agent budget, user 2026-10-13)"
    if plan.get("routes_only"):
        crit += ("; references are routes only: the prefix strategy branches from their rows, but they are recorded and judged "
                 "(success, x2 sha) on their own profile, not here (a reference recorded on another profile, such as blanket's "
                 "GPU step for the CPU twin, need not succeed on this one)")
    lines += [f"# {title}", "", f"{crit}, search protocol {ps.get('protocol', 1)}. "
              f"Search: {size}, seeds {ps['seed0'] + 1}-{ps['seed0'] + len(ps['jobs'])} "
              f"(an extension starts at --seed0 {ps['seed0'] + len(ps['jobs'])}).", "",
              "| instance | corpus | reference | nop | reproducers | search | confirmed BREAKDOWN " + " / ".join(READ_ALL)
              + " | pops | P(miss) 0.1% / 0.5% | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for b, rd in rows.items():
        ch = rd["checks"]
        ref = ch["reference"]
        shat = {True: "same sha", False: "DIFFERENT SHA", None: "second run missing"}
        reft = ("routes only (judged on the recording profile)" if plan.get("routes_only") else "none") if not ref else (f"{'success' if ref['success'] else 'NO SUCCESS'}, {ref['verdict']}, "
                                    f"{shat[ref['deterministic']]}, {ref['wall_min']} min")
        nop = ch.get("nop")
        nopt = "-" if not nop else f"{'SUCCESS' if nop['success'] else 'no success'}, {nop['verdict']}"
        rept = ", ".join(f"{n}: {x['verdict']}{'' if x['deterministic'] else ' NONDET'}" for n, x in ch["reproducers"].items()) or "-"
        s = ch["search"]
        rr = rd["readings"]["confirm"]
        d = s["detection"]["p_miss"]
        lines.append(f"| {b} | {ch['corpus']['streams']} streams, {len(ch['corpus']['breakdown'])} BREAKDOWN | {reft} | {nopt} | {rept} | "
                     f"{s['controls']:,} controls, {s['episodes']} episodes, {s['grasp_slips']} grasp slips | "
                     + " / ".join(str(rr[x]) for x in READ_ALL) + f" | {rd['pops']['count']}, max {rd['pops']['max']:.2f} m/s | "
                     f"{d['0.1%']:.3f} / {d['0.5%']:.1e} | **{rd['verdict']}** |")
    pp = {b: rd["checks"]["search"].get("pressed_pile") for b, rd in rows.items()}
    if any(v and (v["warnings"] or v["thrown"]) for v in pp.values()):
        lines += ["", "Pressed-pile warnings (VBD cloth under a guarded press; in-search, at most "
                  f"{PRESS_WARN_MAX:.0%} of the episodes):", ""]
        lines += [f"- {b}: {v['warnings']} warnings in {rows[b]['checks']['search']['episodes']} episodes "
                  f"({v['rate']:.2%}), {v['thrown']} thrown (VIOLENT, agent-caused)" for b, v in pp.items() if v]
    names = list(dict.fromkeys(plan["search"]["strategies"]))   # each strategy once (repeated ones are summed)
    lines += ["", "Search by strategy (in-search verdicts; the confirmed column above decides):", "",
              "| instance | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for b, rd in rows.items():
        per = rd["checks"]["search"]["per_strategy"]
        cells = [f"{per[st]['episodes']} ep, {per[st]['controls']:,} ctl, {per[st]['BREAKDOWN']} BD / {per[st]['VIOLENT']} V"
                 if st in per else "-" for st in names]
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    lines.append("")
    for b, rd in rows.items():
        for k in rd["checks"]["corpus"]["breakdown"] + rd["checks"]["search"]["confirmed_breakdown"]:
            lines.append(f"- {b}: BREAKDOWN {k}")
        if rd["missing"]:
            lines.append(f"- {b}: not finished: {', '.join(rd['missing'][:8])}{' ...' if len(rd['missing']) > 8 else ''}")
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    (out / "report.json").write_text(json.dumps(dict(plan_criteria=plan["criteria"], self_check=plan["self_check"], instances=rows), indent=1))
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--instances", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--refs")
    r.add_argument("--routes-only", action="store_true", help="the --refs streams are the prefix strategy's routes only: no "
                   "reference replays, no reference check (judged on the profile they were recorded on)")
    r.add_argument("--corpus")
    r.add_argument("--reproducers", nargs="*", help="directories of reproducers (<name>.npz + <name>.json with its base)")
    r.add_argument("--seed0", type=int, default=0)
    r.add_argument("--episodes", type=int, help="episodes per instance, split equally over the strategies (acceptance: ~1000)")
    r.add_argument("--replicas", type=int, default=1, help="workers per strategy name")
    r.add_argument("--budget", type=int, default=25000, help="controls per search worker, when --episodes is not given")
    r.add_argument("--procs", type=int, default=8, help="search workers at once")
    r.add_argument("--replay-workers", type=int, default=8)
    r.add_argument("--self-check", action="store_true")
    r.add_argument("--strategies", help="comma list; default random,directed,mutate,cem,prefix,periodic (search protocol 3); a "
                   "name given n times gets n workers: with --episodes they split its one share, with --budget each "
                   "gets a budget (e.g. ...,prefix,prefix,prefix for a 3/8 prefix share of the controls)")
    p = sub.add_parser("report")
    p.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "run":
        run_round(a)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
