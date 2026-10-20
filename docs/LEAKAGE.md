# Answer-key leakage

How RoboARC keeps a task's solution out of the agent's reach (PROTOCOL Appendix H.2), and what a reviewer of a trial
or of a new image should look at.

> **Removing an answer key from what we ship is not the same as the agent being unable to get it.**
> The first is a set we control. The second is everything the container can reach.

The honour rule in every instruction (`## External material`, PROTOCOL §5) lets the agent install packages and read
anything installed in its container, the simulator's and the benchmark's own source included; it forbids only
searching the web for, or downloading, a ready-made solution. Reading the image is therefore never a violation: the
only protection of the answer is that the image does not hold it.

## What the agent can reach

| Place | What RoboARC does | Check |
|---|---|---|
| The task directory | reference solutions exist only under `solution/`, which Harbor mounts for the oracle agent only; the agent's `environment/` holds the tooling and its runtime binding (the frozen instance is staged from the image) | `ci_checks/check-agent-pointers.py` (no URL, dataset file, demonstration id or source key in any file the agent reads), `ci_checks/check-standard-agent-image.sh` (a standard agent image copies nothing from `sim/`) |
| The image | the simulator package `topogym` without its planners, scripted agents and recorded data (`source/images/topogym/prune.py` removes `agents/`, `planning/`, `data/`, `deferred/`, `legacy/`, `redundants/`, `rollout_simulation/` and the authoring-only generator of the cast-coil geometry, which embeds a reference motion; `source/.dockerignore` keeps them out of the build context), the robot assets and the frozen instances | review the image when it is built: grep it for `https?://`, `*.hdf5`, recorded trajectories and the planners' module names; `SOURCE.json` in the image lists the hash of every simulator file it holds |
| Earlier image layers, `.git`, bytecode caches, package metadata | a layer is immutable (a later `rm` does not unwrite it), `.git` reconstructs every committed file, a `*.pyc` built before a removal keeps the old code, `*.egg-info/SOURCES.txt` lists deleted files | remove in the same `RUN` that adds; never copy `.git`; clear caches before removing; check with `docker save` |
| Instance metadata | a frozen instance that records where it came from (a `source_url`, a demonstration file, the expert's module) hands the agent the address of its solution | provenance lives in `scripts/roboarc/` and PROVENANCE.md, never in a file under `environment/` |
| The network | the agent's egress is the allowlist `HOSTS` of `scripts/set_agent_egress.py`: the model APIs and the package registries (PyPI, the Debian / Ubuntu apt archives). GitHub, Hugging Face and the general web do not answer (PROTOCOL H.6) | `ci_checks/check-verifier-isolation.sh` requires exactly that list in every task |
| Packages | RoboARC is not published on PyPI under any name found; a trial's `pip install` of a package that carries a benchmark's solution would still be looked at | read the trial's install commands |
| The model API | tools that run at the provider, reached through an allowed model-API host: Codex's `web_search`, Claude Code's `WebSearch` / `WebFetch`, and under a ChatGPT login the account's connectors, apps, plugins and image generation | `scripts/harbor/run_agent_batch.sh` turns them off unless `WEB_SEARCH=allowed` (`--ak web_search=disabled`, `--ak disallowed_tools=WebSearch,WebFetch`, and a Codex config from `scripts/harbor/codex_tools_off.py`); grep a Codex transcript for `mcp__codex_apps__` |
| The model-API key | Harbor's native agents put the key (or the login) in the agent's container, so an agent can write its own request to a search-enabled model or tool | one key per batch with a spending cap; read the transcript for direct API calls. An optional usage gateway that forwards model calls only (RUNNING.md) narrows this further; it is an external component, not part of this repository |
| The model's memory | public sources are in pretraining data | nothing in the harness removes it; report it with the result |

Layers and `.git` are invisible from inside the container (overlayfs shows only the merged view), but they leak to
whoever pulls the image.

**Read the agent's transcript.** Leaks are found by watching an agent go looking: `grep -rn` over the package, `cat` on
the likely file, `import dis, marshal` against bytecode caches.

## Deciding what to remove from an image

A name match (`*reward*`, `*success*`, `*solve*`, `*oracle*`, `*planner*`) proposes candidates; it never decides.
Settle each candidate with a rule a script can evaluate, so the removal rediscovers itself on every build:

- **Falls through to a safe base**: delete an override when the language guarantees a harmless replacement. Assert that
  the package still imports and that the same actions give the same observations and terminations.
- **Dead because of the removal**: delete a definition when something removed read it and nothing that remains does.
- **Kept only by a surviving reader**: in a container that mixes grading and operational data, keep exactly the entries
  surviving code reads.

Make the removal idempotent (running it again is a no-op, which turns "did we miss anything?" into a build-time
assertion), and derive whatever the verifier needs back from the real removal, never from a second copy of the rule.

**What stays:** termination conditions, observations, the pass mark, the scene layout and constants surviving code
reads. These describe the task; the grader scores it. When in doubt: would the task statement name it anyway?

## The task statement is then the only account

- **Say** which quantities are judged and how they combine, the order of magnitude of a target, what ends the episode.
- **Do not say** what any particular policy scores, or anything that lets the agent skip the robot and solve for the
  number. RoboARC's standard texts also state no world coordinates, no table height and no robot base pose
  (docs/benchmarks/roboarc.md, "Standard mode, world frame, network").
