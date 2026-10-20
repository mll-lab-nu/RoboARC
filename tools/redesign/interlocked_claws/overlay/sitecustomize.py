"""Overlay for tools that load redesigned claws instances in fresh processes (stress search workers, replays): on
interpreter start, register the five redesign variations in-process (topogym.tasks.interlocked_claws.redesign.task.
register). Active only when this directory is on PYTHONPATH; it changes no file and no registry on disk."""
try:
    from topogym.tasks.interlocked_claws.redesign.task import register as _register
    for _t in ("very_easy", "easier", "easy", "medium", "hard"):
        _register(_t)
except Exception as _e:          # the tool still runs; a redesign instance will then fail to load, loudly
    import sys as _sys
    print(f"[claws overlay] registration failed: {_e!r}", file=_sys.stderr)
