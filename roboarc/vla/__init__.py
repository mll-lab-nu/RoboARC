"""RoboARC for VLA policies served the openpi way (docs/VLA.md): an openpi policy server (openpi_client's websocket
protocol) drives graded RoboARC episodes, on the same physics and grader as the Harbor tasks.

    python -m roboarc.vla.run --task <task> --policy ws://HOST:8000 --seed 0 --episodes 1 --out <dir>

- conventions.py: openpi's ALOHA and DROID observation / action conventions on RoboARC's robots;
- wire.py: the websocket + msgpack-numpy protocol (no openpi_client import: it pins numpy < 2);
- run.py: the episode runner (live episode, the grader's fresh-process replay, results);
- stub_server.py: a stand-in policy server (hold, or a replayed action sequence) for checking a setup."""
from .conventions import (ALOHA_CAMERAS, CONVENTIONS, AlohaConvention, Convention, DroidConvention, make_convention,
                          resize_with_pad, task_prompt)

__all__ = ["ALOHA_CAMERAS", "CONVENTIONS", "AlohaConvention", "Convention", "DroidConvention", "make_convention",
           "resize_with_pad", "task_prompt"]
