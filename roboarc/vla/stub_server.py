"""A stand-in openpi policy server for checking a setup end to end without a model: the websocket protocol of openpi's
WebsocketPolicyServer (metadata first, then one observation -> one action chunk per message, GET /healthz), serving

    hold      every action = the observed state (aloha, droid): the robot holds where it is. Expected verdict: failure.
    replay    a fixed action sequence (an .npz with `actions`, (T, D) in the convention's units: `python -m
              roboarc.vla.run --export-reference` writes a task's reference that way), chunked from the episode's
              control step (the observation's "roboarc"/"control_step"; without it, a counter advancing by --horizon
              per request); past the end the last action repeats.

    python -m roboarc.vla.stub_server --policy hold --port 8000
    python -m roboarc.vla.stub_server --policy replay --actions ref.npz --horizon 10 --port 8000

The policies need numpy only, so they also run under openpi's own server (openpi.serving.websocket_policy_server)."""
from __future__ import annotations

import argparse
import asyncio
import http
import logging
import time
import traceback

import numpy as np


class HoldPolicy:
    """Actions that hold the observed state: aloha's `state`, droid's joint + gripper position, (H, D) repeated."""

    def __init__(self, horizon: int = 10):
        self.horizon = int(horizon)

    def infer(self, obs: dict) -> dict:
        if "state" in obs:
            s = np.asarray(obs["state"], dtype=np.float64)
        else:
            s = np.concatenate([np.asarray(obs["observation/joint_position"], dtype=np.float64).reshape(-1),
                                np.asarray(obs["observation/gripper_position"], dtype=np.float64).reshape(-1)])
        return {"actions": np.repeat(s[None, :], self.horizon, axis=0)}

    def reset(self) -> None:
        pass


class ReplayPolicy:
    """A fixed (T, D) action sequence served in chunks of `horizon` from the episode's control step."""

    def __init__(self, actions: np.ndarray, horizon: int = 10):
        self.actions = np.asarray(actions, dtype=np.float64)
        if self.actions.ndim != 2 or not len(self.actions):
            raise ValueError("actions must be a non-empty (T, D) array")
        self.horizon = int(horizon)
        self._cursor = 0

    def infer(self, obs: dict) -> dict:
        meta = obs.get("roboarc") if isinstance(obs.get("roboarc"), dict) else {}
        k = int(meta["control_step"]) if "control_step" in meta else self._cursor
        idx = np.minimum(np.arange(k, k + self.horizon), len(self.actions) - 1)
        self._cursor = k + self.horizon
        return {"actions": self.actions[idx]}

    def reset(self) -> None:
        self._cursor = 0


def serve(policy, host: str = "0.0.0.0", port: int = 8000, metadata: dict | None = None) -> None:
    """Serve `policy` (anything with infer(obs) -> dict) over openpi's websocket protocol until interrupted."""
    asyncio.run(_run(policy, host, port, metadata or {}))


async def _run(policy, host, port, metadata, ready=None):
    from websockets.asyncio import server as _server
    import websockets
    from .wire import Packer, unpackb

    async def handler(ws):
        packer = Packer()
        await ws.send(packer.pack(metadata))
        prev_total = None
        while True:
            try:
                t0 = time.monotonic()
                obs = unpackb(await ws.recv())
                t1 = time.monotonic()
                out = policy.infer(obs)
                out["server_timing"] = {"infer_ms": (time.monotonic() - t1) * 1000}
                if prev_total is not None:
                    out["server_timing"]["prev_total_ms"] = prev_total * 1000
                await ws.send(packer.pack(out))
                prev_total = time.monotonic() - t0
            except websockets.ConnectionClosed:
                break
            except Exception:  # noqa: BLE001 - as openpi: the traceback as a text frame, then close
                await ws.send(traceback.format_exc())
                await ws.close(code=1011, reason="Internal server error. Traceback included in previous frame.")
                raise

    def health(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async with _server.serve(handler, host, port, compression=None, max_size=None, process_request=health) as srv:
        if ready is not None:
            ready(srv)
        await srv.serve_forever()


def serve_in_thread(policy, host: str = "127.0.0.1", port: int = 0, metadata: dict | None = None):
    """Start the server on a background thread (tests, notebooks). -> (port, stop)."""
    import threading
    box, started = {}, threading.Event()

    def ready(srv):
        box["port"] = next(iter(srv.sockets)).getsockname()[1]
        box["srv"] = srv
        box["loop"] = asyncio.get_running_loop()
        started.set()

    def main():
        try:
            asyncio.run(_run(policy, host, port, metadata or {}, ready))
        except BaseException:  # noqa: BLE001 - stop() closes the server: serve_forever ends cancelled
            pass

    t = threading.Thread(target=main, daemon=True)
    t.start()
    if not started.wait(30):
        raise RuntimeError("the stub policy server did not start")

    def stop():
        box["loop"].call_soon_threadsafe(box["srv"].close)
        t.join(10)
    return box["port"], stop


def main(argv=None, prog: str = "python -m roboarc.vla.stub_server") -> int:
    """`prog`: the command's name in its help (`roboarc vla stub` is this function under another name)."""
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.replace("python -m roboarc.vla.stub_server", prog),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=("hold", "replay"), default="hold",
                    help="hold: every action is the observed state; replay: the actions of --actions (default: hold)")
    ap.add_argument("--actions", help="replay: an .npz with `actions` (T, D) in the convention's units")
    ap.add_argument("--horizon", type=int, default=10, help="actions per chunk (openpi's action_horizon; default: 10)")
    ap.add_argument("--host", default="0.0.0.0", help="the address to listen on (default: 0.0.0.0, every interface)")
    ap.add_argument("--port", type=int, default=8000, help="the port to listen on (default: 8000)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if a.policy == "replay":
        if not a.actions:
            ap.error("--policy replay needs --actions")
        with np.load(a.actions, allow_pickle=False) as z:
            policy = ReplayPolicy(z["actions"], a.horizon)
    else:
        policy = HoldPolicy(a.horizon)
    serve(policy, a.host, a.port, {"policy": f"roboarc stub ({a.policy})", "action_horizon": a.horizon})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
