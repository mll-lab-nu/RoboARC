"""The openpi policy-server wire protocol (openpi_client 0.1.x): a websocket carrying msgpack messages in which numpy
arrays are maps {b"__ndarray__": True, b"data": <bytes>, b"dtype": <dtype.str>, b"shape": <shape>} (numpy scalars:
{b"__npgeneric__": True, b"data": <item>, b"dtype": <dtype.str>}).

    server -> client   the server's metadata (one message, right after the connection opens)
    client -> server   an observation dict
    server -> client   the policy's output dict ({"actions": (H, D) array, ...}); a text message instead is the server's
                       traceback (an error)

This module speaks the protocol without importing openpi_client, whose package pins numpy < 2 (the RoboARC images run
numpy 2.3.5 / 2.5.4, which the graded runtime checks): msgpack and websockets only, the versions openpi's own lock file
pins (`pip install -e .[vla]`). tests/api/test_vla.py checks the bytes against openpi_client's own packer when it is
installed, and the end-to-end test runs openpi's WebsocketPolicyServer against this client."""
from __future__ import annotations

import functools
import logging
import time

import numpy as np


def _need(module: str):
    try:
        return __import__(module)
    except ImportError as exc:      # pragma: no cover - depends on the environment
        raise ImportError(f"roboarc.vla needs {module!r}: pip install msgpack==1.1.0 websockets==15.0.1 "
                          "(or `pip install -e .[vla]`)") from exc


def pack_array(obj):
    """msgpack `default` hook: numpy arrays and scalars as openpi_client.msgpack_numpy encodes them."""
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    """msgpack `object_hook`: the inverse of pack_array."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def packb(obj) -> bytes:
    msgpack = _need("msgpack")
    return msgpack.packb(obj, default=pack_array)


def unpackb(data: bytes):
    msgpack = _need("msgpack")
    return msgpack.unpackb(data, object_hook=unpack_array)


def Packer():  # noqa: N802 - the name openpi_client uses
    msgpack = _need("msgpack")
    return functools.partial(msgpack.Packer, default=pack_array)()


class WebsocketClientPolicy:
    """A policy served over a websocket (openpi's WebsocketPolicyServer or anything speaking its protocol), with the
    interface of openpi_client.websocket_client_policy.WebsocketClientPolicy: infer(obs) -> dict, reset(),
    get_server_metadata(). `wait_s`: how long to wait for the server to accept the connection (openpi waits forever)."""

    def __init__(self, host: str = "0.0.0.0", port: int | None = None, api_key: str | None = None,
                 wait_s: float = 300.0) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._api_key = api_key
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server(wait_s)

    @property
    def uri(self) -> str:
        return self._uri

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self, wait_s: float):
        _need("websockets")
        from websockets.sync import client
        deadline = time.monotonic() + wait_s
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        while True:
            try:
                conn = client.connect(self._uri, compression=None, max_size=None, additional_headers=headers)
                return conn, unpackb(conn.recv())
            except (ConnectionRefusedError, OSError) as exc:
                if time.monotonic() > deadline:
                    raise ConnectionError(f"no policy server at {self._uri} after {wait_s:g} s: {exc}") from exc
                logging.info("waiting for the policy server at %s ...", self._uri)
                time.sleep(2.0)

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass
