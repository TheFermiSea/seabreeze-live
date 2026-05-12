from __future__ import annotations

import json
import sys
import threading
from typing import Any, Callable, TextIO

from seabreeze_live.acquisition import Streamer
from seabreeze_live.consumers import NdjsonStdoutEmitter
from seabreeze_live.device import SpectrometerDevice, TriggerMode


# JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class _LockedStream:
    """Wrap a text stream so concurrent writers serialize line-level writes.

    The streamer thread (via NdjsonStdoutEmitter) and the RPC dispatcher
    both publish to stdout. Holding the lock for the full ``write``+``flush``
    sequence keeps frame and response objects from interleaving on a line.
    """

    def __init__(self, stream: TextIO, lock: threading.Lock) -> None:
        self._stream = stream
        self._lock = lock
        self._buf: list[str] = []

    def write(self, data: str) -> int:
        # Buffer until we see a newline, then flush atomically under the lock.
        self._buf.append(data)
        if "\n" not in data:
            return len(data)
        line = "".join(self._buf)
        self._buf.clear()
        with self._lock:
            self._stream.write(line)
            self._stream.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()


def _trigger_mode_from_str(mode: str) -> TriggerMode:
    try:
        return TriggerMode[mode.upper()]
    except KeyError as e:
        raise ValueError(f"unknown trigger mode {mode!r}") from e


class RpcServer:
    """Line-delimited JSON-RPC 2.0 server bound to a spectrometer device.

    Reads requests one per line from ``stdin`` and writes one response per
    line to ``stdout``. Responses always carry a ``"jsonrpc": "2.0"`` key
    so a Rust parent can discriminate them from NDJSON spectrum frames
    (which never carry that key).

    The server is single-threaded for dispatch; long-running operations
    (streaming) run on the existing ``Streamer`` background thread. Stdout
    writes from the streamer and from RPC responses share a lock so lines
    never interleave.
    """

    def __init__(
        self,
        device: SpectrometerDevice,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.device = device
        self._stdin: TextIO = stdin if stdin is not None else sys.stdin
        self._raw_stdout: TextIO = stdout if stdout is not None else sys.stdout
        self._stdout_lock = threading.Lock()
        self._locked_stdout = _LockedStream(self._raw_stdout, self._stdout_lock)

        self._streamer: Streamer | None = None
        self._shutdown = False

        self._methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "set_integration_time_us": self._m_set_integration_time_us,
            "set_trigger_mode": self._m_set_trigger_mode,
            "get_wavelengths": self._m_get_wavelengths,
            "get_device_info": self._m_get_device_info,
            "start_stream": self._m_start_stream,
            "stop_stream": self._m_stop_stream,
            "shutdown": self._m_shutdown,
        }

    # ----- public API ---------------------------------------------------

    def device_info(self) -> dict[str, Any]:
        """Snapshot of static device metadata, used in the ``ready`` banner."""
        serial = getattr(self.device, "serial_number", None)
        return {
            "model": getattr(self.device, "model", "unknown"),
            "serial": serial,
            "pixel_count": int(getattr(self.device, "pixels", 0)),
            "integration_time_limits_us": list(
                getattr(self.device, "integration_time_limits_us", (0, 0))
            ),
        }

    def emit_ready(self) -> None:
        """Write the single startup banner line announcing the device."""
        banner = {
            "jsonrpc": "2.0",
            "ready": True,
            "device": self.device_info(),
        }
        self._write_line(banner)

    def serve_forever(self) -> None:
        """Block reading stdin until EOF or a ``shutdown`` RPC."""
        for raw in self._stdin:
            line = raw.strip()
            if not line:
                continue
            self._handle_line(line)
            if self._shutdown:
                break
        # If EOF was reached without an explicit shutdown, still tear down.
        self._teardown()

    # ----- request handling --------------------------------------------

    def _handle_line(self, line: str) -> None:
        req_id: Any = None
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            self._write_error(None, PARSE_ERROR, f"parse error: {e}")
            return

        if not isinstance(req, dict):
            self._write_error(None, INVALID_REQUEST, "request must be an object")
            return

        req_id = req.get("id")
        if req.get("jsonrpc") != "2.0":
            self._write_error(req_id, INVALID_REQUEST, "missing jsonrpc=='2.0'")
            return

        method = req.get("method")
        if not isinstance(method, str):
            self._write_error(req_id, INVALID_REQUEST, "missing method")
            return

        params = req.get("params") or {}
        if not isinstance(params, dict):
            self._write_error(
                req_id, INVALID_PARAMS, "params must be an object or omitted"
            )
            return

        handler = self._methods.get(method)
        if handler is None:
            self._write_error(req_id, METHOD_NOT_FOUND, f"unknown method {method!r}")
            return

        try:
            result = handler(params)
        except NotImplementedError as e:
            self._write_error(req_id, INTERNAL_ERROR, f"not implemented: {e}")
            return
        except (TypeError, ValueError, KeyError) as e:
            self._write_error(req_id, INVALID_PARAMS, str(e))
            return
        except Exception as e:  # noqa: BLE001 — surface to the client
            self._write_error(req_id, INTERNAL_ERROR, str(e))
            return

        self._write_result(req_id, result)

    # ----- methods ------------------------------------------------------

    def _m_set_integration_time_us(self, params: dict[str, Any]) -> dict[str, Any]:
        value = params.get("value")
        if not isinstance(value, int):
            raise ValueError("param 'value' must be an integer (microseconds)")
        self.device.set_integration_time(value)
        return {"ok": True}

    def _m_set_trigger_mode(self, params: dict[str, Any]) -> dict[str, Any]:
        mode = params.get("mode")
        if not isinstance(mode, str):
            raise ValueError("param 'mode' must be a string")
        tm = _trigger_mode_from_str(mode)
        # Device may raise NotImplementedError for unsupported modes; that
        # propagates up to _handle_line and becomes a JSON-RPC error.
        self.device.set_trigger_mode(tm)
        return {"ok": True}

    def _m_get_wavelengths(self, _params: dict[str, Any]) -> list[float]:
        return [float(x) for x in self.device.wavelengths()]

    def _m_get_device_info(self, _params: dict[str, Any]) -> dict[str, Any]:
        return self.device_info()

    def _m_start_stream(self, params: dict[str, Any]) -> dict[str, Any]:
        if self._streamer is not None and self._streamer.is_running:
            raise RuntimeError("stream already running")

        integration = params.get("integration_time_us")
        if integration is not None:
            if not isinstance(integration, int):
                raise ValueError("param 'integration_time_us' must be an integer")
            self.device.set_integration_time(integration)

        emitter = NdjsonStdoutEmitter(stream=self._locked_stdout)
        self._streamer = Streamer(self.device, [emitter])
        self._streamer.start()
        return {"ok": True}

    def _m_stop_stream(self, _params: dict[str, Any]) -> dict[str, Any]:
        if self._streamer is None or not self._streamer.is_running:
            return {"ok": True, "was_running": False}
        self._streamer.stop()
        self._streamer = None
        return {"ok": True}

    def _m_shutdown(self, _params: dict[str, Any]) -> dict[str, Any]:
        self._shutdown = True
        return {"ok": True}

    # ----- teardown -----------------------------------------------------

    def _teardown(self) -> None:
        if self._streamer is not None and self._streamer.is_running:
            try:
                self._streamer.stop()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            self._streamer = None
        try:
            self.device.close()
        except Exception:  # noqa: BLE001
            pass

    # ----- framing ------------------------------------------------------

    def _write_line(self, obj: dict[str, Any]) -> None:
        data = json.dumps(obj, separators=(",", ":"))
        with self._stdout_lock:
            self._raw_stdout.write(data)
            self._raw_stdout.write("\n")
            self._raw_stdout.flush()

    def _write_result(self, req_id: Any, result: Any) -> None:
        self._write_line({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _write_error(self, req_id: Any, code: int, message: str) -> None:
        self._write_line(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": code, "message": message},
            }
        )
