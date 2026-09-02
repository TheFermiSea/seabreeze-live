"""Line-delimited JSON-RPC service consumed by rust-daq's SeaBreeze driver."""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any, TextIO

from seabreeze_live.acquisition import Streamer
from seabreeze_live.consumers.ndjson_stdout import NdjsonStdoutEmitter
from seabreeze_live.device import TriggerMode
from seabreeze_live.interfaces import Spectrometer, TextWriter

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.0"
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RpcMethod(str, Enum):
    SET_INTEGRATION_TIME_US = "set_integration_time_us"
    SET_TRIGGER_MODE = "set_trigger_mode"
    GET_WAVELENGTHS = "get_wavelengths"
    GET_DEVICE_INFO = "get_device_info"
    START_STREAM = "start_stream"
    STOP_STREAM = "stop_stream"
    SHUTDOWN = "shutdown"


class _LockedStream:
    """Serialize complete NDJSON lines written by the acquisition thread."""

    def __init__(self, stream: TextIO, lock: threading.Lock) -> None:
        self._stream = stream
        self._lock = lock
        self._buffer: list[str] = []

    def write(self, data: str) -> int:
        self._buffer.append(data)
        if "\n" not in data:
            return len(data)
        line = "".join(self._buffer)
        self._buffer.clear()
        with self._lock:
            self._stream.write(line)
            self._stream.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()


def _trigger_mode_from_str(mode: str) -> TriggerMode:
    aliases = {
        "EXTERNAL_SOFTWARE": TriggerMode.SOFTWARE,
        "EXTERNAL_SYNCHRONIZATION": TriggerMode.EXTERNAL_SYNC,
    }
    normalized = mode.upper()
    if normalized in aliases:
        return aliases[normalized]
    try:
        return TriggerMode[normalized]
    except KeyError as error:
        raise ValueError(f"unknown trigger mode {mode!r}") from error


class RpcServer:
    """Stdio service implementing rust-daq's ``seabreeze-protocol`` v1."""

    def __init__(
        self,
        device: Spectrometer,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self.device = device
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout
        self._stdout_lock = threading.Lock()
        self._frame_stream: TextWriter = _LockedStream(self._stdout, self._stdout_lock)
        self._streamer: Streamer | None = None
        self._shutdown = False
        self._methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            RpcMethod.SET_INTEGRATION_TIME_US.value: self._set_integration_time,
            RpcMethod.SET_TRIGGER_MODE.value: self._set_trigger_mode,
            RpcMethod.GET_WAVELENGTHS.value: self._get_wavelengths,
            RpcMethod.GET_DEVICE_INFO.value: self._get_device_info,
            RpcMethod.START_STREAM.value: self._start_stream,
            RpcMethod.STOP_STREAM.value: self._stop_stream,
            RpcMethod.SHUTDOWN.value: self._request_shutdown,
        }

    def device_info(self) -> dict[str, Any]:
        return {
            "model": self.device.model,
            "serial": self.device.serial_number or None,
            "pixel_count": self.device.pixels,
            "integration_time_limits_us": list(self.device.integration_time_limits_us),
        }

    def emit_ready(self) -> None:
        self._write_line(
            {
                "jsonrpc": "2.0",
                "ready": True,
                "protocol_version": PROTOCOL_VERSION,
                "device": self.device_info(),
            }
        )

    def serve_forever(self) -> None:
        try:
            for raw in self._stdin:
                line = raw.strip()
                if line:
                    self._handle_line(line)
                if self._shutdown:
                    break
        finally:
            self._teardown()

    def _handle_line(self, line: str) -> None:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            self._write_error(None, PARSE_ERROR, f"parse error: {error}")
            return

        if not isinstance(request, dict):
            self._write_error(None, INVALID_REQUEST, "request must be an object")
            return

        request_id = request.get("id")
        if request.get("jsonrpc") != "2.0":
            self._write_error(request_id, INVALID_REQUEST, "missing jsonrpc=='2.0'")
            return
        if not isinstance(request_id, int) or isinstance(request_id, bool):
            self._write_error(None, INVALID_REQUEST, "id must be an integer")
            return

        method = request.get("method")
        if not isinstance(method, str):
            self._write_error(request_id, INVALID_REQUEST, "missing method")
            return

        params = request.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            self._write_error(
                request_id, INVALID_PARAMS, "params must be an object or omitted"
            )
            return

        handler = self._methods.get(method)
        if handler is None:
            self._write_error(
                request_id, METHOD_NOT_FOUND, f"unknown method {method!r}"
            )
            return

        try:
            result = handler(params)
        except NotImplementedError as error:
            self._write_error(request_id, INTERNAL_ERROR, f"not implemented: {error}")
            return
        except (TypeError, ValueError, KeyError) as error:
            self._write_error(request_id, INVALID_PARAMS, str(error))
            return
        except Exception as error:
            logger.exception("SeaBreeze RPC method %s failed", method)
            self._write_error(request_id, INTERNAL_ERROR, str(error))
            return

        self._write_result(request_id, result)

    def _set_integration_time(self, params: dict[str, Any]) -> None:
        value = params.get("value")
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("param 'value' must be an integer (microseconds)")
        self.device.set_integration_time(value)

    def _set_trigger_mode(self, params: dict[str, Any]) -> None:
        mode = params.get("mode")
        if not isinstance(mode, str):
            raise TypeError("param 'mode' must be a string")
        self.device.set_trigger_mode(int(_trigger_mode_from_str(mode)))

    def _get_wavelengths(self, _params: dict[str, Any]) -> list[float]:
        return self.device.wavelengths().tolist()

    def _get_device_info(self, _params: dict[str, Any]) -> dict[str, Any]:
        return self.device_info()

    def _start_stream(self, params: dict[str, Any]) -> None:
        if self._streamer is not None:
            if self._streamer.is_running:
                raise RuntimeError("stream already running")
            self._streamer.stop()

        integration = params.get("integration_time_us")
        if integration is not None:
            if not isinstance(integration, int) or isinstance(integration, bool):
                raise TypeError("param 'integration_time_us' must be an integer")
            self.device.set_integration_time(integration)

        # rust-daq caches the wavelength axis during the handshake. Its event
        # schema needs only timing + values, halving steady-state JSON traffic.
        emitter = NdjsonStdoutEmitter(stream=self._frame_stream, include_context=False)
        self._streamer = Streamer(self.device, [emitter])
        self._streamer.start()

    def _stop_stream(self, _params: dict[str, Any]) -> None:
        if self._streamer is not None:
            self._streamer.stop()
            self._streamer = None

    def _request_shutdown(self, _params: dict[str, Any]) -> None:
        self._shutdown = True

    def _teardown(self) -> None:
        if self._streamer is not None:
            try:
                self._streamer.stop(timeout=None)
            except Exception as error:
                logger.exception("failed to stop SeaBreeze stream", exc_info=error)
            self._streamer = None
        try:
            self.device.close()
        except Exception as error:
            logger.exception("failed to close SeaBreeze device", exc_info=error)

    def _write_line(self, value: dict[str, Any]) -> None:
        data = json.dumps(value, separators=(",", ":"), allow_nan=False)
        with self._stdout_lock:
            self._stdout.write(data)
            self._stdout.write("\n")
            self._stdout.flush()

    def _write_result(self, request_id: int, result: Any) -> None:
        self._write_line({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _write_error(self, request_id: Any, code: int, message: str) -> None:
        self._write_line(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )
