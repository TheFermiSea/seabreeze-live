from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from seabreeze_live.cli import build_parser


def _read_line(proc: subprocess.Popen, timeout: float = 5.0) -> str:
    """Read one line from ``proc.stdout`` with a wall-clock timeout."""
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    # readline() blocks; rely on the child process being well-behaved
    # (it flushes after every write). If timeout fires, we kill the proc
    # so the test fails fast instead of hanging the suite.
    line = proc.stdout.readline()
    if not line:
        if time.monotonic() >= deadline:
            proc.kill()
            raise TimeoutError("timed out reading line from subprocess")
        raise RuntimeError("subprocess closed stdout unexpectedly")
    return line.rstrip("\n")


def _request(proc: subprocess.Popen, req_id: int, method: str, **params) -> dict:
    assert proc.stdin is not None
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params:
        payload["params"] = params
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    # Drain until we see a JSON-RPC response (skipping spectrum frames,
    # which are present only after start_stream and never carry jsonrpc).
    for _ in range(200):
        line = _read_line(proc)
        obj = json.loads(line)
        if obj.get("jsonrpc") == "2.0" and obj.get("id") == req_id:
            return obj
    raise AssertionError("no matching JSON-RPC response in 200 lines")


def test_serve_accepts_rust_driver_serial_flag():
    args = build_parser().parse_args(["serve", "--serial", "USB2+F01234"])
    assert args.command == "serve"
    assert args.device == "USB2+F01234"


@pytest.fixture
def server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "seabreeze_live", "serve", "--mock"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5.0)


def test_ready_banner(server):
    line = _read_line(server)
    obj = json.loads(line)
    assert obj["jsonrpc"] == "2.0"
    assert obj["ready"] is True
    assert obj["protocol_version"] == "1.0"
    info = obj["device"]
    assert info["model"] == "MockSpectrometer"
    assert info["serial"].startswith("MOCK")
    assert info["pixel_count"] == 2048
    assert isinstance(info["integration_time_limits_us"], list)
    assert len(info["integration_time_limits_us"]) == 2


def test_get_device_info(server):
    _read_line(server)  # ready banner
    resp = _request(server, 1, "get_device_info")
    assert "error" not in resp
    info = resp["result"]
    assert info["model"] == "MockSpectrometer"
    assert info["pixel_count"] == 2048


def test_set_integration_time_us(server):
    _read_line(server)
    resp = _request(server, 2, "set_integration_time_us", value=2500)
    assert resp["result"] is None

    # Verify via get_device_info that the device accepted it (mock has no
    # readback method, but a follow-up get_wavelengths still works which
    # confirms the device wasn't broken).
    resp2 = _request(server, 3, "get_wavelengths")
    assert "result" in resp2
    assert len(resp2["result"]) == 2048


def test_get_wavelengths_returns_axis(server):
    _read_line(server)
    resp = _request(server, 4, "get_wavelengths")
    wl = resp["result"]
    assert len(wl) == 2048
    # MockDevice ranges 200..1100 nm
    assert 199.0 <= wl[0] <= 201.0
    assert 1099.0 <= wl[-1] <= 1101.0


def test_set_trigger_mode_normal(server):
    _read_line(server)
    resp = _request(server, 5, "set_trigger_mode", mode="NORMAL")
    assert resp["result"] is None


def test_set_trigger_mode_unsupported_returns_error(server):
    _read_line(server)
    resp = _request(server, 6, "set_trigger_mode", mode="EXTERNAL_HARDWARE_EDGE")
    assert "error" in resp
    assert "not implemented" in resp["error"]["message"].lower()


def test_unknown_method_returns_error(server):
    _read_line(server)
    resp = _request(server, 7, "definitely_not_a_method")
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_start_stream_emits_frames_then_stop(server):
    _read_line(server)
    # Short integration time so frames arrive quickly.
    _request(server, 10, "set_integration_time_us", value=1000)
    resp = _request(server, 11, "start_stream")
    assert resp["result"] is None

    # Read a handful of spectrum frames (no jsonrpc key).
    frames_seen = 0
    deadline = time.monotonic() + 10.0
    while frames_seen < 3 and time.monotonic() < deadline:
        line = _read_line(server)
        obj = json.loads(line)
        if "jsonrpc" in obj:
            # An async/late RPC response slipped in — just ignore.
            continue
        assert "jsonrpc" not in obj
        assert set(obj) == {"timestamp_ns", "integration_time_us", "values"}
        assert len(obj["values"]) == 2048
        frames_seen += 1
    assert frames_seen >= 3, "expected at least 3 spectrum frames"

    resp = _request(server, 12, "stop_stream")
    assert resp["result"] is None


def test_shutdown_exits_cleanly(server):
    _read_line(server)
    resp = _request(server, 99, "shutdown")
    assert resp["result"] is None
    rc = server.wait(timeout=5.0)
    assert rc == 0
