from __future__ import annotations

import json
import sys
from typing import TextIO

from seabreeze_live.frame import SpectrumFrame


class NdjsonStdoutEmitter:
    """Emit each frame as a single JSON line on a text stream (stdout by default).

    Intended as the transport when a Rust parent (e.g. rust-daq) spawns this
    package as a subprocess: parse one JSON object per line from our stdout.
    The schema mirrors rust-daq's `SpectrumData` plus a fixed envelope:

        {"type": "spectrum",
         "timestamp_ns": int, "frame_number": int,
         "integration_time_us": int, "device_serial": str,
         "value_units": str, "axis_units": str,
         "values": [float, ...], "axis": [float, ...]}
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream: TextIO = stream if stream is not None else sys.stdout

    def on_frame(self, frame: SpectrumFrame) -> None:
        payload = {
            "type": "spectrum",
            "timestamp_ns": frame.timestamp_ns,
            "frame_number": frame.frame_number,
            "integration_time_us": frame.integration_time_us,
            "device_serial": frame.device_serial,
            "value_units": frame.value_units,
            "axis_units": frame.axis_units,
            "values": frame.values.tolist(),
            "axis": frame.axis.tolist(),
        }
        self.stream.write(json.dumps(payload, separators=(",", ":")))
        self.stream.write("\n")
        self.stream.flush()

    def close(self) -> None:
        # Never close stdout; flush in case the parent reads at exit.
        try:
            self.stream.flush()
        except (ValueError, OSError):
            pass
