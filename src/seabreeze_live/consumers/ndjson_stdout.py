from __future__ import annotations

import json
import sys

from seabreeze_live.frame import SpectrumFrame
from seabreeze_live.interfaces import TextWriter


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

    def __init__(
        self,
        stream: TextWriter | None = None,
        *,
        include_context: bool = True,
    ) -> None:
        self.stream: TextWriter = stream if stream is not None else sys.stdout
        self.include_context = include_context

    def on_frame(self, frame: SpectrumFrame) -> None:
        payload = frame.to_wire(include_context=self.include_context)
        self.stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False))
        self.stream.write("\n")
        self.stream.flush()

    def close(self) -> None:
        # Never close stdout; flush in case the parent reads at exit.
        try:
            self.stream.flush()
        except (ValueError, OSError):
            pass
