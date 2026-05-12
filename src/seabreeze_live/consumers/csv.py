from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, TextIO

from seabreeze_live.frame import SpectrumFrame


class CsvWriter:
    """Append-only CSV: one row per spectrum.

    Header: timestamp_ns, frame_number, integration_time_us, then one
    column per pixel labelled with its wavelength in nm to 4 decimals.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._f: TextIO | None = None
        self._writer: Any = None

    def on_frame(self, frame: SpectrumFrame) -> None:
        if self._writer is None:
            self._f = self.path.open("w", newline="")
            self._writer = csv.writer(self._f)
            header = ["timestamp_ns", "frame_number", "integration_time_us"] + [
                f"{w:.4f}" for w in frame.axis
            ]
            self._writer.writerow(header)
        self._writer.writerow(
            [
                frame.timestamp_ns,
                frame.frame_number,
                frame.integration_time_us,
                *frame.values.tolist(),
            ]
        )

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
            self._writer = None
