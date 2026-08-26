"""Recording adapter for live clients.

It deliberately reuses the public consumer writers so the TUI, CLI, and
library API emit the same durable schemas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from seabreeze_live.acquisition import Consumer
from seabreeze_live.consumers import CsvWriter, Hdf5Writer
from seabreeze_live.frame import SpectrumFrame


class SpectrumRecorder:
    """Write raw frames through the selected project-standard consumer."""

    def __init__(self, path: str | Path, fmt: str) -> None:
        normalized = fmt.lower()
        if normalized == "csv":
            self._consumer: Consumer = CsvWriter(path)
        elif normalized in {"h5", "hdf5"}:
            # UI rendering is independent of recording, so write HDF5 in
            # chunks instead of forcing a resize and flush for every frame.
            self._consumer = Hdf5Writer(path, flush_every=64)
        else:
            raise ValueError(f"unsupported recording format: {fmt!r}")
        self.path = Path(path)
        self._closed = False

    def write(
        self,
        *,
        values: np.ndarray,
        axis: np.ndarray,
        timestamp_ns: int,
        frame_number: int,
        integration_time_us: int,
        device_serial: str,
    ) -> None:
        if self._closed:
            raise RuntimeError("recorder is closed")
        self._consumer.on_frame(
            SpectrumFrame(
                values=values,
                axis=axis,
                timestamp_ns=timestamp_ns,
                frame_number=frame_number,
                integration_time_us=integration_time_us,
                device_serial=device_serial,
            )
        )

    def close(self) -> None:
        if not self._closed:
            self._consumer.close()
            self._closed = True
