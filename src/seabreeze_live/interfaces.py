"""Stable interfaces shared by acquisition, UI, and the Rust bridge."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Spectrometer(Protocol):
    """Minimum device contract required by the acquisition core.

    Keeping this structural means the Python SeaBreeze adapter, the simulator,
    and future transports can be substituted without inheriting from a shared
    base class. It maps directly to rust-daq's ``SpectrumReadable`` plus
    ``ExposureControl`` capabilities.
    """

    @property
    def serial_number(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def pixels(self) -> int: ...

    @property
    def integration_time_limits_us(self) -> tuple[int, int]: ...

    @property
    def integration_time_us(self) -> int: ...

    def wavelengths(self) -> np.ndarray: ...

    def read_intensities(
        self,
        correct_dark: bool = False,
        correct_nonlinearity: bool = False,
    ) -> np.ndarray: ...

    def set_integration_time(self, microseconds: int) -> int | None: ...

    def set_trigger_mode(self, mode: int) -> None: ...

    def close(self) -> None: ...


class TextWriter(Protocol):
    """The small text-stream surface needed by NDJSON output."""

    def write(self, data: str, /) -> int: ...

    def flush(self) -> None: ...
