from __future__ import annotations

from enum import IntEnum
from typing import Protocol, runtime_checkable

import numpy as np


class TriggerMode(IntEnum):
    """Seabreeze trigger modes. Numeric values follow the most common Ocean
    Optics convention; consult the device datasheet for hardware-specific
    differences. Only NORMAL is fully supported in this release; the others
    are accepted but not exercised end-to-end yet.
    """

    NORMAL = 0
    EXTERNAL_SOFTWARE = 1
    EXTERNAL_SYNCHRONIZATION = 2
    EXTERNAL_HARDWARE_EDGE = 3


@runtime_checkable
class SpectrometerDevice(Protocol):
    serial_number: str
    model: str
    pixels: int
    max_intensity: float
    integration_time_limits_us: tuple[int, int]
    integration_time_us: int

    def set_integration_time(self, microseconds: int) -> None: ...
    def set_trigger_mode(self, mode: TriggerMode) -> None: ...
    def wavelengths(self) -> np.ndarray: ...
    def read_intensities(
        self,
        correct_dark: bool = False,
        correct_nonlinearity: bool = False,
    ) -> np.ndarray: ...
    def close(self) -> None: ...


class SeabreezeDevice:
    """Adapter over `seabreeze.spectrometers.Spectrometer`.

    `seabreeze` is imported lazily so the package can be used (and tested)
    with `MockDevice` without the hardware extra installed.
    """

    def __init__(self, serial: str | None = None) -> None:
        from seabreeze.spectrometers import Spectrometer

        if serial is None:
            self._spec = Spectrometer.from_first_available()
        else:
            self._spec = Spectrometer.from_serial_number(serial)

        self.serial_number: str = self._spec.serial_number
        self.model: str = self._spec.model
        self.pixels: int = self._spec.pixels
        self.max_intensity: float = float(self._spec.max_intensity)
        self.integration_time_limits_us: tuple[int, int] = tuple(
            self._spec.integration_time_micros_limits
        )
        self.integration_time_us: int = self.integration_time_limits_us[0]
        self._spec.integration_time_micros(self.integration_time_us)

    def set_integration_time(self, microseconds: int) -> None:
        lo, hi = self.integration_time_limits_us
        if not lo <= microseconds <= hi:
            raise ValueError(
                f"integration_time {microseconds} us outside device limits [{lo}, {hi}]"
            )
        self._spec.integration_time_micros(microseconds)
        self.integration_time_us = microseconds

    def set_trigger_mode(self, mode: TriggerMode) -> None:
        self._spec.trigger_mode(int(mode))

    def wavelengths(self) -> np.ndarray:
        return np.asarray(self._spec.wavelengths(), dtype=np.float64)

    def read_intensities(
        self,
        correct_dark: bool = False,
        correct_nonlinearity: bool = False,
    ) -> np.ndarray:
        return np.asarray(
            self._spec.intensities(
                correct_dark_counts=correct_dark,
                correct_nonlinearity=correct_nonlinearity,
            ),
            dtype=np.float64,
        )

    def close(self) -> None:
        self._spec.close()


def open_device(serial: str | None = None, *, mock: bool = False) -> SpectrometerDevice:
    """Open the first available device (or one by serial), or a MockDevice."""
    if mock:
        from seabreeze_live.mock import MockDevice

        return MockDevice()
    return SeabreezeDevice(serial=serial)


def list_devices() -> list[dict[str, str]]:
    """Return a list of {serial, model} for connected devices.

    Returns [] if `seabreeze` is not installed.
    """
    try:
        from seabreeze.spectrometers import list_devices as _list
    except ImportError:
        return []
    return [{"serial": d.serial_number, "model": d.model} for d in _list()]
