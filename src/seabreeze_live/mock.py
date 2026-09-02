"""Deterministic, protocol-compatible spectrometer simulator."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


class MockLamp:
    def __init__(self, update: Callable[[bool], None]) -> None:
        self.enabled = True
        self._update = update

    def set_lamp_enable(self, state: bool) -> None:
        self.enabled = bool(state)
        self._update(self.enabled)


class MockShutter:
    def __init__(self, update: Callable[[bool], None]) -> None:
        self.is_open = True
        self._update = update

    def set_shutter_open(self, state: bool) -> None:
        self.is_open = bool(state)
        self._update(self.is_open)


class MockDevice:
    """Synthetic spectrometer implementing the public device interface.

    The RNG is seeded, the wavelength axis is fixed, and exposure delay can be
    disabled. This makes it suitable for unit tests as well as interactive UI
    development without a second mock implementation.
    """

    def __init__(
        self,
        *,
        pixels: int = 2048,
        wavelength_range_nm: tuple[float, float] = (200.0, 1100.0),
        n_peaks: int = 3,
        noise_sigma: float = 5.0,
        max_intensity: float = 65535.0,
        seed: int = 0,
        simulate_exposure: bool = True,
        serial_number: str | None = None,
        model: str = "MockSpectrometer",
    ) -> None:
        if pixels < 1:
            raise ValueError("pixels must be positive")
        self.serial_number = serial_number or f"MOCK{seed:06d}"
        self.model = model
        self.pixels = pixels
        self.max_intensity = max_intensity
        self.integration_time_limits_us = (1_000, 10_000_000)
        self.integration_time_us = 100_000
        self._wavelengths = np.linspace(*wavelength_range_nm, pixels)
        self._rng = np.random.default_rng(seed)
        low, high = wavelength_range_nm
        self._peak_centers = self._rng.uniform(low + 50, high - 50, n_peaks)
        self._peak_amps = self._rng.uniform(5_000, 30_000, n_peaks)
        self._peak_widths = self._rng.uniform(2.0, 10.0, n_peaks)
        self._noise_sigma = noise_sigma
        self._simulate_exposure = simulate_exposure
        self._lamp_enabled = True
        self._shutter_open = True
        self._closed = False
        self.lamp = MockLamp(self.set_lamp_enable)
        self.shutter = MockShutter(self.set_shutter_open)

    def set_integration_time(self, microseconds: int) -> int:
        lower, upper = self.integration_time_limits_us
        if not lower <= microseconds <= upper:
            raise ValueError(
                f"integration_time {microseconds} us outside mock limits [{lower}, {upper}]"
            )
        self.integration_time_us = microseconds
        return microseconds

    def set_trigger_mode(self, mode: int) -> None:
        if int(mode) != 0:
            raise NotImplementedError("MockDevice only supports TriggerMode.NORMAL")

    def set_lamp_enable(self, state: bool) -> None:
        self._lamp_enabled = bool(state)

    def set_shutter_open(self, state: bool) -> None:
        self._shutter_open = bool(state)

    def get_temperatures(self) -> dict[str, float]:
        now = time.time()
        return {
            "Detector": round(22.5 + 0.8 * np.sin(now / 15.0), 2),
            "PCB": round(28.3 + 0.3 * np.cos(now / 20.0), 2),
        }

    def read_eeprom_slot(self, slot_index: int) -> str:
        return f"mock-eeprom-{slot_index}"

    def wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

    def read_intensities(
        self, correct_dark: bool = False, correct_nonlinearity: bool = False
    ) -> np.ndarray:
        if self._closed:
            raise RuntimeError("mock spectrometer is closed")
        if self._simulate_exposure:
            time.sleep(self.integration_time_us / 1_000_000)

        scale = self.integration_time_us / 1_000_000
        signal = np.zeros(self.pixels)
        if self._shutter_open:
            lamp_scale = 1.0 if self._lamp_enabled else 0.05
            for center, amplitude, width in zip(
                self._peak_centers, self._peak_amps, self._peak_widths, strict=True
            ):
                signal += (
                    lamp_scale
                    * amplitude
                    * np.exp(-0.5 * ((self._wavelengths - center) / width) ** 2)
                )
        dark_level = 0.0 if correct_dark else 50.0
        signal = signal * scale + dark_level
        signal += self._rng.normal(
            0.0, self._noise_sigma * np.sqrt(max(scale, 1e-9)), self.pixels
        )
        if not correct_nonlinearity:
            above = np.maximum(signal - 45_000.0, 0.0)
            signal -= 0.000002 * above**2
        return np.clip(signal, 0.0, self.max_intensity).astype(np.float64, copy=False)

    def close(self) -> None:
        self._closed = True


class MockSpectrometer(MockDevice):
    """Compatibility adapter exposing python-seabreeze-style method names."""

    def __init__(self, serial_number: str = "MOCK-001", model: str = "Flame-S"):
        super().__init__(serial_number=serial_number, model=model)

    @property
    def integration_time_micros_limits(self) -> tuple[int, int]:
        return self.integration_time_limits_us

    @property
    def current_integration_us(self) -> int:
        return self.integration_time_us

    def integration_time_micros(self, microseconds: int) -> None:
        self.set_integration_time(microseconds)

    def trigger_mode(self, mode: int) -> None:
        self.set_trigger_mode(mode)

    def intensities(
        self,
        correct_dark_pixels: bool = False,
        correct_nonlinearity: bool = False,
    ) -> np.ndarray:
        return self.read_intensities(
            correct_dark=correct_dark_pixels,
            correct_nonlinearity=correct_nonlinearity,
        )
