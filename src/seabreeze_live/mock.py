"""Realistic Ocean Optics Spectrometer Simulator."""

from __future__ import annotations

import time

import numpy as np


class MockLamp:
    def __init__(self):
        self.enabled = True

    def set_lamp_enable(self, state: bool):
        self.enabled = bool(state)


class MockShutter:
    def __init__(self):
        self.is_open = True

    def set_shutter_open(self, state: bool):
        self.is_open = bool(state)


class MockSpectrometer:
    """Simulates realistic spectroscopy signals with lamp, shutter, and noise."""

    def __init__(self, serial_number: str = "MOCK-001", model: str = "Flame-S"):
        self.serial_number = serial_number
        self.model = model
        self.integration_time_micros_limits: tuple[int, int] = (1_000, 10_000_000)
        self.current_integration_us: int = 100_000
        self._pixels: int = 2048
        self._wavelengths: np.ndarray = np.linspace(350.0, 1000.0, self._pixels)

        self.lamp = MockLamp()
        self.shutter = MockShutter()
        self._trigger_mode = 0
        self._phase = 0.0

    def wavelengths(self) -> np.ndarray:
        return self._wavelengths

    def integration_time_micros(self, us: int):
        self.current_integration_us = us

    def trigger_mode(self, mode: int):
        self._trigger_mode = mode

    def get_temperatures(self) -> dict[str, float]:
        t = time.time()
        return {
            "Detector": round(22.5 + 0.8 * np.sin(t / 15.0), 2),
            "PCB": round(28.3 + 0.3 * np.cos(t / 20.0), 2),
        }

    def intensities(
        self,
        correct_dark_pixels: bool = False,
        correct_nonlinearity: bool = False,
    ) -> np.ndarray:
        # Shutter closed = pure dark noise
        if not self.shutter.is_open:
            return np.clip(np.random.normal(900, 15, size=self._pixels), 0, 65535)

        wl = self._wavelengths
        self._phase += 0.04

        scale = self.current_integration_us / 100_000.0
        dark_floor = 1100.0 if not correct_dark_pixels else 0.0

        # Broad halogen lamp background
        lamp_power = 1.0 if self.lamp.enabled else 0.05
        broadband = lamp_power * 14000.0 * np.exp(-(((wl - 600.0) / 180.0) ** 2))

        # Characteristic spectral lines
        line_532 = 28000.0 * np.exp(-(((wl - 532.0) / 4.0) ** 2))
        line_656 = 19000.0 * np.exp(-(((wl - 656.3) / 8.0) ** 2))
        line_808 = 24000.0 * np.exp(-(((wl - 808.0) / 3.5) ** 2))

        # Dynamic absorption dip
        dip = 0.6 + 0.3 * np.sin(self._phase)
        absorption = 1.0 - dip * np.exp(-(((wl - 580.0) / 25.0) ** 2))

        signal = (broadband + line_532 + line_656 + line_808) * absorption * scale
        noise = np.random.normal(0, 45 * np.sqrt(scale), size=self._pixels)

        raw = dark_floor + signal + noise
        if not correct_nonlinearity:
            # Simulate detector saturation rolloff near 65k counts
            raw = np.where(raw > 45000, raw - 0.000002 * (raw - 45000) ** 2, raw)

        return np.clip(raw, 0.0, 65535.0)

    def close(self):
        pass


class MockDevice:
    """Protocol-compatible synthetic device retained for library users and tests."""

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
    ) -> None:
        self.serial_number = f"MOCK{seed:06d}"
        self.model = "MockSpectrometer"
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
        self._closed = False

    def set_integration_time(self, microseconds: int) -> None:
        lower, upper = self.integration_time_limits_us
        if not lower <= microseconds <= upper:
            raise ValueError(
                f"integration_time {microseconds} us outside mock limits [{lower}, {upper}]"
            )
        self.integration_time_us = microseconds

    def set_trigger_mode(self, mode: object) -> None:
        if int(mode) != 0:
            raise NotImplementedError("MockDevice only supports TriggerMode.NORMAL")

    def wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

    def read_intensities(
        self, correct_dark: bool = False, correct_nonlinearity: bool = False
    ) -> np.ndarray:
        if self._simulate_exposure:
            time.sleep(self.integration_time_us / 1_000_000)
        scale = min(1.0, self.integration_time_us / 1_000_000)
        signal = np.zeros(self.pixels)
        for center, amplitude, width in zip(
            self._peak_centers, self._peak_amps, self._peak_widths
        ):
            signal += amplitude * np.exp(
                -0.5 * ((self._wavelengths - center) / width) ** 2
            )
        signal *= scale
        signal += self._rng.normal(0.0, self._noise_sigma, self.pixels)
        if correct_dark:
            signal -= 50.0
        return np.clip(signal, 0.0, self.max_intensity)

    def close(self) -> None:
        self._closed = True
