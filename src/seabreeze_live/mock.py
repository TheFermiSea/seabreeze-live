"""Realistic Ocean Optics Spectrometer Simulator."""

from __future__ import annotations

import time
from typing import Dict, Tuple
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
        self.integration_time_micros_limits: Tuple[int, int] = (1_000, 10_000_000)
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

    def get_temperatures(self) -> Dict[str, float]:
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
