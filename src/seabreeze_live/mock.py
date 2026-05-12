from __future__ import annotations

import time

import numpy as np

from seabreeze_live.device import TriggerMode


class MockDevice:
    """Synthetic spectrometer for off-hardware development and tests.

    Produces a sum of Gaussians + Gaussian noise. `simulate_exposure=True`
    sleeps for the configured integration time so streaming behaviour
    matches real hardware; set False in tests for speed.
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
    ) -> None:
        self.serial_number: str = f"MOCK{seed:06d}"
        self.model: str = "MockSpectrometer"
        self.pixels: int = pixels
        self.max_intensity: float = max_intensity
        self.integration_time_limits_us: tuple[int, int] = (1_000, 10_000_000)
        self.integration_time_us: int = 100_000

        self._wavelengths = np.linspace(*wavelength_range_nm, pixels)
        rng = np.random.default_rng(seed)
        lo, hi = wavelength_range_nm
        self._peak_centers = rng.uniform(lo + 50, hi - 50, n_peaks)
        self._peak_amps = rng.uniform(5_000, 30_000, n_peaks)
        self._peak_widths = rng.uniform(2.0, 10.0, n_peaks)
        self._rng = rng
        self._noise_sigma = noise_sigma
        self._simulate_exposure = simulate_exposure
        self._closed = False

    def set_integration_time(self, microseconds: int) -> None:
        lo, hi = self.integration_time_limits_us
        if not lo <= microseconds <= hi:
            raise ValueError(
                f"integration_time {microseconds} us outside mock limits [{lo}, {hi}]"
            )
        self.integration_time_us = microseconds

    def set_trigger_mode(self, mode: TriggerMode) -> None:
        if mode is not TriggerMode.NORMAL:
            raise NotImplementedError(
                f"MockDevice only supports TriggerMode.NORMAL, got {mode.name}"
            )

    def wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

    def read_intensities(
        self,
        correct_dark: bool = False,
        correct_nonlinearity: bool = False,
    ) -> np.ndarray:
        if self._simulate_exposure:
            time.sleep(self.integration_time_us / 1_000_000)

        scale = min(1.0, self.integration_time_us / 1_000_000)
        signal = np.zeros(self.pixels)
        for c, a, w in zip(self._peak_centers, self._peak_amps, self._peak_widths):
            signal += a * np.exp(-0.5 * ((self._wavelengths - c) / w) ** 2)
        signal *= scale
        signal += self._rng.normal(0.0, self._noise_sigma, self.pixels)

        if correct_dark:
            signal -= 50.0
        return np.clip(signal, 0.0, self.max_intensity)

    def close(self) -> None:
        self._closed = True
