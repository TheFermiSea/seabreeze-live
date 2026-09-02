"""Pure spectrum-processing helpers shared by interactive and headless clients."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DISPLAY_MODES = (
    "Raw Counts",
    "Dark Subtracted",
    "Transmission (%)",
    "Absorbance (AU)",
)


def average_scans(read: Callable[[], np.ndarray], count: int) -> np.ndarray:
    """Read and average spectra without allocating an accumulator for one scan."""
    if count < 1:
        raise ValueError("scan count must be positive")
    first = read()
    if count == 1:
        return first
    total = np.asarray(first, dtype=np.float64).copy()
    for _ in range(count - 1):
        total += read()
    return total / count


def smooth_boxcar(values: np.ndarray, width: int) -> np.ndarray:
    """Apply symmetric boxcar smoothing, preserving the input for width zero."""
    if width <= 0:
        return values
    kernel = np.full(2 * width + 1, 1.0 / (2 * width + 1))
    return np.convolve(values, kernel, mode="same")


def display_values(
    raw: np.ndarray,
    mode: str,
    dark: np.ndarray | None = None,
    white: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Calculate display-only values without changing recorded raw spectra."""
    if mode == "Raw Counts":
        return raw, "Intensity (Counts)"
    if dark is None or dark.shape != raw.shape:
        return raw, "Raw (Dark Baseline Missing!)"
    if mode == "Dark Subtracted":
        return np.maximum(raw - dark, 0.0), "Dark Subtracted (Counts)"
    if white is None or white.shape != raw.shape:
        return raw, "Raw (Take Dark & White Refs First!)"

    transmission = np.clip(
        np.maximum(raw - dark, 0.0) / np.maximum(white - dark, 1.0), 1e-4, 10.0
    )
    if mode == "Transmission (%)":
        return np.clip(transmission * 100.0, 0.0, 200.0), "% Transmission"
    if mode == "Absorbance (AU)":
        return np.clip(-np.log10(transmission), -0.5, 4.0), "Absorbance (AU)"
    raise ValueError(f"unknown display mode: {mode!r}")


def wavelength_mask(wavelengths: np.ndarray, region: str) -> np.ndarray:
    """Return a boolean mask for the named display region."""
    limits = {"vis": (380.0, 750.0), "uv": (200.0, 400.0), "nir": (700.0, 1050.0)}
    if region == "full":
        return np.ones(wavelengths.shape, dtype=bool)
    try:
        lower, upper = limits[region]
    except KeyError as error:
        raise ValueError(f"unknown wavelength region: {region!r}") from error
    return (wavelengths >= lower) & (wavelengths <= upper)
