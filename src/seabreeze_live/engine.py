"""Spectrometer hardware abstraction and acquisition pipeline."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np

try:
    import seabreeze.spectrometers as sb

    SEABREEZE_AVAILABLE = True
except ImportError:
    SEABREEZE_AVAILABLE = False


@dataclass
class SpectrometerInfo:
    serial_number: str
    model: str
    min_integration_us: int
    max_integration_us: int
    pixels: int
    has_lamp: bool = False
    has_shutter: bool = False
    has_temperature: bool = False
    is_mock: bool = False


@dataclass
class SpectrumFrame:
    wavelengths: np.ndarray
    intensities: np.ndarray
    timestamp: float
    integration_time_us: int
    dark_subtracted: bool = False
    saturation_level: float = 65535.0
    is_saturated: bool = False
    fps: float = 0.0
    temperatures: Dict[str, float] = field(default_factory=dict)


class SpectrometerEngine:
    """Universal controller wrapping SeaBreeze hardware and Mock fallback."""

    def __init__(self):
        self.device: Any = None
        self.info: Optional[SpectrometerInfo] = None
        self.is_running: bool = False

        # Acquisition settings
        self.integration_time_us: int = 100_000
        self.scans_to_average: int = 1
        self.boxcar_width: int = 0
        self.correct_dark_pixels: bool = False
        self.correct_nonlinearity: bool = False
        self.trigger_mode: int = 0

        # Baselines
        self.dark_spectrum: Optional[np.ndarray] = None
        self.white_reference: Optional[np.ndarray] = None

        # Processing & metrics
        self._last_time: float = time.perf_counter()
        self._frame_count: int = 0
        self._fps: float = 0.0

        # Active recording streams
        self.recording_active: bool = False
        self.recording_format: str = "CSV"
        self.recording_path: Optional[Path] = None
        self._csv_writer = None
        self._csv_file = None
        self._h5_file: Optional[h5py.File] = None
        self._h5_dataset = None

    @staticmethod
    def list_devices() -> List[Dict[str, str]]:
        """Return list of available real and mock spectrometers."""
        devices = []
        if SEABREEZE_AVAILABLE:
            try:
                for dev in sb.list_devices():
                    devices.append(
                        {
                            "id": dev.serial_number,
                            "label": f"{dev.model} [{dev.serial_number}]",
                            "mock": False,
                        }
                    )
            except Exception:
                pass

        # Always provide Mock device option
        devices.append(
            {
                "id": "MOCK-001",
                "label": "Mock OceanOptics Flame-S (Simulator)",
                "mock": True,
            }
        )
        return devices

    def connect(self, device_id: str) -> SpectrometerInfo:
        """Connect to real spectrometer or instantiate simulator."""
        self.disconnect()

        if device_id == "MOCK-001" or not SEABREEZE_AVAILABLE:
            self.device = None
            self.info = SpectrometerInfo(
                serial_number="MOCK-001",
                model="Flame-S (Simulated)",
                min_integration_us=1_000,
                max_integration_us=10_000_000,
                pixels=2048,
                has_lamp=True,
                has_shutter=True,
                has_temperature=True,
                is_mock=True,
            )
            self._mock_wavelengths = np.linspace(350.0, 1000.0, 2048)
            self._mock_phase = 0.0
        else:
            self.device = sb.Spectrometer.from_serial_number(device_id)
            min_t, max_t = self.device.integration_time_micros_limits

            # Detect hardware features
            has_lamp = hasattr(self.device, "lamp") and self.device.lamp is not None
            has_shutter = (
                hasattr(self.device, "shutter") and self.device.shutter is not None
            )
            has_temp = hasattr(self.device, "f") and hasattr(
                self.device.f, "temperature"
            )

            self.info = SpectrometerInfo(
                serial_number=self.device.serial_number,
                model=self.device.model,
                min_integration_us=int(min_t),
                max_integration_us=int(max_t),
                pixels=len(self.device.wavelengths()),
                has_lamp=has_lamp,
                has_shutter=has_shutter,
                has_temperature=has_temp,
                is_mock=False,
            )
            self.set_integration_time(self.integration_time_us)

        return self.info

    def disconnect(self):
        self.stop_recording()
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None
        self.info = None

    def set_integration_time(self, integration_time_us: int):
        self.integration_time_us = integration_time_us
        if self.device is not None:
            clamped = max(
                self.info.min_integration_us,
                min(integration_time_us, self.info.max_integration_us),
            )
            self.device.integration_time_micros(clamped)

    def set_trigger_mode(self, mode: int):
        self.trigger_mode = mode
        if self.device is not None and hasattr(self.device, "trigger_mode"):
            try:
                self.device.trigger_mode(mode)
            except Exception:
                pass

    def set_lamp(self, state: bool):
        if (
            self.device is not None
            and hasattr(self.device, "lamp")
            and self.device.lamp
        ):
            try:
                self.device.lamp.set_lamp_enable(state)
            except Exception:
                pass

    def set_shutter(self, state: bool):
        if (
            self.device is not None
            and hasattr(self.device, "shutter")
            and self.device.shutter
        ):
            try:
                self.device.shutter.set_shutter_open(state)
            except Exception:
                pass

    def read_temperatures(self) -> Dict[str, float]:
        temps = {}
        if self.device is not None and self.info and self.info.has_temperature:
            try:
                for idx, t in enumerate(self.device.f.temperature.get_temperatures()):
                    temps[f"Sensor {idx}"] = float(t)
            except Exception:
                pass
        elif self.info and self.info.is_mock:
            temps["Detector"] = 23.4 + 0.5 * np.sin(time.time() / 10)
            temps["PCB"] = 28.1
        return temps

    def acquire_frame(self) -> SpectrumFrame:
        """Acquire, average, correct, and return a single spectral frame."""
        now = time.perf_counter()
        dt = now - self._last_time
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._last_time = now
        self._frame_count += 1

        if self.info is None or self.info.is_mock:
            # Generate simulated absorption & emission peaks
            wl = self._mock_wavelengths
            self._mock_phase += 0.05
            base = 1200 + 400 * np.sin(wl / 100.0 + self._mock_phase)
            peak1 = 35000 * np.exp(-(((wl - 532.0) / 12.0) ** 2))
            peak2 = 22000 * np.exp(-(((wl - 650.0) / 25.0) ** 2))
            peak3 = 18000 * np.exp(-(((wl - 808.0) / 8.0) ** 2))
            noise = np.random.normal(0, 150, size=len(wl))
            intensities = np.clip(base + peak1 + peak2 + peak3 + noise, 0, 65535)
        else:
            accum = np.zeros(self.info.pixels, dtype=np.float64)
            for _ in range(max(1, self.scans_to_average)):
                accum += self.device.intensities(
                    correct_dark_pixels=self.correct_dark_pixels,
                    correct_nonlinearity=self.correct_nonlinearity,
                )
            intensities = accum / max(1, self.scans_to_average)
            wl = self.device.wavelengths()

        # Boxcar smoothing
        if self.boxcar_width > 0:
            kernel = np.ones(self.boxcar_width * 2 + 1) / (self.boxcar_width * 2 + 1)
            intensities = np.convolve(intensities, kernel, mode="same")

        sat_limit = 65000.0
        is_sat = np.any(intensities >= sat_limit)
        temps = self.read_temperatures()

        frame = SpectrumFrame(
            wavelengths=wl,
            intensities=intensities,
            timestamp=now,
            integration_time_us=self.integration_time_us,
            dark_subtracted=self.dark_spectrum is not None,
            saturation_level=sat_limit,
            is_saturated=bool(is_sat),
            fps=self._fps,
            temperatures=temps,
        )

        # Handle active recording
        if self.recording_active:
            self._record_frame(frame)

        return frame

    def calculate_display_data(
        self, frame: SpectrumFrame, mode: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Convert raw intensities into Absorbance, Transmission, or Subtracted spectra."""
        wl = frame.wavelengths
        raw = frame.intensities.copy()

        if mode == "Raw Counts":
            return wl, raw, "Intensity (Counts)"

        if mode == "Dark Subtracted":
            if self.dark_spectrum is not None and len(self.dark_spectrum) == len(raw):
                sub = np.maximum(raw - self.dark_spectrum, 0.0)
                return wl, sub, "Dark-Subtracted Counts"
            return wl, raw, "Intensity (Dark Ref Missing!)"

        if mode == "Transmission (%)":
            if (
                self.dark_spectrum is not None
                and self.white_reference is not None
                and len(self.dark_spectrum) == len(raw)
            ):
                denom = np.maximum(self.white_reference - self.dark_spectrum, 1e-6)
                numer = np.maximum(raw - self.dark_spectrum, 0.0)
                trans = (numer / denom) * 100.0
                return wl, np.clip(trans, 0.0, 200.0), "% Transmission"
            return wl, raw, "Intensity (Refs Missing!)"

        if mode == "Absorbance (AU)":
            if (
                self.dark_spectrum is not None
                and self.white_reference is not None
                and len(self.dark_spectrum) == len(raw)
            ):
                denom = np.maximum(self.white_reference - self.dark_spectrum, 1e-6)
                numer = np.maximum(raw - self.dark_spectrum, 1e-6)
                trans = np.clip(numer / denom, 1e-6, 10.0)
                abs_val = -np.log10(trans)
                return wl, np.clip(abs_val, -0.5, 4.0), "Absorbance (AU)"
            return wl, raw, "Intensity (Refs Missing!)"

        return wl, raw, "Intensity"

    def start_recording(self, path: Path, fmt: str = "CSV"):
        self.recording_path = path
        self.recording_format = fmt
        self.recording_active = True

        if fmt == "CSV":
            self._csv_file = open(path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(
                ["timestamp", "integration_us"]
                + [
                    f"wl_{w:.2f}"
                    for w in (
                        self.device.wavelengths()
                        if self.device
                        else self._mock_wavelengths
                    )
                ]
            )
        elif fmt == "HDF5":
            self._h5_file = h5py.File(path, "w")
            num_px = self.info.pixels if self.info else 2048
            self._h5_dataset = self._h5_file.create_dataset(
                "spectra",
                shape=(0, num_px),
                maxshape=(None, num_px),
                dtype="float32",
                chunks=True,
            )
            self._h5_file.create_dataset(
                "wavelengths",
                data=self.device.wavelengths()
                if self.device
                else self._mock_wavelengths,
            )

    def _record_frame(self, frame: SpectrumFrame):
        if self.recording_format == "CSV" and self._csv_writer:
            row = [frame.timestamp, frame.integration_time_us] + list(frame.intensities)
            self._csv_writer.writerow(row)
        elif self.recording_format == "HDF5" and self._h5_file and self._h5_dataset:
            curr_len = self._h5_dataset.shape[0]
            self._h5_dataset.resize(curr_len + 1, axis=0)
            self._h5_dataset[curr_len] = frame.intensities.astype("float32")

    def stop_recording(self):
        self.recording_active = False
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None
        if self._h5_file:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None
            self._h5_dataset = None

    def export_snapshot(self, path: Path, frame: SpectrumFrame):
        """Export current frame to CSV or JSON snapshot."""
        if path.suffix.lower() == ".json":
            data = {
                "timestamp": frame.timestamp,
                "integration_time_us": frame.integration_time_us,
                "model": self.info.model if self.info else "Mock",
                "serial": self.info.serial_number if self.info else "MOCK",
                "wavelengths": frame.wavelengths.tolist(),
                "intensities": frame.intensities.tolist(),
                "temperatures": frame.temperatures,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Wavelength_nm", "Intensity_Counts"])
                for w, i in zip(frame.wavelengths, frame.intensities):
                    writer.writerow([f"{w:.4f}", f"{i:.4f}"])
