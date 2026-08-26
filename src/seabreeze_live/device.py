"""SeaBreeze hardware wrapper exposing all hardware features and backwards compatibility."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Self

import numpy as np

try:
    import seabreeze.spectrometers as sb

    SEABREEZE_AVAILABLE = True
except ImportError:
    SEABREEZE_AVAILABLE = False

from seabreeze_live.mock import MockSpectrometer

logger = logging.getLogger(__name__)


class TriggerMode(IntEnum):
    NORMAL = 0
    SOFTWARE = 1
    EXTERNAL_SYNC = 2
    EXTERNAL_HARDWARE = 3
    EXTERNAL_HARDWARE_EDGE = 3


class HardwareConnectionError(RuntimeError):
    """A real spectrometer could not be discovered or opened."""


@dataclass
class DeviceMetadata:
    serial_number: str
    model: str
    min_integration_us: int
    max_integration_us: int
    pixels: int
    has_lamp: bool = False
    has_shutter: bool = False
    has_temperature: bool = False
    has_eeprom: bool = False
    is_mock: bool = False


class SpectrometerDevice:
    """Unified wrapper around physical SeaBreeze and Mock spectrometers."""

    def __init__(self, device_id: str | None = None, use_mock: bool = False):
        self.device: Any = None
        self.meta: DeviceMetadata | None = None
        self._wavelengths: np.ndarray | None = None
        self._integration_time_us = 100_000
        self.open(device_id, use_mock=use_mock)

    @property
    def serial_number(self) -> str:
        return self.meta.serial_number if self.meta else ""

    @property
    def model(self) -> str:
        return self.meta.model if self.meta else ""

    @property
    def pixels(self) -> int:
        return self.meta.pixels if self.meta else 0

    @property
    def integration_time_limits_us(self) -> tuple[int, int]:
        if self.meta is None:
            return (0, 0)
        return (self.meta.min_integration_us, self.meta.max_integration_us)

    @property
    def integration_time_us(self) -> int:
        if self.device is not None and hasattr(self.device, "current_integration_us"):
            return int(self.device.current_integration_us)
        return self._integration_time_us

    @staticmethod
    def list_available_devices() -> list[dict[str, Any]]:
        """List discoverable hardware plus the explicit mock simulator option."""
        devices: list[dict[str, Any]] = []
        if SEABREEZE_AVAILABLE:
            try:
                for dev in sb.list_devices():
                    devices.append(
                        {
                            "id": dev.serial_number,
                            "label": f"{dev.model} [{dev.serial_number}]",
                            "model": dev.model,
                            "mock": False,
                        }
                    )
            except Exception as error:
                # Discovery must never make the TUI unusable; opening a real
                # device still raises a visible HardwareConnectionError.
                logger.debug("SeaBreeze device discovery failed", exc_info=error)

        devices.append(
            {
                "id": "MOCK-SIMULATOR",
                "label": "Ocean Optics Flame-S (Simulated)",
                "model": "Flame-S",
                "mock": True,
            }
        )
        return devices

    def open(
        self, device_id: str | None = None, use_mock: bool = False
    ) -> DeviceMetadata:
        self.close()

        if use_mock or device_id == "MOCK-SIMULATOR":
            self.device = MockSpectrometer()
            min_t, max_t = self.device.integration_time_micros_limits
            self._wavelengths = self.device.wavelengths()
            self.meta = DeviceMetadata(
                serial_number=self.device.serial_number,
                model=self.device.model,
                min_integration_us=int(min_t),
                max_integration_us=int(max_t),
                pixels=len(self._wavelengths),
                has_lamp=True,
                has_shutter=True,
                has_temperature=True,
                has_eeprom=True,
                is_mock=True,
            )
            self.set_integration_time_micros(self._integration_time_us)
            return self.meta

        if not SEABREEZE_AVAILABLE:
            raise HardwareConnectionError(
                "python-seabreeze is unavailable; install it and its USB backend, "
                "or pass --mock to use the simulator"
            )

        try:
            if device_id:
                self.device = sb.Spectrometer.from_serial_number(device_id)
            else:
                self.device = sb.Spectrometer.from_first_available()
        except Exception as error:
            target = (
                f"serial {device_id!r}" if device_id else "the first available device"
            )
            raise HardwareConnectionError(
                f"could not open {target}; run `seabreeze-live devices` to verify "
                "discovery, then check the USB cable, permissions, and SeaBreeze driver"
            ) from error

        try:
            min_t, max_t = self.device.integration_time_micros_limits
            self._wavelengths = np.asarray(self.device.wavelengths(), dtype=np.float64)
        except Exception as error:
            self.close()
            raise HardwareConnectionError(
                "device opened but could not read integration-time limits or wavelengths"
            ) from error

        has_lamp = hasattr(self.device, "lamp") and self.device.lamp is not None
        has_shutter = (
            hasattr(self.device, "shutter") and self.device.shutter is not None
        )
        has_temp = hasattr(self.device, "f") and hasattr(self.device.f, "temperature")
        has_eeprom = hasattr(self.device, "f") and hasattr(self.device.f, "eeprom")

        self.meta = DeviceMetadata(
            serial_number=self.device.serial_number,
            model=self.device.model,
            min_integration_us=int(min_t),
            max_integration_us=int(max_t),
            pixels=len(self._wavelengths),
            has_lamp=has_lamp,
            has_shutter=has_shutter,
            has_temperature=has_temp,
            has_eeprom=has_eeprom,
            is_mock=False,
        )
        # A requested setting is applied only after limits are known. Clamp to
        # the hardware range rather than leaving an unknown driver default.
        self.set_integration_time_micros(self._integration_time_us)
        return self.meta

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self):
        if self.device is not None:
            try:
                self.device.close()
            except Exception as error:
                logger.debug("SeaBreeze device close failed", exc_info=error)
            self.device = None
            self.meta = None
            self._wavelengths = None

    def get_wavelengths(self) -> np.ndarray:
        if self._wavelengths is None:
            if self.device is not None:
                self._wavelengths = self.device.wavelengths()
            else:
                self._wavelengths = np.linspace(350.0, 1000.0, 2048)
        return self._wavelengths

    def wavelengths(self) -> np.ndarray:
        return self.get_wavelengths()

    def get_intensities(
        self,
        correct_dark_pixels: bool = False,
        correct_nonlinearity: bool = False,
    ) -> np.ndarray:
        if self.device is None:
            return np.zeros(2048, dtype=np.float64)
        kwargs = {"correct_nonlinearity": correct_nonlinearity}
        if self.meta is not None and self.meta.is_mock:
            kwargs["correct_dark_pixels"] = correct_dark_pixels
        else:
            # python-seabreeze calls this hardware correction
            # ``correct_dark_counts`` rather than ``correct_dark_pixels``.
            kwargs["correct_dark_counts"] = correct_dark_pixels
        return np.asarray(self.device.intensities(**kwargs), dtype=np.float64)

    def read_intensities(
        self, correct_dark: bool = False, correct_nonlinearity: bool = False
    ) -> np.ndarray:
        """Compatibility spelling used by the acquisition and RPC APIs."""
        return self.get_intensities(
            correct_dark_pixels=correct_dark,
            correct_nonlinearity=correct_nonlinearity,
        )

    def intensities(
        self,
        correct_dark_pixels: bool = False,
        correct_nonlinearity: bool = False,
    ) -> np.ndarray:
        return self.get_intensities(
            correct_dark_pixels=correct_dark_pixels,
            correct_nonlinearity=correct_nonlinearity,
        )

    def set_integration_time_micros(self, integration_time_us: int) -> int:
        if self.device is not None and self.meta is not None:
            clamped = max(
                self.meta.min_integration_us,
                min(integration_time_us, self.meta.max_integration_us),
            )
            self.device.integration_time_micros(clamped)
            self._integration_time_us = clamped
            return clamped
        return integration_time_us

    def set_integration_time(self, microseconds: int) -> int:
        return self.set_integration_time_micros(microseconds)

    def integration_time_micros(self, us: int) -> int:
        return self.set_integration_time_micros(us)

    def set_trigger_mode(self, mode: int | TriggerMode):
        val = int(mode)
        if self.device is not None and hasattr(self.device, "trigger_mode"):
            try:
                self.device.trigger_mode(val)
            except Exception as error:
                logger.debug("Could not set trigger mode", exc_info=error)

    def trigger_mode(self, mode: int | TriggerMode):
        self.set_trigger_mode(mode)

    def set_lamp_enable(self, state: bool):
        if (
            self.device is not None
            and hasattr(self.device, "lamp")
            and self.device.lamp
        ):
            try:
                self.device.lamp.set_lamp_enable(state)
            except Exception as error:
                logger.debug("Could not set lamp state", exc_info=error)

    def set_shutter_open(self, state: bool):
        if (
            self.device is not None
            and hasattr(self.device, "shutter")
            and self.device.shutter
        ):
            try:
                self.device.shutter.set_shutter_open(state)
            except Exception as error:
                logger.debug("Could not set shutter state", exc_info=error)

    def read_temperatures(self) -> dict[str, float]:
        temps: dict[str, float] = {}
        if self.device is None:
            return temps
        if self.meta and self.meta.is_mock:
            return self.device.get_temperatures()
        if self.meta and self.meta.has_temperature:
            try:
                raw = self.device.f.temperature.get_temperatures()
                for i, t in enumerate(raw):
                    temps[f"Sensor {i}"] = float(t)
            except Exception as error:
                logger.debug("Could not read temperatures", exc_info=error)
        return temps

    def read_eeprom_slot(self, slot_index: int) -> str:
        if self.device is not None and self.meta and self.meta.has_eeprom:
            try:
                return str(self.device.f.eeprom.read_eeprom_slot(slot_index))
            except Exception as error:
                logger.debug("Could not read EEPROM slot", exc_info=error)
        return ""


# Compatibility aliases
SeabreezeDevice = SpectrometerDevice


def open_device(
    serial_number: str | None = None,
    mock: bool = False,
    use_mock: bool = False,
) -> SpectrometerDevice:
    """Convenience helper to instantiate SpectrometerDevice."""
    return SpectrometerDevice(device_id=serial_number, use_mock=(mock or use_mock))


def list_devices() -> list[dict[str, Any]]:
    """Module-level alias for enumerating devices."""
    return SpectrometerDevice.list_available_devices()
