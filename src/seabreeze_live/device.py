"""SeaBreeze hardware wrapper exposing all hardware features and backwards compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import seabreeze.spectrometers as sb

    SEABREEZE_AVAILABLE = True
except ImportError:
    SEABREEZE_AVAILABLE = False

from seabreeze_live.mock import MockSpectrometer


class TriggerMode(IntEnum):
    NORMAL = 0
    SOFTWARE = 1
    EXTERNAL_SYNC = 2
    EXTERNAL_HARDWARE = 3


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

    def __init__(self, device_id: Optional[str] = None, use_mock: bool = False):
        self.device: Any = None
        self.meta: Optional[DeviceMetadata] = None
        self._wavelengths: Optional[np.ndarray] = None
        self.open(device_id, use_mock=use_mock)

    @property
    def serial_number(self) -> str:
        return self.meta.serial_number if self.meta else ""

    @property
    def model(self) -> str:
        return self.meta.model if self.meta else ""

    @staticmethod
    def list_available_devices() -> List[Dict[str, Any]]:
        """List all connected spectrometers and the mock simulator."""
        devices: List[Dict[str, Any]] = []
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
            except Exception:
                pass

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
        self, device_id: Optional[str] = None, use_mock: bool = False
    ) -> DeviceMetadata:
        self.close()

        if use_mock or device_id == "MOCK-SIMULATOR" or not SEABREEZE_AVAILABLE:
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
            return self.meta

        if device_id:
            self.device = sb.Spectrometer.from_serial_number(device_id)
        else:
            self.device = sb.Spectrometer.from_first_available()

        min_t, max_t = self.device.integration_time_micros_limits
        self._wavelengths = self.device.wavelengths()

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
        return self.meta

    def close(self):
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
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
        return self.device.intensities(
            correct_dark_pixels=correct_dark_pixels,
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
            return clamped
        return integration_time_us

    def integration_time_micros(self, us: int) -> int:
        return self.set_integration_time_micros(us)

    def set_trigger_mode(self, mode: int | TriggerMode):
        val = int(mode)
        if self.device is not None and hasattr(self.device, "trigger_mode"):
            try:
                self.device.trigger_mode(val)
            except Exception:
                pass

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
            except Exception:
                pass

    def set_shutter_open(self, state: bool):
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
        temps: Dict[str, float] = {}
        if self.device is None:
            return temps
        if self.meta and self.meta.is_mock:
            return self.device.get_temperatures()
        if self.meta and self.meta.has_temperature:
            try:
                raw = self.device.f.temperature.get_temperatures()
                for i, t in enumerate(raw):
                    temps[f"Sensor {i}"] = float(t)
            except Exception:
                pass
        return temps

    def read_eeprom_slot(self, slot_index: int) -> str:
        if self.device is not None and self.meta and self.meta.has_eeprom:
            try:
                return str(self.device.f.eeprom.read_eeprom_slot(slot_index))
            except Exception:
                pass
        return ""


# Compatibility aliases
SeabreezeDevice = SpectrometerDevice


def open_device(
    serial_number: Optional[str] = None,
    mock: bool = False,
    use_mock: bool = False,
) -> SpectrometerDevice:
    """Convenience helper to instantiate SpectrometerDevice."""
    return SpectrometerDevice(device_id=serial_number, use_mock=(mock or use_mock))


def list_devices() -> List[Dict[str, Any]]:
    """Module-level alias for enumerating devices."""
    return SpectrometerDevice.list_available_devices()
