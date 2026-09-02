"""seabreeze-live package."""

from seabreeze_live.acquisition import Acquirer, AcquisitionSettings, Streamer, acquire
from seabreeze_live.device import (
    DeviceMetadata,
    HardwareConnectionError,
    HardwareOperationError,
    SeabreezeDevice,
    SpectrometerDevice,
    TriggerMode,
    list_devices,
    open_device,
)
from seabreeze_live.frame import SpectrumFrame
from seabreeze_live.interfaces import Spectrometer
from seabreeze_live.mock import MockDevice

__all__ = [
    "Acquirer",
    "AcquisitionSettings",
    "DeviceMetadata",
    "HardwareConnectionError",
    "HardwareOperationError",
    "MockDevice",
    "SeabreezeDevice",
    "Spectrometer",
    "SpectrometerDevice",
    "SpectrumFrame",
    "Streamer",
    "TriggerMode",
    "acquire",
    "list_devices",
    "open_device",
]
