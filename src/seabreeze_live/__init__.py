"""seabreeze-live package."""

from seabreeze_live.acquisition import Streamer, acquire
from seabreeze_live.device import (
    DeviceMetadata,
    HardwareConnectionError,
    SeabreezeDevice,
    SpectrometerDevice,
    TriggerMode,
    list_devices,
    open_device,
)
from seabreeze_live.frame import SpectrumFrame
from seabreeze_live.mock import MockDevice

__all__ = [
    "DeviceMetadata",
    "HardwareConnectionError",
    "MockDevice",
    "SeabreezeDevice",
    "SpectrometerDevice",
    "SpectrumFrame",
    "Streamer",
    "TriggerMode",
    "acquire",
    "list_devices",
    "open_device",
]
