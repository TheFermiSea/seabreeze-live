from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class SpectrumFrame:
    """One spectrum + the context needed to interpret it.

    Field layout mirrors rust-daq's `SpectrumData` (values / axis / *_units)
    plus capture metadata used by its existing subprocess driver.
    """

    values: np.ndarray
    axis: np.ndarray
    timestamp_ns: int
    frame_number: int
    integration_time_us: int
    device_serial: str
    value_units: str = "counts"
    axis_units: str = "nm"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 1 or self.axis.ndim != 1:
            raise ValueError("spectrum values and axis must be one-dimensional")
        if self.values.shape != self.axis.shape:
            raise ValueError(
                f"spectrum values and axis differ: {self.values.shape} != {self.axis.shape}"
            )
        if self.timestamp_ns < 0 or self.frame_number < 0:
            raise ValueError("timestamps and frame numbers cannot be negative")
        if self.integration_time_us <= 0:
            raise ValueError("integration_time_us must be positive")

    def to_wire(self, *, include_context: bool = True) -> dict[str, Any]:
        """Return an NDJSON-ready payload.

        ``include_context=False`` emits exactly the compact event consumed by
        rust-daq's ``seabreeze-protocol`` crate. The wavelength axis is fetched
        once during its handshake, so repeating it on every frame only wastes
        subprocess bandwidth.
        """
        payload: dict[str, Any] = {
            "timestamp_ns": self.timestamp_ns,
            "integration_time_us": self.integration_time_us,
            "values": self.values.tolist(),
        }
        if include_context:
            payload.update(
                {
                    "type": "spectrum",
                    "frame_number": self.frame_number,
                    "device_serial": self.device_serial,
                    "value_units": self.value_units,
                    "axis_units": self.axis_units,
                    "axis": self.axis.tolist(),
                }
            )
        return payload
