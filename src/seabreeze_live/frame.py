from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class SpectrumFrame:
    """One spectrum + the context needed to interpret it.

    Field layout mirrors rust-daq's `SpectrumData` (values / axis / *_units)
    plus capture metadata, so a future Rust driver can map 1:1.
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
