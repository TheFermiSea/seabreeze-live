from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from seabreeze_live.frame import SpectrumFrame


def save_snapshot(
    frame: SpectrumFrame,
    directory: str | Path,
    fmt: str = "csv",
) -> Path:
    """Write a single frame to a timestamped file.

    Filename pattern:
        spectrum_YYYYMMDD_HHMMSS_microsec.{csv|h5}

    CSV format: a metadata header (lines starting with `#`), then two
    columns `wavelength_nm,intensity`. Friendly for ad-hoc plotting and
    distinct from the per-frame-row layout that `CsvWriter` produces for
    continuous capture.

    HDF5 format: flat datasets `wavelengths` and `intensities` plus the
    same metadata as root attrs. Single-frame, no time dimension.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    stem = f"spectrum_{stamp}"

    if fmt == "csv":
        path = directory / f"{stem}.csv"
        with path.open("w", newline="") as f:
            f.write(f"# device_serial={frame.device_serial}\n")
            f.write(f"# frame_number={frame.frame_number}\n")
            f.write(f"# integration_time_us={frame.integration_time_us}\n")
            f.write(f"# timestamp_ns={frame.timestamp_ns}\n")
            f.write(f"# axis_units={frame.axis_units}\n")
            f.write(f"# value_units={frame.value_units}\n")
            w = csv.writer(f)
            w.writerow(["wavelength_nm", "intensity"])
            for x, y in zip(frame.axis, frame.values):
                w.writerow([f"{x:.4f}", f"{y:.6g}"])
        return path

    if fmt in ("h5", "hdf5"):
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "HDF5 snapshot requires h5py. "
                "Install with `pip install seabreeze-live[hdf5]`."
            ) from e
        path = directory / f"{stem}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("wavelengths", data=frame.axis)
            f.create_dataset("intensities", data=frame.values)
            f.attrs["device_serial"] = frame.device_serial
            f.attrs["frame_number"] = frame.frame_number
            f.attrs["integration_time_us"] = frame.integration_time_us
            f.attrs["timestamp_ns"] = frame.timestamp_ns
            f.attrs["axis_units"] = frame.axis_units
            f.attrs["value_units"] = frame.value_units
            f.attrs["schema_version"] = 1
        return path

    raise ValueError(f"unknown snapshot format {fmt!r}; use 'csv' or 'h5'")
