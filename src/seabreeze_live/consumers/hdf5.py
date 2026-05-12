from __future__ import annotations

from pathlib import Path
from typing import Any

from seabreeze_live.frame import SpectrumFrame


class Hdf5Writer:
    """Chunked, SWMR-enabled HDF5 writer.

    Layout:
        /wavelengths        : float64[pixels]   (written once on first frame)
        /intensities        : float64[N, pixels] (resizable, chunked)
        /timestamp_ns       : int64[N]
        /frame_number       : int64[N]
        /integration_time_us: int64[N]
        root attrs: device_serial, value_units, axis_units, schema_version

    Requires h5py: `pip install seabreeze-live[hdf5]`.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, *, chunk_size: int = 64) -> None:
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "Hdf5Writer requires h5py. Install with `pip install seabreeze-live[hdf5]`."
            ) from e
        self._h5py = h5py
        self.path = Path(path)
        self.chunk_size = chunk_size
        self._file: Any = None
        self._values_ds: Any = None
        self._ts_ds: Any = None
        self._fnum_ds: Any = None
        self._integ_ds: Any = None
        self._n_written = 0

    def _ensure_open(self, frame: SpectrumFrame) -> None:
        if self._file is not None:
            return
        f = self._h5py.File(self.path, "w", libver="latest")
        n_pix = int(frame.values.shape[0])
        chunk = (self.chunk_size, n_pix)
        self._values_ds = f.create_dataset(
            "intensities",
            shape=(0, n_pix),
            maxshape=(None, n_pix),
            chunks=chunk,
            dtype="float64",
        )
        self._ts_ds = f.create_dataset(
            "timestamp_ns",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_size,),
            dtype="int64",
        )
        self._fnum_ds = f.create_dataset(
            "frame_number",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_size,),
            dtype="int64",
        )
        self._integ_ds = f.create_dataset(
            "integration_time_us",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_size,),
            dtype="int64",
        )
        f.create_dataset("wavelengths", data=frame.axis)
        f.attrs["device_serial"] = frame.device_serial
        f.attrs["value_units"] = frame.value_units
        f.attrs["axis_units"] = frame.axis_units
        f.attrs["schema_version"] = self.SCHEMA_VERSION
        # SWMR lets a live viewer open the file while we keep writing.
        f.swmr_mode = True
        self._file = f

    def on_frame(self, frame: SpectrumFrame) -> None:
        self._ensure_open(frame)
        n = self._n_written
        new_n = n + 1
        self._values_ds.resize((new_n, frame.values.shape[0]))
        self._values_ds[n] = frame.values
        self._ts_ds.resize((new_n,))
        self._ts_ds[n] = frame.timestamp_ns
        self._fnum_ds.resize((new_n,))
        self._fnum_ds[n] = frame.frame_number
        self._integ_ds.resize((new_n,))
        self._integ_ds[n] = frame.integration_time_us
        for ds in (self._values_ds, self._ts_ds, self._fnum_ds, self._integ_ds):
            ds.flush()
        self._n_written = new_n

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
