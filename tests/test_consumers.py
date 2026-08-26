import csv as csv_mod
import io
import json

import numpy as np
import pytest

from seabreeze_live import MockDevice, Streamer
from seabreeze_live.consumers import CsvWriter, Hdf5Writer, NdjsonStdoutEmitter


def _fast_mock(**kwargs):
    return MockDevice(simulate_exposure=False, **kwargs)


def test_csv_roundtrip(tmp_path):
    out = tmp_path / "run.csv"
    d = _fast_mock()
    with Streamer(d, [CsvWriter(out)], max_frames=4) as s:
        s.wait(timeout=5.0)

    with out.open() as f:
        rows = list(csv_mod.reader(f))
    header, *data = rows
    assert header[:3] == ["timestamp_ns", "frame_number", "integration_time_us"]
    assert len(header) == 3 + d.pixels
    assert len(data) == 4
    fnums = [int(r[1]) for r in data]
    assert fnums == [0, 1, 2, 3]


def test_hdf5_roundtrip(tmp_path):
    import h5py

    out = tmp_path / "run.h5"
    d = _fast_mock()
    with Streamer(d, [Hdf5Writer(out)], max_frames=6) as s:
        s.wait(timeout=5.0)

    with h5py.File(out, "r") as f:
        assert f["intensities"].shape == (6, d.pixels)
        assert f["wavelengths"].shape == (d.pixels,)
        assert f["timestamp_ns"].shape == (6,)
        assert f["frame_number"].shape == (6,)
        assert f["integration_time_us"].shape == (6,)
        assert list(f["frame_number"][:]) == list(range(6))
        assert f.attrs["device_serial"] == d.serial_number
        assert f.attrs["value_units"] == "counts"
        assert f.attrs["axis_units"] == "nm"
        assert int(f.attrs["schema_version"]) == Hdf5Writer.SCHEMA_VERSION
        # Wavelengths match what the device produced.
        np.testing.assert_array_equal(f["wavelengths"][:], d.wavelengths())


def test_hdf5_writer_flushes_buffer_when_closed(tmp_path):
    import h5py

    out = tmp_path / "buffered.h5"
    d = _fast_mock()
    with Streamer(d, [Hdf5Writer(out, flush_every=64)], max_frames=3) as s:
        s.wait(timeout=5.0)
    with h5py.File(out, "r") as f:
        assert f["intensities"].shape == (3, d.pixels)


def test_ndjson_emitter_schema():
    buf = io.StringIO()
    emitter = NdjsonStdoutEmitter(stream=buf)
    d = _fast_mock()
    with Streamer(d, [emitter], max_frames=3) as s:
        s.wait(timeout=5.0)

    lines = [ln for ln in buf.getvalue().splitlines() if ln]
    assert len(lines) == 3
    for i, line in enumerate(lines):
        obj = json.loads(line)
        assert obj["type"] == "spectrum"
        assert obj["frame_number"] == i
        assert obj["device_serial"] == d.serial_number
        assert obj["value_units"] == "counts"
        assert obj["axis_units"] == "nm"
        assert len(obj["values"]) == d.pixels
        assert len(obj["axis"]) == d.pixels
        assert isinstance(obj["timestamp_ns"], int)


def test_hdf5_writer_requires_h5py(monkeypatch):
    # Simulate h5py missing by hiding it from import.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("forced")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="h5py"):
        Hdf5Writer("ignored.h5")
