import re

import numpy as np
import pytest

from seabreeze_live import MockDevice
from seabreeze_live.consumers import MatplotlibLiveView
from seabreeze_live.frame import SpectrumFrame
from seabreeze_live.snapshot import save_snapshot


def _make_frame(d: MockDevice) -> SpectrumFrame:
    return SpectrumFrame(
        values=d.read_intensities(),
        axis=d.wavelengths(),
        timestamp_ns=1_700_000_000_000_000_000,
        frame_number=42,
        integration_time_us=d.integration_time_us,
        device_serial=d.serial_number,
    )


def test_csv_snapshot_filename_pattern(tmp_path):
    d = MockDevice(simulate_exposure=False)
    path = save_snapshot(_make_frame(d), tmp_path, "csv")
    assert path.parent == tmp_path
    assert path.suffix == ".csv"
    assert re.match(r"spectrum_\d{8}_\d{6}_\d{6}\.csv", path.name), path.name


def test_csv_snapshot_metadata_and_data(tmp_path):
    d = MockDevice(simulate_exposure=False)
    frame = _make_frame(d)
    path = save_snapshot(frame, tmp_path, "csv")

    text = path.read_text()
    assert f"# device_serial={frame.device_serial}" in text
    assert f"# frame_number={frame.frame_number}" in text
    assert f"# integration_time_us={frame.integration_time_us}" in text
    assert f"# timestamp_ns={frame.timestamp_ns}" in text
    assert "wavelength_nm,intensity" in text

    data_lines = [
        ln for ln in text.splitlines()
        if ln and not ln.startswith("#") and not ln.startswith("wavelength_nm")
    ]
    assert len(data_lines) == d.pixels
    first_w, first_i = data_lines[0].split(",")
    assert float(first_w) == pytest.approx(frame.axis[0], rel=1e-6)
    assert float(first_i) == pytest.approx(frame.values[0], rel=1e-4)


def test_hdf5_snapshot_attrs_and_datasets(tmp_path):
    import h5py

    d = MockDevice(simulate_exposure=False)
    frame = _make_frame(d)
    path = save_snapshot(frame, tmp_path, "h5")

    assert path.suffix == ".h5"
    with h5py.File(path, "r") as f:
        assert f["wavelengths"].shape == (d.pixels,)
        assert f["intensities"].shape == (d.pixels,)
        np.testing.assert_array_equal(f["wavelengths"][:], frame.axis)
        np.testing.assert_array_equal(f["intensities"][:], frame.values)
        assert f.attrs["device_serial"] == frame.device_serial
        assert int(f.attrs["frame_number"]) == frame.frame_number
        assert int(f.attrs["integration_time_us"]) == frame.integration_time_us
        assert int(f.attrs["timestamp_ns"]) == frame.timestamp_ns
        assert f.attrs["axis_units"] == "nm"
        assert f.attrs["value_units"] == "counts"


def test_snapshot_unknown_format(tmp_path):
    d = MockDevice(simulate_exposure=False)
    with pytest.raises(ValueError, match="unknown snapshot format"):
        save_snapshot(_make_frame(d), tmp_path, "txt")


def test_save_snapshot_now_returns_none_before_first_frame(tmp_path):
    view = MatplotlibLiveView(snapshot_dir=tmp_path)
    assert view.save_snapshot_now() is None
    assert view._last_saved is None


def test_save_snapshot_now_writes_and_records_filename(tmp_path):
    d = MockDevice(simulate_exposure=False)
    view = MatplotlibLiveView(snapshot_dir=tmp_path, snapshot_format="csv")
    view.on_frame(_make_frame(d))
    path = view.save_snapshot_now()
    assert path is not None
    assert path.exists()
    assert path.parent == tmp_path
    assert view._last_saved == path.name


def test_save_snapshot_now_each_call_makes_a_new_file(tmp_path):
    import time

    d = MockDevice(simulate_exposure=False)
    view = MatplotlibLiveView(snapshot_dir=tmp_path)
    view.on_frame(_make_frame(d))
    p1 = view.save_snapshot_now()
    time.sleep(0.001)  # avoid same-microsecond collision
    view.on_frame(_make_frame(d))
    p2 = view.save_snapshot_now()
    assert p1 is not None and p2 is not None
    assert p1 != p2
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([p1.name, p2.name])


def test_save_snapshot_now_writes_png_when_figure_attached(tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    d = MockDevice(simulate_exposure=False)
    view = MatplotlibLiveView(snapshot_dir=tmp_path, snapshot_format="csv")
    view.on_frame(_make_frame(d))
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    view._fig = fig
    try:
        data_path = view.save_snapshot_now()
        assert data_path is not None
        png_path = data_path.with_suffix(".png")
        assert png_path.exists()
        assert png_path.stat().st_size > 0
        # Stems match — same timestamped basename for the pair.
        assert data_path.stem == png_path.stem
    finally:
        plt.close(fig)
