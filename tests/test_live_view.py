"""Tests for MatplotlibLiveView's consumer-side behavior.

The actual GUI (`run()` / `plt.show()`) is not exercised here — it
requires a display and is matplotlib's responsibility. We verify:
  * import-error path when matplotlib is missing,
  * Consumer protocol (`on_frame` updates `latest_frame`, `close` flags),
  * the view composes with Streamer end-to-end (frames arrive).
"""

import asyncio
import builtins

import numpy as np
import pytest

from seabreeze_live import MockDevice, Streamer
from seabreeze_live.consumers import MatplotlibLiveView
from seabreeze_live.processing import display_values, wavelength_mask
from seabreeze_live.recording import SpectrumRecorder
from seabreeze_live.tui import LiveSpectrometerApp


def _fast_mock(**kw):
    return MockDevice(simulate_exposure=False, **kw)


def test_live_view_requires_matplotlib(monkeypatch):
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("forced")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="matplotlib"):
        MatplotlibLiveView()


def test_on_frame_stores_latest_and_close_flags():
    view = MatplotlibLiveView()
    assert view.latest_frame is None
    d = _fast_mock()
    wavelengths = d.wavelengths()
    from seabreeze_live.frame import SpectrumFrame

    f0 = SpectrumFrame(
        values=d.read_intensities(),
        axis=wavelengths,
        timestamp_ns=1,
        frame_number=0,
        integration_time_us=d.integration_time_us,
        device_serial=d.serial_number,
    )
    f1 = SpectrumFrame(
        values=d.read_intensities(),
        axis=wavelengths,
        timestamp_ns=2,
        frame_number=1,
        integration_time_us=d.integration_time_us,
        device_serial=d.serial_number,
    )
    view.on_frame(f0)
    assert view.latest_frame is f0
    view.on_frame(f1)
    assert view.latest_frame is f1
    view.close()
    assert view._closed is True


def test_live_view_receives_frames_from_streamer():
    view = MatplotlibLiveView()
    d = _fast_mock()
    with Streamer(d, [view], max_frames=5) as s:
        s.wait(timeout=5.0)
    assert view.latest_frame is not None
    assert view.latest_frame.frame_number == 4


def test_display_processing_handles_reference_and_region_boundaries():
    raw = np.array([10.0, 50.0, 110.0])
    dark = np.array([10.0, 10.0, 10.0])
    white = np.array([110.0, 110.0, 110.0])
    transmission, label = display_values(raw, "Transmission (%)", dark, white)
    assert label == "% Transmission"
    np.testing.assert_allclose(transmission, [0.01, 40.0, 100.0])
    assert wavelength_mask(np.array([350.0, 500.0, 800.0]), "vis").tolist() == [
        False,
        True,
        False,
    ]


def test_recorder_uses_standard_csv_schema(tmp_path):
    d = _fast_mock()
    recorder = SpectrumRecorder(tmp_path / "tui.csv", "csv")
    recorder.write(
        values=d.read_intensities(),
        axis=d.wavelengths(),
        timestamp_ns=1,
        frame_number=0,
        integration_time_us=d.integration_time_us,
        device_serial=d.serial_number,
    )
    recorder.close()
    header = (tmp_path / "tui.csv").read_text().splitlines()[0].split(",")
    assert header[:3] == ["timestamp_ns", "frame_number", "integration_time_us"]


def test_textual_tui_runs_and_stops_with_mock_device():
    async def exercise() -> None:
        app = LiveSpectrometerApp(use_mock=True, integration_time_us=1_000)
        async with app.run_test() as pilot:
            await pilot.pause(0.2)
            assert app.dev.meta is not None and app.dev.meta.is_mock
            assert app.latest_intensities.shape == (app.dev.meta.pixels,)

    asyncio.run(exercise())
