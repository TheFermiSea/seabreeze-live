import time

import pytest

from seabreeze_live import (
    Acquirer,
    AcquisitionSettings,
    MockDevice,
    Spectrometer,
    SpectrumFrame,
    Streamer,
    acquire,
)


def _fast_mock(**kwargs):
    return MockDevice(simulate_exposure=False, **kwargs)


def test_acquire_returns_count_frames():
    d = _fast_mock()
    frames = acquire(d, count=5)
    assert len(frames) == 5
    for i, f in enumerate(frames):
        assert isinstance(f, SpectrumFrame)
        assert f.frame_number == i
        assert f.values.shape == (d.pixels,)
        assert f.axis.shape == (d.pixels,)
        assert f.device_serial == d.serial_number


def test_acquire_timestamps_monotonic():
    d = _fast_mock()
    frames = acquire(d, count=10)
    ts = [f.timestamp_ns for f in frames]
    assert ts == sorted(ts)


def test_acquire_applies_integration_time_override():
    d = _fast_mock()
    acquire(d, count=1, integration_time_us=250_000)
    assert d.integration_time_us == 250_000


def test_mock_satisfies_structural_device_contract():
    assert isinstance(_fast_mock(), Spectrometer)


def test_acquirer_applies_processing_and_preserves_frame_contract():
    d = _fast_mock(seed=8, noise_sigma=0)
    acquirer = Acquirer(d, AcquisitionSettings(scans_to_average=2, boxcar_width=2))
    frame = acquirer.capture()
    assert frame.frame_number == 0
    assert frame.values.shape == frame.axis.shape == (d.pixels,)
    assert frame.values.dtype.name == "float64"
    assert frame.axis.flags.writeable is False


def test_acquirer_rejects_shape_mismatch():
    class BadShape(MockDevice):
        def read_intensities(self, *args, **kwargs):
            return super().read_intensities(*args, **kwargs)[:-1]

    with pytest.raises(ValueError, match="does not match wavelength axis"):
        Acquirer(BadShape(simulate_exposure=False)).capture()


def test_acquire_zero_count_rejected():
    d = _fast_mock()
    with pytest.raises(ValueError):
        acquire(d, count=0)


class _Collector:
    def __init__(self) -> None:
        self.frames: list[SpectrumFrame] = []
        self.closed = False

    def on_frame(self, frame: SpectrumFrame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


def test_streamer_max_frames_and_consumer_close():
    d = _fast_mock()
    collector = _Collector()
    s = Streamer(d, [collector], max_frames=7)
    s.start()
    assert s.wait(timeout=5.0)
    s.stop()
    assert collector.closed is True
    assert s.frame_count == 7
    assert len(collector.frames) == 7
    fnums = [f.frame_number for f in collector.frames]
    assert fnums == list(range(7))


def test_streamer_context_manager_stops_cleanly():
    d = _fast_mock()
    collector = _Collector()
    with Streamer(d, [collector], max_frames=3) as s:
        s.wait(timeout=5.0)
    assert collector.closed is True
    assert len(collector.frames) == 3


def test_streamer_stop_without_max_frames():
    d = _fast_mock()
    collector = _Collector()
    s = Streamer(d, [collector])
    s.start()
    # Let it accumulate a few frames.
    while collector.frames and len(collector.frames) < 3:
        pass
    s.stop()
    assert s.is_running is False
    assert collector.closed is True


def test_streamer_add_consumer_rejected_while_running():
    d = _fast_mock()
    s = Streamer(d, [], max_frames=1000)
    s.start()
    try:
        with pytest.raises(RuntimeError):
            s.add_consumer(_Collector())
    finally:
        s.stop()


def test_streamer_pause_and_resume():
    d = MockDevice(simulate_exposure=True)
    d.set_integration_time(1_000)
    collector = _Collector()
    s = Streamer(d, [collector])
    s.start()
    deadline = time.monotonic() + 2
    while len(collector.frames) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(collector.frames) >= 2
    s.pause()
    paused_at = len(collector.frames)
    time.sleep(0.03)
    assert len(collector.frames) <= paused_at + 1
    s.resume()
    deadline = time.monotonic() + 2
    while len(collector.frames) <= paused_at + 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(collector.frames) > paused_at + 1
    s.stop()


def test_streamer_propagates_consumer_exception():
    class _Bad:
        def on_frame(self, frame):
            raise ValueError("boom")

        def close(self):
            pass

    d = _fast_mock()
    s = Streamer(d, [_Bad()], max_frames=5)
    s.start()
    s.wait(timeout=5.0)
    with pytest.raises(ValueError, match="boom"):
        s.stop()
