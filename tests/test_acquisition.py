import pytest

from seabreeze_live import MockDevice, SpectrumFrame, Streamer, acquire


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
