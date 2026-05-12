from __future__ import annotations

import threading
import time
from typing import Iterable, Protocol, runtime_checkable

from seabreeze_live.device import SpectrometerDevice
from seabreeze_live.frame import SpectrumFrame


@runtime_checkable
class Consumer(Protocol):
    def on_frame(self, frame: SpectrumFrame) -> None: ...
    def close(self) -> None: ...


def acquire(
    device: SpectrometerDevice,
    count: int,
    *,
    correct_dark: bool = False,
    correct_nonlinearity: bool = False,
    integration_time_us: int | None = None,
) -> list[SpectrumFrame]:
    """Capture exactly `count` frames sequentially. Blocking."""
    if count <= 0:
        raise ValueError("count must be positive")
    if integration_time_us is not None:
        device.set_integration_time(integration_time_us)

    wavelengths = device.wavelengths()
    frames: list[SpectrumFrame] = []
    for i in range(count):
        intensities = device.read_intensities(
            correct_dark=correct_dark,
            correct_nonlinearity=correct_nonlinearity,
        )
        frames.append(
            SpectrumFrame(
                values=intensities,
                axis=wavelengths,
                timestamp_ns=time.time_ns(),
                frame_number=i,
                integration_time_us=device.integration_time_us,
                device_serial=device.serial_number,
            )
        )
    return frames


class Streamer:
    """Continuous acquisition in a background thread, dispatched to consumers.

    Consumers are called inline on the acquisition thread. Each call is
    synchronous and blocks the next frame, which provides natural
    backpressure. If a consumer is slower than the integration time, wrap
    it in a queued forwarder (not provided here).
    """

    def __init__(
        self,
        device: SpectrometerDevice,
        consumers: Iterable[Consumer] = (),
        *,
        max_frames: int | None = None,
        correct_dark: bool = False,
        correct_nonlinearity: bool = False,
    ) -> None:
        self.device = device
        self.consumers: list[Consumer] = list(consumers)
        self.max_frames = max_frames
        self._correct_dark = correct_dark
        self._correct_nonlinearity = correct_nonlinearity

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frame_count = 0
        self._error: BaseException | None = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def add_consumer(self, consumer: Consumer) -> None:
        if self.is_running:
            raise RuntimeError("cannot add consumer while streaming")
        self.consumers.append(consumer)

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("streamer already running")
        self._stop.clear()
        self._error = None
        self._frame_count = 0
        self._thread = threading.Thread(
            target=self._loop, name="seabreeze-live", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        for c in self.consumers:
            c.close()
        if self._error is not None:
            err = self._error
            self._error = None
            raise err

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the streamer finishes naturally (max_frames reached)
        or `timeout` elapses. Returns True if the thread is done.

        Polls with a short interval so SIGINT can be delivered between
        polls — a bare `Thread.join(None)` on POSIX is not interrupted by
        signals, so KeyboardInterrupt would never fire.
        """
        if self._thread is None:
            return True
        poll = 0.1
        if timeout is None:
            while self._thread.is_alive():
                self._thread.join(poll)
            return True
        remaining = timeout
        while self._thread.is_alive() and remaining > 0:
            step = min(poll, remaining)
            self._thread.join(step)
            remaining -= step
        return not self._thread.is_alive()

    def __enter__(self) -> "Streamer":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _loop(self) -> None:
        try:
            wavelengths = self.device.wavelengths()
            while not self._stop.is_set():
                if self.max_frames is not None and self._frame_count >= self.max_frames:
                    break
                intensities = self.device.read_intensities(
                    correct_dark=self._correct_dark,
                    correct_nonlinearity=self._correct_nonlinearity,
                )
                frame = SpectrumFrame(
                    values=intensities,
                    axis=wavelengths,
                    timestamp_ns=time.time_ns(),
                    frame_number=self._frame_count,
                    integration_time_us=self.device.integration_time_us,
                    device_serial=self.device.serial_number,
                )
                self._frame_count += 1
                for c in self.consumers:
                    c.on_frame(frame)
        except BaseException as e:  # noqa: BLE001 — surface in stop()
            self._error = e
