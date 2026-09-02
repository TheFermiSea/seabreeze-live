"""Transport-neutral spectrum acquisition and streaming."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np
from typing_extensions import Self

from seabreeze_live.frame import SpectrumFrame
from seabreeze_live.interfaces import Spectrometer
from seabreeze_live.processing import average_scans, smooth_boxcar


@runtime_checkable
class Consumer(Protocol):
    def on_frame(self, frame: SpectrumFrame) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AcquisitionSettings:
    """Settings applied consistently to every captured frame."""

    scans_to_average: int = 1
    boxcar_width: int = 0
    correct_dark: bool = False
    correct_nonlinearity: bool = False

    def __post_init__(self) -> None:
        if self.scans_to_average < 1:
            raise ValueError("scans_to_average must be positive")
        if self.boxcar_width < 0:
            raise ValueError("boxcar_width cannot be negative")


class Acquirer:
    """Serialize access to one spectrometer and construct canonical frames.

    Device I/O and configuration share one lock. UI controls can therefore
    change exposure or replace a device without racing an in-flight read.
    """

    def __init__(
        self,
        device: Spectrometer,
        settings: AcquisitionSettings | None = None,
    ) -> None:
        self._device = device
        self._settings = settings or AcquisitionSettings()
        self._lock = threading.RLock()
        self._frame_number = 0
        self._axis = self._read_axis(device)

    @staticmethod
    def _read_axis(device: Spectrometer) -> np.ndarray:
        axis = np.asarray(device.wavelengths(), dtype=np.float64)
        if axis.ndim != 1 or axis.size == 0:
            raise ValueError("spectrometer wavelength axis must be a non-empty vector")
        axis.setflags(write=False)
        return axis

    @property
    def device(self) -> Spectrometer:
        return self._device

    @property
    def settings(self) -> AcquisitionSettings:
        with self._lock:
            return self._settings

    @property
    def frame_number(self) -> int:
        with self._lock:
            return self._frame_number

    def configure(self, **changes: object) -> AcquisitionSettings:
        """Atomically update capture settings and return the new snapshot."""
        with self._lock:
            self._settings = replace(self._settings, **changes)
            return self._settings

    def set_integration_time(self, microseconds: int) -> int:
        with self._lock:
            self._device.set_integration_time(microseconds)
            return self._device.integration_time_us

    def set_trigger_mode(self, mode: int) -> None:
        with self._lock:
            self._device.set_trigger_mode(mode)

    def replace_device(
        self, device: Spectrometer, *, close_previous: bool = True
    ) -> None:
        """Atomically switch devices and restart frame numbering."""
        axis = self._read_axis(device)
        with self._lock:
            previous = self._device
            self._device = device
            self._axis = axis
            self._frame_number = 0
            if close_previous and previous is not device:
                previous.close()

    def close(self) -> None:
        """Close the active device after any in-flight capture completes."""
        with self._lock:
            self._device.close()

    def capture(self) -> SpectrumFrame:
        """Acquire one processed frame while excluding configuration races."""
        with self._lock:
            settings = self._settings
            values = average_scans(
                lambda: self._device.read_intensities(
                    correct_dark=settings.correct_dark,
                    correct_nonlinearity=settings.correct_nonlinearity,
                ),
                settings.scans_to_average,
            )
            values = np.asarray(
                smooth_boxcar(values, settings.boxcar_width), dtype=np.float64
            )
            if values.ndim != 1 or values.shape != self._axis.shape:
                raise ValueError(
                    "spectrum shape does not match wavelength axis: "
                    f"{values.shape} != {self._axis.shape}"
                )
            frame = SpectrumFrame(
                values=values,
                axis=self._axis,
                timestamp_ns=time.time_ns(),
                frame_number=self._frame_number,
                integration_time_us=self._device.integration_time_us,
                device_serial=self._device.serial_number,
            )
            self._frame_number += 1
            return frame


def acquire(
    device: Spectrometer,
    count: int,
    *,
    correct_dark: bool = False,
    correct_nonlinearity: bool = False,
    integration_time_us: int | None = None,
    scans_to_average: int = 1,
    boxcar_width: int = 0,
) -> list[SpectrumFrame]:
    """Capture exactly ``count`` frames sequentially."""
    if count <= 0:
        raise ValueError("count must be positive")
    acquirer = Acquirer(
        device,
        AcquisitionSettings(
            scans_to_average=scans_to_average,
            boxcar_width=boxcar_width,
            correct_dark=correct_dark,
            correct_nonlinearity=correct_nonlinearity,
        ),
    )
    if integration_time_us is not None:
        acquirer.set_integration_time(integration_time_us)
    return [acquirer.capture() for _ in range(count)]


class Streamer:
    """Continuous acquisition in a background thread with backpressure."""

    def __init__(
        self,
        device: Spectrometer,
        consumers: Iterable[Consumer] = (),
        *,
        max_frames: int | None = None,
        correct_dark: bool = False,
        correct_nonlinearity: bool = False,
        scans_to_average: int = 1,
        boxcar_width: int = 0,
    ) -> None:
        self.acquirer = Acquirer(
            device,
            AcquisitionSettings(
                scans_to_average=scans_to_average,
                boxcar_width=boxcar_width,
                correct_dark=correct_dark,
                correct_nonlinearity=correct_nonlinearity,
            ),
        )
        self.consumers: list[Consumer] = list(consumers)
        self.max_frames = max_frames
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._error: BaseException | None = None
        self._consumers_closed = False

    @property
    def device(self) -> Spectrometer:
        return self.acquirer.device

    @property
    def frame_count(self) -> int:
        return self.acquirer.frame_number

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    def add_consumer(self, consumer: Consumer) -> None:
        if self.is_running:
            raise RuntimeError("cannot add consumer while streaming")
        self.consumers.append(consumer)

    def configure(self, **changes: object) -> AcquisitionSettings:
        return self.acquirer.configure(**changes)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def start(self) -> None:
        if self.is_running:
            raise RuntimeError("streamer already running")
        if self._consumers_closed:
            raise RuntimeError(
                "cannot restart a streamer after its consumers are closed"
            )
        self._stop.clear()
        self._pause.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._loop, name="seabreeze-live", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 15.0) -> None:
        self._stop.set()
        self._pause.clear()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)
            if thread.is_alive():
                raise TimeoutError(
                    "spectrometer acquisition did not stop before timeout"
                )
            self._thread = None

        close_error: BaseException | None = None
        if not self._consumers_closed:
            for consumer in self.consumers:
                try:
                    consumer.close()
                except BaseException as error:  # noqa: BLE001 - close every consumer
                    close_error = close_error or error
            self._consumers_closed = True

        worker_error, self._error = self._error, None
        if worker_error is not None:
            raise worker_error
        if close_error is not None:
            raise close_error

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for natural completion while remaining interruptible."""
        thread = self._thread
        if thread is None:
            return True
        deadline = None if timeout is None else time.monotonic() + timeout
        while thread.is_alive():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                thread.join(min(0.1, remaining))
            else:
                thread.join(0.1)
        return True

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                if self.max_frames is not None and self.frame_count >= self.max_frames:
                    break
                if self._pause.is_set():
                    self._stop.wait(0.01)
                    continue
                frame = self.acquirer.capture()
                for consumer in self.consumers:
                    consumer.on_frame(frame)
        except BaseException as error:  # noqa: BLE001 - re-raised by stop()
            self._error = error
