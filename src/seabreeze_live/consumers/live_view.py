from __future__ import annotations

import time
from typing import Any

from seabreeze_live.frame import SpectrumFrame


class MatplotlibLiveView:
    """Live spectrum plot as a Consumer.

    The Streamer calls `on_frame` from its background thread and we store
    the latest frame in a single attribute. The GUI runs on the main
    thread via `run()`, polled by `matplotlib.animation.FuncAnimation`.

    Frame handoff is lock-free because:
      * `SpectrumFrame` is frozen,
      * assigning a single attribute is atomic under the CPython GIL,
      * the reader only ever sees a fully-constructed frame.

    Requires matplotlib: `pip install seabreeze-live[plot]`.
    """

    def __init__(
        self,
        *,
        refresh_ms: int = 50,
        title: str | None = None,
        wait_for_first_frame_s: float = 5.0,
    ) -> None:
        try:
            import matplotlib  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "MatplotlibLiveView requires matplotlib. "
                "Install with `pip install seabreeze-live[plot]`."
            ) from e
        self.refresh_ms = refresh_ms
        self.title = title
        self.wait_for_first_frame_s = wait_for_first_frame_s
        self._latest: SpectrumFrame | None = None
        self._closed = False

    @property
    def latest_frame(self) -> SpectrumFrame | None:
        return self._latest

    # --- Consumer protocol ---

    def on_frame(self, frame: SpectrumFrame) -> None:
        self._latest = frame

    def close(self) -> None:
        self._closed = True

    # --- GUI ---

    def run(self) -> None:
        """Block the calling thread on the matplotlib event loop until the
        window is closed. MUST be called from the main thread.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        deadline = time.monotonic() + self.wait_for_first_frame_s
        while self._latest is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._latest is None:
            raise RuntimeError(
                "no frame received within "
                f"{self.wait_for_first_frame_s}s; is the streamer running?"
            )

        frame0 = self._latest
        fig, ax = plt.subplots()
        (line,) = ax.plot(frame0.axis, frame0.values)
        ax.set_xlabel(f"wavelength ({frame0.axis_units})")
        ax.set_ylabel(frame0.value_units)
        ax.set_title(self.title or f"{frame0.device_serial}")
        info = ax.text(
            0.02, 0.95, "",
            transform=ax.transAxes, va="top",
            family="monospace", fontsize=9,
        )

        def update(_: Any):
            f = self._latest
            if f is None:
                return ()
            line.set_ydata(f.values)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            info.set_text(
                f"frame {f.frame_number}\n"
                f"int_time {f.integration_time_us / 1000:.1f} ms\n"
                f"max {float(f.values.max()):.0f}"
            )
            return ()

        anim = FuncAnimation(
            fig,
            update,
            interval=self.refresh_ms,
            blit=False,
            cache_frame_data=False,
        )
        try:
            plt.show()
        finally:
            del anim
