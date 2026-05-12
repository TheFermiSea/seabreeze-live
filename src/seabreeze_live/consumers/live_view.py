from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from seabreeze_live.frame import SpectrumFrame
from seabreeze_live.snapshot import save_snapshot


class MatplotlibLiveView:
    """Live spectrum plot as a Consumer.

    The Streamer calls `on_frame` from its background thread and we store
    the latest frame in a single attribute. The GUI runs on the main
    thread via `run()`, polled by `matplotlib.animation.FuncAnimation`.

    A "Save snapshot" button (and the `s` key) writes the most recent
    frame to a timestamped file under `snapshot_dir`.

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
        snapshot_dir: str | Path | None = None,
        snapshot_format: str = "csv",
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
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else Path.cwd()
        self.snapshot_format = snapshot_format
        self._latest: SpectrumFrame | None = None
        self._closed = False
        self._last_saved: str | None = None
        self._fig: Any = None  # set by run(); enables PNG dump alongside data

    @property
    def latest_frame(self) -> SpectrumFrame | None:
        return self._latest

    # --- Consumer protocol ---

    def on_frame(self, frame: SpectrumFrame) -> None:
        self._latest = frame

    def close(self) -> None:
        self._closed = True

    # --- Snapshot ---

    def save_snapshot_now(self) -> Path | None:
        """Save the most recent frame to `snapshot_dir`. Returns the data
        path (None if no frame has arrived yet). When called from the GUI
        (i.e. `_fig` is set), a matching `.png` of the current plot is
        written alongside the data file with the same timestamped stem.
        """
        frame = self._latest
        if frame is None:
            return None
        data_path = save_snapshot(frame, self.snapshot_dir, self.snapshot_format)
        png_path: Path | None = None
        if self._fig is not None:
            png_path = data_path.with_suffix(".png")
            self._fig.savefig(png_path, dpi=150, bbox_inches="tight")
        self._last_saved = data_path.name
        if png_path is not None:
            print(f"saved {data_path} + {png_path.name}", file=sys.stderr, flush=True)
        else:
            print(f"saved {data_path}", file=sys.stderr, flush=True)
        return data_path

    # --- GUI ---

    def run(self) -> None:
        """Block the calling thread on the matplotlib event loop until the
        window is closed. MUST be called from the main thread.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Button

        # Matplotlib's default keymap binds 's' to "save figure" (which
        # opens a PNG save dialog and would shadow our snapshot handler).
        # Reclaim it.
        save_keymap = list(mpl.rcParams.get("keymap.save", []))
        if "s" in save_keymap:
            mpl.rcParams["keymap.save"] = [k for k in save_keymap if k != "s"]

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
        self._fig = fig
        fig.subplots_adjust(bottom=0.18)
        (line,) = ax.plot(frame0.axis, frame0.values)
        ax.set_xlabel(f"wavelength ({frame0.axis_units})")
        ax.set_ylabel(frame0.value_units)
        ax.set_title(self.title or f"{frame0.device_serial}")
        info = ax.text(
            0.02, 0.95, "",
            transform=ax.transAxes, va="top",
            family="monospace", fontsize=9,
        )

        btn_ax = fig.add_axes((0.74, 0.03, 0.22, 0.07))
        save_button = Button(
            btn_ax, f"Save {self.snapshot_format} + PNG"
        )

        hint_ax = fig.add_axes((0.04, 0.03, 0.66, 0.07))
        hint_ax.axis("off")
        hint_text = hint_ax.text(
            0.0, 0.5,
            f"snapshots → {self.snapshot_dir}  (button or 's' key)",
            transform=hint_ax.transAxes, va="center",
            family="monospace", fontsize=9,
        )

        def _on_save(_event: Any) -> None:
            self.save_snapshot_now()

        def _on_key(event: Any) -> None:
            if event.key == "s":
                self.save_snapshot_now()

        save_button.on_clicked(_on_save)
        fig.canvas.mpl_connect("key_press_event", _on_key)

        def update(_: Any):
            f = self._latest
            if f is None:
                return ()
            line.set_ydata(f.values)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            txt = (
                f"frame {f.frame_number}\n"
                f"int_time {f.integration_time_us / 1000:.1f} ms\n"
                f"max {float(f.values.max()):.0f}"
            )
            if self._last_saved is not None:
                txt += f"\nlast saved: {self._last_saved}"
            info.set_text(txt)
            return ()

        anim = FuncAnimation(
            fig,
            update,
            interval=self.refresh_ms,
            blit=False,
            cache_frame_data=False,
        )
        # Keep handler references alive for the duration of plt.show().
        _ = (save_button, hint_text)
        try:
            plt.show()
        finally:
            del anim
            self._fig = None
