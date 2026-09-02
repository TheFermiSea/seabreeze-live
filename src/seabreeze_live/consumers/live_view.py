"""High-performance live Textual Spectrometer TUI powered by textual-plotext."""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

import numpy as np
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Log,
    Select,
    Static,
    Switch,
)
from textual_plotext import PlotextPlot

from seabreeze_live.acquisition import Acquirer, AcquisitionSettings
from seabreeze_live.device import (
    HardwareConnectionError,
    HardwareOperationError,
    SpectrometerDevice,
)
from seabreeze_live.frame import SpectrumFrame
from seabreeze_live.processing import (
    display_values,
    wavelength_mask,
)
from seabreeze_live.recording import SpectrumRecorder
from seabreeze_live.snapshot import save_snapshot


class MetricBadge(Static):
    DEFAULT_CSS = """
    MetricBadge {
        background: $surface-lighten-1;
        border: round $primary;
        padding: 0 1;
        margin: 0 1;
        height: 3;
        width: 22;
        content-align: center middle;
    }
    """


class MatplotlibLiveView:
    """Lightweight consumer view retained for programmatic integrations.

    The Textual app is the interactive control surface; this class keeps the
    established ``Streamer(..., [MatplotlibLiveView()])`` composition API.
    """

    def __init__(self, *, snapshot_dir: str | Path = ".", snapshot_format: str = "csv"):
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError(
                "MatplotlibLiveView requires matplotlib. Install seabreeze-live[plot]."
            ) from error
        self._plt = plt
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_format = snapshot_format
        self.latest_frame: SpectrumFrame | None = None
        self._closed = False
        self._fig = None
        self._last_saved: str | None = None

    def on_frame(self, frame: SpectrumFrame) -> None:
        self.latest_frame = frame

    def close(self) -> None:
        self._closed = True

    def save_snapshot_now(self) -> Path | None:
        if self.latest_frame is None:
            return None
        path = save_snapshot(self.latest_frame, self.snapshot_dir, self.snapshot_format)
        if self._fig is not None:
            self._fig.savefig(path.with_suffix(".png"))
        self._last_saved = path.name
        return path

    def run(self) -> None:
        """Run a minimal main-thread matplotlib view until the window closes."""
        fig, axis = self._plt.subplots()
        self._fig = fig
        (line,) = axis.plot([], [])
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Intensity (counts)")

        def update(_: object):
            frame = self.latest_frame
            if frame is not None:
                line.set_data(frame.axis, frame.values)
                axis.relim()
                axis.autoscale_view()
            return (line,)

        from matplotlib.animation import FuncAnimation

        FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
        self._plt.show()


class LiveSpectrometerApp(App):
    """Full-featured live SeaBreeze Spectrometer TUI."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 50;
        dock: left;
        background: $surface;
        border-right: double $primary;
        padding: 1;
    }

    #main-panel {
        width: 1fr;
        height: 100%;
        background: $background;
    }

    #metrics-bar {
        height: 4;
        dock: top;
        layout: horizontal;
        padding: 0 1;
        background: $surface-darken-1;
    }

    #plot-box {
        height: 1fr;
        border: solid $accent;
        margin: 1;
    }

    .section-head {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        margin-bottom: 1;
        border-bottom: solid $accent;
    }

    .ctrl-row {
        layout: horizontal;
        height: 3;
        align: left middle;
        margin-bottom: 1;
    }

    .ctrl-row Label {
        width: 22;
    }

    .saturated-adc {
        background: $error;
        color: $text;
        text-style: bold blink;
    }

    #event-log {
        height: 8;
        border: solid $surface-lighten-2;
        margin: 1;
    }

    Button {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit", show=True),
        Binding("space", "toggle_pause", "Pause/Play", show=True),
        Binding("d", "capture_dark", "Take Dark", show=True),
        Binding("w", "capture_white", "Take White", show=True),
        Binding("c", "clear_refs", "Clear Refs", show=True),
        Binding("s", "take_snapshot", "Snapshot", show=True),
        Binding("r", "toggle_record", "Record Stream", show=True),
        Binding("a", "toggle_autoscale", "Autoscale", show=True),
        Binding("l", "toggle_lamp", "Lamp", show=True),
    ]

    is_paused: reactive[bool] = reactive(False)
    display_mode: reactive[str] = reactive("Raw Counts")
    autoscale_y: reactive[bool] = reactive(True)
    marker_style: reactive[str] = reactive("braille")
    line_color: reactive[str] = reactive("cyan")
    show_grid: reactive[bool] = reactive(True)
    wavelength_zoom: reactive[str] = reactive("full")

    def __init__(
        self,
        device_id: str | None = None,
        integration_time_us: int = 100_000,
        use_mock: bool = False,
        output_dir: str | Path = "spectra",
    ):
        super().__init__()
        self.initial_device_id = device_id
        self.initial_integration_us = integration_time_us
        self.initial_use_mock = use_mock
        self.output_dir = Path(output_dir)
        self._is_active = True

        self.dev = SpectrometerDevice(device_id=device_id, use_mock=use_mock)
        self.integration_time_us = self.dev.set_integration_time_micros(
            integration_time_us
        )
        self._acquirer = Acquirer(self.dev, AcquisitionSettings())

        # Optical Baselines & Processing
        self.dark_spectrum: np.ndarray | None = None
        self.white_spectrum: np.ndarray | None = None
        self.scans_to_average: int = 1
        self.boxcar_width: int = 0
        self.correct_dark_pixels: bool = False
        self.correct_nonlinearity: bool = False

        # Live State
        self.latest_wl: np.ndarray = self.dev.get_wavelengths()
        self.latest_intensities: np.ndarray = np.zeros_like(self.latest_wl)
        self.latest_frame: SpectrumFrame | None = None
        self.fps: float = 0.0
        self._last_frame_t: float = time.perf_counter()
        self._next_render_at: float = 0.0
        self._frame_number = 0

        # Data Logging uses the same public writers as the headless API.
        self.recording_active: bool = False
        self._recorder: SpectrumRecorder | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with VerticalScroll(id="sidebar"):
            yield Label("Spectrometer Selection", classes="section-head")
            yield Select([], id="select-device", prompt="Select Spectrometer")

            yield Label("Plotext Line & Style Menu", classes="section-head")
            with Horizontal(classes="ctrl-row"):
                yield Label("Line Resolution:")
                yield Select(
                    [
                        ("Ultra-Thin Braille (2x4)", "braille"),
                        ("Full HD Matrix (fhd)", "fhd"),
                        ("High Definition (hd)", "hd"),
                        ("Smooth Solid Line", "line"),
                        ("Scatter Points (dot)", "dot"),
                    ],
                    value="braille",
                    id="select-marker",
                )

            with Horizontal(classes="ctrl-row"):
                yield Label("Spectrum Color:")
                yield Select(
                    [
                        ("Cyan Pulse", "cyan"),
                        ("Emerald Green", "green"),
                        ("Solar Yellow", "yellow"),
                        ("Ultraviolet (Magenta)", "magenta"),
                        ("Infrared (Red)", "red"),
                        ("Bright White", "white"),
                    ],
                    value="cyan",
                    id="select-color",
                )

            with Horizontal(classes="ctrl-row"):
                yield Label("Show Gridlines:")
                yield Switch(value=True, id="sw-grid")

            with Horizontal(classes="ctrl-row"):
                yield Label("Spectral Range:")
                yield Select(
                    [
                        ("Full Range", "full"),
                        ("Visible (380 - 750 nm)", "vis"),
                        ("UV Range (200 - 400 nm)", "uv"),
                        ("NIR Range (700 - 1050 nm)", "nir"),
                    ],
                    value="full",
                    id="select-wl-zoom",
                )

            yield Label("Acquisition Settings", classes="section-head")
            with Horizontal(classes="ctrl-row"):
                yield Label("Integration (µs):")
                yield Input(
                    value=str(self.initial_integration_us),
                    id="inp-integration",
                    max_length=10,
                )

            with Horizontal(classes="ctrl-row"):
                yield Label("Scans to Average:")
                yield Input(value="1", id="inp-average", max_length=4)

            with Horizontal(classes="ctrl-row"):
                yield Label("Boxcar Smoothing:")
                yield Input(value="0", id="inp-boxcar", max_length=2)

            with Horizontal(classes="ctrl-row"):
                yield Label("Trigger Mode:")
                yield Select(
                    [
                        ("Normal (0)", 0),
                        ("Software (1)", 1),
                        ("Ext Sync (2)", 2),
                        ("Ext Hardware (3)", 3),
                    ],
                    value=0,
                    id="select-trigger",
                )

            yield Label("Hardware Corrections", classes="section-head")
            with Horizontal(classes="ctrl-row"):
                yield Label("Electric Dark Pix:")
                yield Switch(id="sw-dark-pix")

            with Horizontal(classes="ctrl-row"):
                yield Label("Non-Linearity:")
                yield Switch(id="sw-nonlin")

            yield Label("Optical References", classes="section-head")
            with Horizontal():
                yield Button("Take Dark (D)", id="btn-dark", variant="warning")
                yield Button("Take White (W)", id="btn-white", variant="success")
            yield Button("Clear References (C)", id="btn-clear-refs", variant="default")

            yield Label("Spectrum Display Mode", classes="section-head")
            yield Select(
                [
                    ("Raw Counts", "Raw Counts"),
                    ("Dark Subtracted", "Dark Subtracted"),
                    ("Transmission (%)", "Transmission (%)"),
                    ("Absorbance (AU)", "Absorbance (AU)"),
                ],
                value="Raw Counts",
                id="select-mode",
            )

            yield Label("Peripherals & GPIO", classes="section-head")
            with Horizontal(classes="ctrl-row"):
                yield Label("Lamp Enable (L):")
                yield Switch(value=True, id="sw-lamp")

            with Horizontal(classes="ctrl-row"):
                yield Label("Shutter Open:")
                yield Switch(value=True, id="sw-shutter")

            yield Label("Export & Recording", classes="section-head")
            yield Select(
                [("CSV", "CSV"), ("HDF5", "HDF5")], value="CSV", id="select-rec-fmt"
            )
            yield Button("Start Recording (R)", id="btn-record", variant="primary")
            yield Button("Export Snapshot (S)", id="btn-snapshot", variant="default")

        with Vertical(id="main-panel"):
            with Horizontal(id="metrics-bar"):
                yield MetricBadge(id="badge-fps")
                yield MetricBadge(id="badge-peak")
                yield MetricBadge(id="badge-sat")
                yield MetricBadge(id="badge-temp")

            with Container(id="plot-box"):
                yield PlotextPlot(id="spectrum-plot")

            yield Log(id="event-log", highlight=True)

        yield Footer()

    def on_mount(self):
        devs = self.dev.list_available_devices()
        sel = self.query_one("#select-device", Select)
        sel.set_options([(d["label"], d["id"]) for d in devs])

        target_id = self.initial_device_id or (
            devs[0]["id"] if devs else "MOCK-SIMULATOR"
        )
        sel.value = target_id

        self.log_msg(
            "[bold green]TUI Ready.[/] Live streaming with textual-plotext engine."
        )
        self.start_acquisition_loop()

    def on_unmount(self):
        self._is_active = False
        self._stop_recording()
        self._acquirer.close()

    def log_msg(self, text: str):
        if not self._is_active:
            return
        try:
            self.query_one("#event-log", Log).write_line(
                f"[{time.strftime('%H:%M:%S')}] {text}"
            )
        except NoMatches:
            # Textual may execute a queued callback after it tears down the screen.
            return

    @work(exclusive=True, thread=True)
    def start_acquisition_loop(self):
        while self._is_active:
            if not self.is_paused and self.dev.device is not None:
                try:
                    now = time.perf_counter()
                    dt = now - self._last_frame_t
                    if dt > 0:
                        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                    self._last_frame_t = now

                    frame = self._acquirer.capture()
                    raw = frame.values
                    self.latest_wl = frame.axis
                    self.latest_intensities = raw
                    self.latest_frame = frame

                    if self._recorder is not None:
                        self._recorder.on_frame(frame)
                    self._frame_number = frame.frame_number + 1

                    # Terminal rendering is much slower than acquisition. Coalesce
                    # frames so plotting cannot flood Textual's message queue.
                    if now >= self._next_render_at:
                        self._next_render_at = now + 1 / 12
                        self.app.call_from_thread(
                            self._render_frame, raw, self.dev.read_temperatures()
                        )
                except Exception as ex:  # noqa: BLE001 - report asynchronous acquisition failures
                    if self._is_active:
                        self.app.call_from_thread(
                            self.log_msg, f"[red]Acquisition error: {ex}[/]"
                        )
            time.sleep(0.001)

    def _render_frame(self, raw: np.ndarray, temps: dict[str, float]):
        # A frame may already be queued when Textual starts unmounting.
        if not self._is_active:
            return
        try:
            fps_badge = self.query_one("#badge-fps", MetricBadge)
        except NoMatches:
            return
        wl = self.latest_wl
        y_data, y_label = self._calculate_spectrum(raw)

        mask = wavelength_mask(wl, self.wavelength_zoom)

        wl_plot = wl[mask].tolist()
        y_plot = y_data[mask].tolist()

        if len(wl_plot) == 0:
            wl_plot = wl.tolist()
            y_plot = y_data.tolist()

        # Update Badges
        peak_i = int(np.argmax(y_data))
        fps_badge.update(f"⚡ {self.fps:.1f} FPS")
        self.query_one("#badge-peak", MetricBadge).update(
            f"🎯 {wl[peak_i]:.1f}nm ({y_data[peak_i]:.0f})"
        )

        sat_badge = self.query_one("#badge-sat", MetricBadge)
        if np.any(raw >= 65000.0):
            sat_badge.update("⚠️ SATURATED")
            sat_badge.add_class("saturated-adc")
        else:
            sat_badge.update("✅ ADC Normal")
            sat_badge.remove_class("saturated-adc")

        t_str = " | ".join([f"{v:.1f}°C" for v in temps.values()]) or "N/A"
        self.query_one("#badge-temp", MetricBadge).update(f"🌡️ {t_str}")

        # Render via textual-plotext
        plot = self.query_one("#spectrum-plot", PlotextPlot)
        plt = plot.plt
        plt.clf()

        # Choose marker resolution / style
        if self.marker_style == "line":
            plt.plot(wl_plot, y_plot, color=self.line_color, label=self.display_mode)
        elif self.marker_style == "dot":
            plt.scatter(
                wl_plot,
                y_plot,
                color=self.line_color,
                label=self.display_mode,
                marker="dot",
            )
        else:
            plt.plot(
                wl_plot,
                y_plot,
                color=self.line_color,
                label=self.display_mode,
                marker=self.marker_style,
            )

        # Baseline Overlays
        if self.display_mode == "Raw Counts":
            if self.dark_spectrum is not None:
                plt.plot(
                    wl_plot,
                    self.dark_spectrum[mask].tolist(),
                    color="gray",
                    label="Dark Baseline",
                    marker=self.marker_style if self.marker_style != "line" else None,
                )
            if self.white_spectrum is not None:
                plt.plot(
                    wl_plot,
                    self.white_spectrum[mask].tolist(),
                    color="yellow",
                    label="White Ref",
                    marker=self.marker_style if self.marker_style != "line" else None,
                )

        dev_title = (
            f"{self.dev.meta.model} ({self.dev.meta.serial_number})"
            if self.dev.meta
            else "Disconnected"
        )
        plt.title(f"Live Spectra | {dev_title} | Display: {self.display_mode}")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel(y_label)

        if self.show_grid:
            plt.grid(True, True)

        if not self.autoscale_y:
            if self.display_mode == "Raw Counts":
                plt.ylim(0, 65535)
            elif self.display_mode == "Transmission (%)":
                plt.ylim(-5, 110)
            elif self.display_mode == "Absorbance (AU)":
                plt.ylim(-0.2, 3.5)

        plt.xlim(float(wl_plot[0]), float(wl_plot[-1]))
        plot.refresh()

    def _calculate_spectrum(self, raw: np.ndarray) -> tuple[np.ndarray, str]:
        return display_values(
            raw, self.display_mode, self.dark_spectrum, self.white_spectrum
        )

    def _stop_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._recorder = None
        self.recording_active = False

    # --- UI Event Handlers ---

    @on(Select.Changed, "#select-marker")
    def on_marker_change(self, event: Select.Changed):
        if event.value != Select.BLANK:
            self.marker_style = str(event.value)
            self.log_msg(f"Line Resolution: [cyan]{event.value}[/]")

    @on(Select.Changed, "#select-color")
    def on_color_change(self, event: Select.Changed):
        if event.value != Select.BLANK:
            self.line_color = str(event.value)
            self.log_msg(f"Trace Color: [cyan]{event.value}[/]")

    @on(Switch.Changed, "#sw-grid")
    def on_grid_switch(self, event: Switch.Changed):
        self.show_grid = event.value
        self.log_msg(f"Gridlines: {'ON' if event.value else 'OFF'}")

    @on(Select.Changed, "#select-wl-zoom")
    def on_wl_zoom_change(self, event: Select.Changed):
        if event.value != Select.BLANK:
            self.wavelength_zoom = str(event.value)
            self.log_msg(f"Spectral Range: [cyan]{event.value}[/]")

    @on(Select.Changed, "#select-device")
    def on_device_select(self, event: Select.Changed):
        if event.value != Select.BLANK:
            target = str(event.value)
            current = (
                "MOCK-SIMULATOR"
                if self.dev.meta and self.dev.meta.is_mock
                else self.dev.serial_number
            )
            if target == current:
                return
            try:
                replacement = SpectrometerDevice(device_id=target)
                self.integration_time_us = replacement.set_integration_time_micros(
                    self.integration_time_us
                )
            except (HardwareConnectionError, HardwareOperationError) as error:
                self.log_msg(f"[red]Could not connect: {error}[/]")
                return
            self._acquirer.replace_device(replacement)
            self.dev = replacement
            self.latest_wl = replacement.get_wavelengths()
            self.latest_intensities = np.zeros_like(self.latest_wl)
            self.latest_frame = None
            self._frame_number = 0
            self.log_msg(f"Connected to spectrometer: [bold]{target}[/]")

    @on(Input.Submitted, "#inp-integration")
    def on_integration_change(self, event: Input.Submitted):
        if event.value.isdigit():
            val = int(event.value)
            try:
                self.integration_time_us = self._acquirer.set_integration_time(val)
            except (HardwareConnectionError, HardwareOperationError) as error:
                self.log_msg(f"[red]Integration-time update failed: {error}[/]")
                return
            self.log_msg(f"Integration time: [bold]{self.integration_time_us:,}[/] µs")

    @on(Input.Changed, "#inp-average")
    def on_average_change(self, event: Input.Changed):
        if event.value.isdigit() and int(event.value) > 0:
            self.scans_to_average = int(event.value)
            self._acquirer.configure(scans_to_average=self.scans_to_average)

    @on(Input.Changed, "#inp-boxcar")
    def on_boxcar_change(self, event: Input.Changed):
        if event.value.isdigit():
            self.boxcar_width = int(event.value)
            self._acquirer.configure(boxcar_width=self.boxcar_width)

    @on(Select.Changed, "#select-trigger")
    def on_trigger_change(self, event: Select.Changed):
        if event.value != Select.BLANK:
            try:
                self._acquirer.set_trigger_mode(int(str(event.value)))
            except (NotImplementedError, HardwareOperationError) as error:
                self.log_msg(f"[red]Trigger update failed: {error}[/]")
                return
            self.log_msg(f"Trigger mode set to {event.value}")

    @on(Switch.Changed, "#sw-dark-pix")
    def on_dark_pix_switch(self, event: Switch.Changed):
        self.correct_dark_pixels = event.value
        self._acquirer.configure(correct_dark=event.value)

    @on(Switch.Changed, "#sw-nonlin")
    def on_nonlin_switch(self, event: Switch.Changed):
        self.correct_nonlinearity = event.value
        self._acquirer.configure(correct_nonlinearity=event.value)

    @on(Switch.Changed, "#sw-lamp")
    def on_lamp_switch(self, event: Switch.Changed):
        try:
            self.dev.set_lamp_enable(event.value)
        except (NotImplementedError, HardwareOperationError) as error:
            self.log_msg(f"[red]Lamp update failed: {error}[/]")
            return
        self.log_msg(f"Light Source: {'ON' if event.value else 'OFF'}")

    @on(Switch.Changed, "#sw-shutter")
    def on_shutter_switch(self, event: Switch.Changed):
        try:
            self.dev.set_shutter_open(event.value)
        except (NotImplementedError, HardwareOperationError) as error:
            self.log_msg(f"[red]Shutter update failed: {error}[/]")
            return
        self.log_msg(f"Shutter: {'OPEN' if event.value else 'CLOSED'}")

    @on(Select.Changed, "#select-mode")
    def on_mode_change(self, event: Select.Changed):
        if event.value != Select.BLANK:
            self.display_mode = str(event.value)
            self.log_msg(f"Display Mode: [bold cyan]{self.display_mode}[/]")

    @on(Button.Pressed, "#btn-dark")
    def action_capture_dark(self):
        self.dark_spectrum = self.latest_intensities.copy()
        self.log_msg("[yellow]Captured dark baseline spectrum.[/]")

    @on(Button.Pressed, "#btn-white")
    def action_capture_white(self):
        self.white_spectrum = self.latest_intensities.copy()
        self.log_msg("[green]Captured 100% white reference spectrum.[/]")

    @on(Button.Pressed, "#btn-clear-refs")
    def action_clear_refs(self):
        self.dark_spectrum = None
        self.white_spectrum = None
        self.log_msg("Optical references cleared.")

    @on(Button.Pressed, "#btn-snapshot")
    def action_take_snapshot(self):
        if self.latest_frame is None:
            self.log_msg("[yellow]No spectrum is available to save yet.[/]")
            return
        path = save_snapshot(self.latest_frame, self.output_dir)
        self.log_msg(f"[bold green]Saved snapshot to {path.resolve()}[/]")

    @on(Button.Pressed, "#btn-record")
    def action_toggle_record(self):
        btn = self.query_one("#btn-record", Button)
        if not self.recording_active:
            fmt = str(self.query_one("#select-rec-fmt", Select).value or "CSV")
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"spectrum_log_{ts}.{fmt.lower()}"
            self._recorder = SpectrumRecorder(path, fmt)
            self.recording_active = True
            btn.label = "Stop Recording"
            btn.variant = "error"
            self.log_msg(f"[bold red]Recording started: {path.name}[/]")
        else:
            self._stop_recording()
            btn.label = "Start Recording (R)"
            btn.variant = "primary"
            self.log_msg("[bold green]Recording stopped.[/]")

    def action_toggle_pause(self):
        self.is_paused = not self.is_paused
        self.log_msg(f"Stream: {'PAUSED' if self.is_paused else 'RESUMED'}")

    def action_toggle_autoscale(self):
        self.autoscale_y = not self.autoscale_y
        self.log_msg(f"Y-Autoscale: {'ENABLED' if self.autoscale_y else 'FIXED'}")

    def action_toggle_lamp(self):
        sw = self.query_one("#sw-lamp", Switch)
        sw.value = not sw.value


def run_tui(
    device: str | None = None,
    integration_time_us: int = 100_000,
    use_mock: bool = False,
):
    app = LiveSpectrometerApp(
        device_id=device,
        integration_time_us=integration_time_us,
        use_mock=use_mock,
    )
    app.run()
