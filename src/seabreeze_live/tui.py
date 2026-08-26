"""Full-featured Textual TUI for SeaBreeze Spectrometer live control & plotting."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    RadioSet,
    RadioButton,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual_plot import Plot

from seabreeze_live.engine import SpectrometerEngine, SpectrumFrame


class StatBadge(Static):
    """Reusable visual badge for live metrics."""

    DEFAULT_CSS = """
    StatBadge {
        background: $surface-lighten-1;
        border: round $primary;
        padding: 0 1;
        margin: 0 1;
        height: 3;
        width: 18;
        content-align: center middle;
    }
    """


class LiveSpectrometerApp(App):
    """Ocean Optics / SeaBreeze Spectrum Analyzer TUI."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 44;
        dock: left;
        background: $surface;
        border-right: double $primary;
        padding: 1;
    }

    #main-area {
        width: 1fr;
        height: 100%;
        background: $background;
    }

    #plot-container {
        height: 1fr;
        border: solid $accent;
        margin: 1;
    }

    #stats-bar {
        height: 4;
        dock: top;
        layout: horizontal;
        padding: 0 1;
        background: $surface-darken-1;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        margin-bottom: 1;
        border-bottom: solid $accent;
    }

    .row {
        layout: horizontal;
        height: 3;
        align: left middle;
        margin-bottom: 1;
    }

    .row Label {
        width: 18;
    }

    .saturated {
        background: $error;
        color: $text;
        text-style: bold blink;
    }

    Button {
        width: 100%;
        margin-top: 1;
    }

    #log-view {
        height: 10;
        border: solid $surface-lighten-2;
        margin: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("space", "toggle_pause", "Pause/Resume", show=True),
        Binding("d", "capture_dark", "Take Dark", show=True),
        Binding("w", "capture_white", "Take White", show=True),
        Binding("s", "snapshot", "Save Snapshot", show=True),
        Binding("r", "toggle_record", "Record Stream", show=True),
        Binding("a", "autoscale", "Autoscale", show=True),
    ]

    is_paused: reactive[bool] = reactive(False)
    is_recording: reactive[bool] = reactive(False)
    display_mode: reactive[str] = reactive("Raw Counts")
    fps_text: reactive[str] = reactive("0.0 FPS")
    peak_text: reactive[str] = reactive("--- nm")
    sat_text: reactive[str] = reactive("ADC OK")

    def __init__(self):
        super().__init__()
        self.engine = SpectrometerEngine()
        self.latest_frame: Optional[SpectrumFrame] = None
        self.autoscale_y: bool = True

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with VerticalScroll(id="sidebar"):
            yield Label("Spectrometer Hardware", classes="section-title")
            yield Select([], id="device-select", prompt="Select Spectrometer")

            yield Label("Acquisition Settings", classes="section-title")
            with Horizontal(classes="row"):
                yield Label("Integ Time (µs):")
                yield Input(value="100000", id="integration-input", max_length=10)

            with Horizontal(classes="row"):
                yield Label("Scans Avg:")
                yield Input(value="1", id="avg-input", max_length=4)

            with Horizontal(classes="row"):
                yield Label("Boxcar Width:")
                yield Input(value="0", id="boxcar-input", max_length=2)

            yield Label("Corrections & Baselines", classes="section-title")
            with Horizontal(classes="row"):
                yield Label("Dark Pix Correct:")
                yield Switch(id="switch-dark-pix")

            with Horizontal(classes="row"):
                yield Label("Non-Linearity:")
                yield Switch(id="switch-nonlin")

            with Horizontal():
                yield Button("Take Dark (D)", id="btn-dark", variant="warning")
                yield Button("Take White (W)", id="btn-white", variant="success")
            yield Button("Clear References", id="btn-clear-refs", variant="default")

            yield Label("Display Calculation", classes="section-title")
            yield Select(
                [
                    ("Raw Counts", "Raw Counts"),
                    ("Dark Subtracted", "Dark Subtracted"),
                    ("Transmission (%)", "Transmission (%)"),
                    ("Absorbance (AU)", "Absorbance (AU)"),
                ],
                value="Raw Counts",
                id="calc-select",
            )

            yield Label("Hardware Peripherals", classes="section-title")
            with Horizontal(classes="row"):
                yield Label("Lamp/Light:")
                yield Switch(id="switch-lamp")
            with Horizontal(classes="row"):
                yield Label("Shutter Open:")
                yield Switch(id="switch-shutter")

            yield Label("Data Logging & Snapshots", classes="section-title")
            yield Select(
                [("CSV", "CSV"), ("HDF5", "HDF5")], value="CSV", id="rec-format"
            )
            yield Button("Start Recording (R)", id="btn-record", variant="primary")
            yield Button("Export Snapshot (S)", id="btn-snapshot", variant="secondary")

        with Vertical(id="main-area"):
            with Horizontal(id="stats-bar"):
                yield StatBadge(id="badge-fps")
                yield StatBadge(id="badge-peak")
                yield StatBadge(id="badge-sat")
                yield StatBadge(id="badge-temp")

            with Container(id="plot-container"):
                yield Plot(id="spectral-plot")

            yield Log(id="log-view", highlight=True)

        yield Footer()

    def on_mount(self):
        """Populate devices and start hardware polling worker."""
        dev_list = self.engine.list_devices()
        select_widget = self.query_one("#device-select", Select)
        select_widget.set_options([(d["label"], d["id"]) for d in dev_list])

        if dev_list:
            select_widget.value = dev_list[0]["id"]
            self._connect_device(dev_list[0]["id"])

        self.log_message(
            "[bold green]TUI initialized.[/] Press [cyan]Space[/] to pause, [cyan]D[/] for Dark, [cyan]W[/] for Reference."
        )
        self.acquire_loop()

    def log_message(self, msg: str):
        log_view = self.query_one("#log-view", Log)
        log_view.write_line(f"[{time.strftime('%H:%M:%S')}] {msg}")

    def _connect_device(self, dev_id: str):
        info = self.engine.connect(dev_id)
        self.log_message(
            f"Connected to [bold]{info.model}[/] (S/N: {info.serial_number})"
        )
        self.log_message(
            f"Limits: {info.min_integration_us}µs - {info.max_integration_us}µs | Pixels: {info.pixels}"
        )

        # Update switch capabilities
        self.query_one("#switch-lamp", Switch).disabled = not info.has_lamp
        self.query_one("#switch-shutter", Switch).disabled = not info.has_shutter

    @work(exclusive=True, thread=True)
    def acquire_loop(self):
        """Background acquisition thread feeding frames to Textual."""
        while self.is_running:
            if not self.is_paused and self.engine.info:
                try:
                    frame = self.engine.acquire_frame()
                    self.latest_frame = frame
                    self.app.call_from_thread(self._update_ui, frame)
                except Exception as ex:
                    self.app.call_from_thread(
                        self.log_message, f"[red]Acquisition error: {ex}[/]"
                    )
            time.sleep(0.02)

    def _update_ui(self, frame: SpectrumFrame):
        """Render frame to `textual_plot.Plot` and update metrics."""
        wl, y_data, y_label = self.engine.calculate_display_data(
            frame, self.display_mode
        )

        # Peak detection
        peak_idx = int(np.argmax(y_data))
        peak_wl = wl[peak_idx]
        peak_val = y_data[peak_idx]

        # Update Badges
        self.query_one("#badge-fps", StatBadge).update(f"⚡ {frame.fps:.1f} FPS")
        self.query_one("#badge-peak", StatBadge).update(
            f"🎯 {peak_wl:.1f}nm\n({peak_val:.0f})"
        )

        sat_badge = self.query_one("#badge-sat", StatBadge)
        if frame.is_saturated:
            sat_badge.update("⚠️ SATURATED")
            sat_badge.add_class("saturated")
        else:
            sat_badge.update("✅ ADC Normal")
            sat_badge.remove_class("saturated")

        temp_str = (
            " | ".join([f"{v:.1f}°C" for v in frame.temperatures.values()]) or "N/A"
        )
        self.query_one("#badge-temp", StatBadge).update(f"🌡️ {temp_str}")

        # Update textual-plot widget
        plot_widget = self.query_one("#spectral-plot", Plot)
        plot_widget.plt.clear_data()
        plot_widget.plt.plot(wl, y_data, color="cyan", label=self.display_mode)

        # Optional baseline overlay
        if self.engine.dark_spectrum is not None and self.display_mode == "Raw Counts":
            plot_widget.plt.plot(
                wl,
                self.engine.dark_spectrum,
                color="bright_black",
                label="Dark Baseline",
            )
        if (
            self.engine.white_reference is not None
            and self.display_mode == "Raw Counts"
        ):
            plot_widget.plt.plot(
                wl, self.engine.white_reference, color="yellow", label="White Reference"
            )

        plot_widget.plt.title(
            f"Live Spectrum - {self.engine.info.model if self.engine.info else ''}"
        )
        plot_widget.plt.xlabel("Wavelength (nm)")
        plot_widget.plt.ylabel(y_label)
        plot_widget.plt.grid(True, True)

        if not self.autoscale_y and self.display_mode == "Raw Counts":
            plot_widget.plt.ylim(0, 65535)
        elif not self.autoscale_y and self.display_mode == "Transmission (%)":
            plot_widget.plt.ylim(-5, 120)
        elif not self.autoscale_y and self.display_mode == "Absorbance (AU)":
            plot_widget.plt.ylim(-0.2, 3.0)

        plot_widget.refresh()

    # --- UI Event Handlers ---

    @on(Select.Changed, "#device-select")
    def on_device_change(self, event: Select.Changed):
        if event.value != Select.BLANK:
            self._connect_device(str(event.value))

    @on(Input.Changed, "#integration-input")
    def on_integration_change(self, event: Input.Changed):
        if event.value.isdigit():
            val = int(event.value)
            self.engine.set_integration_time(val)
            self.log_message(f"Integration time set to [bold]{val:,}[/] µs")

    @on(Input.Changed, "#avg-input")
    def on_avg_change(self, event: Input.Changed):
        if event.value.isdigit() and int(event.value) > 0:
            self.engine.scans_to_average = int(event.value)

    @on(Input.Changed, "#boxcar-input")
    def on_boxcar_change(self, event: Input.Changed):
        if event.value.isdigit():
            self.engine.boxcar_width = int(event.value)

    @on(Switch.Changed, "#switch-dark-pix")
    def on_dark_pixel_switch(self, event: Switch.Changed):
        self.engine.correct_dark_pixels = event.value

    @on(Switch.Changed, "#switch-nonlin")
    def on_nonlin_switch(self, event: Switch.Changed):
        self.engine.correct_nonlinearity = event.value

    @on(Switch.Changed, "#switch-lamp")
    def on_lamp_switch(self, event: Switch.Changed):
        self.engine.set_lamp(event.value)
        self.log_message(f"Lamp state: {'ON' if event.value else 'OFF'}")

    @on(Switch.Changed, "#switch-shutter")
    def on_shutter_switch(self, event: Switch.Changed):
        self.engine.set_shutter(event.value)
        self.log_message(f"Shutter state: {'OPEN' if event.value else 'CLOSED'}")

    @on(Select.Changed, "#calc-select")
    def on_calc_mode_change(self, event: Select.Changed):
        if event.value != Select.BLANK:
            self.display_mode = str(event.value)
            self.log_message(
                f"Display mode changed to [bold cyan]{self.display_mode}[/]"
            )

    @on(Button.Pressed, "#btn-dark")
    def action_capture_dark(self):
        if self.latest_frame:
            self.engine.dark_spectrum = self.latest_frame.intensities.copy()
            self.log_message("[yellow]Dark baseline spectrum captured.[/]")

    @on(Button.Pressed, "#btn-white")
    def action_capture_white(self):
        if self.latest_frame:
            self.engine.white_reference = self.latest_frame.intensities.copy()
            self.log_message("[green]White reference spectrum captured.[/]")

    @on(Button.Pressed, "#btn-clear-refs")
    def action_clear_refs(self):
        self.engine.dark_spectrum = None
        self.engine.white_reference = None
        self.log_message("Cleared dark & white reference spectra.")

    @on(Button.Pressed, "#btn-record")
    def action_toggle_record(self):
        if not self.engine.recording_active:
            fmt = str(self.query_one("#rec-format", Select).value or "CSV")
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = Path(f"spectrum_stream_{ts}.{fmt.lower()}")
            self.engine.start_recording(filename, fmt)
            self.query_one("#btn-record", Button).label = "Stop Recording"
            self.query_one("#btn-record", Button).variant = "error"
            self.log_message(f"[bold red]Recording streaming data to {filename}...[/]")
        else:
            self.engine.stop_recording()
            self.query_one("#btn-record", Button).label = "Start Recording (R)"
            self.query_one("#btn-record", Button).variant = "primary"
            self.log_message("[bold green]Recording stopped and file closed.[/]")

    @on(Button.Pressed, "#btn-snapshot")
    def action_snapshot(self):
        if self.latest_frame:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = Path(f"snapshot_{ts}.csv")
            self.engine.export_snapshot(path, self.latest_frame)
            self.log_message(f"[bold green]Snapshot saved to {path.resolve()}[/]")

    def action_toggle_pause(self):
        self.is_paused = not self.is_paused
        self.log_message(f"Acquisition {'PAUSED' if self.is_paused else 'RESUMED'}")

    def action_autoscale(self):
        self.autoscale_y = not self.autoscale_y
        self.log_message(
            f"Autoscale Y: {'ENABLED' if self.autoscale_y else 'FIXED RANGE'}"
        )


def run_tui():
    app = LiveSpectrometerApp()
    app.run()


if __name__ == "__main__":
    run_tui()
