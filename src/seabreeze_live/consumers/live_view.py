"""High-performance live Textual Spectrometer TUI with textual-plot integration."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
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

try:
    from textual_plot.plot_widget import PlotWidget
except ImportError:
    from textual_plot import PlotWidget

from seabreeze_live.device import SpectrometerDevice


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


class LiveSpectrometerApp(App):
    """Full-featured live SeaBreeze Spectrometer TUI."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 46;
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
        width: 20;
    }

    .saturated-adc {
        background: $error;
        color: $text;
        text-style: bold blink;
    }

    #event-log {
        height: 9;
        border: solid $surface-lighten-2;
        margin: 1;
    }

    Button {
        width: 100%;
        margin-top: 1;
    }
    """

    BINDINGS = [
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

    def __init__(
        self,
        device_id: Optional[str] = None,
        integration_time_us: int = 100_000,
        use_mock: bool = False,
    ):
        super().__init__()
        self.initial_device_id = device_id
        self.initial_integration_us = integration_time_us
        self.initial_use_mock = use_mock

        self.dev = SpectrometerDevice(device_id=device_id, use_mock=use_mock)
        self.dev.set_integration_time_micros(integration_time_us)

        # Baselines & Processing
        self.dark_spectrum: Optional[np.ndarray] = None
        self.white_spectrum: Optional[np.ndarray] = None
        self.scans_to_average: int = 1
        self.boxcar_width: int = 0
        self.correct_dark_pixels: bool = False
        self.correct_nonlinearity: bool = False

        # Live State
        self.latest_wl: np.ndarray = self.dev.get_wavelengths()
        self.latest_intensities: np.ndarray = np.zeros_like(self.latest_wl)
        self.fps: float = 0.0
        self._last_frame_t: float = time.perf_counter()

        # Recording
        self.recording_active: bool = False
        self.recording_format: str = "CSV"
        self._csv_file = None
        self._csv_writer = None
        self._h5_file = None
        self._h5_dataset = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with VerticalScroll(id="sidebar"):
            yield Label("Spectrometer Selection", classes="section-head")
            yield Select([], id="select-device", prompt="Select Spectrometer")

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
                yield PlotWidget(id="spectrum-plot")

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

        self.log_msg("[bold green]TUI Ready.[/] Live streaming active.")
        self.start_acquisition_loop()

    def log_msg(self, text: str):
        self.query_one("#event-log", Log).write_line(
            f"[{time.strftime('%H:%M:%S')}] {text}"
        )

    @work(exclusive=True, thread=True)
    def start_acquisition_loop(self):
        """Asynchronous spectrometer polling worker."""
        while self.is_running:
            if not self.is_paused and self.dev.device is not None:
                try:
                    now = time.perf_counter()
                    dt = now - self._last_frame_t
                    if dt > 0:
                        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                    self._last_frame_t = now

                    # Scan Averaging
                    accum = np.zeros(
                        self.dev.meta.pixels if self.dev.meta else 2048,
                        dtype=np.float64,
                    )
                    for _ in range(max(1, self.scans_to_average)):
                        accum += self.dev.get_intensities(
                            correct_dark_pixels=self.correct_dark_pixels,
                            correct_nonlinearity=self.correct_nonlinearity,
                        )
                    raw = accum / max(1, self.scans_to_average)

                    # Boxcar smoothing
                    if self.boxcar_width > 0:
                        k = np.ones(self.boxcar_width * 2 + 1) / (
                            self.boxcar_width * 2 + 1
                        )
                        raw = np.convolve(raw, k, mode="same")

                    self.latest_wl = self.dev.get_wavelengths()
                    self.latest_intensities = raw

                    # Stream writer
                    if self.recording_active:
                        self._write_recording_frame(now, raw)

                    temps = self.dev.read_temperatures()
                    self.app.call_from_thread(self._render_frame, raw, temps)
                except Exception as ex:
                    self.app.call_from_thread(
                        self.log_msg, f"[red]Acquisition error: {ex}[/]"
                    )
            time.sleep(0.015)

    def _render_frame(self, raw: np.ndarray, temps: Dict[str, float]):
        """Render processed spectrum and update UI metrics."""
        wl = self.latest_wl
        y_data, y_label = self._calculate_spectrum(raw)

        # Status Badges
        peak_i = int(np.argmax(y_data))
        self.query_one("#badge-fps", MetricBadge).update(f"⚡ {self.fps:.1f} FPS")
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

        # Render to PlotWidget
        plot = self.query_one("#spectrum-plot", PlotWidget)
        if hasattr(plot, "clear"):
            plot.clear()

        if hasattr(plot, "plot"):
            plot.plot(wl, y_data, color="cyan", label=self.display_mode)
            if self.display_mode == "Raw Counts":
                if self.dark_spectrum is not None:
                    plot.plot(
                        wl, self.dark_spectrum, color="gray", label="Dark Baseline"
                    )
                if self.white_spectrum is not None:
                    plot.plot(
                        wl, self.white_spectrum, color="yellow", label="White Reference"
                    )

        dev_title = (
            f"{self.dev.meta.model} ({self.dev.meta.serial_number})"
            if self.dev.meta
            else "Disconnected"
        )
        if hasattr(plot, "title"):
            plot.title = f"Live Spectra | {dev_title}"
        if hasattr(plot, "xlabel"):
            plot.xlabel = "Wavelength (nm)"
        if hasattr(plot, "ylabel"):
            plot.ylabel = y_label

        if not self.autoscale_y:
            if self.display_mode == "Raw Counts":
                if hasattr(plot, "set_ylim"):
                    plot.set_ylim(0, 65535)
                elif hasattr(plot, "ylim"):
                    plot.ylim = (0, 65535)
            elif self.display_mode == "Transmission (%)":
                if hasattr(plot, "set_ylim"):
                    plot.set_ylim(-5, 110)
                elif hasattr(plot, "ylim"):
                    plot.ylim = (-5, 110)
            elif self.display_mode == "Absorbance (AU)":
                if hasattr(plot, "set_ylim"):
                    plot.set_ylim(-0.2, 3.5)
                elif hasattr(plot, "ylim"):
                    plot.ylim = (-0.2, 3.5)
        else:
            if hasattr(plot, "set_ylim"):
                plot.set_ylim(None, None)
            elif hasattr(plot, "ylim"):
                plot.ylim = None

        plot.refresh()

    def _calculate_spectrum(self, raw: np.ndarray) -> Tuple[np.ndarray, str]:
        if self.display_mode == "Raw Counts":
            return raw, "Intensity (Counts)"

        if self.display_mode == "Dark Subtracted":
            if self.dark_spectrum is not None and len(self.dark_spectrum) == len(raw):
                return np.maximum(
                    raw - self.dark_spectrum, 0.0
                ), "Dark Subtracted (Counts)"
            return raw, "Raw (Dark Baseline Missing!)"

        if self.display_mode in ("Transmission (%)", "Absorbance (AU)"):
            if self.dark_spectrum is not None and self.white_spectrum is not None:
                denom = np.maximum(self.white_spectrum - self.dark_spectrum, 1.0)
                numer = np.maximum(raw - self.dark_spectrum, 0.0)
                trans = numer / denom
                if self.display_mode == "Transmission (%)":
                    return np.clip(trans * 100.0, 0.0, 200.0), "% Transmission"
                else:
                    abs_val = -np.log10(np.clip(trans, 1e-4, 10.0))
                    return np.clip(abs_val, -0.5, 4.0), "Absorbance (AU)"
            return raw, "Raw (Take Dark & White Refs First!)"

        return raw, "Intensity"

    def _write_recording_frame(self, t: float, raw: np.ndarray):
        if self.recording_format == "CSV" and self._csv_writer:
            self._csv_writer.writerow([t] + list(raw))
        elif self.recording_format == "HDF5" and self._h5_dataset:
            curr = self._h5_dataset.shape[0]
            self._h5_dataset.resize(curr + 1, axis=0)
            self._h5_dataset[curr] = raw.astype("float32")

    # --- UI Event Handlers ---

    @on(Select.Changed, "#select-device")
    def on_device_select(self, event: Select.Changed):
        if event.value != Select.BLANK:
            self.dev.open(str(event.value))
            self.log_msg(f"Connected to spectrometer: [bold]{event.value}[/]")

    @on(Input.Changed, "#inp-integration")
    def on_integration_change(self, event: Input.Changed):
        if event.value.isdigit():
            val = int(event.value)
            actual = self.dev.set_integration_time_micros(val)
            self.log_msg(f"Integration time: [bold]{actual:,}[/] µs")

    @on(Input.Changed, "#inp-average")
    def on_average_change(self, event: Input.Changed):
        if event.value.isdigit() and int(event.value) > 0:
            self.scans_to_average = int(event.value)

    @on(Input.Changed, "#inp-boxcar")
    def on_boxcar_change(self, event: Input.Changed):
        if event.value.isdigit():
            self.boxcar_width = int(event.value)

    @on(Select.Changed, "#select-trigger")
    def on_trigger_change(self, event: Select.Changed):
        if event.value != Select.BLANK:
            self.dev.set_trigger_mode(int(event.value))
            self.log_msg(f"Trigger mode set to {event.value}")

    @on(Switch.Changed, "#sw-dark-pix")
    def on_dark_pix_switch(self, event: Switch.Changed):
        self.correct_dark_pixels = event.value

    @on(Switch.Changed, "#sw-nonlin")
    def on_nonlin_switch(self, event: Switch.Changed):
        self.correct_nonlinearity = event.value

    @on(Switch.Changed, "#sw-lamp")
    def on_lamp_switch(self, event: Switch.Changed):
        self.dev.set_lamp_enable(event.value)
        self.log_msg(f"Light Source: {'ON' if event.value else 'OFF'}")

    @on(Switch.Changed, "#sw-shutter")
    def on_shutter_switch(self, event: Switch.Changed):
        self.dev.set_shutter_open(event.value)
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
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = Path(f"snapshot_{ts}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Wavelength_nm", "Raw_Counts", "Display_Value"])
            disp, _ = self._calculate_spectrum(self.latest_intensities)
            for wl, raw, d in zip(self.latest_wl, self.latest_intensities, disp):
                w.writerow([f"{wl:.4f}", f"{raw:.2f}", f"{d:.4f}"])
        self.log_msg(f"[bold green]Saved snapshot to {path.resolve()}[/]")

    @on(Button.Pressed, "#btn-record")
    def action_toggle_record(self):
        btn = self.query_one("#btn-record", Button)
        if not self.recording_active:
            fmt = str(self.query_one("#select-rec-fmt", Select).value or "CSV")
            self.recording_format = fmt
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = Path(f"spectrum_log_{ts}.{fmt.lower()}")

            if fmt == "CSV":
                self._csv_file = open(path, "w", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(
                    ["timestamp"] + [f"wl_{w:.2f}" for w in self.latest_wl]
                )
            else:
                self._h5_file = h5py.File(path, "w")
                px = len(self.latest_wl)
                self._h5_dataset = self._h5_file.create_dataset(
                    "spectra",
                    shape=(0, px),
                    maxshape=(None, px),
                    dtype="float32",
                    chunks=True,
                )
                self._h5_file.create_dataset("wavelengths", data=self.latest_wl)

            self.recording_active = True
            btn.label = "Stop Recording"
            btn.variant = "error"
            self.log_msg(f"[bold red]Recording started: {path.name}[/]")
        else:
            self.recording_active = False
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
            if self._h5_file:
                self._h5_file.close()
                self._h5_file = None
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
    device: Optional[str] = None,
    integration_time_us: int = 100_000,
    use_mock: bool = False,
):
    app = LiveSpectrometerApp(
        device_id=device,
        integration_time_us=integration_time_us,
        use_mock=use_mock,
    )
    app.run()
