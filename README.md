# seabreeze-live

A live acquisition wrapper around [python-seabreeze](https://python-seabreeze.readthedocs.io/) for Ocean Optics spectrometers. Layered API (blocking `acquire()` + threaded `Streamer`), pluggable consumers (HDF5/CSV/NDJSON), and a `MockDevice` so you can develop and test off-hardware.

The on-the-wire `SpectrumFrame` shape mirrors [rust-daq](https://github.com/) `SpectrumData` (`values`/`axis`/`*_units` + capture metadata) so a future native Rust driver can map 1:1.

## Install

```bash
uv sync
```

For a real instrument, install the SeaBreeze USB driver and platform-specific
permissions/rules described in the [python-seabreeze installation guide](https://python-seabreeze.readthedocs.io/en/latest/install.html).

## Preparing a real spectrometer

Start by discovering the instrument, then select it explicitly by serial number:

```bash
seabreeze-live devices
seabreeze-live view --device USB2+F01234 --integration-time 100000
```

`--device` avoids accidentally attaching to the first available instrument
when more than one is connected. Omitting it opens the first discovered device.
The real-hardware path never falls back to the mock simulator: a missing
SeaBreeze backend, inaccessible USB device, or invalid serial produces a clear
connection error with next steps. Use `--mock` only when simulation is intended.

At connection, the application reads and caches the wavelength axis, probes
available lamp/shutter/temperature/EEPROM capabilities, reads the hardware
integration-time limits, and clamps the configured integration time to that
range. The device is closed on TUI exit, command termination, or context-manager
exit.

## Quick start

```python
from seabreeze_live import MockDevice, Streamer, acquire
from seabreeze_live.consumers import Hdf5Writer, NdjsonStdoutEmitter

device = MockDevice()                  # or open_device(serial=...) for real HW
device.set_integration_time(100_000)   # microseconds

# Blocking: grab N frames.
frames = acquire(device, count=10)
print(frames[0].values.shape, frames[0].axis_units)

# Streaming: background thread fans out to consumers.
with Streamer(device, [Hdf5Writer("run.h5"), NdjsonStdoutEmitter()],
              max_frames=100) as s:
    s.wait()
```

## CLI

```bash
seabreeze-live devices                                  # enumerate devices
seabreeze-live --mock                                   # launch the Textual TUI
seabreeze-live view --mock --integration-time 50000     # launch the TUI
seabreeze-live stream --mock --integration-time 50000   # headless peak monitor
python -m seabreeze_live serve --mock                    # JSON-RPC bridge
```

The TUI supports CSV and HDF5 recording and CSV snapshots from its controls.

## Live visualization

The default interactive experience is a Textual TUI, with live metrics,
spectral-region selection, trace color/style controls, optical references,
hardware corrections, and keyboard shortcuts. Plot updates are coalesced to
12 Hz so terminal rendering cannot build an unbounded backlog while capture
and recording proceed at the device rate.

**CLI**:
```bash
seabreeze-live view --mock --integration-time 50000
```

**Programmatic** — compose the lightweight Matplotlib consumer with other
stream consumers when a GUI plot is preferred:
```python
from seabreeze_live import MockDevice, Streamer
from seabreeze_live.consumers import Hdf5Writer, MatplotlibLiveView

device = MockDevice()
device.set_integration_time(50_000)

view = MatplotlibLiveView()
streamer = Streamer(device, [view, Hdf5Writer("run.h5")])
streamer.start()
try:
    view.run()         # blocks the main thread on the matplotlib event loop
finally:
    streamer.stop()    # also closes the HDF5 file
```

Acquisition runs on the Streamer's background thread; `view.run()` must be
called from the main thread. The view stores only the latest frame, so a slow
display cannot block acquisition.

## HDF5 layout

```
/wavelengths         float64[pixels]      written once
/intensities         float64[N, pixels]   chunked, resizable
/timestamp_ns        int64[N]
/frame_number        int64[N]
/integration_time_us int64[N]

root attrs: device_serial, value_units, axis_units, schema_version
```

The standalone `Hdf5Writer` opens files in
[SWMR](https://docs.h5py.org/en/stable/swmr.html) mode and flushes each frame
by default, so another process can tail the datasets while capture is active.
For TUI recording, frames are written in batches of 64 to avoid a dataset
resize and flush on every acquisition; pending frames are flushed on stop.

## NDJSON schema (Rust handoff)

One JSON object per line on stdout, fixed envelope:

```json
{"type":"spectrum","timestamp_ns":1715520000000000000,"frame_number":0,
 "integration_time_us":100000,"device_serial":"USB2+H02749",
 "value_units":"counts","axis_units":"nm",
 "values":[…],"axis":[…]}
```

A Rust parent process (e.g. a future `crates/driver-seabreeze` in rust-daq that spawns this CLI) can read `stdout` line-by-line and map each object to `SpectrumData` directly.

## JSON-RPC stdio protocol (Rust bridge)

For tighter integration with a Rust parent (e.g. `crates/driver-seabreeze` in rust-daq), the `serve` subcommand replaces the implicit lifecycle of `stream`: the parent opens the spectrometer once, drives it via line-delimited JSON-RPC 2.0 on `stdin`, and reads spectrum frames + RPC responses from `stdout`.

```bash
python -m seabreeze_live serve --mock
```

On startup the server writes one banner line:

```json
{"jsonrpc":"2.0","ready":true,"device":{"model":"MockSpectrometer","serial":"MOCK000000","pixel_count":2048,"integration_time_limits_us":[1000,10000000]}}
```

Then it reads one JSON-RPC request per `stdin` line and writes exactly one JSON line per response. While streaming, NDJSON spectrum frames (same schema as `seabreeze-live stream`) interleave with RPC responses on `stdout`.

**Discriminator:** RPC responses always carry `"jsonrpc": "2.0"`. NDJSON spectrum frames never carry that key — they always carry `"type": "spectrum"`. A Rust parent should branch on the presence of `jsonrpc` before parsing.

### Methods

| Method | Params | Result |
|---|---|---|
| `set_integration_time_us` | `{value: int}` | `{ok: true}` |
| `set_trigger_mode` | `{mode: "NORMAL"\|...}` | `{ok: true}` — non-`NORMAL` modes return a JSON-RPC error on the mock |
| `get_wavelengths` | `{}` | `[float, ...]` (length = `pixel_count`) |
| `get_device_info` | `{}` | `{model, serial, pixel_count, integration_time_limits_us: [int, int]}` |
| `start_stream` | `{integration_time_us?: int}` | `{ok: true}` — begins emitting NDJSON spectrum frames on stdout |
| `stop_stream` | `{}` | `{ok: true}` (or `{ok: true, was_running: false}`) |
| `shutdown` | `{}` | `{ok: true}` — server stops the stream, closes the device, exits 0 |

Errors use the standard envelope:

```json
{"jsonrpc":"2.0","id":7,"error":{"code":-32601,"message":"unknown method 'foo'"}}
```

### How `serve` differs from `stream`

- `stream` opens the device, applies a fixed integration time from CLI args, and streams until the parent kills it (SIGINT/SIGTERM/EOF). No mid-run control.
- `serve` opens the device and waits — no streaming until the parent calls `start_stream`. Integration time, trigger mode, and stream lifecycle are all driven over `stdin`. `serve` is the supported entry point for the Rust driver.

## Architecture

```
                ┌──────────────┐
                │   device     │   SeabreezeDevice (hardware)
                │   (Protocol) │   MockDevice       (synthetic)
                └──────┬───────┘
                       │ frames (NDArray)
                ┌──────▼───────┐
                │  acquire()   │   blocking, returns list[SpectrumFrame]
                │  Streamer    │   threaded, push to consumers
                └──────┬───────┘
                       │ SpectrumFrame
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
   Hdf5Writer     CsvWriter     NdjsonStdout    (your consumer)
                                  Emitter
```

The Textual TUI uses the same consumer writers through `SpectrumRecorder`, so
interactive and headless recordings have identical schemas. Pure acquisition
averaging, boxcar smoothing, display transforms, and wavelength-region
selection live in `seabreeze_live.processing` and are independently testable.

Consumers implement a two-method `Protocol` (`on_frame`, `close`) — the
Streamer dispatches synchronously on its acquisition thread, so a slow consumer
naturally throttles capture. Wrap in a queued forwarder if you need true
decoupling.

## Status

Single-device, software-timed acquisition only. The following are stubbed for forward-compatibility and will raise `NotImplementedError`:

- `TriggerMode.EXTERNAL_*` on `MockDevice`
- multi-device groups (no class yet; open one device at a time)

## Tests

```bash
uv run pytest
```

The complete test suite runs against `MockDevice`; no hardware is required.
It covers acquisition, stream consumers, buffered HDF5 close behavior,
snapshots, display processing, the JSON-RPC bridge, and TUI recording schema.
