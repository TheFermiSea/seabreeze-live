# seabreeze-live

A live acquisition service around [python-seabreeze](https://python-seabreeze.readthedocs.io/) for Ocean Optics spectrometers. It provides a transport-neutral Python acquisition core, pluggable HDF5/CSV/NDJSON consumers, an interactive TUI, and the stdio protocol used by rust-daq's existing `driver-seabreeze` crate.

The core `SpectrumFrame` maps directly to rust-daq's `SpectrumData`. The Rust bridge caches the wavelength axis during startup and receives compact frames containing only `timestamp_ns`, `integration_time_us`, and `values`.

## Install

```bash
uv sync --extra ui --extra hdf5
```

For a headless rust-daq bridge, the base installation is sufficient. The UI,
Matplotlib view, and HDF5 support are separate `ui`, `plot`, and `hdf5` extras,
so a daemon host does not need to install graphical or storage dependencies.

For a real instrument, install the SeaBreeze USB driver and platform-specific
permissions/rules described in the [python-seabreeze installation guide](https://python-seabreeze.readthedocs.io/en/latest/install.html).

## Preparing a real spectrometer

Start by discovering the instrument, then select it explicitly by serial number:

```bash
seabreeze-live devices
seabreeze-live view --serial USB2+F01234 --integration-time 100000
```

`--serial` (also accepted as `--device`) avoids accidentally attaching to the first available instrument
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
It writes them under `spectra/` by default rather than cluttering the working
directory.

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

This full envelope is emitted by the standalone `stream --format ndjson`
command. The rust-daq service path uses the compact event described below.

## JSON-RPC stdio protocol (Rust bridge)

The `serve` subcommand is the subprocess entry point used by rust-daq's
`driver-seabreeze` crate. The parent opens the spectrometer once, drives it via
line-delimited JSON-RPC 2.0 on `stdin`, and reads compact spectrum frames plus
RPC responses from `stdout`.

```bash
python -m seabreeze_live serve --mock
```

On startup the server writes one banner line:

```json
{"jsonrpc":"2.0","ready":true,"protocol_version":"1.0","device":{"model":"MockSpectrometer","serial":"MOCK000000","pixel_count":2048,"integration_time_limits_us":[1000,10000000]}}
```

Then it reads one JSON-RPC request per `stdin` line and writes exactly one JSON
line per response. While streaming, compact spectrum events interleave with RPC
responses on `stdout`:

```json
{"timestamp_ns":1715520000000000000,"integration_time_us":100000,"values":[…]}
```

The axis is fetched once with `get_wavelengths`. Avoiding its repetition cuts
the dominant steady-state subprocess serialization and pipe overhead roughly in
half for typical devices.

**Discriminator:** RPC responses always carry `"jsonrpc": "2.0"`. Spectrum
events never carry that key. A Rust parent branches on the presence of
`jsonrpc` before parsing, exactly as `driver-seabreeze` does.

### Methods

| Method | Params | Result |
|---|---|---|
| `set_integration_time_us` | `{value: int}` | `null` |
| `set_trigger_mode` | `{mode: "NORMAL"\|...}` | `null` — non-`NORMAL` modes return a JSON-RPC error on the mock |
| `get_wavelengths` | `{}` | `[float, ...]` (length = `pixel_count`) |
| `get_device_info` | `{}` | `{model, serial, pixel_count, integration_time_limits_us: [int, int]}` |
| `start_stream` | `{integration_time_us?: int}` | `null` — begins emitting compact spectrum events |
| `stop_stream` | `{}` | `null` |
| `shutdown` | `{}` | `null` — server stops the stream, closes the device, exits 0 |

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
                │ Spectrometer │   structural Protocol
                │   device     │   SeaBreeze adapter / one MockDevice
                └──────┬───────┘
                       │ synchronized device I/O
                ┌──────▼───────┐
                │   Acquirer   │   validates and constructs SpectrumFrame
                │ acquire/     │   blocking and threaded front ends
                │ Streamer     │
                └──────┬───────┘
                       │ SpectrumFrame
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
   Hdf5Writer     CsvWriter     NdjsonStdout    (your consumer)
                                  Emitter
```

The `Acquirer` serializes reads and configuration changes, preventing exposure
or device-switch operations from racing an in-flight hardware read. Blocking
acquisition, threaded streaming, the TUI, and the Rust bridge all construct the
same validated `SpectrumFrame`. The Textual TUI uses the public consumer writers
through `SpectrumRecorder`, so interactive and headless recordings have
identical schemas.

Consumers implement a two-method `Protocol` (`on_frame`, `close`) — the
Streamer dispatches synchronously on its acquisition thread, so a slow consumer
naturally throttles capture. Wrap in a queued forwarder if you need true
decoupling.

## Status

Single-device acquisition only. Hardware trigger support depends on the selected
Ocean Optics model and python-seabreeze backend. The simulator intentionally
raises `NotImplementedError` for non-normal trigger modes.

- multi-device groups are not implemented; open one device per service process

## Tests

```bash
uv run pytest
```

The complete test suite runs against `MockDevice`; no hardware is required.
It covers acquisition, stream consumers, buffered HDF5 close behavior,
snapshots, display processing, the JSON-RPC bridge, and TUI recording schema.
