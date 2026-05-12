# seabreeze-live

A live acquisition wrapper around [python-seabreeze](https://python-seabreeze.readthedocs.io/) for Ocean Optics spectrometers. Layered API (blocking `acquire()` + threaded `Streamer`), pluggable consumers (HDF5/CSV/NDJSON), and a `MockDevice` so you can develop and test off-hardware.

The on-the-wire `SpectrumFrame` shape mirrors [rust-daq](https://github.com/) `SpectrumData` (`values`/`axis`/`*_units` + capture metadata) so a future native Rust driver can map 1:1.

## Install

```bash
uv sync                          # core (mock + CSV + NDJSON only)
uv sync --extra hdf5             # add HDF5 support
uv sync --extra plot             # add live matplotlib viewer
uv sync --extra hardware         # add the real seabreeze backend
uv sync --extra hdf5 --extra plot --extra hardware --extra dev
```

If you use the hardware extra, you'll also need the seabreeze udev rules / driver per the [seabreeze install docs](https://python-seabreeze.readthedocs.io/en/latest/install.html).

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
seabreeze-live list                                       # connected devices
seabreeze-live acquire --mock --count 100 -o run.h5       # capture to HDF5
seabreeze-live acquire --count 50 -o run.csv              # capture to CSV
seabreeze-live stream --mock --max-frames 100             # NDJSON on stdout
seabreeze-live view --mock --integration-us 50000         # live plot
seabreeze-live view --mock --save run.h5                  # live plot + record
```

Output extension picks the writer: `.h5`/`.hdf5` → HDF5, `.csv` → CSV.

## Live visualization

Requires the `[plot]` extra (matplotlib). Two equivalent ways:

**CLI** — closing the window stops acquisition:
```bash
seabreeze-live view --mock --integration-us 50000 --refresh-ms 50
```

**Programmatic** — compose the viewer with other consumers in one `Streamer`:
```python
from seabreeze_live import MockDevice, Streamer
from seabreeze_live.consumers import Hdf5Writer, MatplotlibLiveView

device = MockDevice()
device.set_integration_time(50_000)

view = MatplotlibLiveView(refresh_ms=50)
streamer = Streamer(device, [view, Hdf5Writer("run.h5")])
streamer.start()
try:
    view.run()         # blocks the main thread on the matplotlib event loop
finally:
    streamer.stop()    # also closes the HDF5 file
```

Acquisition runs on the Streamer's background thread; `view.run()` must be called from the main thread (matplotlib GUI requirement). The view stores only the latest frame, so a slow display can't block acquisition — but any other consumer in the same Streamer (e.g. `Hdf5Writer`) does throttle the loop synchronously, which is what you want for not dropping data.

## HDF5 layout

```
/wavelengths         float64[pixels]      written once
/intensities         float64[N, pixels]   chunked, resizable
/timestamp_ns        int64[N]
/frame_number        int64[N]
/integration_time_us int64[N]

root attrs: device_serial, value_units, axis_units, schema_version
```

The file is opened in [SWMR](https://docs.h5py.org/en/stable/swmr.html) mode, so a separate process can open it read-only and tail the datasets while the writer is still running.

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

Consumers implement a two-method `Protocol` (`on_frame`, `close`) — the Streamer dispatches synchronously on its acquisition thread, so a slow consumer naturally throttles capture. Wrap in a queued forwarder if you need true decoupling.

## Status

Single-device, software-timed acquisition only. The following are stubbed for forward-compatibility and will raise `NotImplementedError`:

- `TriggerMode.EXTERNAL_*` on `MockDevice`
- multi-device groups (no class yet; open one device at a time)

## Tests

```bash
uv run pytest
```

The entire test suite runs against `MockDevice`; no hardware required.
