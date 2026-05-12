from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from seabreeze_live.acquisition import Streamer
from seabreeze_live.device import list_devices, open_device
from seabreeze_live.consumers import CsvWriter, NdjsonStdoutEmitter


def _pick_writer(output: Path):
    suffix = output.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        from seabreeze_live.consumers import Hdf5Writer

        return Hdf5Writer(output)
    if suffix == ".csv":
        return CsvWriter(output)
    raise SystemExit(
        f"unsupported output extension {suffix!r}; use .h5/.hdf5 or .csv"
    )


def _cmd_list(_args: argparse.Namespace) -> int:
    devs = list_devices()
    if not devs:
        print("no spectrometers found (or `seabreeze` not installed)", file=sys.stderr)
        return 1
    for d in devs:
        print(f"{d['serial']}\t{d['model']}")
    return 0


def _cmd_acquire(args: argparse.Namespace) -> int:
    device = open_device(serial=args.serial, mock=args.mock)
    try:
        device.set_integration_time(args.integration_us)
        writer = _pick_writer(args.output)
        with Streamer(device, [writer], max_frames=args.count) as s:
            try:
                s.wait()
            except KeyboardInterrupt:
                pass
        print(
            f"wrote {s.frame_count} frames to {args.output}",
            file=sys.stderr,
        )
        return 0
    finally:
        device.close()


def _cmd_stream(args: argparse.Namespace) -> int:
    device = open_device(serial=args.serial, mock=args.mock)
    try:
        device.set_integration_time(args.integration_us)
        emitter = NdjsonStdoutEmitter()
        with Streamer(device, [emitter], max_frames=args.max_frames) as s:
            try:
                s.wait()
            except KeyboardInterrupt:
                pass
        return 0
    finally:
        device.close()


def _cmd_view(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    from seabreeze_live.consumers import MatplotlibLiveView

    device = open_device(serial=args.serial, mock=args.mock)
    try:
        device.set_integration_time(args.integration_us)
        view = MatplotlibLiveView(
            refresh_ms=args.refresh_ms,
            snapshot_dir=args.snapshot_dir,
            snapshot_format=args.snapshot_format,
        )
        print(
            f"snapshots will be saved to {view.snapshot_dir} "
            f"(button or 's' key, format={args.snapshot_format})",
            file=sys.stderr,
        )
        consumers = [view]
        if args.save is not None:
            consumers.append(_pick_writer(args.save))

        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _shutdown(_signum, _frame):
            # Closing all figures makes plt.show() return, so the `with`
            # block can stop the streamer and close files cleanly.
            plt.close("all")

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
        try:
            with Streamer(device, consumers, max_frames=args.max_frames) as s:
                view.run()
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)
        if args.save is not None:
            print(
                f"wrote {s.frame_count} frames to {args.save}",
                file=sys.stderr,
            )
        return 0
    finally:
        device.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="seabreeze-live")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list connected spectrometers")

    acq = sub.add_parser("acquire", help="capture N spectra to a file")
    acq.add_argument("--count", type=int, required=True)
    acq.add_argument("--integration-us", type=int, default=100_000)
    acq.add_argument("--output", "-o", type=Path, required=True,
                     help="output path; .h5/.hdf5 -> HDF5, .csv -> CSV")
    acq.add_argument("--serial", default=None)
    acq.add_argument("--mock", action="store_true")

    strm = sub.add_parser("stream", help="emit spectra as NDJSON on stdout")
    strm.add_argument("--integration-us", type=int, default=100_000)
    strm.add_argument("--serial", default=None)
    strm.add_argument("--mock", action="store_true")
    strm.add_argument("--max-frames", type=int, default=None)

    vw = sub.add_parser("view", help="live spectrum plot (matplotlib)")
    vw.add_argument("--integration-us", type=int, default=100_000)
    vw.add_argument("--serial", default=None)
    vw.add_argument("--mock", action="store_true")
    vw.add_argument("--max-frames", type=int, default=None)
    vw.add_argument("--refresh-ms", type=int, default=50,
                    help="plot refresh interval in ms (default 50)")
    vw.add_argument("--save", type=Path, default=None,
                    help="also record every frame to this file (.h5/.hdf5 or .csv)")
    vw.add_argument("--snapshot-dir", type=Path, default=Path.cwd(),
                    help="directory for button/keypress snapshots (default: CWD)")
    vw.add_argument("--snapshot-format", choices=["csv", "h5"], default="csv",
                    help="snapshot file format (default: csv)")

    args = p.parse_args(argv)
    handlers = {
        "list": _cmd_list,
        "acquire": _cmd_acquire,
        "stream": _cmd_stream,
        "view": _cmd_view,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
