"""Command line interface routing TUI, CLI streaming, snapshot, and RPC modes."""

import argparse
import time

from seabreeze_live.consumers.live_view import run_tui
from seabreeze_live.device import HardwareConnectionError, SpectrometerDevice
from seabreeze_live.mock import MockDevice
from seabreeze_live.rpc import RpcServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seabreeze-live",
        description="High-performance live Ocean Optics spectrometer TUI & acquisition engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # TUI Command (Default)
    tui_parser = subparsers.add_parser(
        "view", help="Launch interactive Textual TUI live view"
    )
    _add_device_args(tui_parser)
    tui_parser.set_defaults(
        func=lambda args: run_tui(args.device, args.integration_time, args.mock)
    )

    # CLI Stream Command
    stream_parser = subparsers.add_parser(
        "stream", help="Headless terminal streaming output"
    )
    _add_device_args(stream_parser)
    stream_parser.add_argument(
        "--format", choices=["stdout", "ndjson"], default="stdout"
    )
    stream_parser.set_defaults(func=_handle_stream)

    # List Devices Command
    list_parser = subparsers.add_parser(
        "devices", help="List all connected spectrometers"
    )
    list_parser.set_defaults(func=_handle_list_devices)

    serve_parser = subparsers.add_parser("serve", help="Run the JSON-RPC device bridge")
    serve_parser.add_argument(
        "--mock", action="store_true", help="Use the mock spectrometer"
    )
    serve_parser.add_argument("-d", "--device", type=str, default=None)
    serve_parser.set_defaults(func=_handle_serve)

    # Direct fallback options for root command
    _add_device_args(parser)
    return parser


def _add_device_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-d", "--device", type=str, default=None, help="Spectrometer serial number"
    )
    parser.add_argument(
        "-i",
        "--integration-time",
        type=int,
        default=100_000,
        help="Integration time in µs (default: 100000)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Force mock spectrometer simulator"
    )


def _handle_list_devices(args: argparse.Namespace):
    devices = SpectrometerDevice.list_available_devices()
    print("\nAvailable Spectrometers:")
    for d in devices:
        kind = "SIMULATOR" if d["mock"] else "HARDWARE"
        print(f"  • {d['label']} [{kind}]")
    print()


def _handle_stream(args: argparse.Namespace):
    dev = SpectrometerDevice(device_id=args.device, use_mock=args.mock)
    dev.set_integration_time_micros(args.integration_time)
    meta = dev.meta
    print(
        f"Streaming from {meta.model} ({meta.serial_number})... Press Ctrl+C to stop.\n"
    )
    try:
        while True:
            intensities = dev.get_intensities()
            wl = dev.get_wavelengths()
            peak_idx = int(intensities.argmax())
            print(
                f"\rPeak: {wl[peak_idx]:.2f} nm | Counts: {intensities[peak_idx]:.1f} | Saturated: {bool(intensities.max() >= 65000)}",
                end="",
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStreaming stopped.")
    finally:
        dev.close()


def _handle_serve(args: argparse.Namespace) -> None:
    device = MockDevice() if args.mock else SpectrometerDevice(device_id=args.device)
    server = RpcServer(device)
    try:
        server.emit_ready()
        server.serve_forever()
    finally:
        device.close()


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command is None:
            # Default behavior: run TUI
            run_tui(
                device=args.device,
                integration_time_us=args.integration_time,
                use_mock=args.mock,
            )
        else:
            args.func(args)
    except HardwareConnectionError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
