from typing import TYPE_CHECKING, Any

from seabreeze_live.consumers.csv import CsvWriter
from seabreeze_live.consumers.ndjson_stdout import NdjsonStdoutEmitter

if TYPE_CHECKING:
    from seabreeze_live.consumers.hdf5 import Hdf5Writer
    from seabreeze_live.consumers.live_view import MatplotlibLiveView

__all__ = ["CsvWriter", "Hdf5Writer", "MatplotlibLiveView", "NdjsonStdoutEmitter"]


def __getattr__(name: str) -> Any:
    if name == "Hdf5Writer":
        from seabreeze_live.consumers.hdf5 import Hdf5Writer

        return Hdf5Writer
    if name == "MatplotlibLiveView":
        from seabreeze_live.consumers.live_view import MatplotlibLiveView

        return MatplotlibLiveView
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
