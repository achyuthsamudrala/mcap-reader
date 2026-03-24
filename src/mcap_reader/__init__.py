"""
mcap-reader: Python library for reading MCAP files (ROS 2 recording format).

A learning project for understanding how robot data is stored, structured,
time-synchronized, and spatially referenced.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mcap-reader")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__all__ = [
    "McapReader",
    "TimeSynchronizer",
    "EpisodeDetector",
    "TransformBuffer",
    "CameraModel",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading heavy modules until needed."""
    if name == "McapReader":
        from mcap_reader.reader import McapReader
        return McapReader
    if name == "TimeSynchronizer":
        from mcap_reader.sync import TimeSynchronizer
        return TimeSynchronizer
    if name == "EpisodeDetector":
        from mcap_reader.episode import EpisodeDetector
        return EpisodeDetector
    if name == "TransformBuffer":
        from mcap_reader.transforms.buffer import TransformBuffer
        return TransformBuffer
    if name == "CameraModel":
        from mcap_reader.calibration import CameraModel
        return CameraModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
