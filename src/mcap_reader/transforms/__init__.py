"""
Transform tree module — TF2-style transform buffer with quaternion math.

Provides coordinate frame management for robot sensor data:
- Quaternion operations (SLERP, composition, inversion)
- Frame graph with path finding
- Transform buffer with timestamp-based lookup and interpolation
"""

from mcap_reader.transforms.math import Quaternion, Transform, Vector3

__all__ = [
    "Quaternion",
    "Vector3",
    "Transform",
    "TransformBuffer",
]


def __getattr__(name: str):
    if name == "TransformBuffer":
        from mcap_reader.transforms.buffer import TransformBuffer
        return TransformBuffer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
