"""
Message type deserializers for ROS 2 sensor messages.

Each module handles one ROS 2 message type, converting from either decoded
ROS messages (via mcap-ros2-support) or raw CDR bytes into structured Python
objects with numpy array fields.

The MESSAGE_REGISTRY maps ROS type strings to wrapper classes for automatic
dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# Registry mapping ROS 2 message type strings to wrapper classes.
# Populated lazily to avoid circular imports.
MESSAGE_REGISTRY: dict[str, type] = {}

_REGISTRY_POPULATED = False


def _populate_registry() -> None:
    """Lazily populate the message registry on first use."""
    global _REGISTRY_POPULATED
    if _REGISTRY_POPULATED:
        return
    from mcap_reader.messages.camera_info import CameraInfo
    from mcap_reader.messages.compressed_image import CompressedImage
    from mcap_reader.messages.image import Image
    from mcap_reader.messages.imu import Imu
    from mcap_reader.messages.joint_state import JointState
    from mcap_reader.messages.pointcloud import PointCloud2
    from mcap_reader.messages.transform import TFMessage

    MESSAGE_REGISTRY.update({
        "sensor_msgs/msg/Image": Image,
        "sensor_msgs/msg/CompressedImage": CompressedImage,
        "sensor_msgs/msg/CameraInfo": CameraInfo,
        "sensor_msgs/msg/PointCloud2": PointCloud2,
        "sensor_msgs/msg/Imu": Imu,
        "sensor_msgs/msg/JointState": JointState,
        "tf2_msgs/msg/TFMessage": TFMessage,
    })
    _REGISTRY_POPULATED = True


def wrap_message(ros_type: str, msg: Any) -> Any:
    """Convert a decoded ROS message to the appropriate typed wrapper.

    Args:
        ros_type: ROS 2 message type string (e.g. "sensor_msgs/msg/Image").
        msg: Decoded message object from mcap-ros2-support.

    Returns:
        Typed wrapper instance, or the original message if no wrapper is registered.
    """
    _populate_registry()
    wrapper_cls = MESSAGE_REGISTRY.get(ros_type)
    if wrapper_cls is None:
        return msg
    return wrapper_cls.from_ros_msg(msg)
