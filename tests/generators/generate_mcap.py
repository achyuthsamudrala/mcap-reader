"""Synthetic MCAP file generators for testing.

Uses the mcap writer library to create valid MCAP files with
CDR-encoded ROS 2 messages built via struct.pack.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from mcap.writer import Writer


# ---------------------------------------------------------------------------
# CDR encoding helpers
# ---------------------------------------------------------------------------

def _cdr_header_mcap_ros2() -> bytes:
    """4-byte CDR little-endian encapsulation header for mcap-ros2-support.

    The mcap_ros2 CdrReader checks ``data[1] & 1`` to determine endianness.
    Byte[1] = 0x01 means little-endian (CDR_LE = 0x0001).
    """
    return b"\x00\x01\x00\x00"


def _cdr_header() -> bytes:
    """4-byte CDR little-endian encapsulation header for our CdrDeserializer.

    Our deserializer interprets 0x0002 as little-endian.
    """
    return b"\x00\x02\x00\x00"


def _cdr_uint32(val: int) -> bytes:
    return struct.pack("<I", val)


def _cdr_uint8(val: int) -> bytes:
    return struct.pack("<B", val)


def _cdr_float64(val: float) -> bytes:
    return struct.pack("<d", val)


def _cdr_float32(val: float) -> bytes:
    return struct.pack("<f", val)


def _cdr_string(s: str) -> bytes:
    """CDR string: uint32 length (including null terminator) + chars + null."""
    encoded = s.encode("utf-8") + b"\x00"
    return struct.pack("<I", len(encoded)) + encoded


def _cdr_ros_header(sec: int, nanosec: int, frame_id: str) -> bytes:
    """Encode a std_msgs/Header in CDR."""
    return _cdr_uint32(sec) + _cdr_uint32(nanosec) + _cdr_string(frame_id)


def _align_to(data: bytes, alignment: int) -> bytes:
    """Pad data to the next multiple of alignment (relative to data start after CDR header)."""
    # offset from start of data portion (after the 4-byte encapsulation header)
    # When building a full CDR payload, call this considering the current payload length
    remainder = len(data) % alignment
    if remainder != 0:
        data += b"\x00" * (alignment - remainder)
    return data


def _pad_to_align(payload_len: int, alignment: int) -> bytes:
    """Return padding bytes needed to align payload_len to alignment."""
    remainder = payload_len % alignment
    if remainder != 0:
        return b"\x00" * (alignment - remainder)
    return b""


def _build_imu_cdr(
    sec: int,
    nanosec: int,
    frame_id: str,
    orientation: tuple[float, float, float, float],
    angular_velocity: tuple[float, float, float],
    linear_acceleration: tuple[float, float, float],
    orientation_cov: list[float] | None = None,
    angular_velocity_cov: list[float] | None = None,
    linear_acceleration_cov: list[float] | None = None,
    encap_header: bytes | None = None,
) -> bytes:
    """Build a complete CDR-encoded sensor_msgs/msg/Imu."""
    if orientation_cov is None:
        orientation_cov = [0.0] * 9
    if angular_velocity_cov is None:
        angular_velocity_cov = [0.0] * 9
    if linear_acceleration_cov is None:
        linear_acceleration_cov = [0.0] * 9

    payload = bytearray()
    # Header
    payload += _cdr_ros_header(sec, nanosec, frame_id)
    # Align to 8 for float64 orientation
    payload += _pad_to_align(len(payload), 8)
    # orientation (x, y, z, w)
    for v in orientation:
        payload += _cdr_float64(v)
    # orientation_covariance (9 float64)
    for v in orientation_cov:
        payload += _cdr_float64(v)
    # angular_velocity (x, y, z)
    for v in angular_velocity:
        payload += _cdr_float64(v)
    # angular_velocity_covariance
    for v in angular_velocity_cov:
        payload += _cdr_float64(v)
    # linear_acceleration (x, y, z)
    for v in linear_acceleration:
        payload += _cdr_float64(v)
    # linear_acceleration_covariance
    for v in linear_acceleration_cov:
        payload += _cdr_float64(v)

    return (encap_header or _cdr_header_mcap_ros2()) + bytes(payload)


def _build_joint_state_cdr(
    sec: int,
    nanosec: int,
    frame_id: str,
    names: list[str],
    positions: list[float],
    velocities: list[float],
    efforts: list[float],
    encap_header: bytes | None = None,
) -> bytes:
    """Build a complete CDR-encoded sensor_msgs/msg/JointState."""
    payload = bytearray()
    # Header
    payload += _cdr_ros_header(sec, nanosec, frame_id)
    # name: sequence<string>
    # Align to 4 for the sequence count uint32
    payload += _pad_to_align(len(payload), 4)
    payload += _cdr_uint32(len(names))
    for name in names:
        # Each string's length prefix is uint32, needs 4-byte alignment
        payload += _pad_to_align(len(payload), 4)
        payload += _cdr_string(name)
    # position: sequence<float64>
    payload += _pad_to_align(len(payload), 4)
    payload += _cdr_uint32(len(positions))
    if positions:
        payload += _pad_to_align(len(payload), 8)
        for v in positions:
            payload += _cdr_float64(v)
    # velocity: sequence<float64>
    payload += _pad_to_align(len(payload), 4)
    payload += _cdr_uint32(len(velocities))
    if velocities:
        payload += _pad_to_align(len(payload), 8)
        for v in velocities:
            payload += _cdr_float64(v)
    # effort: sequence<float64>
    payload += _pad_to_align(len(payload), 4)
    payload += _cdr_uint32(len(efforts))
    if efforts:
        payload += _pad_to_align(len(payload), 8)
        for v in efforts:
            payload += _cdr_float64(v)

    return (encap_header or _cdr_header_mcap_ros2()) + bytes(payload)


def _build_tf_message_cdr(
    transforms: list[dict],
    encap_header: bytes | None = None,
) -> bytes:
    """Build a CDR-encoded tf2_msgs/msg/TFMessage.

    Each transform dict has: sec, nanosec, parent_frame, child_frame,
    translation (x,y,z), rotation (x,y,z,w).
    """
    payload = bytearray()
    # transforms: sequence<TransformStamped>
    payload += _cdr_uint32(len(transforms))
    for tf in transforms:
        # header: the stamp fields are uint32 which need 4-byte alignment
        payload += _pad_to_align(len(payload), 4)
        payload += _cdr_ros_header(tf["sec"], tf["nanosec"], tf["parent_frame"])
        # child_frame_id: string length is uint32, needs 4-byte alignment
        payload += _pad_to_align(len(payload), 4)
        payload += _cdr_string(tf["child_frame"])
        # align to 8 for float64
        payload += _pad_to_align(len(payload), 8)
        # translation (x, y, z)
        for v in tf["translation"]:
            payload += _cdr_float64(v)
        # rotation (x, y, z, w)
        for v in tf["rotation"]:
            payload += _cdr_float64(v)

    return (encap_header or _cdr_header_mcap_ros2()) + bytes(payload)


def _build_image_cdr(
    sec: int,
    nanosec: int,
    frame_id: str,
    height: int,
    width: int,
    encoding: str,
    is_bigendian: int,
    step: int,
    pixel_data: bytes,
    encap_header: bytes | None = None,
) -> bytes:
    """Build a CDR-encoded sensor_msgs/msg/Image."""
    payload = bytearray()
    # Header
    payload += _cdr_ros_header(sec, nanosec, frame_id)
    # height, width
    payload += _pad_to_align(len(payload), 4)
    payload += _cdr_uint32(height)
    payload += _cdr_uint32(width)
    # encoding (string)
    payload += _cdr_string(encoding)
    # is_bigendian (uint8)
    payload += _cdr_uint8(is_bigendian)
    # step (uint32, align to 4)
    payload += _pad_to_align(len(payload), 4)
    payload += _cdr_uint32(step)
    # data: sequence<uint8>
    payload += _cdr_uint32(len(pixel_data))
    payload += pixel_data

    return (encap_header or _cdr_header_mcap_ros2()) + bytes(payload)


# ---------------------------------------------------------------------------
# ROS 2 message schema strings (simplified .msg definitions)
# ---------------------------------------------------------------------------

IMU_SCHEMA = """\
std_msgs/Header header
geometry_msgs/Quaternion orientation
float64[9] orientation_covariance
geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance
geometry_msgs/Vector3 linear_acceleration
float64[9] linear_acceleration_covariance

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
uint32 sec
uint32 nanosec

================================================================================
MSG: geometry_msgs/Quaternion
float64 x 0
float64 y 0
float64 z 0
float64 w 1

================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z"""

JOINT_STATE_SCHEMA = """\
std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
uint32 sec
uint32 nanosec"""

TF_MESSAGE_SCHEMA = """\
geometry_msgs/TransformStamped[] transforms

================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
uint32 sec
uint32 nanosec

================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation

================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z

================================================================================
MSG: geometry_msgs/Quaternion
float64 x 0
float64 y 0
float64 z 0
float64 w 1"""

IMAGE_SCHEMA = """\
std_msgs/Header header
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data

================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id

================================================================================
MSG: builtin_interfaces/Time
uint32 sec
uint32 nanosec"""


# ---------------------------------------------------------------------------
# Generator functions
# ---------------------------------------------------------------------------

def _sec_nanosec(t: float) -> tuple[int, int]:
    """Convert a float timestamp to (sec, nanosec)."""
    sec = int(t)
    nanosec = int((t - sec) * 1e9)
    return sec, nanosec


def _timestamp_ns(t: float) -> int:
    """Convert float seconds to nanoseconds."""
    return int(t * 1e9)


def generate_imu_mcap(
    path: str | Path,
    num_messages: int = 100,
    rate_hz: float = 200,
) -> Path:
    """Generate a synthetic MCAP file with IMU messages.

    Produces sine wave angular velocity and constant acceleration.
    """
    path = Path(path)
    dt = 1.0 / rate_hz
    base_time = 1700000000.0  # a fixed epoch

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()

        schema_id = writer.register_schema(
            name="sensor_msgs/msg/Imu",
            encoding="ros2msg",
            data=IMU_SCHEMA.encode(),
        )
        channel_id = writer.register_channel(
            topic="/imu/data",
            message_encoding="cdr",
            schema_id=schema_id,
        )

        for i in range(num_messages):
            t = base_time + i * dt
            sec, nanosec = _sec_nanosec(t)
            phase = 2.0 * math.pi * i / num_messages

            data = _build_imu_cdr(
                sec=sec,
                nanosec=nanosec,
                frame_id="imu_link",
                orientation=(0.0, 0.0, 0.0, 1.0),
                angular_velocity=(
                    math.sin(phase),
                    math.cos(phase),
                    0.0,
                ),
                linear_acceleration=(0.0, 0.0, 9.81),
            )

            writer.add_message(
                channel_id=channel_id,
                log_time=_timestamp_ns(t),
                data=data,
                publish_time=_timestamp_ns(t),
            )

        writer.finish()

    return path


def generate_joint_state_mcap(
    path: str | Path,
    num_messages: int = 100,
    rate_hz: float = 50,
    joint_names: list[str] | None = None,
) -> Path:
    """Generate a synthetic MCAP file with JointState messages.

    Produces sine wave positions for each joint.
    """
    if joint_names is None:
        joint_names = ["joint1", "joint2"]

    path = Path(path)
    dt = 1.0 / rate_hz
    base_time = 1700000000.0

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()

        schema_id = writer.register_schema(
            name="sensor_msgs/msg/JointState",
            encoding="ros2msg",
            data=JOINT_STATE_SCHEMA.encode(),
        )
        channel_id = writer.register_channel(
            topic="/joint_states",
            message_encoding="cdr",
            schema_id=schema_id,
        )

        for i in range(num_messages):
            t = base_time + i * dt
            sec, nanosec = _sec_nanosec(t)
            phase = 2.0 * math.pi * i / num_messages

            positions = [
                math.sin(phase + j * math.pi / len(joint_names))
                for j in range(len(joint_names))
            ]
            velocities = [
                math.cos(phase + j * math.pi / len(joint_names))
                for j in range(len(joint_names))
            ]

            data = _build_joint_state_cdr(
                sec=sec,
                nanosec=nanosec,
                frame_id="",
                names=joint_names,
                positions=positions,
                velocities=velocities,
                efforts=[0.0] * len(joint_names),
            )

            writer.add_message(
                channel_id=channel_id,
                log_time=_timestamp_ns(t),
                data=data,
                publish_time=_timestamp_ns(t),
            )

        writer.finish()

    return path


def generate_image_mcap(
    path: str | Path,
    num_messages: int = 10,
    height: int = 480,
    width: int = 640,
    encoding: str = "rgb8",
) -> Path:
    """Generate a synthetic MCAP file with Image messages.

    Produces gradient test images.
    """
    path = Path(path)
    base_time = 1700000000.0
    dt = 1.0 / 30.0  # 30 Hz

    channels = 3 if encoding in ("rgb8", "bgr8") else 1
    bytes_per_pixel = channels
    step = width * bytes_per_pixel

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()

        schema_id = writer.register_schema(
            name="sensor_msgs/msg/Image",
            encoding="ros2msg",
            data=IMAGE_SCHEMA.encode(),
        )
        channel_id = writer.register_channel(
            topic="/camera/image_raw",
            message_encoding="cdr",
            schema_id=schema_id,
        )

        for i in range(num_messages):
            t = base_time + i * dt
            sec, nanosec = _sec_nanosec(t)

            # Create a gradient pattern
            pixel_data = bytearray()
            for row in range(height):
                for col in range(width):
                    val = int(255 * col / max(width - 1, 1)) & 0xFF
                    if channels == 3:
                        r = val
                        g = int(255 * row / max(height - 1, 1)) & 0xFF
                        b = (i * 25) & 0xFF
                        pixel_data += bytes([r, g, b])
                    else:
                        pixel_data += bytes([val])

            data = _build_image_cdr(
                sec=sec,
                nanosec=nanosec,
                frame_id="camera_optical",
                height=height,
                width=width,
                encoding=encoding,
                is_bigendian=0,
                step=step,
                pixel_data=bytes(pixel_data),
            )

            writer.add_message(
                channel_id=channel_id,
                log_time=_timestamp_ns(t),
                data=data,
                publish_time=_timestamp_ns(t),
            )

        writer.finish()

    return path


def generate_tf_mcap(
    path: str | Path,
    num_messages: int = 50,
    frames: list[tuple[str, str]] | None = None,
) -> Path:
    """Generate a synthetic MCAP file with TF messages.

    Produces rotating transforms between frame pairs.
    """
    if frames is None:
        frames = [("world", "base_link"), ("base_link", "camera")]

    path = Path(path)
    base_time = 1700000000.0
    dt = 1.0 / 50.0

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()

        schema_id = writer.register_schema(
            name="tf2_msgs/msg/TFMessage",
            encoding="ros2msg",
            data=TF_MESSAGE_SCHEMA.encode(),
        )
        channel_id = writer.register_channel(
            topic="/tf",
            message_encoding="cdr",
            schema_id=schema_id,
        )

        for i in range(num_messages):
            t = base_time + i * dt
            sec, nanosec = _sec_nanosec(t)
            angle = 2.0 * math.pi * i / num_messages

            transforms = []
            for parent, child in frames:
                # Rotation about Z axis
                qz = math.sin(angle / 2.0)
                qw = math.cos(angle / 2.0)
                transforms.append({
                    "sec": sec,
                    "nanosec": nanosec,
                    "parent_frame": parent,
                    "child_frame": child,
                    "translation": (1.0, 0.0, 0.0),
                    "rotation": (0.0, 0.0, qz, qw),
                })

            data = _build_tf_message_cdr(transforms)

            writer.add_message(
                channel_id=channel_id,
                log_time=_timestamp_ns(t),
                data=data,
                publish_time=_timestamp_ns(t),
            )

        writer.finish()

    return path


def generate_multi_topic_mcap(path: str | Path) -> Path:
    """Generate a synthetic MCAP file with IMU at 200Hz + JointState at 50Hz + TF at 50Hz.

    Includes some clock drift on the JointState topic.
    """
    path = Path(path)
    base_time = 1700000000.0
    duration = 2.0  # 2 seconds

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()

        # Register schemas
        imu_schema_id = writer.register_schema(
            name="sensor_msgs/msg/Imu",
            encoding="ros2msg",
            data=IMU_SCHEMA.encode(),
        )
        js_schema_id = writer.register_schema(
            name="sensor_msgs/msg/JointState",
            encoding="ros2msg",
            data=JOINT_STATE_SCHEMA.encode(),
        )
        tf_schema_id = writer.register_schema(
            name="tf2_msgs/msg/TFMessage",
            encoding="ros2msg",
            data=TF_MESSAGE_SCHEMA.encode(),
        )

        imu_channel = writer.register_channel(
            topic="/imu/data",
            message_encoding="cdr",
            schema_id=imu_schema_id,
        )
        js_channel = writer.register_channel(
            topic="/joint_states",
            message_encoding="cdr",
            schema_id=js_schema_id,
        )
        tf_channel = writer.register_channel(
            topic="/tf",
            message_encoding="cdr",
            schema_id=tf_schema_id,
        )

        # Collect all messages with their timestamps, then sort
        messages = []

        # IMU at 200 Hz
        imu_dt = 1.0 / 200.0
        n_imu = int(duration / imu_dt)
        for i in range(n_imu):
            t = base_time + i * imu_dt
            sec, nanosec = _sec_nanosec(t)
            phase = 2.0 * math.pi * i / n_imu
            data = _build_imu_cdr(
                sec=sec, nanosec=nanosec, frame_id="imu_link",
                orientation=(0.0, 0.0, 0.0, 1.0),
                angular_velocity=(math.sin(phase), 0.0, 0.0),
                linear_acceleration=(0.0, 0.0, 9.81),
            )
            messages.append((_timestamp_ns(t), imu_channel, data))

        # JointState at 50 Hz with slight clock drift
        js_dt = 1.0 / 50.0
        n_js = int(duration / js_dt)
        drift_per_sample = 0.0001  # 100 microseconds drift per sample
        for i in range(n_js):
            t = base_time + i * js_dt + i * drift_per_sample
            sec, nanosec = _sec_nanosec(t)
            data = _build_joint_state_cdr(
                sec=sec, nanosec=nanosec, frame_id="",
                names=["joint1", "joint2"],
                positions=[math.sin(2 * math.pi * i / n_js), 0.0],
                velocities=[],
                efforts=[],
            )
            messages.append((_timestamp_ns(t), js_channel, data))

        # TF at 50 Hz
        tf_dt = 1.0 / 50.0
        n_tf = int(duration / tf_dt)
        for i in range(n_tf):
            t = base_time + i * tf_dt
            sec, nanosec = _sec_nanosec(t)
            angle = 2.0 * math.pi * i / n_tf
            qz = math.sin(angle / 2.0)
            qw = math.cos(angle / 2.0)
            data = _build_tf_message_cdr([{
                "sec": sec, "nanosec": nanosec,
                "parent_frame": "world", "child_frame": "base_link",
                "translation": (1.0, 0.0, 0.0),
                "rotation": (0.0, 0.0, qz, qw),
            }])
            messages.append((_timestamp_ns(t), tf_channel, data))

        # Sort by timestamp and write
        messages.sort(key=lambda x: x[0])
        for log_time, channel_id, data in messages:
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=data,
                publish_time=log_time,
            )

        writer.finish()

    return path


def generate_multi_episode_mcap(
    path: str | Path,
    num_episodes: int = 3,
    gap_seconds: float = 10.0,
) -> Path:
    """Generate a synthetic MCAP file with multiple episodes separated by gaps.

    Each episode has IMU and JointState data for 2 seconds, with
    gap_seconds of silence between episodes.
    """
    path = Path(path)
    base_time = 1700000000.0
    episode_duration = 2.0

    with open(path, "wb") as f:
        writer = Writer(f)
        writer.start()

        imu_schema_id = writer.register_schema(
            name="sensor_msgs/msg/Imu",
            encoding="ros2msg",
            data=IMU_SCHEMA.encode(),
        )
        js_schema_id = writer.register_schema(
            name="sensor_msgs/msg/JointState",
            encoding="ros2msg",
            data=JOINT_STATE_SCHEMA.encode(),
        )

        imu_channel = writer.register_channel(
            topic="/imu/data",
            message_encoding="cdr",
            schema_id=imu_schema_id,
        )
        js_channel = writer.register_channel(
            topic="/joint_states",
            message_encoding="cdr",
            schema_id=js_schema_id,
        )

        messages = []

        for ep in range(num_episodes):
            ep_start = base_time + ep * (episode_duration + gap_seconds)

            # IMU at 100 Hz
            imu_dt = 1.0 / 100.0
            n_imu = int(episode_duration / imu_dt)
            for i in range(n_imu):
                t = ep_start + i * imu_dt
                sec, nanosec = _sec_nanosec(t)
                data = _build_imu_cdr(
                    sec=sec, nanosec=nanosec, frame_id="imu_link",
                    orientation=(0.0, 0.0, 0.0, 1.0),
                    angular_velocity=(0.0, 0.0, 0.0),
                    linear_acceleration=(0.0, 0.0, 9.81),
                )
                messages.append((_timestamp_ns(t), imu_channel, data))

            # JointState at 50 Hz
            js_dt = 1.0 / 50.0
            n_js = int(episode_duration / js_dt)
            for i in range(n_js):
                t = ep_start + i * js_dt
                sec, nanosec = _sec_nanosec(t)
                data = _build_joint_state_cdr(
                    sec=sec, nanosec=nanosec, frame_id="",
                    names=["joint1"],
                    positions=[float(ep)],
                    velocities=[],
                    efforts=[],
                )
                messages.append((_timestamp_ns(t), js_channel, data))

        # Sort and write
        messages.sort(key=lambda x: x[0])
        for log_time, channel_id, data in messages:
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=data,
                publish_time=log_time,
            )

        writer.finish()

    return path
