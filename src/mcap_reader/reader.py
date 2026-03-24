"""Core MCAP file reader with ROS 2 message decoding.

MCAP Format Overview
====================
MCAP is a modular, performant, and self-contained container format for
heterogeneous timestamped data. It was designed by Foxglove as the successor
to ROS 1 bag files and is the default recording format for ROS 2 (Iron+).

An MCAP file is organized into the following structural elements:

  Magic | Header | [ Data Section ] | [ Summary Section ] | Footer | Magic

**Data Section** — a sequence of *Chunks*, each containing one or more
compressed *Messages*. Chunks may be individually compressed (lz4 / zstd)
so the reader only decompresses what it needs.

**Summary Section** — written at the end of the file after recording
completes. Contains:
  - *Statistics* — total message counts, channel counts, time range.
  - *Schema* records — serialized type definitions (e.g. ROS 2 .msg text).
  - *Channel* records — mapping from numeric channel IDs to topic names,
    message types, and which schema they reference.
  - *Chunk Index* records — byte offsets to each chunk for random access.

Every record carries a CRC-32 so readers can detect corruption without
reading the entire file.

Schemas vs. Messages
--------------------
A **Schema** is the type definition for a message — the textual `.msg`
description (for ROS 2) or a protobuf/flatbuffer definition. It tells the
deserializer how to interpret raw bytes.

A **Message** is a timestamped blob of serialized data belonging to a
particular *Channel* (topic). To decode a message you need its channel's
schema.

ROS 2 Topic Mapping
--------------------
In ROS 2, every *topic* has a fixed *message type* (e.g.
``/camera/image_raw`` carries ``sensor_msgs/msg/Image``). When rosbag2
writes an MCAP file it creates one Channel per topic, and one Schema per
unique message type. Multiple channels can share the same schema (e.g.
two image topics both use ``sensor_msgs/msg/Image``).

ROS 2 serializes messages using CDR (Common Data Representation) from the
DDS middleware layer. The ``mcap-ros2-support`` package provides a
``DecoderFactory`` that understands CDR encoding and can reconstruct
Python objects from the raw bytes.

header.stamp vs. log_time
--------------------------
MCAP messages carry two timestamps:

* **log_time** — the wall-clock time when the message was *recorded* by the
  logger. This is monotonically increasing and is used for ordering and
  seeking within the file.
* **publish_time** — typically the same as log_time for ROS 2 bags.

Many ROS 2 messages also contain a ``header.stamp`` field *inside* the
message payload. This is the **sensor time** — when the data was actually
captured by the hardware (camera shutter, LiDAR spin, IMU sample). It may
differ from log_time due to:

  - Network transport latency
  - Software pipeline buffering
  - Hardware clock drift

For time-synchronization across sensors you almost always want
``header.stamp``. For seeking within the file, use ``log_time``.

Why Iterate, Not Load
---------------------
Robot recordings routinely contain millions of messages totaling tens of
gigabytes. Loading everything into memory is impractical and unnecessary —
most analyses touch only a subset of topics or a time window. The iterator
pattern lets the caller:

  1. Filter by topic and time range at the read level (skipping chunks).
  2. Process messages one at a time with constant memory.
  3. Short-circuit early (e.g. ``itertools.islice``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

NANOSECONDS_PER_SECOND = 1e9


@dataclass(frozen=True, slots=True)
class TopicInfo:
    """Metadata about a single topic in the MCAP file.

    Attributes:
        name: The ROS 2 topic name (e.g. ``/camera/image_raw``).
        message_type: Fully qualified message type
            (e.g. ``sensor_msgs/msg/Image``).
        message_count: Number of messages recorded on this topic.
        frequency: Estimated publishing frequency in Hz, or ``None`` if the
            topic has fewer than two messages (frequency cannot be computed).
    """

    name: str
    message_type: str
    message_count: int
    frequency: float | None


@dataclass(frozen=True, slots=True)
class RawMessage:
    """A single decoded message from the MCAP file.

    Carries both the raw serialized bytes and the deserialized ROS 2 Python
    object so the caller can choose which to work with.

    Attributes:
        topic: The ROS 2 topic name this message was published on.
        timestamp: The ``header.stamp`` time in seconds if the message has
            a ``header`` field, otherwise falls back to ``log_time``.
        log_time: The recorder's wall-clock time in seconds — when the
            message was written to disk. Monotonically increasing.
        data: The raw CDR-serialized bytes of the message payload.
        ros_msg: The deserialized ROS 2 message as a Python object.
            The exact type depends on the message schema (e.g.
            ``sensor_msgs.msg.Image``).
        schema_name: The message type name
            (e.g. ``sensor_msgs/msg/Image``).
    """

    topic: str
    timestamp: float
    log_time: float
    data: bytes
    ros_msg: Any
    schema_name: str


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class McapReader:
    """High-level reader for MCAP files containing ROS 2 data.

    Opens an MCAP file, reads its summary section (channels, schemas,
    statistics), and provides convenient access to topic metadata and
    message iteration with ROS 2 CDR decoding.

    Usage::

        with McapReader("recording.mcap") as reader:
            print(reader.duration, "seconds")
            for msg in reader.iter_messages(topics=["/imu/data"]):
                print(msg.timestamp, msg.ros_msg.linear_acceleration)

    The reader uses the summary section for metadata queries (topics,
    duration, message count) without scanning the data section, making
    those operations O(1) regardless of file size.

    Parameters:
        path: Path to the ``.mcap`` file. Accepts a string or
            :class:`pathlib.Path`.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"MCAP file not found: {self._path}")
        if not self._path.suffix == ".mcap":
            raise ValueError(
                f"Expected an .mcap file, got: {self._path.suffix!r}"
            )

        self._file = open(self._path, "rb")  # noqa: SIM115
        self._reader = make_reader(self._file, decoder_factories=[DecoderFactory()])
        self._summary = self._reader.get_summary()

        if self._summary is None:
            raise ValueError(
                f"MCAP file has no summary section: {self._path}. "
                "The file may be truncated or was not finalized."
            )

        # Build lookup tables from the summary for fast access.
        self._schemas: dict[int, Any] = {
            sid: schema for sid, schema in self._summary.schemas.items()
        }
        self._channels: dict[int, Any] = {
            cid: channel for cid, channel in self._summary.channels.items()
        }
        self._statistics = self._summary.statistics

        # Map topic name -> channel id for quick lookups.
        self._topic_to_channel: dict[str, int] = {
            ch.topic: cid for cid, ch in self._channels.items()
        }

        logger.info(
            "Opened %s: %d topics, %d messages, %.1fs duration",
            self._path.name,
            len(self._channels),
            self.message_count,
            self.duration,
        )

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> McapReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file and not self._file.closed:
            self._file.close()

    # -- Properties ----------------------------------------------------------

    @property
    def topics(self) -> list[TopicInfo]:
        """All topics in the recording with message counts and frequencies.

        Returns a list of :class:`TopicInfo` dataclasses sorted by topic
        name. Frequency is estimated from the message count and recording
        duration; it is ``None`` for topics with fewer than 2 messages.
        """
        result: list[TopicInfo] = []
        channel_stats: dict[int, int] = {}
        if self._statistics and self._statistics.channel_message_counts:
            channel_stats = dict(self._statistics.channel_message_counts)

        duration = self.duration

        for cid, channel in self._channels.items():
            count = channel_stats.get(cid, 0)
            schema = self._schemas.get(channel.schema_id)
            message_type = schema.name if schema else ""

            if count >= 2 and duration > 0:
                frequency = round((count - 1) / duration, 2)
            else:
                frequency = None

            result.append(
                TopicInfo(
                    name=channel.topic,
                    message_type=message_type,
                    message_count=count,
                    frequency=frequency,
                )
            )

        result.sort(key=lambda t: t.name)
        return result

    @property
    def duration(self) -> float:
        """Total recording duration in seconds.

        Computed from the difference between the last and first ``log_time``
        values in the statistics summary. Returns ``0.0`` if the file
        contains no messages.
        """
        if not self._statistics:
            return 0.0
        start_ns = self._statistics.message_start_time
        end_ns = self._statistics.message_end_time
        if start_ns == 0 and end_ns == 0:
            return 0.0
        return (end_ns - start_ns) / NANOSECONDS_PER_SECOND

    @property
    def start_time(self) -> float:
        """Timestamp of the first recorded message in seconds (from log_time).

        This is the wall-clock time when recording began, converted from
        MCAP's native nanosecond representation to floating-point seconds.
        """
        if not self._statistics:
            return 0.0
        return self._statistics.message_start_time / NANOSECONDS_PER_SECOND

    @property
    def end_time(self) -> float:
        """Timestamp of the last recorded message in seconds (from log_time).

        This is the wall-clock time when the final message was written,
        converted from MCAP's native nanosecond representation to
        floating-point seconds.
        """
        if not self._statistics:
            return 0.0
        return self._statistics.message_end_time / NANOSECONDS_PER_SECOND

    @property
    def message_count(self) -> int:
        """Total number of messages across all topics."""
        if not self._statistics:
            return 0
        return self._statistics.message_count

    @property
    def topic_names(self) -> list[str]:
        """Sorted list of all topic names in the recording."""
        return sorted(ch.topic for ch in self._channels.values())

    # -- Message iteration ---------------------------------------------------

    def iter_messages(
        self,
        topics: list[str] | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> Iterator[RawMessage]:
        """Iterate over decoded messages, optionally filtered by topic and time.

        Messages are yielded in ``log_time`` order. Each yielded
        :class:`RawMessage` contains both the raw CDR bytes and the
        deserialized ROS 2 Python object.

        Parameters:
            topics: If provided, only yield messages from these topic names.
                If ``None``, yield messages from all topics.
            start_time: If provided, only yield messages with
                ``log_time >= start_time`` (in seconds).
            end_time: If provided, only yield messages with
                ``log_time <= end_time`` (in seconds).

        Yields:
            :class:`RawMessage` instances in chronological order.

        Note:
            The MCAP library applies topic and time filters at the chunk
            level when possible, skipping entire chunks that fall outside
            the requested range. This makes filtered iteration significantly
            faster than reading everything and discarding in Python.
        """
        # Convert seconds back to nanoseconds for the MCAP reader API.
        start_ns = int(start_time * NANOSECONDS_PER_SECOND) if start_time is not None else None
        end_ns = int(end_time * NANOSECONDS_PER_SECOND) if end_time is not None else None

        for schema, channel, message, ros_msg in self._reader.iter_decoded_messages(
            topics=topics,
            start_time=start_ns,
            end_time=end_ns,
        ):
            log_time_sec = message.log_time / NANOSECONDS_PER_SECOND

            # Try to extract header.stamp from the ROS message for the
            # sensor timestamp. Fall back to log_time if unavailable.
            timestamp = _extract_header_stamp(ros_msg, fallback=log_time_sec)

            schema_name = schema.name if schema else ""

            yield RawMessage(
                topic=channel.topic,
                timestamp=timestamp,
                log_time=log_time_sec,
                data=message.data,
                ros_msg=ros_msg,
                schema_name=schema_name,
            )

    # -- Schema access -------------------------------------------------------

    def get_schema(self, topic: str) -> str:
        """Get the message definition (schema text) for a topic.

        For ROS 2 messages this is the ``.msg`` file content that defines
        the message fields and types.

        Parameters:
            topic: The topic name (e.g. ``/camera/image_raw``).

        Returns:
            The schema definition as a string.

        Raises:
            KeyError: If the topic is not found in the recording.
            ValueError: If the topic has no associated schema.
        """
        cid = self._topic_to_channel.get(topic)
        if cid is None:
            raise KeyError(
                f"Topic {topic!r} not found. "
                f"Available topics: {self.topic_names}"
            )
        channel = self._channels[cid]
        schema = self._schemas.get(channel.schema_id)
        if schema is None:
            raise ValueError(f"No schema found for topic {topic!r}")
        return schema.data.decode("utf-8", errors="replace")

    # -- Summary -------------------------------------------------------------

    def summary(self) -> dict:
        """Return a structured summary of the MCAP file.

        Returns:
            A dictionary with the following keys:

            - ``file``: File name.
            - ``duration_seconds``: Recording duration in seconds.
            - ``start_time``: First message timestamp in seconds.
            - ``end_time``: Last message timestamp in seconds.
            - ``message_count``: Total messages across all topics.
            - ``topics``: List of dicts, one per topic, each containing
              ``name``, ``message_type``, ``message_count``, and
              ``frequency``.
        """
        return {
            "file": self._path.name,
            "duration_seconds": round(self.duration, 3),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "message_count": self.message_count,
            "topics": [
                {
                    "name": t.name,
                    "message_type": t.message_type,
                    "message_count": t.message_count,
                    "frequency": t.frequency,
                }
                for t in self.topics
            ],
        }

    # -- Topic pairing -------------------------------------------------------

    def find_paired_topic(self, topic: str, suffix: str) -> str | None:
        """Find a companion topic that shares the same prefix.

        Many ROS 2 sensor drivers publish related data on paired topics
        that share a common prefix. For example, a camera node might
        publish:

        - ``/camera/image_raw`` — the image data
        - ``/camera/camera_info`` — the camera calibration parameters

        This method strips the last path component from *topic* and
        searches for another topic ending with *suffix* under the same
        namespace.

        Parameters:
            topic: The reference topic (e.g. ``/camera/image_raw``).
            suffix: The suffix to search for (e.g. ``camera_info``).

        Returns:
            The matched topic name, or ``None`` if no match is found.

        Example::

            >>> reader.find_paired_topic("/front_camera/image_raw", "camera_info")
            '/front_camera/camera_info'
        """
        # Strip the last component to get the namespace prefix.
        if "/" not in topic:
            return None
        prefix = topic.rsplit("/", 1)[0]

        # Normalize suffix: ensure it does not start with '/'.
        suffix = suffix.lstrip("/")

        candidate = f"{prefix}/{suffix}"
        if candidate in self._topic_to_channel:
            return candidate

        # Broader search: look for any topic under the prefix ending with
        # the suffix.
        for name in self.topic_names:
            if name.startswith(prefix + "/") and name.endswith(suffix):
                return name

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_header_stamp(ros_msg: Any, fallback: float) -> float:
    """Extract ``header.stamp`` from a ROS 2 message, if present.

    ROS 2 ``std_msgs/msg/Header`` stores time as two integers:
    ``stamp.sec`` (seconds since epoch) and ``stamp.nanosec`` (nanosecond
    fraction). This function combines them into a single float.

    Parameters:
        ros_msg: The deserialized ROS 2 message object.
        fallback: Value to return if the message has no header or stamp.

    Returns:
        Sensor timestamp in seconds, or *fallback* if unavailable.
    """
    try:
        header = ros_msg.header
        stamp = header.stamp
        return stamp.sec + stamp.nanosec / NANOSECONDS_PER_SECOND
    except AttributeError:
        return fallback
