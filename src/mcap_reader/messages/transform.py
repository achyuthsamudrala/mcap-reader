"""
Wrapper for tf2_msgs/msg/TFMessage — coordinate frame transform broadcasts.

ROS 2 Message Definition
=========================
The ``tf2_msgs/msg/TFMessage`` message carries a list of coordinate frame
transforms. Each element is a ``geometry_msgs/msg/TransformStamped``. The
full IDL is::

    geometry_msgs/TransformStamped[] transforms

Where ``TransformStamped`` is::

    std_msgs/msg/Header header
    string child_frame_id
    geometry_msgs/msg/Transform transform

And ``geometry_msgs/msg/Transform`` is::

    geometry_msgs/msg/Vector3 translation
    geometry_msgs/msg/Quaternion rotation

CDR Byte Layout
===============
After the 4-byte encapsulation header:

  1. transforms — CDR sequence of TransformStamped:
       [uint32 count] then for each element:
         a. header (stamp.sec uint32, stamp.nanosec uint32, frame_id string)
         b. child_frame_id (CDR string)
         c. transform.translation (x, y, z as float64)
         d. transform.rotation (x, y, z, w as float64)

All float64 fields require 8-byte alignment. After each variable-length
string, the deserializer must align to 8 before reading the float64 fields.

Learning Notes
==============
* **/tf vs /tf_static:** ROS 2 has two TF topics:

    - ``/tf`` — dynamic transforms that change over time (e.g., joint
      positions, odometry). Published at high frequency (10-1000 Hz).
    - ``/tf_static`` — static transforms that never change (e.g., sensor
      mount offsets). Published once with a latched/transient-local QoS.

  Both topics carry ``tf2_msgs/msg/TFMessage`` messages. The difference is
  purely semantic and in the QoS settings:
    - ``/tf`` uses default (volatile) durability — new subscribers only see
      messages published after they connect.
    - ``/tf_static`` uses TRANSIENT_LOCAL durability — the last published
      message is replayed to new subscribers immediately. This ensures
      static transforms are always available.

  In MCAP files, ``/tf_static`` typically has only one or a few messages
  (all published at startup), while ``/tf`` has thousands or millions.

* **Transform tree:** The collection of all transforms forms a directed
  tree (forest). Each TransformStamped defines an edge from
  ``header.frame_id`` (parent) to ``child_frame_id`` (child). The TF2
  library looks up chains of transforms to convert coordinates between
  any two frames in the tree.

  Example tree::

      map -> odom -> base_link -> camera_link -> camera_optical

  To transform a point from ``camera_optical`` to ``map``, TF2 chains
  all four intermediate transforms.

* **Timestamp semantics:** The ``header.stamp`` on each TransformStamped
  indicates the time at which the transform is valid. For dynamic
  transforms, TF2 interpolates between the two closest timestamps to
  answer queries at arbitrary times. For static transforms, the timestamp
  is typically zero (meaning "always valid").

* **Frame naming conventions:** ROS 2 frame IDs follow conventions:
    - No leading slash (unlike ROS 1): ``"base_link"`` not ``"/base_link"``
    - ``*_link`` for rigid body frames
    - ``*_optical`` for camera optical frames (Z-forward, X-right, Y-down)
    - ``"map"`` for the global fixed frame
    - ``"odom"`` for the odometry frame (may drift over time)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcap_reader.deserializer import CdrDeserializer, Header, deserialize_header
from mcap_reader.transforms.math import Quaternion, Transform, Vector3


@dataclass
class TransformStamped:
    """A single timestamped coordinate frame transform.

    Represents the transform from ``header.frame_id`` (parent) to
    ``child_frame_id`` (child). Applying this transform to a point
    expressed in the child frame yields the point in the parent frame.

    Attributes
    ----------
    header : Header
        Timestamp and parent frame ID from std_msgs/Header.
        ``header.frame_id`` is the parent (target) frame.
    child_frame_id : str
        The child (source) frame ID.
    transform : Transform
        The rigid-body transform (translation + rotation) from parent
        to child frame.
    """

    header: Header
    child_frame_id: str
    transform: Transform


@dataclass
class TFMessage:
    """Deserialized tf2_msgs/msg/TFMessage — a batch of transforms.

    A TFMessage carries one or more TransformStamped entries. A single
    message often contains transforms for multiple joints or links that
    were all computed at the same time.

    Attributes
    ----------
    transforms : list[TransformStamped]
        The list of coordinate frame transforms in this message.
    """

    transforms: list[TransformStamped]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ros_msg(cls, msg: Any) -> TFMessage:
        """Create a TFMessage wrapper from a decoded mcap-ros2-support message.

        The mcap-ros2-support decoder produces an object with a ``transforms``
        attribute that is a list of TransformStamped-like objects, each with
        ``header``, ``child_frame_id``, and ``transform`` sub-objects.

        Parameters
        ----------
        msg : Any
            A decoded ``tf2_msgs/msg/TFMessage`` object.

        Returns
        -------
        TFMessage
            The wrapped message with our typed dataclasses.
        """
        transforms: list[TransformStamped] = []
        for t in msg.transforms:
            header = Header(
                sec=t.header.stamp.sec,
                nanosec=t.header.stamp.nanosec,
                frame_id=t.header.frame_id,
            )
            translation = Vector3(
                x=t.transform.translation.x,
                y=t.transform.translation.y,
                z=t.transform.translation.z,
            )
            rotation = Quaternion(
                x=t.transform.rotation.x,
                y=t.transform.rotation.y,
                z=t.transform.rotation.z,
                w=t.transform.rotation.w,
            )
            transforms.append(TransformStamped(
                header=header,
                child_frame_id=t.child_frame_id,
                transform=Transform(translation=translation, rotation=rotation),
            ))
        return cls(transforms=transforms)

    @classmethod
    def from_cdr(cls, data: bytes) -> TFMessage:
        """Deserialize a TFMessage from raw CDR bytes.

        The CDR layout is a single sequence of TransformStamped structs.
        Each struct contains a header, child_frame_id string, and a
        Transform (translation Vector3 + rotation Quaternion).

        Parameters
        ----------
        data : bytes
            Raw CDR payload including the 4-byte encapsulation header.

        Returns
        -------
        TFMessage
            The deserialized message.
        """
        cdr = CdrDeserializer(data)

        # transforms: sequence<TransformStamped>
        count = cdr.read_uint32()
        transforms: list[TransformStamped] = []

        for _ in range(count):
            # header: std_msgs/Header
            header = deserialize_header(cdr)

            # child_frame_id: string
            child_frame_id = cdr.read_string()

            # transform.translation: geometry_msgs/Vector3 (3x float64)
            translation = Vector3(
                x=cdr.read_float64(),
                y=cdr.read_float64(),
                z=cdr.read_float64(),
            )

            # transform.rotation: geometry_msgs/Quaternion (x, y, z, w as float64)
            rotation = Quaternion(
                x=cdr.read_float64(),
                y=cdr.read_float64(),
                z=cdr.read_float64(),
                w=cdr.read_float64(),
            )

            transforms.append(TransformStamped(
                header=header,
                child_frame_id=child_frame_id,
                transform=Transform(translation=translation, rotation=rotation),
            ))

        return cls(transforms=transforms)
