"""
Wrapper for sensor_msgs/msg/JointState — robot joint positions, velocities, and efforts.

ROS 2 Message Definition
=========================
The ``sensor_msgs/msg/JointState`` message reports the state of a set of
joints (revolute, prismatic, or other). The full IDL is::

    std_msgs/msg/Header header

    string[] name
    float64[] position
    float64[] velocity
    float64[] effort

CDR Byte Layout
===============
After the 4-byte encapsulation header:

  1. header (stamp.sec uint32, stamp.nanosec uint32, frame_id string)
  2. name — CDR sequence of strings:
       [uint32 count] then for each element: [uint32 str_len] [chars + null]
  3. position — CDR sequence of float64:
       [uint32 count] [count * float64]
  4. velocity — CDR sequence of float64:
       [uint32 count] [count * float64]
  5. effort — CDR sequence of float64:
       [uint32 count] [count * float64]

Each sequence is preceded by a uint32 length. The float64 elements within
each sequence require 8-byte alignment.

Learning Notes
==============
* **Parallel arrays:** The four arrays (name, position, velocity, effort)
  are parallel — ``name[i]`` corresponds to ``position[i]``, ``velocity[i]``,
  and ``effort[i]``. The arrays MUST be the same length, or any of position,
  velocity, or effort may be **empty** (length 0) if that data is not
  available.

* **Empty arrays:** It is common for a joint state publisher to omit
  velocity and/or effort entirely (empty arrays). For example, a position-
  only joint encoder will publish ``name`` and ``position`` but leave
  ``velocity`` and ``effort`` as zero-length sequences. Always check
  array length before indexing.

* **Joint ordering:** The order of joints in the ``name`` array is set by
  the publisher and may not match the URDF joint ordering. Never assume a
  fixed index — always look up joints by name.

* **Units:** Position is in **radians** for revolute joints and **meters**
  for prismatic joints. Velocity is in **rad/s** or **m/s** respectively.
  Effort is in **N*m** (torque) for revolute and **N** (force) for prismatic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mcap_reader.deserializer import CdrDeserializer, Header, deserialize_header


@dataclass
class JointState:
    """Deserialized sensor_msgs/msg/JointState message.

    Attributes
    ----------
    header : Header
        Timestamp and coordinate frame from std_msgs/Header.
    name : list[str]
        Joint names. The ordering matches position/velocity/effort arrays.
    position : np.ndarray
        Joint positions in radians (revolute) or meters (prismatic).
        May be empty (length 0) if not provided.
    velocity : np.ndarray
        Joint velocities in rad/s or m/s. May be empty.
    effort : np.ndarray
        Joint efforts (torque in N*m or force in N). May be empty.
    """

    header: Header
    name: list[str]
    position: np.ndarray
    velocity: np.ndarray
    effort: np.ndarray

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ros_msg(cls, msg: Any) -> JointState:
        """Create a JointState wrapper from a decoded mcap-ros2-support message.

        Parameters
        ----------
        msg : Any
            A decoded ``sensor_msgs/msg/JointState`` object.

        Returns
        -------
        JointState
            The wrapped message with numpy arrays.
        """
        header = Header(
            sec=msg.header.stamp.sec,
            nanosec=msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
        )
        return cls(
            header=header,
            name=list(msg.name),
            position=np.array(msg.position, dtype=np.float64),
            velocity=np.array(msg.velocity, dtype=np.float64),
            effort=np.array(msg.effort, dtype=np.float64),
        )

    @classmethod
    def from_cdr(cls, data: bytes) -> JointState:
        """Deserialize a JointState from raw CDR bytes.

        Reads the four parallel arrays in order: name (sequence of strings),
        position, velocity, effort (each a sequence of float64).

        Parameters
        ----------
        data : bytes
            Raw CDR payload including the 4-byte encapsulation header.

        Returns
        -------
        JointState
            The deserialized message.
        """
        cdr = CdrDeserializer(data)
        header = deserialize_header(cdr)

        # name: sequence<string>
        name = cdr.read_sequence(cdr.read_string)

        # position: sequence<float64>
        position = np.array(
            cdr.read_sequence(cdr.read_float64), dtype=np.float64
        )

        # velocity: sequence<float64>
        velocity = np.array(
            cdr.read_sequence(cdr.read_float64), dtype=np.float64
        )

        # effort: sequence<float64>
        effort = np.array(
            cdr.read_sequence(cdr.read_float64), dtype=np.float64
        )

        return cls(
            header=header,
            name=name,
            position=position,
            velocity=velocity,
            effort=effort,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_position(self) -> bool:
        """Whether position data is provided (non-empty array).

        Returns
        -------
        bool
            True if the position array has elements.
        """
        return len(self.position) > 0

    @property
    def has_velocity(self) -> bool:
        """Whether velocity data is provided (non-empty array).

        Returns
        -------
        bool
            True if the velocity array has elements.
        """
        return len(self.velocity) > 0

    @property
    def has_effort(self) -> bool:
        """Whether effort data is provided (non-empty array).

        Returns
        -------
        bool
            True if the effort array has elements.
        """
        return len(self.effort) > 0

    # ------------------------------------------------------------------
    # Joint access
    # ------------------------------------------------------------------

    def get_joint(self, name: str) -> dict:
        """Get position, velocity, and effort for a named joint.

        Looks up the joint by name in the parallel arrays and returns a
        dictionary with the available values. Missing arrays (empty) result
        in ``None`` for that field.

        Parameters
        ----------
        name : str
            The joint name to look up (e.g., ``"shoulder_pan_joint"``).

        Returns
        -------
        dict
            Keys: ``"position"``, ``"velocity"``, ``"effort"``, each
            either a float or None if the corresponding array is empty.

        Raises
        ------
        KeyError
            If the joint name is not found in the ``name`` list.
        """
        if name not in self.name:
            raise KeyError(
                f"Joint {name!r} not found. Available joints: {self.name}"
            )
        idx = self.name.index(name)
        return {
            "position": float(self.position[idx]) if self.has_position else None,
            "velocity": float(self.velocity[idx]) if self.has_velocity else None,
            "effort": float(self.effort[idx]) if self.has_effort else None,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict]:
        """Convert to a dictionary keyed by joint name.

        Each value is a dict with ``"position"``, ``"velocity"``, and
        ``"effort"`` fields. This format is convenient for JSON export
        and programmatic access.

        Returns
        -------
        dict[str, dict]
            Mapping from joint name to its state values.
        """
        result: dict[str, dict] = {}
        for i, joint_name in enumerate(self.name):
            result[joint_name] = {
                "position": float(self.position[i]) if self.has_position else None,
                "velocity": float(self.velocity[i]) if self.has_velocity else None,
                "effort": float(self.effort[i]) if self.has_effort else None,
            }
        return result

    def to_pandas_row(self) -> dict:
        """Flatten into a single-row dict for DataFrame construction.

        Produces columns like ``position_shoulder_pan_joint``,
        ``velocity_shoulder_pan_joint``, etc. Joint names with spaces or
        special characters are preserved as-is in column names.

        Returns
        -------
        dict
            Flat key-value pairs suitable for ``pd.DataFrame([row1, row2, ...])``.
        """
        row: dict = {
            "timestamp": self.header.to_timestamp(),
            "frame_id": self.header.frame_id,
        }
        for i, joint_name in enumerate(self.name):
            if self.has_position:
                row[f"position_{joint_name}"] = float(self.position[i])
            if self.has_velocity:
                row[f"velocity_{joint_name}"] = float(self.velocity[i])
            if self.has_effort:
                row[f"effort_{joint_name}"] = float(self.effort[i])
        return row
