"""
Wrapper for sensor_msgs/msg/Imu — Inertial Measurement Unit data.

ROS 2 Message Definition
=========================
The ``sensor_msgs/msg/Imu`` message carries orientation, angular velocity,
and linear acceleration from an IMU sensor. The full IDL is::

    std_msgs/msg/Header header

    geometry_msgs/msg/Quaternion orientation
    float64[9] orientation_covariance

    geometry_msgs/msg/Vector3 angular_velocity
    float64[9] angular_velocity_covariance

    geometry_msgs/msg/Vector3 linear_acceleration
    float64[9] linear_acceleration_covariance

CDR Byte Layout
===============
After the 4-byte encapsulation header, the fields are serialized in
declaration order:

  1. header (stamp.sec uint32, stamp.nanosec uint32, frame_id string)
  2. orientation (x, y, z, w as four float64, each 8-byte aligned)
  3. orientation_covariance (9 consecutive float64 = 72 bytes)
  4. angular_velocity (x, y, z as three float64)
  5. angular_velocity_covariance (9 float64)
  6. linear_acceleration (x, y, z as three float64)
  7. linear_acceleration_covariance (9 float64)

All float64 fields require 8-byte alignment. After the variable-length
string in the header, the deserializer must align to 8 before reading
the first quaternion component.

Learning Notes
==============
* **Orientation frame:** The orientation quaternion describes the sensor's
  rotation relative to a fixed reference frame (often ENU — East-North-Up
  or the frame specified in ``header.frame_id``). It is in the **sensor's
  local frame**, not the world frame. The reference frame convention varies
  by driver — always check the driver documentation.

* **Covariance "not provided" sentinel:** If a sensor does not provide a
  measurement (e.g., an accelerometer-only IMU has no orientation), the
  first element of the corresponding covariance array is set to **-1**.
  This is the official ROS convention documented in the message definition.
  Checking ``covariance[0, 0] == -1`` tells you whether the field is valid.

* **Covariance layout:** The 9-element array is a row-major 3x3 covariance
  matrix. For orientation it corresponds to (roll, pitch, yaw) uncertainty;
  for angular velocity and linear acceleration it is (x, y, z).

* **Units:** Angular velocity is in **rad/s**, linear acceleration is in
  **m/s^2**, and orientation is a unit quaternion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mcap_reader.deserializer import CdrDeserializer, Header, deserialize_header
from mcap_reader.transforms.math import Quaternion, Vector3


@dataclass
class Imu:
    """Deserialized sensor_msgs/msg/Imu message.

    Attributes
    ----------
    header : Header
        Timestamp and coordinate frame from std_msgs/Header.
    orientation : Quaternion
        Orientation of the sensor in quaternion form (x, y, z, w).
        May be invalid — check :meth:`has_orientation`.
    orientation_covariance : np.ndarray
        3x3 row-major covariance matrix for orientation (roll, pitch, yaw).
        If ``[0, 0] == -1``, orientation is not provided by this sensor.
    angular_velocity : Vector3
        Angular velocity about x, y, z axes in rad/s.
    angular_velocity_covariance : np.ndarray
        3x3 covariance matrix for angular velocity.
    linear_acceleration : Vector3
        Linear acceleration along x, y, z axes in m/s^2.
    linear_acceleration_covariance : np.ndarray
        3x3 covariance matrix for linear acceleration.
    """

    header: Header
    orientation: Quaternion
    orientation_covariance: np.ndarray
    angular_velocity: Vector3
    angular_velocity_covariance: np.ndarray
    linear_acceleration: Vector3
    linear_acceleration_covariance: np.ndarray

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ros_msg(cls, msg: Any) -> Imu:
        """Create an Imu wrapper from a decoded mcap-ros2-support message.

        The mcap-ros2-support ``DecoderFactory`` produces Python objects whose
        attribute names mirror the ROS 2 message fields. This constructor
        extracts those attributes and wraps them in our typed dataclasses.

        Parameters
        ----------
        msg : Any
            A decoded ``sensor_msgs/msg/Imu`` object from mcap-ros2-support.

        Returns
        -------
        Imu
            The wrapped message with numpy covariance matrices.
        """
        header = Header(
            sec=msg.header.stamp.sec,
            nanosec=msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
        )
        orientation = Quaternion(
            x=msg.orientation.x,
            y=msg.orientation.y,
            z=msg.orientation.z,
            w=msg.orientation.w,
        )
        angular_velocity = Vector3(
            x=msg.angular_velocity.x,
            y=msg.angular_velocity.y,
            z=msg.angular_velocity.z,
        )
        linear_acceleration = Vector3(
            x=msg.linear_acceleration.x,
            y=msg.linear_acceleration.y,
            z=msg.linear_acceleration.z,
        )
        return cls(
            header=header,
            orientation=orientation,
            orientation_covariance=np.array(
                msg.orientation_covariance, dtype=np.float64
            ).reshape(3, 3),
            angular_velocity=angular_velocity,
            angular_velocity_covariance=np.array(
                msg.angular_velocity_covariance, dtype=np.float64
            ).reshape(3, 3),
            linear_acceleration=linear_acceleration,
            linear_acceleration_covariance=np.array(
                msg.linear_acceleration_covariance, dtype=np.float64
            ).reshape(3, 3),
        )

    @classmethod
    def from_cdr(cls, data: bytes) -> Imu:
        """Deserialize an Imu from raw CDR bytes.

        Reads fields in the exact order they appear in the CDR wire format:
        header, orientation quaternion (x, y, z, w), orientation covariance,
        angular velocity, angular velocity covariance, linear acceleration,
        linear acceleration covariance.

        Parameters
        ----------
        data : bytes
            Raw CDR payload including the 4-byte encapsulation header.

        Returns
        -------
        Imu
            The deserialized message.
        """
        cdr = CdrDeserializer(data)
        header = deserialize_header(cdr)

        # orientation: geometry_msgs/Quaternion (x, y, z, w as float64)
        orientation = Quaternion(
            x=cdr.read_float64(),
            y=cdr.read_float64(),
            z=cdr.read_float64(),
            w=cdr.read_float64(),
        )

        # orientation_covariance: float64[9] fixed-size array
        orientation_cov = np.array(
            [cdr.read_float64() for _ in range(9)], dtype=np.float64
        ).reshape(3, 3)

        # angular_velocity: geometry_msgs/Vector3 (x, y, z as float64)
        angular_velocity = Vector3(
            x=cdr.read_float64(),
            y=cdr.read_float64(),
            z=cdr.read_float64(),
        )

        angular_velocity_cov = np.array(
            [cdr.read_float64() for _ in range(9)], dtype=np.float64
        ).reshape(3, 3)

        # linear_acceleration: geometry_msgs/Vector3
        linear_acceleration = Vector3(
            x=cdr.read_float64(),
            y=cdr.read_float64(),
            z=cdr.read_float64(),
        )

        linear_acceleration_cov = np.array(
            [cdr.read_float64() for _ in range(9)], dtype=np.float64
        ).reshape(3, 3)

        return cls(
            header=header,
            orientation=orientation,
            orientation_covariance=orientation_cov,
            angular_velocity=angular_velocity,
            angular_velocity_covariance=angular_velocity_cov,
            linear_acceleration=linear_acceleration,
            linear_acceleration_covariance=linear_acceleration_cov,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_orientation(self) -> bool:
        """Whether the IMU provides orientation data.

        Per the ROS 2 convention, if ``orientation_covariance[0, 0] == -1``
        the orientation field is not populated by the sensor driver and
        should be ignored. Many IMUs (e.g., accelerometer-only units) do
        not estimate orientation.

        Returns
        -------
        bool
            True if orientation data is valid.
        """
        return self.orientation_covariance[0, 0] != -1

    @property
    def has_angular_velocity(self) -> bool:
        """Whether the IMU provides angular velocity data.

        Same sentinel convention: ``angular_velocity_covariance[0, 0] == -1``
        means the angular velocity field is not provided.

        Returns
        -------
        bool
            True if angular velocity data is valid.
        """
        return self.angular_velocity_covariance[0, 0] != -1

    @property
    def has_linear_acceleration(self) -> bool:
        """Whether the IMU provides linear acceleration data.

        Same sentinel convention: ``linear_acceleration_covariance[0, 0] == -1``
        means the linear acceleration field is not provided.

        Returns
        -------
        bool
            True if linear acceleration data is valid.
        """
        return self.linear_acceleration_covariance[0, 0] != -1

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a nested dictionary suitable for JSON serialization.

        Returns
        -------
        dict
            Contains ``header``, ``orientation``, ``angular_velocity``,
            ``linear_acceleration``, and their covariance matrices
            (as nested lists).
        """
        return {
            "header": {
                "sec": self.header.sec,
                "nanosec": self.header.nanosec,
                "frame_id": self.header.frame_id,
            },
            "orientation": {
                "x": self.orientation.x,
                "y": self.orientation.y,
                "z": self.orientation.z,
                "w": self.orientation.w,
            },
            "orientation_covariance": self.orientation_covariance.tolist(),
            "angular_velocity": {
                "x": self.angular_velocity.x,
                "y": self.angular_velocity.y,
                "z": self.angular_velocity.z,
            },
            "angular_velocity_covariance": self.angular_velocity_covariance.tolist(),
            "linear_acceleration": {
                "x": self.linear_acceleration.x,
                "y": self.linear_acceleration.y,
                "z": self.linear_acceleration.z,
            },
            "linear_acceleration_covariance": self.linear_acceleration_covariance.tolist(),
        }

    def to_pandas_row(self) -> dict:
        """Flatten into a single-row dict for DataFrame construction.

        Produces a flat dictionary with keys like ``orientation_x``,
        ``angular_velocity_z``, ``linear_acceleration_x``, etc. This
        format is ideal for appending rows to a pandas DataFrame for
        time-series analysis.

        Returns
        -------
        dict
            Flat key-value pairs suitable for ``pd.DataFrame([row1, row2, ...])``.
        """
        return {
            "timestamp": self.header.to_timestamp(),
            "frame_id": self.header.frame_id,
            "orientation_x": self.orientation.x,
            "orientation_y": self.orientation.y,
            "orientation_z": self.orientation.z,
            "orientation_w": self.orientation.w,
            "angular_velocity_x": self.angular_velocity.x,
            "angular_velocity_y": self.angular_velocity.y,
            "angular_velocity_z": self.angular_velocity.z,
            "linear_acceleration_x": self.linear_acceleration.x,
            "linear_acceleration_y": self.linear_acceleration.y,
            "linear_acceleration_z": self.linear_acceleration.z,
        }
