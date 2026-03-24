"""
Wrapper for sensor_msgs/msg/CameraInfo — camera calibration and projection parameters.

ROS 2 Message Definition
=========================
The ``sensor_msgs/msg/CameraInfo`` message carries intrinsic calibration,
distortion coefficients, and projection matrices for a camera. The full
IDL is::

    std_msgs/msg/Header header

    uint32 height
    uint32 width

    string distortion_model

    float64[] d
    float64[9] k
    float64[9] r
    float64[12] p

    uint32 binning_x
    uint32 binning_y

    sensor_msgs/RegionOfInterest roi

CDR Byte Layout
===============
After the 4-byte encapsulation header:

  1. header (stamp.sec uint32, stamp.nanosec uint32, frame_id string)
  2. height (uint32)
  3. width (uint32)
  4. distortion_model (CDR string)
  5. D (CDR sequence of float64: variable-length distortion coefficients)
  6. K (9 float64 values: 3x3 intrinsic matrix, fixed-size array)
  7. R (9 float64 values: 3x3 rectification matrix, fixed-size)
  8. P (12 float64 values: 3x4 projection matrix, fixed-size)
  9. binning_x (uint32)
  10. binning_y (uint32)
  11. roi (RegionOfInterest: 4x uint32 + 1x bool)

Learning Notes
==============
* **CameraInfo as a sibling topic:** CameraInfo is almost always published
  as a companion to an Image or CompressedImage topic, sharing the same
  namespace prefix. For example:
    - ``/camera/image_raw`` (the image data)
    - ``/camera/camera_info`` (calibration for that camera)

  The ``header.stamp`` of the CameraInfo message matches the corresponding
  Image message, enabling time-synchronized lookup. The
  ``McapReader.find_paired_topic()`` method can discover these pairings
  automatically.

* **Intrinsic matrix K (3x3):** The camera intrinsic matrix maps 3D camera
  coordinates to 2D pixel coordinates::

      K = | fx  0  cx |
          |  0  fy cy |
          |  0   0  1 |

  Where:
    - ``fx``, ``fy`` = focal lengths in pixels
    - ``cx``, ``cy`` = principal point (optical center) in pixels

  For a monocular camera, K is all you need for projection (plus distortion).

* **Distortion coefficients D:** The distortion model (usually ``"plumb_bob"``
  for standard radial-tangential, or ``"rational_polynomial"`` for fisheye)
  specifies how to interpret D. For ``plumb_bob``, D has 5 elements:
  ``[k1, k2, t1, t2, k3]`` where k1-k3 are radial and t1-t2 are tangential
  distortion coefficients. D is a **variable-length** array because different
  distortion models require different numbers of coefficients.

* **Rectification matrix R (3x3):** For monocular cameras this is the
  identity matrix. For stereo pairs, R is the rotation matrix that aligns
  the camera to the rectified (epipolar-aligned) coordinate system. After
  rectification, corresponding points in left/right images lie on the same
  pixel row, which simplifies stereo matching.

* **Projection matrix P (3x4):** The full projection matrix from 3D
  rectified camera coordinates to 2D pixel coordinates::

      P = | fx'  0   cx' Tx |
          |  0  fy'  cy' Ty |
          |  0   0    1   0 |

  For monocular cameras, ``Tx = Ty = 0`` and ``fx' = fx``, so P is just K
  with a zero column appended. For stereo, ``Tx = -fx' * baseline`` encodes
  the stereo baseline.

* **OpenCV compatibility:** The K, D, R, P matrices map directly to
  OpenCV's camera calibration functions (``cv2.undistort``,
  ``cv2.initUndistortRectifyMap``, ``cv2.projectPoints``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mcap_reader.deserializer import CdrDeserializer, Header, deserialize_header


@dataclass
class CameraInfo:
    """Deserialized sensor_msgs/msg/CameraInfo message.

    Attributes
    ----------
    header : Header
        Timestamp and coordinate frame from std_msgs/Header.
    height : int
        Image height in pixels.
    width : int
        Image width in pixels.
    distortion_model : str
        Distortion model name (e.g., ``"plumb_bob"``, ``"rational_polynomial"``).
    D : np.ndarray
        Distortion coefficients. Length depends on the distortion model
        (typically 5 for plumb_bob).
    K : np.ndarray
        3x3 intrinsic camera matrix (row-major).
    R : np.ndarray
        3x3 rectification matrix. Identity for monocular cameras.
    P : np.ndarray
        3x4 projection matrix.
    """

    header: Header
    height: int
    width: int
    distortion_model: str
    D: np.ndarray
    K: np.ndarray
    R: np.ndarray
    P: np.ndarray

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ros_msg(cls, msg: Any) -> CameraInfo:
        """Create a CameraInfo wrapper from a decoded mcap-ros2-support message.

        Parameters
        ----------
        msg : Any
            A decoded ``sensor_msgs/msg/CameraInfo`` object.

        Returns
        -------
        CameraInfo
            The wrapped message with numpy matrices.
        """
        header = Header(
            sec=msg.header.stamp.sec,
            nanosec=msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
        )
        return cls(
            header=header,
            height=msg.height,
            width=msg.width,
            distortion_model=msg.distortion_model,
            D=np.array(msg.d, dtype=np.float64),
            K=np.array(msg.k, dtype=np.float64).reshape(3, 3),
            R=np.array(msg.r, dtype=np.float64).reshape(3, 3),
            P=np.array(msg.p, dtype=np.float64).reshape(3, 4),
        )

    @classmethod
    def from_cdr(cls, data: bytes) -> CameraInfo:
        """Deserialize a CameraInfo from raw CDR bytes.

        Reads all fields in CDR wire order: header, height, width,
        distortion_model, D (variable-length), K (9 float64), R (9 float64),
        P (12 float64). Binning and ROI fields are consumed but not stored
        in this wrapper (they are rarely used in practice).

        Parameters
        ----------
        data : bytes
            Raw CDR payload including the 4-byte encapsulation header.

        Returns
        -------
        CameraInfo
            The deserialized message.
        """
        cdr = CdrDeserializer(data)
        header = deserialize_header(cdr)

        height = cdr.read_uint32()
        width = cdr.read_uint32()
        distortion_model = cdr.read_string()

        # D: sequence<float64> (variable length)
        d_coeffs = np.array(
            cdr.read_sequence(cdr.read_float64), dtype=np.float64
        )

        # K: float64[9] (fixed-size, 3x3 intrinsic matrix)
        k_values = np.array(
            [cdr.read_float64() for _ in range(9)], dtype=np.float64
        ).reshape(3, 3)

        # R: float64[9] (fixed-size, 3x3 rectification matrix)
        r_values = np.array(
            [cdr.read_float64() for _ in range(9)], dtype=np.float64
        ).reshape(3, 3)

        # P: float64[12] (fixed-size, 3x4 projection matrix)
        p_values = np.array(
            [cdr.read_float64() for _ in range(12)], dtype=np.float64
        ).reshape(3, 4)

        # binning_x, binning_y: uint32 (consumed but not stored)
        _binning_x = cdr.read_uint32()
        _binning_y = cdr.read_uint32()

        # roi: RegionOfInterest (4x uint32 + 1x bool, consumed but not stored)
        _roi_x_offset = cdr.read_uint32()
        _roi_y_offset = cdr.read_uint32()
        _roi_height = cdr.read_uint32()
        _roi_width = cdr.read_uint32()
        _roi_do_rectify = cdr.read_uint8()

        return cls(
            header=header,
            height=height,
            width=width,
            distortion_model=distortion_model,
            D=d_coeffs,
            K=k_values,
            R=r_values,
            P=p_values,
        )

    # ------------------------------------------------------------------
    # Properties — intrinsic parameter accessors
    # ------------------------------------------------------------------

    @property
    def fx(self) -> float:
        """Focal length in the x-direction (pixels).

        Extracted from ``K[0, 0]``. Represents the horizontal focal length,
        which is the distance from the camera center to the image plane
        measured in pixel units along the x-axis.
        """
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        """Focal length in the y-direction (pixels).

        Extracted from ``K[1, 1]``. For most cameras ``fy`` is very close
        to ``fx`` (square pixels), but they may differ for anamorphic lenses.
        """
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        """Principal point x-coordinate (pixels).

        Extracted from ``K[0, 2]``. This is the x-coordinate of the optical
        axis intersection with the image plane. Ideally near ``width / 2``
        but may differ due to manufacturing tolerances.
        """
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        """Principal point y-coordinate (pixels).

        Extracted from ``K[1, 2]``. This is the y-coordinate of the optical
        axis intersection with the image plane. Ideally near ``height / 2``.
        """
        return float(self.K[1, 2])

    # ------------------------------------------------------------------
    # OpenCV conversion
    # ------------------------------------------------------------------

    def to_cv2_camera_matrix(self) -> np.ndarray:
        """Return the 3x3 camera intrinsic matrix for OpenCV functions.

        This is a copy of the K matrix, suitable for passing directly to
        ``cv2.undistort``, ``cv2.projectPoints``, ``cv2.solvePnP``, etc.

        Returns
        -------
        np.ndarray
            A 3x3 float64 camera matrix.
        """
        return self.K.copy()

    def to_cv2_distortion(self) -> np.ndarray:
        """Return the distortion coefficients as a 1D array for OpenCV.

        OpenCV expects distortion coefficients as a 1D array (or row/column
        vector). For the ``plumb_bob`` model, this is ``[k1, k2, p1, p2, k3]``.

        Returns
        -------
        np.ndarray
            A 1D float64 array of distortion coefficients.
        """
        return self.D.copy()
