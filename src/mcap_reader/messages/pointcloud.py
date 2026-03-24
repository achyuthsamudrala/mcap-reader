"""
Wrapper for sensor_msgs/msg/PointCloud2 — 3D point cloud data.

ROS 2 Message Definition
=========================
The ``sensor_msgs/msg/PointCloud2`` message is the universal container for
3D point cloud data in ROS 2. It can represent any collection of points with
arbitrary per-point attributes. The full IDL is::

    std_msgs/msg/Header header

    uint32 height
    uint32 width

    sensor_msgs/PointField[] fields

    bool is_bigendian
    uint32 point_step
    uint32 row_step
    uint8[] data

    bool is_dense

Where ``PointField`` is::

    string name
    uint32 offset
    uint8 datatype
    uint32 count

CDR Byte Layout
===============
After the 4-byte encapsulation header:

  1. header (stamp.sec uint32, stamp.nanosec uint32, frame_id string)
  2. height (uint32)
  3. width (uint32)
  4. fields (CDR sequence of PointField structs):
       [uint32 count] then for each:
         [string name] [uint32 offset] [uint8 datatype] [uint32 count]
  5. is_bigendian (uint8/bool)
  6. point_step (uint32) — bytes per point
  7. row_step (uint32) — bytes per row
  8. data (CDR sequence of uint8)
  9. is_dense (uint8/bool)

Learning Notes
==============
* **PointCloud2 is extremely general:** Unlike a simple (N, 3) array of XYZ
  coordinates, PointCloud2 carries an embedded schema (the ``fields`` list)
  that describes what data each point contains. Common field configurations:

    - LiDAR: x, y, z, intensity, ring, time
    - RGB-D camera: x, y, z, rgb (packed as uint32)
    - Stereo: x, y, z
    - Semantic: x, y, z, label, confidence

  The ``fields`` list is the schema embedded in each message. Each PointField
  specifies a field name, its byte offset within the point, a datatype enum,
  and a count (usually 1).

* **Organized vs unorganized:** If ``height > 1``, the cloud is "organized"
  — it has a 2D grid structure (like a depth image projected to 3D). This is
  common for structured-light cameras (e.g., Intel RealSense) and scanning
  LiDARs. Unorganized clouds have ``height == 1`` and ``width == N``.

* **is_dense:** If True, all points contain finite values (no NaN/Inf).
  If False, some points may have NaN coordinates (e.g., invalid depth
  readings in an organized cloud). Always filter NaN points before
  processing if ``is_dense`` is False.

* **point_step and row_step:** ``point_step`` is the total bytes per point
  (including padding between fields). ``row_step`` is the bytes per row
  (``width * point_step`` for organized clouds). These are analogous to
  Image's ``step`` field — they account for alignment padding.

* **PointField datatype enum:** The datatype field is an integer enum::

      INT8    = 1
      UINT8   = 2
      INT16   = 3
      UINT16  = 4
      INT32   = 5
      UINT32  = 6
      FLOAT32 = 7
      FLOAT64 = 8

* **Structured numpy arrays:** The most efficient way to decode PointCloud2
  data is to create a numpy structured dtype from the fields list, then
  use ``np.frombuffer`` to view the data buffer as a structured array. This
  gives O(1) decoding time regardless of point count (no per-point loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mcap_reader.deserializer import CdrDeserializer, Header, deserialize_header


# Mapping from PointField datatype enum values to numpy dtype strings.
# These correspond to the sensor_msgs/PointField constants.
DATATYPE_MAP: dict[int, np.dtype] = {
    1: np.dtype(np.int8),       # INT8
    2: np.dtype(np.uint8),      # UINT8
    3: np.dtype(np.int16),      # INT16
    4: np.dtype(np.uint16),     # UINT16
    5: np.dtype(np.int32),      # INT32
    6: np.dtype(np.uint32),     # UINT32
    7: np.dtype(np.float32),    # FLOAT32
    8: np.dtype(np.float64),    # FLOAT64
}


@dataclass
class PointField:
    """Description of a single field within a PointCloud2 point.

    Each point in a PointCloud2 message is a fixed-size binary blob.
    The PointField describes where within that blob a particular named
    value lives, what type it is, and how many elements it has.

    Attributes
    ----------
    name : str
        Field name (e.g., ``"x"``, ``"y"``, ``"z"``, ``"intensity"``).
    offset : int
        Byte offset of this field from the start of the point.
    datatype : int
        Datatype enum value (1-8). See ``DATATYPE_MAP``.
    count : int
        Number of elements. Usually 1. If > 1, the field is an array
        (e.g., ``"rgb"`` is sometimes stored as count=1 UINT32, or
        ``"normal"`` as count=3 FLOAT32).
    """

    name: str
    offset: int
    datatype: int
    count: int

    @property
    def numpy_dtype(self) -> np.dtype:
        """Get the numpy dtype corresponding to this field's datatype.

        Returns
        -------
        np.dtype
            The numpy dtype for this field's element type.

        Raises
        ------
        KeyError
            If the datatype enum value is not recognized.
        """
        if self.datatype not in DATATYPE_MAP:
            raise KeyError(
                f"Unknown PointField datatype {self.datatype}. "
                f"Known types: {list(DATATYPE_MAP.keys())}"
            )
        return DATATYPE_MAP[self.datatype]


@dataclass
class PointCloud2:
    """Deserialized sensor_msgs/msg/PointCloud2 message.

    Attributes
    ----------
    header : Header
        Timestamp and coordinate frame from std_msgs/Header.
    height : int
        Height of the point cloud. If 1, the cloud is unorganized.
        If > 1, the cloud is organized (2D grid structure).
    width : int
        Width of the point cloud. For unorganized clouds, this is the
        total number of points. For organized clouds, it is the number
        of columns.
    fields : list[PointField]
        Description of each per-point field (name, offset, type, count).
    is_bigendian : bool
        Whether multi-byte values in the data buffer are big-endian.
    point_step : int
        Length of a single point in bytes (including padding).
    row_step : int
        Length of a row in bytes (``width * point_step``).
    data : bytes
        Raw point data buffer.
    is_dense : bool
        If True, all points are finite (no NaN/Inf values).
    """

    header: Header
    height: int
    width: int
    fields: list[PointField]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ros_msg(cls, msg: Any) -> PointCloud2:
        """Create a PointCloud2 wrapper from a decoded mcap-ros2-support message.

        Parameters
        ----------
        msg : Any
            A decoded ``sensor_msgs/msg/PointCloud2`` object.

        Returns
        -------
        PointCloud2
            The wrapped message.
        """
        header = Header(
            sec=msg.header.stamp.sec,
            nanosec=msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
        )
        fields = [
            PointField(
                name=f.name,
                offset=f.offset,
                datatype=f.datatype,
                count=f.count,
            )
            for f in msg.fields
        ]
        return cls(
            header=header,
            height=msg.height,
            width=msg.width,
            fields=fields,
            is_bigendian=bool(msg.is_bigendian),
            point_step=msg.point_step,
            row_step=msg.row_step,
            data=bytes(msg.data),
            is_dense=bool(msg.is_dense),
        )

    @classmethod
    def from_cdr(cls, data: bytes) -> PointCloud2:
        """Deserialize a PointCloud2 from raw CDR bytes.

        Parameters
        ----------
        data : bytes
            Raw CDR payload including the 4-byte encapsulation header.

        Returns
        -------
        PointCloud2
            The deserialized message.
        """
        cdr = CdrDeserializer(data)
        header = deserialize_header(cdr)

        height = cdr.read_uint32()
        width = cdr.read_uint32()

        # fields: sequence<PointField>
        field_count = cdr.read_uint32()
        fields: list[PointField] = []
        for _ in range(field_count):
            name = cdr.read_string()
            offset = cdr.read_uint32()
            datatype = cdr.read_uint8()
            count = cdr.read_uint32()
            fields.append(PointField(
                name=name, offset=offset, datatype=datatype, count=count
            ))

        is_bigendian = bool(cdr.read_uint8())
        point_step = cdr.read_uint32()
        row_step = cdr.read_uint32()

        # data: sequence<uint8>
        data_count = cdr.read_uint32()
        point_data = bytes(cdr._buf[cdr._pos : cdr._pos + data_count])
        cdr._pos += data_count

        is_dense = bool(cdr.read_uint8())

        return cls(
            header=header,
            height=height,
            width=width,
            fields=fields,
            is_bigendian=is_bigendian,
            point_step=point_step,
            row_step=row_step,
            data=point_data,
            is_dense=is_dense,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_organized(self) -> bool:
        """Whether the point cloud has a 2D grid structure.

        Organized clouds (``height > 1``) come from structured sensors like
        depth cameras or scanning LiDARs. Each (row, col) position
        corresponds to a specific angular or pixel coordinate.

        Unorganized clouds (``height == 1``) are just a flat list of N
        points with no spatial structure.

        Returns
        -------
        bool
            True if ``height > 1``.
        """
        return self.height > 1

    @property
    def num_points(self) -> int:
        """Total number of points in the cloud.

        Returns
        -------
        int
            ``height * width``.
        """
        return self.height * self.width

    # ------------------------------------------------------------------
    # Conversion to numpy
    # ------------------------------------------------------------------

    def _build_structured_dtype(self) -> np.dtype:
        """Build a numpy structured dtype from the PointField list.

        Creates a dtype that mirrors the binary layout of each point,
        including any padding bytes between fields. This allows
        ``np.frombuffer`` to decode the entire data buffer in one call.

        Returns
        -------
        np.dtype
            A structured numpy dtype with named fields matching the
            PointField names.
        """
        byte_order = ">" if self.is_bigendian else "<"
        fields_list: list[tuple[str, str, tuple[int, ...]] | tuple[str, str]] = []
        current_offset = 0

        for i, field in enumerate(self.fields):
            # Insert padding if there is a gap between the current offset
            # and this field's offset.
            if field.offset > current_offset:
                pad_size = field.offset - current_offset
                fields_list.append((f"_pad{i}", f"|V{pad_size}"))
                current_offset = field.offset

            base_dtype = field.numpy_dtype.newbyteorder(byte_order)
            if field.count > 1:
                fields_list.append((field.name, base_dtype.str, (field.count,)))
            else:
                fields_list.append((field.name, base_dtype.str))

            current_offset = field.offset + field.numpy_dtype.itemsize * field.count

        # Trailing padding to fill out to point_step.
        if current_offset < self.point_step:
            pad_size = self.point_step - current_offset
            fields_list.append(("_pad_end", f"|V{pad_size}"))

        return np.dtype(fields_list)

    def to_numpy(self) -> np.ndarray:
        """Convert the point cloud to a numpy structured array.

        The returned array has one element per point, with named fields
        matching the PointField names. Access fields like::

            cloud = pc.to_numpy()
            x = cloud['x']
            intensity = cloud['intensity']

        This is an O(1) operation (no per-point loop) because it uses
        ``np.frombuffer`` with a structured dtype that matches the binary
        layout.

        Returns
        -------
        np.ndarray
            A structured array of shape ``(height * width,)`` with named
            fields.
        """
        dtype = self._build_structured_dtype()
        return np.frombuffer(self.data, dtype=dtype)

    def to_xyz(self) -> np.ndarray:
        """Extract just the XYZ coordinates as an (N, 3) float array.

        This is a convenience method for the very common case where you
        only need the spatial positions. It extracts the ``x``, ``y``, ``z``
        fields from the structured array and stacks them into a contiguous
        (N, 3) array.

        Returns
        -------
        np.ndarray
            Shape ``(N, 3)`` array of float64 XYZ coordinates.

        Raises
        ------
        KeyError
            If the point cloud does not have ``x``, ``y``, ``z`` fields.
        """
        structured = self.to_numpy()
        try:
            x = structured["x"].astype(np.float64)
            y = structured["y"].astype(np.float64)
            z = structured["z"].astype(np.float64)
        except (ValueError, KeyError) as e:
            available = [f.name for f in self.fields]
            raise KeyError(
                f"Point cloud does not have x/y/z fields. "
                f"Available fields: {available}"
            ) from e

        return np.column_stack([x, y, z])

    def to_pandas(self) -> "pd.DataFrame":
        """Convert the point cloud to a pandas DataFrame.

        Each PointField becomes a column. Padding fields (used for
        alignment) are excluded.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per point and columns for each field.
        """
        import pandas as pd

        structured = self.to_numpy()
        # Filter out padding fields (names starting with '_pad').
        columns = {
            name: structured[name]
            for name in structured.dtype.names
            if not name.startswith("_pad")
        }
        return pd.DataFrame(columns)
