"""
Wrapper for sensor_msgs/msg/Image — raw uncompressed image data.

ROS 2 Message Definition
=========================
The ``sensor_msgs/msg/Image`` message carries raw (uncompressed) pixel data
along with metadata describing the image dimensions and encoding. The full
IDL is::

    std_msgs/msg/Header header
    uint32 height
    uint32 width
    string encoding
    uint8 is_bigendian
    uint32 step
    uint8[] data

CDR Byte Layout
===============
After the 4-byte encapsulation header:

  1. header (stamp.sec uint32, stamp.nanosec uint32, frame_id string)
  2. height (uint32)
  3. width (uint32)
  4. encoding (CDR string: uint32 length + chars + null)
  5. is_bigendian (uint8)
  6. step (uint32, aligned to 4 after the uint8)
  7. data (CDR sequence: uint32 count + count bytes)

Learning Notes
==============
* **step vs. width * bytes_per_pixel:** The ``step`` field is the number
  of bytes per row in the ``data`` buffer. You might expect
  ``step == width * bytes_per_pixel``, but this is NOT always true. Camera
  drivers and GPU pipelines often pad each row to a specific alignment
  (e.g., 64-byte or 128-byte boundaries) for SIMD processing or DMA
  transfer efficiency. The ``step`` field captures this actual row stride
  in bytes, including any trailing padding.

  If you reconstruct the image using ``width * channels`` instead of
  ``step``, you will:
    1. Read the wrong bytes for every row after the first (the offset
       drifts by the padding amount each row).
    2. Get a sheared/corrupted image that looks "almost right" but with
       diagonal artifacts.

  Always reshape using ``step`` as the row stride, then slice off padding
  columns if you need a clean ``(H, W, C)`` array.

* **Encoding strings:** The ``encoding`` field specifies the pixel format.
  Common values include:
    - ``"rgb8"`` — 3 channels, 8 bits each, R-G-B order
    - ``"bgr8"`` — 3 channels, 8 bits each, B-G-R order (OpenCV default)
    - ``"mono8"`` — single channel, 8-bit grayscale
    - ``"mono16"`` — single channel, 16-bit grayscale (depth cameras)
    - ``"16UC1"`` — single channel uint16 (another depth convention)
    - ``"32FC1"`` — single channel float32 (depth in meters)
    - ``"bayer_rggb8"`` — Bayer mosaic pattern, needs demosaicing

* **Depth images:** Depth sensors like the Intel RealSense often publish
  depth as ``mono16`` or ``16UC1`` (millimeters) or ``32FC1`` (meters).
  It is critical to preserve the numeric dtype — do not cast to uint8,
  as you will lose depth information.

* **is_bigendian:** Indicates the byte order of multi-byte pixel values
  (e.g., 16-bit or 32-bit). On x86/ARM this is almost always 0 (little-
  endian), but must be respected for correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mcap_reader.deserializer import CdrDeserializer, Header, deserialize_header


# Mapping from ROS 2 image encoding strings to (numpy dtype, num_channels).
# This covers the most common encodings encountered in real datasets.
ENCODING_DTYPE_MAP: dict[str, tuple[np.dtype, int]] = {
    "rgb8": (np.dtype(np.uint8), 3),
    "bgr8": (np.dtype(np.uint8), 3),
    "rgba8": (np.dtype(np.uint8), 4),
    "bgra8": (np.dtype(np.uint8), 4),
    "mono8": (np.dtype(np.uint8), 1),
    "mono16": (np.dtype(np.uint16), 1),
    "16UC1": (np.dtype(np.uint16), 1),
    "16UC3": (np.dtype(np.uint16), 3),
    "32FC1": (np.dtype(np.float32), 1),
    "32FC3": (np.dtype(np.float32), 3),
    "bayer_rggb8": (np.dtype(np.uint8), 1),
    "bayer_bggr8": (np.dtype(np.uint8), 1),
    "bayer_gbrg8": (np.dtype(np.uint8), 1),
    "bayer_grbg8": (np.dtype(np.uint8), 1),
}


@dataclass
class Image:
    """Deserialized sensor_msgs/msg/Image message.

    Attributes
    ----------
    header : Header
        Timestamp and coordinate frame from std_msgs/Header.
    height : int
        Image height in pixels (number of rows).
    width : int
        Image width in pixels (number of columns).
    encoding : str
        Pixel encoding string (e.g., ``"rgb8"``, ``"mono16"``).
    is_bigendian : bool
        Whether multi-byte pixel values are big-endian.
    step : int
        Full row length in bytes, including any padding. This is the
        actual stride between consecutive rows in ``data``.
    data : bytes
        Raw pixel data buffer. Length should be ``step * height``.
    """

    header: Header
    height: int
    width: int
    encoding: str
    is_bigendian: bool
    step: int
    data: bytes

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ros_msg(cls, msg: Any) -> Image:
        """Create an Image wrapper from a decoded mcap-ros2-support message.

        Parameters
        ----------
        msg : Any
            A decoded ``sensor_msgs/msg/Image`` object.

        Returns
        -------
        Image
            The wrapped message.
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
            encoding=msg.encoding,
            is_bigendian=bool(msg.is_bigendian),
            step=msg.step,
            data=bytes(msg.data),
        )

    @classmethod
    def from_cdr(cls, data: bytes) -> Image:
        """Deserialize an Image from raw CDR bytes.

        Parameters
        ----------
        data : bytes
            Raw CDR payload including the 4-byte encapsulation header.

        Returns
        -------
        Image
            The deserialized message.
        """
        cdr = CdrDeserializer(data)
        header = deserialize_header(cdr)

        height = cdr.read_uint32()
        width = cdr.read_uint32()
        encoding = cdr.read_string()
        is_bigendian = bool(cdr.read_uint8())
        step = cdr.read_uint32()

        # data: sequence<uint8>
        # The sequence length prefix gives byte count, then raw pixel bytes follow.
        pixel_count = cdr.read_uint32()
        pixel_data = bytes(cdr._buf[cdr._pos : cdr._pos + pixel_count])
        cdr._pos += pixel_count

        return cls(
            header=header,
            height=height,
            width=width,
            encoding=encoding,
            is_bigendian=is_bigendian,
            step=step,
            data=pixel_data,
        )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Convert the raw pixel buffer to a numpy array.

        Uses the ``step`` field (not ``width * channels``) to correctly
        handle row padding. The returned array has shape:
          - ``(height, width)`` for single-channel images
          - ``(height, width, channels)`` for multi-channel images

        For depth images (``mono16``, ``16UC1``, ``32FC1``), the original
        dtype is preserved so numeric values (e.g., millimeters or meters)
        are not lost.

        Returns
        -------
        np.ndarray
            The image as a numpy array with the appropriate dtype.

        Raises
        ------
        ValueError
            If the encoding is not recognized in ``ENCODING_DTYPE_MAP``.
        """
        if self.encoding not in ENCODING_DTYPE_MAP:
            raise ValueError(
                f"Unsupported encoding {self.encoding!r}. "
                f"Supported: {sorted(ENCODING_DTYPE_MAP.keys())}"
            )

        dtype, num_channels = ENCODING_DTYPE_MAP[self.encoding]

        # Respect the is_bigendian flag for multi-byte dtypes.
        if dtype.itemsize > 1:
            byte_order = ">" if self.is_bigendian else "<"
            dtype = dtype.newbyteorder(byte_order)

        # Reconstruct the image using step as the row stride.
        # First, create a flat array from the raw data using the step.
        bytes_per_pixel = dtype.itemsize * num_channels
        expected_row_bytes = self.width * bytes_per_pixel

        if self.step == expected_row_bytes:
            # No padding — straightforward reshape.
            flat = np.frombuffer(self.data, dtype=dtype)
            if num_channels == 1:
                return flat.reshape(self.height, self.width)
            else:
                return flat.reshape(self.height, self.width, num_channels)
        else:
            # Row padding present — must parse row by row using step.
            # Create buffer as uint8, reshape by step, then slice valid columns.
            raw = np.frombuffer(self.data, dtype=np.uint8)
            raw = raw.reshape(self.height, self.step)
            # Slice off the padding bytes at the end of each row.
            raw = raw[:, :expected_row_bytes]
            # Reinterpret as the target dtype.
            raw = raw.copy()  # Ensure contiguous before view cast.
            pixel_array = raw.view(dtype)
            if num_channels == 1:
                return pixel_array.reshape(self.height, self.width)
            else:
                return pixel_array.reshape(self.height, self.width, num_channels)
