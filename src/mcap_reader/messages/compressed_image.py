"""
Wrapper for sensor_msgs/msg/CompressedImage — JPEG/PNG compressed image data.

ROS 2 Message Definition
=========================
The ``sensor_msgs/msg/CompressedImage`` message carries compressed image data
(typically JPEG or PNG). The full IDL is::

    std_msgs/msg/Header header
    string format
    uint8[] data

CDR Byte Layout
===============
After the 4-byte encapsulation header:

  1. header (stamp.sec uint32, stamp.nanosec uint32, frame_id string)
  2. format (CDR string: uint32 length + chars + null)
  3. data (CDR sequence: uint32 count + count bytes)

The ``data`` field contains the raw compressed image bytes (e.g., a complete
JPEG or PNG file that could be written to disk and opened in any image viewer).

Learning Notes
==============
* **Format field is freeform:** The ``format`` string has no strict schema.
  Different ROS 2 camera drivers produce wildly inconsistent values. Common
  variants encountered in real datasets include:

    - ``"jpeg"`` — plain JPEG
    - ``"png"`` — plain PNG
    - ``"bgr8; jpeg compressed"`` — source was BGR8, compressed to JPEG
    - ``"rgb8; jpeg compressed"`` — source was RGB8, compressed to JPEG
    - ``"bgr8; png compressed"`` — source was BGR8, compressed to PNG
    - ``"mono8; jpeg compressed"`` — grayscale, compressed to JPEG
    - ``"16UC1; png compressed"`` — 16-bit depth, compressed to PNG
    - ``"rgb8; compressed bgr8"`` — another variant from some drivers

  Because the format string is not standardized, parsing it requires
  heuristic pattern matching. The :meth:`parse_format` method handles
  the common cases.

* **Decompression:** The ``data`` field is a standard JPEG/PNG byte stream.
  OpenCV's ``cv2.imdecode`` can decode it directly from a numpy buffer
  without writing to disk. The decoded image will be in BGR order (OpenCV's
  default), regardless of the original source encoding.

* **Why CompressedImage vs Image?** Raw images are large (a 1920x1080 RGB8
  image is ~6 MB). JPEG compression typically achieves 10-20x reduction.
  For bandwidth-limited scenarios (wireless robots, cloud logging),
  CompressedImage is strongly preferred. The trade-off is lossy compression
  artifacts (for JPEG) and CPU cost for encoding/decoding.

* **Depth images:** Some drivers publish compressed depth as PNG (which is
  lossless). The ``format`` field will indicate the source encoding (e.g.,
  ``"16UC1; png compressed"``). After decoding, the pixel values preserve
  the original depth measurements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mcap_reader.deserializer import CdrDeserializer, Header, deserialize_header


@dataclass
class CompressedImage:
    """Deserialized sensor_msgs/msg/CompressedImage message.

    Attributes
    ----------
    header : Header
        Timestamp and coordinate frame from std_msgs/Header.
    format : str
        Freeform string describing the compression format and source
        encoding (e.g., ``"jpeg"``, ``"bgr8; jpeg compressed"``).
    data : bytes
        Raw compressed image bytes (JPEG, PNG, etc.).
    """

    header: Header
    format: str
    data: bytes

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ros_msg(cls, msg: Any) -> CompressedImage:
        """Create a CompressedImage wrapper from a decoded mcap-ros2-support message.

        Parameters
        ----------
        msg : Any
            A decoded ``sensor_msgs/msg/CompressedImage`` object.

        Returns
        -------
        CompressedImage
            The wrapped message.
        """
        header = Header(
            sec=msg.header.stamp.sec,
            nanosec=msg.header.stamp.nanosec,
            frame_id=msg.header.frame_id,
        )
        return cls(
            header=header,
            format=msg.format,
            data=bytes(msg.data),
        )

    @classmethod
    def from_cdr(cls, data: bytes) -> CompressedImage:
        """Deserialize a CompressedImage from raw CDR bytes.

        Parameters
        ----------
        data : bytes
            Raw CDR payload including the 4-byte encapsulation header.

        Returns
        -------
        CompressedImage
            The deserialized message.
        """
        cdr = CdrDeserializer(data)
        header = deserialize_header(cdr)

        fmt = cdr.read_string()

        # data: sequence<uint8>
        count = cdr.read_uint32()
        img_data = bytes(cdr._buf[cdr._pos : cdr._pos + count])
        cdr._pos += count

        return cls(
            header=header,
            format=fmt,
            data=img_data,
        )

    # ------------------------------------------------------------------
    # Format parsing
    # ------------------------------------------------------------------

    def parse_format(self) -> tuple[str, str]:
        """Parse the freeform format string into (source_encoding, compression).

        Handles the common variants of the format string found in real ROS 2
        datasets. The format field is not standardized, so this method uses
        heuristic pattern matching.

        Returns
        -------
        tuple[str, str]
            A 2-tuple of ``(source_encoding, compression_type)``.

            - ``source_encoding`` is the pixel format before compression
              (e.g., ``"bgr8"``, ``"rgb8"``, ``"mono8"``, ``"16UC1"``).
              If not specified in the format string, defaults to ``"bgr8"``.
            - ``compression_type`` is ``"jpeg"`` or ``"png"``.

        Examples
        --------
        >>> img.format = "jpeg"
        >>> img.parse_format()
        ('bgr8', 'jpeg')

        >>> img.format = "bgr8; jpeg compressed"
        >>> img.parse_format()
        ('bgr8', 'jpeg')

        >>> img.format = "rgb8; compressed bgr8"
        >>> img.parse_format()
        ('rgb8', 'bgr8')

        >>> img.format = "16UC1; png compressed"
        >>> img.parse_format()
        ('16UC1', 'png')
        """
        fmt_lower = self.format.strip().lower()

        # Simple case: just "jpeg" or "png"
        if fmt_lower in ("jpeg", "png"):
            return ("bgr8", fmt_lower)

        # Pattern: "encoding; compression_type compressed"
        # e.g., "bgr8; jpeg compressed", "16UC1; png compressed"
        if ";" in self.format:
            parts = self.format.split(";", 1)
            source_encoding = parts[0].strip()
            compression_part = parts[1].strip().lower()

            if "jpeg" in compression_part:
                return (source_encoding, "jpeg")
            elif "png" in compression_part:
                return (source_encoding, "png")
            else:
                # Fallback: return the compression part as-is
                return (source_encoding, compression_part.replace("compressed", "").strip())

        # Fallback for other patterns
        if "jpeg" in fmt_lower:
            return ("bgr8", "jpeg")
        if "png" in fmt_lower:
            return ("bgr8", "png")

        return ("bgr8", self.format.strip())

    # ------------------------------------------------------------------
    # Decompression
    # ------------------------------------------------------------------

    def decompress(self) -> np.ndarray:
        """Decompress the image data into a numpy array using OpenCV.

        Uses ``cv2.imdecode`` to decode the JPEG or PNG byte stream. The
        returned image is in BGR color order (OpenCV's default), regardless
        of the original source encoding stated in the format string.

        For 16-bit depth images (e.g., ``"16UC1; png compressed"``), the
        image is decoded with ``cv2.IMREAD_UNCHANGED`` to preserve the
        original bit depth.

        Returns
        -------
        np.ndarray
            The decoded image array. Shape is ``(H, W, 3)`` for color
            images or ``(H, W)`` for grayscale/depth images.

        Raises
        ------
        ImportError
            If OpenCV (cv2) is not installed.
        RuntimeError
            If decompression fails (corrupted data, unsupported format).
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV (cv2) is required for CompressedImage.decompress(). "
                "Install it with: pip install opencv-python"
            )

        buf = np.frombuffer(self.data, dtype=np.uint8)

        # Use IMREAD_UNCHANGED for depth images to preserve 16-bit values.
        source_encoding, _ = self.parse_format()
        if source_encoding.lower() in ("16uc1", "mono16", "32fc1"):
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if img is None:
            raise RuntimeError(
                f"cv2.imdecode failed for format {self.format!r}. "
                f"Data length: {len(self.data)} bytes. "
                "The image data may be corrupted or in an unsupported format."
            )

        return img
