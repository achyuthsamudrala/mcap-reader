"""
CDR (Common Data Representation) byte-level deserializer for ROS 2 messages.

What is CDR and why does ROS 2 use it?
--------------------------------------
CDR is the binary serialization format defined by the OMG (Object Management Group)
as part of the CORBA standard. ROS 2 adopted CDR through its use of DDS
(Data Distribution Service) as the middleware transport layer. Every ROS 2 message
published over DDS is serialized into CDR bytes before hitting the wire.

Why CDR specifically? DDS was chosen for ROS 2 because it provides a mature,
standards-based publish-subscribe transport with QoS (Quality of Service) policies,
discovery, and interoperability across vendors. CDR comes along for the ride as
DDS's native wire format. The alternative would have been inventing a custom
serialization (as ROS 1 did), but reusing CDR means any compliant DDS implementation
can decode the data -- enabling cross-language, cross-vendor interoperability without
a translation layer.

Alignment padding rules and why they exist
------------------------------------------
CDR requires that every primitive value is aligned to a byte boundary equal to its
own size:

    - uint8  / int8  -> 1-byte aligned (no padding ever needed)
    - uint16 / int16 -> 2-byte aligned
    - uint32 / int32 / float32 -> 4-byte aligned
    - uint64 / int64 / float64 -> 8-byte aligned

Alignment is measured from the START of the serialized payload (after the 4-byte
encapsulation header). So if you have just read a 1-byte field and the next field
is a uint32, the deserializer must skip 0-3 padding bytes to reach the next
4-byte-aligned offset.

Why does this matter? Modern CPUs access memory most efficiently when data falls on
its natural alignment boundary. An unaligned 4-byte read on some architectures
(notably ARM) will either fault or silently incur a multi-cycle penalty. CDR's
alignment rules guarantee that a receiver can cast directly into native types
without ever hitting an unaligned access -- even when the receiver is an embedded
system with strict alignment requirements. The cost is a few wasted padding bytes;
the benefit is zero-copy-friendly decoding on every platform.

The 4-byte encapsulation header
-------------------------------
Every ROS 2 CDR-serialized message begins with a 4-byte encapsulation header
BEFORE the actual data fields:

    Byte 0: Encapsulation scheme identifier (high byte)
    Byte 1: Encapsulation scheme identifier (low byte)
              0x00 0x01 = CDR Big-Endian (CDR_BE)
              0x00 0x02 = CDR Little-Endian (CDR_LE)  <-- typical on x86/ARM
    Byte 2: Options (unused, 0x00)
    Byte 3: Options (unused, 0x00)

After these 4 bytes, all alignment offsets are measured relative to byte 4 (the
first data byte). This header lets the receiver know the byte order without any
out-of-band negotiation -- critical for interoperability between big-endian and
little-endian machines.

Why step (not width * channels) must be used for Image reconstruction
---------------------------------------------------------------------
sensor_msgs/Image has both a `width` field and a `step` field. You might assume
step == width * bytes_per_pixel, but this is NOT always true. Camera drivers
and GPU pipelines often pad each row to a specific alignment (e.g., 64-byte or
128-byte boundaries) for SIMD processing or DMA transfer efficiency. The `step`
field captures this actual row stride in bytes, including any trailing padding.

If you reconstruct the image using width * channels instead of step, you will:
  1. Read the wrong bytes for every row after the first (the offset drifts by
     the padding amount each row).
  2. Get a sheared / corrupted image that looks "almost right" but with diagonal
     artifacts.

Always reshape with step as the row stride, then slice off padding columns if
you need a clean width * channels array.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, List, TypeVar

T = TypeVar("T")


@dataclass
class Header:
    """Deserialized std_msgs/Header.

    The Header is the most common ROS 2 message component -- nearly every sensor
    message embeds one. It carries a timestamp and a coordinate frame identifier.

    Fields
    ------
    sec : int
        Seconds portion of the ROS timestamp (uint32). This is typically seconds
        since the Unix epoch, but may use sim-time depending on the system clock.
    nanosec : int
        Nanoseconds portion (uint32, always in [0, 999_999_999]).
    frame_id : str
        The TF frame this data is associated with (e.g., "base_link", "camera_optical").
    """

    sec: int
    nanosec: int
    frame_id: str

    def to_timestamp(self) -> float:
        """Convert to a single float representing seconds since epoch.

        Returns
        -------
        float
            ``sec + nanosec * 1e-9``. Note that Python floats are IEEE 754
            double-precision, giving ~15-17 significant digits. A Unix timestamp
            around 1.7e9 already uses 10 digits, so the nanosecond portion is
            accurate to roughly 7 decimal places (~100 ns resolution). For most
            robotics applications this is more than sufficient, but if you need
            exact nanosecond fidelity, keep sec and nanosec separate.
        """
        return self.sec + self.nanosec * 1e-9


class CdrDeserializer:
    """Stateful cursor-based deserializer for CDR-encoded ROS 2 message payloads.

    This class wraps a raw byte buffer and provides typed read methods that
    respect CDR alignment rules. Each read advances an internal cursor, so
    fields are consumed in declaration order -- matching how ROS 2 message
    definitions map to CDR byte layout.

    Parameters
    ----------
    data : bytes | memoryview
        The raw CDR payload, **including** the 4-byte encapsulation header.
        Typically this is the ``data`` field from an MCAP message record.

    Attributes
    ----------
    _buf : memoryview
        The underlying byte buffer (as a memoryview for zero-copy slicing).
    _pos : int
        Current read cursor position within ``_buf``.
    _little_endian : bool
        Byte order parsed from the encapsulation header.

    Example
    -------
    >>> raw = message_record.data  # bytes from MCAP
    >>> cdr = CdrDeserializer(raw)
    >>> value = cdr.read_uint32()
    """

    def __init__(self, data: bytes | memoryview) -> None:
        if isinstance(data, memoryview):
            self._buf = data
        else:
            self._buf = memoryview(data)

        # --- Parse the 4-byte encapsulation header ---
        # Bytes 0-1: encapsulation kind.  0x0001 = big-endian, 0x0002 = little-endian.
        # Bytes 2-3: options, currently unused by ROS 2 (always 0x0000).
        if len(self._buf) < 4:
            raise ValueError(
                f"CDR payload too short ({len(self._buf)} bytes); need at least "
                f"4 bytes for the encapsulation header."
            )

        encapsulation_kind = (self._buf[0] << 8) | self._buf[1]
        if encapsulation_kind == 0x0001:
            self._little_endian = False
        elif encapsulation_kind == 0x0002:
            self._little_endian = True
        else:
            # Default to little-endian with a warning -- many MCAP files in the
            # wild have a zeroed-out header and are still LE.
            self._little_endian = True

        # Advance past the 4-byte header.  All alignment offsets are now
        # relative to this position (i.e., self._pos == 0 is 4-byte aligned
        # by definition).
        self._pos = 4

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def align(self, n: int) -> None:
        """Advance the cursor to the next *n*-byte-aligned offset.

        CDR alignment is relative to the start of the data portion (byte 4
        of the full payload, right after the encapsulation header).  This
        method computes how many padding bytes to skip so that the next read
        begins on a properly aligned boundary.

        Parameters
        ----------
        n : int
            Required alignment in bytes (1, 2, 4, or 8).

        Why is alignment separate from the read methods?
        Because compound types (structs, sequences) sometimes need explicit
        alignment before a nested structure begins, even when the previous
        field was the "right" size. Keeping it as a callable method gives the
        caller full control.
        """
        # offset relative to data start (byte 4 of the buffer)
        offset = self._pos - 4
        remainder = offset % n
        if remainder != 0:
            self._pos += n - remainder

    # ------------------------------------------------------------------
    # Primitive readers
    # ------------------------------------------------------------------

    def read_uint8(self) -> int:
        """Read a single unsigned 8-bit integer.

        No alignment is needed -- uint8 has 1-byte alignment, which is always
        satisfied regardless of cursor position.
        """
        val = self._buf[self._pos]
        self._pos += 1
        return val

    def read_uint16(self) -> int:
        """Read an unsigned 16-bit integer (2 bytes, aligned to 2).

        CDR requires 2-byte alignment for 16-bit types.  If the cursor is at
        an odd offset, one padding byte is skipped.
        """
        self.align(2)
        fmt = "<H" if self._little_endian else ">H"
        (val,) = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += 2
        return val

    def read_uint32(self) -> int:
        """Read an unsigned 32-bit integer (4 bytes, aligned to 4).

        This is one of the most common CDR types -- it encodes sequence
        lengths, string lengths, and many ROS 2 numeric fields.
        """
        self.align(4)
        fmt = "<I" if self._little_endian else ">I"
        (val,) = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += 4
        return val

    def read_int32(self) -> int:
        """Read a signed 32-bit integer (4 bytes, aligned to 4).

        Uses two's-complement representation, same as C ``int32_t``.
        """
        self.align(4)
        fmt = "<i" if self._little_endian else ">i"
        (val,) = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += 4
        return val

    def read_float32(self) -> float:
        """Read an IEEE 754 single-precision float (4 bytes, aligned to 4).

        Common in ROS 2 for sensor values where double precision is overkill
        (e.g., LaserScan ranges, IMU angular velocities).
        """
        self.align(4)
        fmt = "<f" if self._little_endian else ">f"
        (val,) = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += 4
        return val

    def read_float64(self) -> float:
        """Read an IEEE 754 double-precision float (8 bytes, aligned to 8).

        CDR aligns float64 to 8 bytes -- this is the strictest alignment of
        any common primitive.  On a 64-bit CPU the 8-byte alignment lets the
        hardware load the value in a single aligned memory access.

        ROS 2 uses float64 for high-precision quantities: GPS coordinates,
        joint positions, or any value where float32's ~7 significant digits
        are insufficient.
        """
        self.align(8)
        fmt = "<d" if self._little_endian else ">d"
        (val,) = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += 8
        return val

    # ------------------------------------------------------------------
    # String
    # ------------------------------------------------------------------

    def read_string(self) -> str:
        """Read a CDR string (length-prefixed, null-terminated UTF-8).

        CDR string layout:
            [uint32 length] [length bytes of character data including '\\0']

        The ``length`` field includes the null terminator.  So a 5-character
        ASCII string like "hello" has length == 6 (5 chars + 1 null byte).
        We strip the trailing null before decoding to Python ``str``.

        Why null-terminated AND length-prefixed?
        CDR inherits this from CORBA, which needed C interoperability (null
        termination) while also supporting efficient skip-ahead (length prefix).
        It's redundant, but ROS 2 follows the spec faithfully.
        """
        length = self.read_uint32()
        if length == 0:
            return ""
        # `length` includes the null terminator; grab all bytes then drop it.
        raw = bytes(self._buf[self._pos : self._pos + length])
        self._pos += length
        # Strip the null terminator (last byte).  Some writers may omit it,
        # so use rstrip as a safety net.
        return raw.rstrip(b"\x00").decode("utf-8")

    # ------------------------------------------------------------------
    # Sequence (variable-length array)
    # ------------------------------------------------------------------

    def read_sequence(self, reader_fn: Callable[[], T]) -> List[T]:
        """Read a CDR sequence (variable-length array) of homogeneous elements.

        CDR sequence layout:
            [uint32 count] [count elements, each read by reader_fn]

        The ``count`` field is a uint32 giving the number of elements (NOT
        the byte length).  Each element is then deserialized by calling
        ``reader_fn`` -- which may itself call ``align`` as needed.

        Parameters
        ----------
        reader_fn : Callable[[], T]
            A bound method on this deserializer (e.g., ``cdr.read_float32``)
            or any zero-argument callable that reads the next element from the
            buffer.

        Returns
        -------
        List[T]
            The deserialized elements in order.

        Example
        -------
        >>> points = cdr.read_sequence(cdr.read_float64)
        """
        count = self.read_uint32()
        return [reader_fn() for _ in range(count)]

    # ------------------------------------------------------------------
    # Convenience: current position
    # ------------------------------------------------------------------

    @property
    def pos(self) -> int:
        """Current cursor position in the underlying buffer (including header)."""
        return self._pos

    @property
    def remaining(self) -> int:
        """Number of bytes remaining after the cursor."""
        return len(self._buf) - self._pos


def deserialize_header(cdr: CdrDeserializer) -> Header:
    """Deserialize a ``std_msgs/msg/Header`` from the current cursor position.

    The Header message definition in ROS 2 IDL is::

        builtin_interfaces/Time stamp
            uint32 sec
            uint32 nanosec
        string frame_id

    In CDR byte layout this becomes:

        1. stamp.sec    -- uint32 (4 bytes, 4-byte aligned)
        2. stamp.nanosec -- uint32 (4 bytes, already aligned after sec)
        3. frame_id     -- CDR string (uint32 length + chars + null)

    No explicit alignment call is needed between sec and nanosec because
    two consecutive uint32 values are naturally aligned.  The string's
    length prefix is also uint32, so it too stays aligned after nanosec.

    Parameters
    ----------
    cdr : CdrDeserializer
        A deserializer whose cursor is positioned at the start of a Header
        (i.e., at stamp.sec).

    Returns
    -------
    Header
        A dataclass with ``sec``, ``nanosec``, and ``frame_id`` populated.
    """
    sec = cdr.read_uint32()
    nanosec = cdr.read_uint32()
    frame_id = cdr.read_string()
    return Header(sec=sec, nanosec=nanosec, frame_id=frame_id)
