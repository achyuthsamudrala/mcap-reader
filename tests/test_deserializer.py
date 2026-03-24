"""Tests for CdrDeserializer and related deserialization functions."""

from __future__ import annotations

import struct

import pytest

from mcap_reader.deserializer import CdrDeserializer, Header, deserialize_header


# ---------------------------------------------------------------------------
# Helpers to build CDR payloads
# ---------------------------------------------------------------------------

LE_HEADER = b"\x00\x02\x00\x00"  # Little-endian CDR encapsulation
BE_HEADER = b"\x00\x01\x00\x00"  # Big-endian CDR encapsulation


def _le_payload(*parts: bytes) -> bytes:
    """Concatenate LE header + parts."""
    return LE_HEADER + b"".join(parts)


# ---------------------------------------------------------------------------
# Encapsulation header
# ---------------------------------------------------------------------------


class TestEncapsulationHeader:
    """Tests for CDR encapsulation header parsing."""

    def test_little_endian_header(self):
        data = LE_HEADER + struct.pack("<I", 42)
        cdr = CdrDeserializer(data)
        assert cdr._little_endian is True
        assert cdr.read_uint32() == 42

    def test_big_endian_header(self):
        data = BE_HEADER + struct.pack(">I", 42)
        cdr = CdrDeserializer(data)
        assert cdr._little_endian is False
        assert cdr.read_uint32() == 42

    def test_short_payload_raises(self):
        with pytest.raises(ValueError, match="too short"):
            CdrDeserializer(b"\x00\x02")

    def test_unknown_encapsulation_defaults_to_le(self):
        """Unknown encapsulation kind defaults to little-endian."""
        data = b"\x00\x00\x00\x00" + struct.pack("<I", 99)
        cdr = CdrDeserializer(data)
        assert cdr._little_endian is True
        assert cdr.read_uint32() == 99


# ---------------------------------------------------------------------------
# Primitive type reading
# ---------------------------------------------------------------------------


class TestPrimitiveReads:
    """Test reading each CDR primitive type."""

    def test_read_uint8(self):
        data = _le_payload(b"\xff")
        cdr = CdrDeserializer(data)
        assert cdr.read_uint8() == 255

    def test_read_uint16(self):
        data = _le_payload(struct.pack("<H", 0xBEEF))
        cdr = CdrDeserializer(data)
        assert cdr.read_uint16() == 0xBEEF

    def test_read_uint32(self):
        data = _le_payload(struct.pack("<I", 0xDEADBEEF))
        cdr = CdrDeserializer(data)
        assert cdr.read_uint32() == 0xDEADBEEF

    def test_read_int32_positive(self):
        data = _le_payload(struct.pack("<i", 123456))
        cdr = CdrDeserializer(data)
        assert cdr.read_int32() == 123456

    def test_read_int32_negative(self):
        data = _le_payload(struct.pack("<i", -42))
        cdr = CdrDeserializer(data)
        assert cdr.read_int32() == -42

    def test_read_float32(self):
        data = _le_payload(struct.pack("<f", 3.14))
        cdr = CdrDeserializer(data)
        assert abs(cdr.read_float32() - 3.14) < 1e-5

    def test_read_float64_at_offset_0(self):
        """float64 at offset 0 after header (offset 0 is 8-byte aligned from data start)."""
        data = _le_payload(struct.pack("<d", 2.718281828))
        cdr = CdrDeserializer(data)
        val = cdr.read_float64()
        assert abs(val - 2.718281828) < 1e-9

    def test_read_float64_after_uint32(self):
        """float64 after a uint32 requires alignment padding to next 8-byte boundary."""
        # After header at pos 4, read uint32 (4 bytes) -> pos 8, offset 4.
        # Align to 8 means skip 4 padding bytes -> offset 8 -> pos 12.
        data = _le_payload(
            struct.pack("<I", 1)   # uint32 at offset 0
            + b"\x00" * 4          # padding to align to 8
            + struct.pack("<d", 1.23456789),
        )
        cdr = CdrDeserializer(data)
        assert cdr.read_uint32() == 1
        val = cdr.read_float64()
        assert abs(val - 1.23456789) < 1e-9


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


class TestAlignment:
    """Test CDR alignment padding."""

    def test_align_after_uint8(self):
        """After reading a uint8, aligning to 4 should skip 3 bytes."""
        data = _le_payload(
            b"\x01"  # uint8
            + b"\x00\x00\x00"  # padding
            + struct.pack("<I", 42),  # uint32
        )
        cdr = CdrDeserializer(data)
        assert cdr.read_uint8() == 1
        cdr.align(4)
        assert cdr.read_uint32() == 42

    def test_align_already_aligned(self):
        """Aligning when already aligned should be a no-op."""
        data = _le_payload(struct.pack("<I", 7))
        cdr = CdrDeserializer(data)
        cdr.align(4)  # pos 4, offset 0: already 4-aligned
        assert cdr.read_uint32() == 7

    def test_align_to_8_after_uint32(self):
        """uint32 at offset 0 (4 bytes) then align to 8 should skip 4 bytes."""
        data = _le_payload(
            struct.pack("<I", 1)  # 4 bytes at offset 0
            + b"\x00" * 4  # padding to 8-byte boundary
            + struct.pack("<d", 9.99),
        )
        cdr = CdrDeserializer(data)
        assert cdr.read_uint32() == 1
        cdr.align(8)
        assert abs(cdr.read_float64() - 9.99) < 1e-9

    def test_align_to_2_after_uint8(self):
        data = _le_payload(
            b"\x0A"  # uint8 at offset 0
            + b"\x00"  # padding
            + struct.pack("<H", 500),  # uint16 at offset 2
        )
        cdr = CdrDeserializer(data)
        assert cdr.read_uint8() == 0x0A
        cdr.align(2)
        assert cdr.read_uint16() == 500


# ---------------------------------------------------------------------------
# String reading
# ---------------------------------------------------------------------------


class TestStringReading:
    """Test CDR string deserialization."""

    def test_simple_string(self):
        s = "hello"
        encoded = s.encode("utf-8") + b"\x00"
        data = _le_payload(struct.pack("<I", len(encoded)) + encoded)
        cdr = CdrDeserializer(data)
        assert cdr.read_string() == "hello"

    def test_empty_string(self):
        """A string with length 0 should return empty string."""
        data = _le_payload(struct.pack("<I", 0))
        cdr = CdrDeserializer(data)
        assert cdr.read_string() == ""

    def test_string_with_special_chars(self):
        s = "base_link/camera_optical"
        encoded = s.encode("utf-8") + b"\x00"
        data = _le_payload(struct.pack("<I", len(encoded)) + encoded)
        cdr = CdrDeserializer(data)
        assert cdr.read_string() == s

    def test_single_char_string(self):
        s = "x"
        encoded = s.encode("utf-8") + b"\x00"
        data = _le_payload(struct.pack("<I", len(encoded)) + encoded)
        cdr = CdrDeserializer(data)
        assert cdr.read_string() == "x"


# ---------------------------------------------------------------------------
# Sequence reading
# ---------------------------------------------------------------------------


class TestSequenceReading:
    """Test CDR sequence deserialization."""

    def test_sequence_of_uint32(self):
        items = [10, 20, 30]
        data = _le_payload(
            struct.pack("<I", 3)  # count
            + struct.pack("<III", *items),
        )
        cdr = CdrDeserializer(data)
        result = cdr.read_sequence(cdr.read_uint32)
        assert result == items

    def test_empty_sequence(self):
        data = _le_payload(struct.pack("<I", 0))
        cdr = CdrDeserializer(data)
        result = cdr.read_sequence(cdr.read_uint32)
        assert result == []

    def test_sequence_of_float64(self):
        items = [1.1, 2.2, 3.3]
        # count (uint32) at offset 0, then align to 8 for float64
        payload = struct.pack("<I", 3) + b"\x00" * 4  # pad to 8
        payload += struct.pack("<ddd", *items)
        data = _le_payload(payload)
        cdr = CdrDeserializer(data)
        result = cdr.read_sequence(cdr.read_float64)
        assert len(result) == 3
        for got, expected in zip(result, items):
            assert abs(got - expected) < 1e-9

    def test_sequence_of_strings(self):
        strings = ["joint1", "joint2"]
        # Build payload respecting CDR alignment for each string's uint32 length prefix
        payload = bytearray(struct.pack("<I", 2))  # count
        for s in strings:
            encoded = s.encode("utf-8") + b"\x00"
            # Align to 4 for the uint32 length prefix
            remainder = len(payload) % 4
            if remainder != 0:
                payload += b"\x00" * (4 - remainder)
            payload += struct.pack("<I", len(encoded)) + encoded
        data = _le_payload(bytes(payload))
        cdr = CdrDeserializer(data)
        result = cdr.read_sequence(cdr.read_string)
        assert result == strings


# ---------------------------------------------------------------------------
# deserialize_header
# ---------------------------------------------------------------------------


class TestDeserializeHeader:
    """Test the deserialize_header convenience function."""

    def test_basic_header(self):
        sec, nanosec, frame_id = 1700000000, 123456789, "imu_link"
        encoded_frame = frame_id.encode("utf-8") + b"\x00"
        payload = (
            struct.pack("<I", sec)
            + struct.pack("<I", nanosec)
            + struct.pack("<I", len(encoded_frame))
            + encoded_frame
        )
        data = _le_payload(payload)
        cdr = CdrDeserializer(data)
        header = deserialize_header(cdr)

        assert isinstance(header, Header)
        assert header.sec == sec
        assert header.nanosec == nanosec
        assert header.frame_id == frame_id

    def test_header_to_timestamp(self):
        header = Header(sec=100, nanosec=500_000_000, frame_id="test")
        assert abs(header.to_timestamp() - 100.5) < 1e-9

    def test_header_empty_frame_id(self):
        encoded_frame = b"\x00"  # just null terminator
        payload = (
            struct.pack("<I", 0)
            + struct.pack("<I", 0)
            + struct.pack("<I", 1)
            + encoded_frame
        )
        data = _le_payload(payload)
        cdr = CdrDeserializer(data)
        header = deserialize_header(cdr)
        assert header.frame_id == ""


# ---------------------------------------------------------------------------
# pos and remaining properties
# ---------------------------------------------------------------------------


class TestCursorProperties:
    """Test pos and remaining properties."""

    def test_pos_advances(self):
        data = _le_payload(struct.pack("<I", 0))
        cdr = CdrDeserializer(data)
        assert cdr.pos == 4  # after encapsulation header
        cdr.read_uint32()
        assert cdr.pos == 8

    def test_remaining_decreases(self):
        data = _le_payload(struct.pack("<I", 0))
        cdr = CdrDeserializer(data)
        initial_remaining = cdr.remaining
        cdr.read_uint32()
        assert cdr.remaining == initial_remaining - 4
