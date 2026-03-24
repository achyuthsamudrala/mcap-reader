"""Tests for message wrappers (Imu, JointState, Image, PointCloud2, CameraInfo).

Uses hand-crafted CDR bytes via from_cdr to avoid needing actual mcap-decoded
ROS objects.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from mcap_reader.messages.imu import Imu
from mcap_reader.messages.joint_state import JointState
from mcap_reader.messages.image import Image, ENCODING_DTYPE_MAP
from mcap_reader.messages.pointcloud import PointCloud2, PointField
from mcap_reader.messages.camera_info import CameraInfo
from mcap_reader.messages.compressed_image import CompressedImage
from mcap_reader.messages.transform import TFMessage, TransformStamped

from tests.generators.generate_mcap import (
    _cdr_header,
    _cdr_ros_header,
    _cdr_uint32,
    _cdr_uint8,
    _cdr_float64,
    _cdr_float32,
    _cdr_string,
    _pad_to_align,
    _build_imu_cdr as _build_imu_cdr_raw,
    _build_joint_state_cdr as _build_joint_state_cdr_raw,
    _build_image_cdr as _build_image_cdr_raw,
    _build_tf_message_cdr as _build_tf_message_cdr_raw,
)

# Our CdrDeserializer uses 0x0002 for LE, so we wrap the builders
# to use the correct encapsulation header for our own deserializer.
_OUR_CDR_HEADER = _cdr_header()


def _build_imu_cdr(**kwargs):
    kwargs.setdefault("encap_header", _OUR_CDR_HEADER)
    return _build_imu_cdr_raw(**kwargs)


def _build_joint_state_cdr(**kwargs):
    kwargs.setdefault("encap_header", _OUR_CDR_HEADER)
    return _build_joint_state_cdr_raw(**kwargs)


def _build_image_cdr(**kwargs):
    kwargs.setdefault("encap_header", _OUR_CDR_HEADER)
    return _build_image_cdr_raw(**kwargs)


def _build_tf_message_cdr(transforms, **kwargs):
    kwargs.setdefault("encap_header", _OUR_CDR_HEADER)
    return _build_tf_message_cdr_raw(transforms, **kwargs)


# ---------------------------------------------------------------------------
# Imu
# ---------------------------------------------------------------------------


class TestImu:
    """Test the Imu message wrapper."""

    def test_from_cdr_basic(self):
        data = _build_imu_cdr(
            sec=100, nanosec=500_000_000, frame_id="imu_link",
            orientation=(0.0, 0.0, 0.0, 1.0),
            angular_velocity=(1.0, 2.0, 3.0),
            linear_acceleration=(0.0, 0.0, 9.81),
        )
        imu = Imu.from_cdr(data)

        assert imu.header.sec == 100
        assert imu.header.nanosec == 500_000_000
        assert imu.header.frame_id == "imu_link"
        assert abs(imu.orientation.w - 1.0) < 1e-9
        assert abs(imu.angular_velocity.x - 1.0) < 1e-9
        assert abs(imu.angular_velocity.y - 2.0) < 1e-9
        assert abs(imu.angular_velocity.z - 3.0) < 1e-9
        assert abs(imu.linear_acceleration.z - 9.81) < 1e-6

    def test_has_orientation_true(self):
        """Default zero covariance means orientation IS provided."""
        data = _build_imu_cdr(
            sec=0, nanosec=0, frame_id="",
            orientation=(0.0, 0.0, 0.0, 1.0),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_acceleration=(0.0, 0.0, 0.0),
        )
        imu = Imu.from_cdr(data)
        assert imu.has_orientation == True

    def test_has_orientation_false(self):
        """Covariance[0,0] == -1 means orientation is NOT provided."""
        cov = [-1.0] + [0.0] * 8
        data = _build_imu_cdr(
            sec=0, nanosec=0, frame_id="",
            orientation=(0.0, 0.0, 0.0, 1.0),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_acceleration=(0.0, 0.0, 0.0),
            orientation_cov=cov,
        )
        imu = Imu.from_cdr(data)
        assert imu.has_orientation == False

    def test_has_angular_velocity_false(self):
        cov = [-1.0] + [0.0] * 8
        data = _build_imu_cdr(
            sec=0, nanosec=0, frame_id="",
            orientation=(0.0, 0.0, 0.0, 1.0),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_acceleration=(0.0, 0.0, 0.0),
            angular_velocity_cov=cov,
        )
        imu = Imu.from_cdr(data)
        assert imu.has_angular_velocity == False

    def test_has_linear_acceleration_false(self):
        cov = [-1.0] + [0.0] * 8
        data = _build_imu_cdr(
            sec=0, nanosec=0, frame_id="",
            orientation=(0.0, 0.0, 0.0, 1.0),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_acceleration=(0.0, 0.0, 0.0),
            linear_acceleration_cov=cov,
        )
        imu = Imu.from_cdr(data)
        assert imu.has_linear_acceleration == False

    def test_covariance_shape(self):
        data = _build_imu_cdr(
            sec=0, nanosec=0, frame_id="",
            orientation=(0.0, 0.0, 0.0, 1.0),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_acceleration=(0.0, 0.0, 0.0),
        )
        imu = Imu.from_cdr(data)
        assert imu.orientation_covariance.shape == (3, 3)
        assert imu.angular_velocity_covariance.shape == (3, 3)
        assert imu.linear_acceleration_covariance.shape == (3, 3)

    def test_to_dict(self):
        data = _build_imu_cdr(
            sec=10, nanosec=0, frame_id="test",
            orientation=(0.0, 0.0, 0.0, 1.0),
            angular_velocity=(1.0, 0.0, 0.0),
            linear_acceleration=(0.0, 0.0, 9.81),
        )
        imu = Imu.from_cdr(data)
        d = imu.to_dict()
        assert d["header"]["sec"] == 10
        assert d["angular_velocity"]["x"] == 1.0

    def test_to_pandas_row(self):
        data = _build_imu_cdr(
            sec=10, nanosec=0, frame_id="test",
            orientation=(0.1, 0.2, 0.3, 0.9),
            angular_velocity=(1.0, 2.0, 3.0),
            linear_acceleration=(4.0, 5.0, 6.0),
        )
        imu = Imu.from_cdr(data)
        row = imu.to_pandas_row()
        assert "timestamp" in row
        assert abs(row["angular_velocity_x"] - 1.0) < 1e-9
        assert abs(row["linear_acceleration_z"] - 6.0) < 1e-9


# ---------------------------------------------------------------------------
# JointState
# ---------------------------------------------------------------------------


class TestJointState:
    """Test the JointState message wrapper."""

    def test_from_cdr_basic(self):
        data = _build_joint_state_cdr(
            sec=100, nanosec=0, frame_id="",
            names=["joint1", "joint2"],
            positions=[1.0, 2.0],
            velocities=[0.5, 0.6],
            efforts=[10.0, 20.0],
        )
        js = JointState.from_cdr(data)

        assert js.name == ["joint1", "joint2"]
        assert len(js.position) == 2
        assert abs(js.position[0] - 1.0) < 1e-9
        assert abs(js.position[1] - 2.0) < 1e-9
        assert abs(js.velocity[0] - 0.5) < 1e-9
        assert abs(js.effort[1] - 20.0) < 1e-9

    def test_parallel_arrays(self):
        """Joint arrays should have same length as name list."""
        data = _build_joint_state_cdr(
            sec=0, nanosec=0, frame_id="",
            names=["a", "b", "c"],
            positions=[1.0, 2.0, 3.0],
            velocities=[4.0, 5.0, 6.0],
            efforts=[7.0, 8.0, 9.0],
        )
        js = JointState.from_cdr(data)
        assert len(js.name) == len(js.position) == len(js.velocity) == len(js.effort) == 3

    def test_empty_velocity_and_effort(self):
        """Velocity and effort can be empty arrays."""
        data = _build_joint_state_cdr(
            sec=0, nanosec=0, frame_id="",
            names=["j1"],
            positions=[1.5],
            velocities=[],
            efforts=[],
        )
        js = JointState.from_cdr(data)
        assert js.has_position is True
        assert js.has_velocity is False
        assert js.has_effort is False

    def test_get_joint(self):
        data = _build_joint_state_cdr(
            sec=0, nanosec=0, frame_id="",
            names=["shoulder", "elbow"],
            positions=[0.5, 1.2],
            velocities=[0.1, 0.2],
            efforts=[5.0, 10.0],
        )
        js = JointState.from_cdr(data)
        joint = js.get_joint("elbow")
        assert abs(joint["position"] - 1.2) < 1e-9
        assert abs(joint["velocity"] - 0.2) < 1e-9
        assert abs(joint["effort"] - 10.0) < 1e-9

    def test_get_joint_not_found(self):
        data = _build_joint_state_cdr(
            sec=0, nanosec=0, frame_id="",
            names=["j1"],
            positions=[0.0],
            velocities=[],
            efforts=[],
        )
        js = JointState.from_cdr(data)
        with pytest.raises(KeyError):
            js.get_joint("nonexistent")

    def test_to_pandas_row(self):
        data = _build_joint_state_cdr(
            sec=100, nanosec=500_000_000, frame_id="",
            names=["j1", "j2"],
            positions=[1.0, 2.0],
            velocities=[0.1, 0.2],
            efforts=[],
        )
        js = JointState.from_cdr(data)
        row = js.to_pandas_row()
        assert "timestamp" in row
        assert "position_j1" in row
        assert "position_j2" in row
        assert "velocity_j1" in row
        # effort should not be present since it's empty
        assert "effort_j1" not in row

    def test_to_dict(self):
        data = _build_joint_state_cdr(
            sec=0, nanosec=0, frame_id="",
            names=["j1"],
            positions=[3.14],
            velocities=[1.0],
            efforts=[5.0],
        )
        js = JointState.from_cdr(data)
        d = js.to_dict()
        assert "j1" in d
        assert abs(d["j1"]["position"] - 3.14) < 1e-9


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


class TestImage:
    """Test the Image message wrapper."""

    def test_from_cdr_rgb8(self):
        height, width = 4, 8
        channels = 3
        step = width * channels
        pixel_data = bytes([i % 256 for i in range(height * step)])

        data = _build_image_cdr(
            sec=0, nanosec=0, frame_id="cam",
            height=height, width=width,
            encoding="rgb8", is_bigendian=0,
            step=step, pixel_data=pixel_data,
        )
        img = Image.from_cdr(data)
        assert img.height == height
        assert img.width == width
        assert img.encoding == "rgb8"
        assert img.step == step
        assert len(img.data) == height * step

    def test_to_numpy_rgb8(self):
        height, width = 4, 8
        channels = 3
        step = width * channels
        pixel_data = bytes([i % 256 for i in range(height * step)])

        data = _build_image_cdr(
            sec=0, nanosec=0, frame_id="cam",
            height=height, width=width,
            encoding="rgb8", is_bigendian=0,
            step=step, pixel_data=pixel_data,
        )
        img = Image.from_cdr(data)
        arr = img.to_numpy()
        assert arr.shape == (height, width, channels)
        assert arr.dtype == np.uint8

    def test_to_numpy_mono8(self):
        height, width = 10, 10
        step = width
        pixel_data = bytes([i % 256 for i in range(height * step)])

        data = _build_image_cdr(
            sec=0, nanosec=0, frame_id="cam",
            height=height, width=width,
            encoding="mono8", is_bigendian=0,
            step=step, pixel_data=pixel_data,
        )
        img = Image.from_cdr(data)
        arr = img.to_numpy()
        assert arr.shape == (height, width)
        assert arr.dtype == np.uint8

    def test_to_numpy_mono16(self):
        height, width = 4, 4
        step = width * 2  # 2 bytes per pixel
        pixel_data = b""
        for i in range(height * width):
            pixel_data += struct.pack("<H", i * 100)

        data = _build_image_cdr(
            sec=0, nanosec=0, frame_id="depth",
            height=height, width=width,
            encoding="mono16", is_bigendian=0,
            step=step, pixel_data=pixel_data,
        )
        img = Image.from_cdr(data)
        arr = img.to_numpy()
        assert arr.shape == (height, width)
        assert arr.dtype.kind == "u"  # unsigned integer
        assert arr[0, 0] == 0
        assert arr[0, 1] == 100

    def test_unsupported_encoding_raises(self):
        data = _build_image_cdr(
            sec=0, nanosec=0, frame_id="",
            height=1, width=1,
            encoding="yuv422", is_bigendian=0,
            step=2, pixel_data=b"\x00\x00",
        )
        img = Image.from_cdr(data)
        with pytest.raises(ValueError, match="Unsupported encoding"):
            img.to_numpy()

    def test_gradient_pixel_values(self):
        """Verify pixel values are correctly reconstructed."""
        height, width = 2, 3
        channels = 3
        step = width * channels
        # Construct known pixel values
        pixels = []
        for r in range(height):
            for c in range(width):
                pixels.extend([r * 10, c * 20, 100])
        pixel_data = bytes(pixels)

        data = _build_image_cdr(
            sec=0, nanosec=0, frame_id="",
            height=height, width=width,
            encoding="rgb8", is_bigendian=0,
            step=step, pixel_data=pixel_data,
        )
        img = Image.from_cdr(data)
        arr = img.to_numpy()
        # Check specific pixels
        assert arr[0, 0, 0] == 0   # row 0 -> r=0
        assert arr[0, 1, 1] == 20  # col 1 -> c*20=20
        assert arr[1, 0, 0] == 10  # row 1 -> r*10=10


# ---------------------------------------------------------------------------
# PointCloud2
# ---------------------------------------------------------------------------


class TestPointCloud2:
    """Test the PointCloud2 message wrapper."""

    def _make_simple_cloud(self, n_points: int = 10) -> PointCloud2:
        """Create a simple XYZ point cloud without CDR (direct construction)."""
        from mcap_reader.deserializer import Header

        fields = [
            PointField(name="x", offset=0, datatype=7, count=1),   # FLOAT32
            PointField(name="y", offset=4, datatype=7, count=1),
            PointField(name="z", offset=8, datatype=7, count=1),
        ]
        point_step = 12
        row_step = point_step * n_points

        # Build raw data
        raw = bytearray()
        for i in range(n_points):
            raw += struct.pack("<fff", float(i), float(i * 2), float(i * 3))

        return PointCloud2(
            header=Header(sec=0, nanosec=0, frame_id="lidar"),
            height=1,
            width=n_points,
            fields=fields,
            is_bigendian=False,
            point_step=point_step,
            row_step=row_step,
            data=bytes(raw),
            is_dense=True,
        )

    def test_to_numpy_structured(self):
        cloud = self._make_simple_cloud(5)
        arr = cloud.to_numpy()
        assert len(arr) == 5
        assert "x" in arr.dtype.names
        assert "y" in arr.dtype.names
        assert "z" in arr.dtype.names

    def test_to_xyz(self):
        cloud = self._make_simple_cloud(5)
        xyz = cloud.to_xyz()
        assert xyz.shape == (5, 3)
        assert abs(xyz[0, 0] - 0.0) < 1e-6
        assert abs(xyz[1, 0] - 1.0) < 1e-6
        assert abs(xyz[2, 1] - 4.0) < 1e-6  # y=2*2=4
        assert abs(xyz[3, 2] - 9.0) < 1e-6  # z=3*3=9

    def test_num_points(self):
        cloud = self._make_simple_cloud(20)
        assert cloud.num_points == 20

    def test_is_organized(self):
        cloud = self._make_simple_cloud(10)
        assert cloud.is_organized is False  # height==1

    def test_point_field_numpy_dtype(self):
        pf = PointField(name="x", offset=0, datatype=7, count=1)
        assert pf.numpy_dtype == np.dtype(np.float32)

    def test_point_field_unknown_datatype(self):
        pf = PointField(name="x", offset=0, datatype=99, count=1)
        with pytest.raises(KeyError):
            _ = pf.numpy_dtype


# ---------------------------------------------------------------------------
# CameraInfo
# ---------------------------------------------------------------------------


class TestCameraInfo:
    """Test the CameraInfo message wrapper."""

    def _make_camera_info(self) -> CameraInfo:
        """Build a CameraInfo with known intrinsics."""
        from mcap_reader.deserializer import Header

        K = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        R = np.eye(3)
        P = np.zeros((3, 4))
        P[:3, :3] = K
        D = np.array([0.1, -0.2, 0.001, 0.002, 0.0])

        return CameraInfo(
            header=Header(sec=0, nanosec=0, frame_id="camera_optical"),
            height=480,
            width=640,
            distortion_model="plumb_bob",
            D=D,
            K=K,
            R=R,
            P=P,
        )

    def test_intrinsic_properties(self):
        ci = self._make_camera_info()
        assert abs(ci.fx - 500.0) < 1e-9
        assert abs(ci.fy - 500.0) < 1e-9
        assert abs(ci.cx - 320.0) < 1e-9
        assert abs(ci.cy - 240.0) < 1e-9

    def test_k_matrix_shape(self):
        ci = self._make_camera_info()
        assert ci.K.shape == (3, 3)

    def test_p_matrix_shape(self):
        ci = self._make_camera_info()
        assert ci.P.shape == (3, 4)

    def test_d_coefficients(self):
        ci = self._make_camera_info()
        assert len(ci.D) == 5
        assert abs(ci.D[0] - 0.1) < 1e-9

    def test_to_cv2_camera_matrix(self):
        ci = self._make_camera_info()
        K_cv = ci.to_cv2_camera_matrix()
        np.testing.assert_array_almost_equal(K_cv, ci.K)
        # Should be a copy, not the same object
        assert K_cv is not ci.K

    def test_to_cv2_distortion(self):
        ci = self._make_camera_info()
        D_cv = ci.to_cv2_distortion()
        np.testing.assert_array_almost_equal(D_cv, ci.D)

    def test_from_cdr(self):
        """Test CameraInfo CDR deserialization with hand-crafted bytes."""
        payload = bytearray()
        # Header
        payload += _cdr_ros_header(0, 0, "cam")
        # height, width
        payload += _pad_to_align(len(payload), 4)
        payload += _cdr_uint32(480)
        payload += _cdr_uint32(640)
        # distortion_model
        payload += _cdr_string("plumb_bob")
        # D: sequence<float64> with 5 elements
        payload += _pad_to_align(len(payload), 4)
        payload += _cdr_uint32(5)
        payload += _pad_to_align(len(payload), 8)
        for d in [0.1, -0.2, 0.001, 0.002, 0.0]:
            payload += _cdr_float64(d)
        # K: 9 float64
        for v in [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]:
            payload += _cdr_float64(v)
        # R: 9 float64 (identity)
        for v in [1, 0, 0, 0, 1, 0, 0, 0, 1]:
            payload += _cdr_float64(float(v))
        # P: 12 float64
        for v in [500, 0, 320, 0, 0, 500, 240, 0, 0, 0, 1, 0]:
            payload += _cdr_float64(float(v))
        # binning_x, binning_y
        payload += _pad_to_align(len(payload), 4)
        payload += _cdr_uint32(0)
        payload += _cdr_uint32(0)
        # ROI: 4x uint32 + 1x uint8
        payload += _cdr_uint32(0)
        payload += _cdr_uint32(0)
        payload += _cdr_uint32(0)
        payload += _cdr_uint32(0)
        payload += _cdr_uint8(0)

        data = _cdr_header() + bytes(payload)
        ci = CameraInfo.from_cdr(data)

        assert ci.height == 480
        assert ci.width == 640
        assert ci.distortion_model == "plumb_bob"
        assert abs(ci.fx - 500.0) < 1e-9


# ---------------------------------------------------------------------------
# TFMessage
# ---------------------------------------------------------------------------


class TestTFMessage:
    """Test the TFMessage wrapper."""

    def test_from_cdr_single_transform(self):
        data = _build_tf_message_cdr([{
            "sec": 100, "nanosec": 0,
            "parent_frame": "world",
            "child_frame": "base_link",
            "translation": (1.0, 2.0, 3.0),
            "rotation": (0.0, 0.0, 0.0, 1.0),
        }])
        tf_msg = TFMessage.from_cdr(data)
        assert len(tf_msg.transforms) == 1
        ts = tf_msg.transforms[0]
        assert ts.header.frame_id == "world"
        assert ts.child_frame_id == "base_link"
        assert abs(ts.transform.translation.x - 1.0) < 1e-9
        assert abs(ts.transform.rotation.w - 1.0) < 1e-9

    def test_from_cdr_multiple_transforms(self):
        data = _build_tf_message_cdr([
            {
                "sec": 100, "nanosec": 0,
                "parent_frame": "world",
                "child_frame": "base_link",
                "translation": (1.0, 0.0, 0.0),
                "rotation": (0.0, 0.0, 0.0, 1.0),
            },
            {
                "sec": 100, "nanosec": 0,
                "parent_frame": "base_link",
                "child_frame": "camera",
                "translation": (0.0, 0.0, 0.5),
                "rotation": (0.0, 0.0, 0.0, 1.0),
            },
        ])
        tf_msg = TFMessage.from_cdr(data)
        assert len(tf_msg.transforms) == 2
        assert tf_msg.transforms[0].child_frame_id == "base_link"
        assert tf_msg.transforms[1].child_frame_id == "camera"


# ---------------------------------------------------------------------------
# CompressedImage
# ---------------------------------------------------------------------------


class TestCompressedImage:
    """Test the CompressedImage wrapper."""

    def test_from_cdr(self):
        payload = bytearray()
        payload += _cdr_ros_header(10, 0, "cam")
        payload += _cdr_string("jpeg")
        # data: sequence<uint8>
        fake_jpeg = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        payload += _pad_to_align(len(payload), 4)
        payload += _cdr_uint32(len(fake_jpeg))
        payload += fake_jpeg

        data = _cdr_header() + bytes(payload)
        ci = CompressedImage.from_cdr(data)

        assert ci.header.sec == 10
        assert ci.format == "jpeg"
        assert len(ci.data) == len(fake_jpeg)

    def test_parse_format_simple_jpeg(self):
        from mcap_reader.deserializer import Header
        ci = CompressedImage(
            header=Header(sec=0, nanosec=0, frame_id=""),
            format="jpeg",
            data=b"",
        )
        source, comp = ci.parse_format()
        assert source == "bgr8"
        assert comp == "jpeg"

    def test_parse_format_with_encoding(self):
        from mcap_reader.deserializer import Header
        ci = CompressedImage(
            header=Header(sec=0, nanosec=0, frame_id=""),
            format="bgr8; jpeg compressed",
            data=b"",
        )
        source, comp = ci.parse_format()
        assert source == "bgr8"
        assert comp == "jpeg"

    def test_parse_format_png(self):
        from mcap_reader.deserializer import Header
        ci = CompressedImage(
            header=Header(sec=0, nanosec=0, frame_id=""),
            format="16UC1; png compressed",
            data=b"",
        )
        source, comp = ci.parse_format()
        assert source == "16UC1"
        assert comp == "png"
