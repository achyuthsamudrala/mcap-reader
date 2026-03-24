"""Tests for quaternion math, transform system, frame graph, and transform buffer.

Includes property-based tests with hypothesis for quaternion operations,
and validation against scipy.spatial.transform.Rotation where possible.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcap_reader.transforms.math import (
    Quaternion,
    Transform,
    Vector3,
    interpolate_transform,
    lerp_vector,
    slerp,
)
from mcap_reader.transforms.frames import (
    FrameGraph,
    FrameNotFoundError,
    NoPathError,
)
from mcap_reader.transforms.buffer import TransformBuffer

# Try to import hypothesis and scipy; tests are skipped if unavailable.
try:
    from hypothesis import given, settings, assume
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

try:
    from scipy.spatial.transform import Rotation as ScipyRotation

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Strategy for generating unit quaternions (hypothesis)
# ---------------------------------------------------------------------------

if HAS_HYPOTHESIS:
    @st.composite
    def unit_quaternions(draw):
        """Generate random unit quaternions."""
        x = draw(st.floats(min_value=-1, max_value=1))
        y = draw(st.floats(min_value=-1, max_value=1))
        z = draw(st.floats(min_value=-1, max_value=1))
        w = draw(st.floats(min_value=-1, max_value=1))
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        assume(norm > 0.1)
        return Quaternion(x / norm, y / norm, z / norm, w / norm)


# ---------------------------------------------------------------------------
# Vector3 tests
# ---------------------------------------------------------------------------


class TestVector3:
    """Test Vector3 operations."""

    def test_creation(self):
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_addition(self):
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_subtraction(self):
        v1 = Vector3(4, 5, 6)
        v2 = Vector3(1, 2, 3)
        result = v1 - v2
        assert result.x == 3
        assert result.y == 3
        assert result.z == 3

    def test_scalar_multiply(self):
        v = Vector3(1, 2, 3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_scalar_rmul(self):
        v = Vector3(1, 2, 3)
        result = 3 * v
        assert result.x == 3

    def test_negation(self):
        v = Vector3(1, -2, 3)
        result = -v
        assert result.x == -1
        assert result.y == 2
        assert result.z == -3

    def test_dot_product(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        assert v1.dot(v2) == 0.0

    def test_cross_product(self):
        v1 = Vector3(1, 0, 0)
        v2 = Vector3(0, 1, 0)
        result = v1.cross(v2)
        assert abs(result.z - 1.0) < 1e-9

    def test_norm(self):
        v = Vector3(3, 4, 0)
        assert abs(v.norm() - 5.0) < 1e-9

    def test_to_numpy(self):
        v = Vector3(1, 2, 3)
        arr = v.to_numpy()
        np.testing.assert_array_almost_equal(arr, [1, 2, 3])

    def test_from_numpy(self):
        arr = np.array([4.0, 5.0, 6.0])
        v = Vector3.from_numpy(arr)
        assert v.x == 4.0
        assert v.y == 5.0
        assert v.z == 6.0

    def test_from_numpy_wrong_size(self):
        with pytest.raises(ValueError):
            Vector3.from_numpy(np.array([1.0, 2.0]))

    def test_division(self):
        v = Vector3(6, 8, 10)
        result = v / 2
        assert result.x == 3.0


# ---------------------------------------------------------------------------
# Quaternion tests
# ---------------------------------------------------------------------------


class TestQuaternion:
    """Test Quaternion operations."""

    def test_identity(self):
        q = Quaternion.identity()
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0
        assert q.w == 1.0

    def test_norm_unit(self):
        q = Quaternion(0, 0, 0, 1)
        assert abs(q.norm() - 1.0) < 1e-12

    def test_normalize(self):
        q = Quaternion(1, 1, 1, 1)
        qn = q.normalize()
        assert abs(qn.norm() - 1.0) < 1e-12

    def test_normalize_zero_raises(self):
        q = Quaternion(0, 0, 0, 0)
        with pytest.raises(ValueError):
            q.normalize()

    def test_conjugate(self):
        q = Quaternion(1, 2, 3, 4)
        qc = q.conjugate()
        assert qc.x == -1
        assert qc.y == -2
        assert qc.z == -3
        assert qc.w == 4

    def test_inverse_unit(self):
        """For unit quaternion, inverse == conjugate."""
        q = Quaternion(0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        qi = q.inverse()
        qc = q.conjugate()
        assert abs(qi.x - qc.x) < 1e-9
        assert abs(qi.y - qc.y) < 1e-9
        assert abs(qi.z - qc.z) < 1e-9
        assert abs(qi.w - qc.w) < 1e-9

    def test_inverse_zero_raises(self):
        q = Quaternion(0, 0, 0, 0)
        with pytest.raises(ValueError):
            q.inverse()

    def test_hamilton_product_identity(self):
        """q * identity = q."""
        q = Quaternion(0.1, 0.2, 0.3, 0.9).normalize()
        identity = Quaternion.identity()
        result = q * identity
        assert abs(result.x - q.x) < 1e-9
        assert abs(result.w - q.w) < 1e-9

    def test_angular_distance_same(self):
        q = Quaternion(0, 0, 0, 1)
        assert abs(q.angular_distance(q)) < 1e-9

    def test_angular_distance_180(self):
        """180 degree rotation should give distance ~pi."""
        q1 = Quaternion(0, 0, 0, 1)
        q2 = Quaternion(0, 0, 1, 0)  # 180 deg about Z
        dist = q1.angular_distance(q2)
        assert abs(dist - math.pi) < 1e-6

    def test_to_rotation_matrix_identity(self):
        q = Quaternion.identity()
        R = q.to_rotation_matrix()
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_rotation_matrix_determinant(self):
        """Rotation matrix should have det = +1."""
        q = Quaternion(0.1, 0.2, 0.3, 0.9).normalize()
        R = q.to_rotation_matrix()
        assert abs(np.linalg.det(R) - 1.0) < 1e-9

    def test_rotation_matrix_orthogonal(self):
        """R @ R.T should be identity."""
        q = Quaternion(0.1, 0.2, 0.3, 0.9).normalize()
        R = q.to_rotation_matrix()
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=9)

    def test_from_rotation_matrix_roundtrip(self):
        q_orig = Quaternion(0.1, 0.2, 0.3, 0.9).normalize()
        R = q_orig.to_rotation_matrix()
        q_recovered = Quaternion.from_rotation_matrix(R)
        # Check that they represent the same rotation (q or -q)
        dist = q_orig.angular_distance(q_recovered)
        assert dist < 1e-6

    def test_to_numpy(self):
        q = Quaternion(1, 2, 3, 4)
        arr = q.to_numpy()
        np.testing.assert_array_equal(arr, [1, 2, 3, 4])

    def test_from_numpy(self):
        q = Quaternion.from_numpy(np.array([0.1, 0.2, 0.3, 0.9]))
        assert abs(q.x - 0.1) < 1e-9
        assert abs(q.w - 0.9) < 1e-9

    def test_from_numpy_wrong_size(self):
        with pytest.raises(ValueError):
            Quaternion.from_numpy(np.array([1, 2, 3]))


# ---------------------------------------------------------------------------
# Quaternion property-based tests (hypothesis)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestQuaternionProperties:
    """Property-based tests for quaternion operations."""

    @given(q=unit_quaternions())
    @settings(max_examples=50)
    def test_compose_with_inverse_is_identity(self, q):
        """q * q^{-1} should be approximately the identity quaternion."""
        q_inv = q.inverse()
        result = q * q_inv
        result = result.normalize()
        # The identity quaternion is (0, 0, 0, 1) or (0, 0, 0, -1)
        assert abs(abs(result.w) - 1.0) < 1e-6
        assert abs(result.x) < 1e-6
        assert abs(result.y) < 1e-6
        assert abs(result.z) < 1e-6

    @given(q1=unit_quaternions(), q2=unit_quaternions())
    @settings(max_examples=50)
    def test_slerp_at_0_returns_q1(self, q1, q2):
        """slerp(q1, q2, 0.0) should return q1."""
        result = slerp(q1, q2, 0.0)
        dist = q1.angular_distance(result)
        assert dist < 1e-4

    @given(q1=unit_quaternions(), q2=unit_quaternions())
    @settings(max_examples=50)
    def test_slerp_at_1_returns_q2(self, q1, q2):
        """slerp(q1, q2, 1.0) should return q2."""
        result = slerp(q1, q2, 1.0)
        dist = q2.angular_distance(result)
        assert dist < 1e-4

    @given(q=unit_quaternions())
    @settings(max_examples=50)
    def test_rotation_matrix_det_is_1(self, q):
        """to_rotation_matrix should produce det=+1."""
        R = q.to_rotation_matrix()
        assert abs(np.linalg.det(R) - 1.0) < 1e-6

    @given(q=unit_quaternions())
    @settings(max_examples=50)
    def test_rotation_matrix_orthogonal_property(self, q):
        """R @ R.T should be identity."""
        R = q.to_rotation_matrix()
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=5)

    @given(q1=unit_quaternions(), q2=unit_quaternions())
    @settings(max_examples=50)
    def test_angular_distance_symmetric(self, q1, q2):
        """angular_distance(q1, q2) == angular_distance(q2, q1)."""
        d1 = q1.angular_distance(q2)
        d2 = q2.angular_distance(q1)
        assert abs(d1 - d2) < 1e-9

    @given(q1=unit_quaternions(), q2=unit_quaternions())
    @settings(max_examples=50)
    def test_angular_distance_non_negative(self, q1, q2):
        """Angular distance should be >= 0."""
        assert q1.angular_distance(q2) >= -1e-12


# ---------------------------------------------------------------------------
# scipy validation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestQuaternionScipyValidation:
    """Validate quaternion operations against scipy.spatial.transform.Rotation."""

    def test_rotation_matrix_matches_scipy(self):
        q = Quaternion(0.1, 0.2, 0.3, 0.9).normalize()
        R_ours = q.to_rotation_matrix()
        # scipy uses (x, y, z, w) order
        R_scipy = ScipyRotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        np.testing.assert_array_almost_equal(R_ours, R_scipy, decimal=9)

    def test_hamilton_product_matches_scipy(self):
        q1 = Quaternion(0.1, 0.2, 0.3, 0.9).normalize()
        q2 = Quaternion(0.4, -0.1, 0.2, 0.8).normalize()
        result = (q1 * q2).normalize()

        r1 = ScipyRotation.from_quat([q1.x, q1.y, q1.z, q1.w])
        r2 = ScipyRotation.from_quat([q2.x, q2.y, q2.z, q2.w])
        r_combined = r1 * r2
        q_scipy = r_combined.as_quat()  # (x, y, z, w)

        # Compare rotation matrices (avoids sign ambiguity)
        R_ours = result.to_rotation_matrix()
        R_scipy = r_combined.as_matrix()
        np.testing.assert_array_almost_equal(R_ours, R_scipy, decimal=9)

    def test_from_rotation_matrix_matches_scipy(self):
        """Round-trip through rotation matrix should match scipy."""
        angle = 0.7
        axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
        r_scipy = ScipyRotation.from_rotvec(angle * axis)
        R = r_scipy.as_matrix()

        q_ours = Quaternion.from_rotation_matrix(R)
        R_ours = q_ours.to_rotation_matrix()
        np.testing.assert_array_almost_equal(R_ours, R, decimal=9)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class TestTransform:
    """Test rigid-body Transform operations."""

    def test_identity(self):
        t = Transform.identity()
        assert t.translation.x == 0
        assert t.rotation.w == 1.0

    def test_apply_identity(self):
        t = Transform.identity()
        p = Vector3(1, 2, 3)
        result = t.apply(p)
        assert abs(result.x - 1) < 1e-9
        assert abs(result.y - 2) < 1e-9
        assert abs(result.z - 3) < 1e-9

    def test_apply_translation_only(self):
        t = Transform(Vector3(10, 20, 30), Quaternion.identity())
        p = Vector3(1, 2, 3)
        result = t.apply(p)
        assert abs(result.x - 11) < 1e-9
        assert abs(result.y - 22) < 1e-9
        assert abs(result.z - 33) < 1e-9

    def test_composition(self):
        t1 = Transform(Vector3(1, 0, 0), Quaternion.identity())
        t2 = Transform(Vector3(0, 1, 0), Quaternion.identity())
        t12 = t1 * t2
        # Pure translations compose additively
        assert abs(t12.translation.x - 1) < 1e-9
        assert abs(t12.translation.y - 1) < 1e-9

    def test_inverse(self):
        t = Transform(Vector3(1, 2, 3), Quaternion.identity())
        t_inv = t.inverse()
        # T * T_inv should be identity
        result = t * t_inv
        assert abs(result.translation.x) < 1e-9
        assert abs(result.translation.y) < 1e-9
        assert abs(result.translation.z) < 1e-9

    def test_to_matrix(self):
        t = Transform(Vector3(1, 2, 3), Quaternion.identity())
        m = t.to_matrix()
        assert m.shape == (4, 4)
        assert abs(m[0, 3] - 1) < 1e-9
        assert abs(m[3, 3] - 1) < 1e-9
        np.testing.assert_array_almost_equal(m[:3, :3], np.eye(3))

    def test_from_matrix_roundtrip(self):
        t = Transform(
            Vector3(1, 2, 3),
            Quaternion(0.1, 0.2, 0.3, 0.9).normalize(),
        )
        m = t.to_matrix()
        t2 = Transform.from_matrix(m)
        p = Vector3(5, 6, 7)
        r1 = t.apply(p)
        r2 = t2.apply(p)
        assert abs(r1.x - r2.x) < 1e-6
        assert abs(r1.y - r2.y) < 1e-6
        assert abs(r1.z - r2.z) < 1e-6


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


class TestInterpolation:
    """Test SLERP and linear interpolation."""

    def test_slerp_identity(self):
        q = Quaternion.identity()
        result = slerp(q, q, 0.5)
        assert abs(result.w - 1.0) < 1e-6

    def test_slerp_midpoint_90_deg(self):
        """SLERP midpoint of 0-degree and 90-degree rotation about Z."""
        q1 = Quaternion.identity()
        # 90 degrees about Z: sin(45 deg) = cos(45 deg) = sqrt(2)/2
        q2 = Quaternion(0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4))
        mid = slerp(q1, q2, 0.5)
        # Should be 45 degrees about Z
        expected_angle = math.pi / 4
        dist_from_identity = q1.angular_distance(mid)
        assert abs(dist_from_identity - expected_angle) < 1e-4

    def test_lerp_vector(self):
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 20, 30)
        mid = lerp_vector(v1, v2, 0.5)
        assert abs(mid.x - 5) < 1e-9
        assert abs(mid.y - 10) < 1e-9
        assert abs(mid.z - 15) < 1e-9

    def test_lerp_vector_endpoints(self):
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        at_0 = lerp_vector(v1, v2, 0.0)
        at_1 = lerp_vector(v1, v2, 1.0)
        assert abs(at_0.x - v1.x) < 1e-9
        assert abs(at_1.x - v2.x) < 1e-9

    def test_interpolate_transform(self):
        t1 = Transform(Vector3(0, 0, 0), Quaternion.identity())
        t2 = Transform(Vector3(10, 0, 0), Quaternion.identity())
        mid = interpolate_transform(t1, t2, 0.5)
        assert abs(mid.translation.x - 5) < 1e-9


# ---------------------------------------------------------------------------
# FrameGraph
# ---------------------------------------------------------------------------


class TestFrameGraph:
    """Test FrameGraph BFS path finding."""

    def _build_graph(self) -> FrameGraph:
        g = FrameGraph()
        g.add_edge("map", "odom")
        g.add_edge("odom", "base_link")
        g.add_edge("base_link", "camera_link")
        g.add_edge("camera_link", "camera_optical")
        return g

    def test_has_frame(self):
        g = self._build_graph()
        assert g.has_frame("map")
        assert g.has_frame("base_link")
        assert not g.has_frame("nonexistent")

    def test_get_chain_direct(self):
        g = self._build_graph()
        path = g.get_chain("map", "odom")
        assert path == ["map", "odom"]

    def test_get_chain_multi_hop(self):
        g = self._build_graph()
        path = g.get_chain("map", "camera_optical")
        assert path == ["map", "odom", "base_link", "camera_link", "camera_optical"]

    def test_get_chain_reverse(self):
        """BFS should work in reverse direction too."""
        g = self._build_graph()
        path = g.get_chain("camera_optical", "map")
        assert path[0] == "camera_optical"
        assert path[-1] == "map"
        assert len(path) == 5

    def test_get_chain_same_frame(self):
        g = self._build_graph()
        path = g.get_chain("base_link", "base_link")
        assert path == ["base_link"]

    def test_frame_not_found_source(self):
        g = self._build_graph()
        with pytest.raises(FrameNotFoundError):
            g.get_chain("nonexistent", "map")

    def test_frame_not_found_target(self):
        g = self._build_graph()
        with pytest.raises(FrameNotFoundError):
            g.get_chain("map", "nonexistent")

    def test_no_path_disconnected(self):
        g = FrameGraph()
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        with pytest.raises(NoPathError):
            g.get_chain("a", "d")

    def test_get_parent(self):
        g = self._build_graph()
        assert g.get_parent("odom") == "map"
        assert g.get_parent("map") is None

    def test_get_children(self):
        g = self._build_graph()
        children = g.get_children("base_link")
        assert "camera_link" in children

    def test_all_frames(self):
        g = self._build_graph()
        frames = g.all_frames()
        assert frames == {"map", "odom", "base_link", "camera_link", "camera_optical"}

    def test_to_ascii_tree(self):
        g = self._build_graph()
        tree = g.to_ascii_tree()
        assert "map" in tree
        assert "camera_optical" in tree


# ---------------------------------------------------------------------------
# TransformBuffer
# ---------------------------------------------------------------------------


class TestTransformBuffer:
    """Test TransformBuffer lookup with interpolation."""

    def test_static_transform(self):
        buf = TransformBuffer()
        tf = Transform(Vector3(1, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf, timestamp=0.0, is_static=True)

        result = buf.lookup_transform("world", "base_link", timestamp=100.0)
        assert abs(result.translation.x - 1.0) < 1e-9

    def test_dynamic_exact_time(self):
        buf = TransformBuffer()
        tf1 = Transform(Vector3(0, 0, 0), Quaternion.identity())
        tf2 = Transform(Vector3(10, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf1, timestamp=1.0)
        buf.add_transform("world", "base_link", tf2, timestamp=2.0)

        result = buf.lookup_transform("world", "base_link", timestamp=1.0)
        assert abs(result.translation.x) < 1e-9

    def test_dynamic_interpolation(self):
        buf = TransformBuffer()
        tf1 = Transform(Vector3(0, 0, 0), Quaternion.identity())
        tf2 = Transform(Vector3(10, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf1, timestamp=1.0)
        buf.add_transform("world", "base_link", tf2, timestamp=2.0)

        result = buf.lookup_transform("world", "base_link", timestamp=1.5)
        assert abs(result.translation.x - 5.0) < 1e-6

    def test_lookup_same_frame(self):
        buf = TransformBuffer()
        tf = Transform(Vector3(1, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf, timestamp=1.0)

        result = buf.lookup_transform("world", "world", timestamp=1.0)
        assert abs(result.translation.x) < 1e-9

    def test_lookup_chain(self):
        """Look up transform through a chain of frames."""
        buf = TransformBuffer()
        t1 = Transform(Vector3(1, 0, 0), Quaternion.identity())
        t2 = Transform(Vector3(0, 2, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", t1, timestamp=1.0)
        buf.add_transform("base_link", "camera", t2, timestamp=1.0)

        result = buf.lookup_transform("world", "camera", timestamp=1.0)
        # Translation should compose: (1,0,0) + identity_rot @ (0,2,0) = (1,2,0)
        assert abs(result.translation.x - 1.0) < 1e-6
        assert abs(result.translation.y - 2.0) < 1e-6

    def test_lookup_reverse_direction(self):
        buf = TransformBuffer()
        tf = Transform(Vector3(1, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf, timestamp=1.0)

        result = buf.lookup_transform("base_link", "world", timestamp=1.0)
        assert abs(result.translation.x - (-1.0)) < 1e-6

    def test_can_transform(self):
        buf = TransformBuffer()
        tf = Transform(Vector3(1, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf, timestamp=1.0)

        assert buf.can_transform("world", "base_link", 1.0) is True
        assert buf.can_transform("world", "nonexistent", 1.0) is False

    def test_get_frames(self):
        buf = TransformBuffer()
        tf = Transform(Vector3(0, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf, timestamp=1.0)
        frames = buf.get_frames()
        assert "world" in frames
        assert "base_link" in frames

    def test_clear(self):
        buf = TransformBuffer()
        tf = Transform(Vector3(0, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf, timestamp=1.0)
        buf.clear()
        assert len(buf.get_frames()) == 0

    def test_nearest_neighbor_no_interpolation(self):
        buf = TransformBuffer()
        tf1 = Transform(Vector3(0, 0, 0), Quaternion.identity())
        tf2 = Transform(Vector3(10, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf1, timestamp=1.0)
        buf.add_transform("world", "base_link", tf2, timestamp=2.0)

        result = buf.lookup_transform(
            "world", "base_link", timestamp=1.3, interpolate=False
        )
        # Nearest to 1.3 is 1.0
        assert abs(result.translation.x - 0.0) < 1e-9

    def test_clamp_before_range(self):
        """Requesting time before first entry returns first entry."""
        buf = TransformBuffer()
        tf1 = Transform(Vector3(5, 0, 0), Quaternion.identity())
        buf.add_transform("world", "base_link", tf1, timestamp=10.0)

        result = buf.lookup_transform("world", "base_link", timestamp=1.0)
        assert abs(result.translation.x - 5.0) < 1e-9

    def test_buffer_duration_pruning(self):
        """Old entries should be pruned when buffer_duration is set."""
        buf = TransformBuffer()
        buf.set_buffer_duration(1.0)

        for i in range(10):
            tf = Transform(Vector3(float(i), 0, 0), Quaternion.identity())
            buf.add_transform("world", "base_link", tf, timestamp=float(i))

        # Only entries within 1.0 second of the newest (9.0) should remain
        # So entries with t >= 8.0 should be present
        entries = buf._dynamic[("world", "base_link")]
        assert all(e[0] >= 8.0 for e in entries)
