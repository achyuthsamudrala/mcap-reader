"""
Quaternion and rigid-body transform math for robotics coordinate frames.

This module provides from-scratch implementations of:
- 3-D vectors
- Unit quaternions (rotation representation)
- Rigid-body transforms (rotation + translation)
- Spherical linear interpolation (SLERP)

Why quaternions?
~~~~~~~~~~~~~~~~
Every 3-D rotation can be described by a unit quaternion q with ||q|| = 1.
The set of unit quaternions forms a double cover of the rotation group SO(3):
both q and -q map to the same rotation.  Compared with rotation matrices,
quaternions are more compact (4 floats vs 9), numerically stable under
repeated composition (just re-normalise), and free of gimbal lock.

A quaternion q = w + xi + yj + zk can also be written as (vector, scalar)
= ((x, y, z), w).  Rotating a pure-vector quaternion p = (v, 0) by q gives
    p' = q * p * q*
where q* is the conjugate (same as the inverse for unit quaternions).

ROS vs SciPy ordering
~~~~~~~~~~~~~~~~~~~~~
ROS stores quaternions as (x, y, z, w).  SciPy stores them as (w, x, y, z).
This module uses the **ROS convention** (x, y, z, w) everywhere.

Composition order
~~~~~~~~~~~~~~~~~
Transform composition follows the "right-to-left" convention used by TF2:
    T_AC = T_AB * T_BC
reads as "the transform *from* frame A *to* frame C is obtained by first
going A -> B, then B -> C."  Internally this is:
    R_AC = R_AB @ R_BC
    t_AC = R_AB @ t_BC + t_AB

Hamilton product
~~~~~~~~~~~~~~~~
The Hamilton product of two quaternions q1 and q2 yields a quaternion whose
corresponding rotation is the composition of the two rotations (q1 applied
first in world frame, or equivalently q2 applied first in body frame):
    (q1 * q2) corresponds to R(q1) @ R(q2)

Why SLERP instead of LERP for rotations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Linearly interpolating two quaternions and re-normalising (NLERP) does not
produce constant angular velocity — the rotation speeds up in the middle
and slows at the ends.  Spherical Linear Interpolation (SLERP) traverses
the great-circle arc on the unit 4-sphere at constant speed, which is
essential for smooth, physically meaningful motion interpolation.

All dataclasses are **frozen** (immutable) to prevent accidental mutation
of shared transform data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Vector3
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Vector3:
    """An immutable 3-D vector (x, y, z).

    Used for positions, translations, and angular velocities throughout the
    transform system.

    Parameters
    ----------
    x, y, z : float
        Cartesian components.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # -- NumPy conversion ---------------------------------------------------

    def to_numpy(self) -> NDArray[np.float64]:
        """Return a shape-(3,) numpy array ``[x, y, z]``."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_numpy(cls, arr: NDArray[np.float64]) -> Vector3:
        """Create a Vector3 from any array-like with 3 elements.

        Parameters
        ----------
        arr : array-like
            Must be convertible to a flat array of length 3.
        """
        a = np.asarray(arr, dtype=np.float64).ravel()
        if a.shape[0] != 3:
            raise ValueError(f"Expected 3 elements, got {a.shape[0]}")
        return cls(float(a[0]), float(a[1]), float(a[2]))

    # -- Arithmetic ---------------------------------------------------------

    def __add__(self, other: Vector3) -> Vector3:
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        if not isinstance(other, Vector3):
            return NotImplemented
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        """Scalar multiplication: ``v * s``."""
        if isinstance(scalar, (int, float)):
            return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
        return NotImplemented

    def __rmul__(self, scalar: float) -> Vector3:
        """Scalar multiplication: ``s * v``."""
        return self.__mul__(scalar)

    def __neg__(self) -> Vector3:
        return Vector3(-self.x, -self.y, -self.z)

    def __truediv__(self, scalar: float) -> Vector3:
        """Scalar division: ``v / s``."""
        if isinstance(scalar, (int, float)):
            return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
        return NotImplemented

    def dot(self, other: Vector3) -> float:
        """Euclidean dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vector3) -> Vector3:
        """Cross product ``self x other``."""
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        """Euclidean length."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def __repr__(self) -> str:
        return f"Vector3(x={self.x:.6f}, y={self.y:.6f}, z={self.z:.6f})"


# ---------------------------------------------------------------------------
# Quaternion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Quaternion:
    """An immutable quaternion in **ROS (x, y, z, w)** order.

    A unit quaternion (||q|| = 1) represents a rotation in 3-D space.
    The mapping to the rotation group SO(3) is a 2-to-1 covering: both
    ``q`` and ``-q`` encode the same physical rotation.

    Parameters
    ----------
    x, y, z : float
        Vector (imaginary) part of the quaternion.
    w : float
        Scalar (real) part of the quaternion.

    Notes
    -----
    The **Hamilton product** ``q1 * q2`` composes rotations so that the
    resulting rotation matrix equals ``R(q1) @ R(q2)``.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    # -- Class methods ------------------------------------------------------

    @classmethod
    def identity(cls) -> Quaternion:
        """Return the identity quaternion (no rotation).

        The identity quaternion is ``(0, 0, 0, 1)`` — a zero-angle rotation
        about any axis.
        """
        return cls(0.0, 0.0, 0.0, 1.0)

    # -- NumPy conversion ---------------------------------------------------

    def to_numpy(self) -> NDArray[np.float64]:
        """Return a shape-(4,) array in ROS order ``[x, y, z, w]``."""
        return np.array([self.x, self.y, self.z, self.w], dtype=np.float64)

    @classmethod
    def from_numpy(cls, arr: NDArray[np.float64]) -> Quaternion:
        """Create a Quaternion from an array-like of 4 elements in (x, y, z, w) order.

        Parameters
        ----------
        arr : array-like
            Must be convertible to a flat array of length 4.
        """
        a = np.asarray(arr, dtype=np.float64).ravel()
        if a.shape[0] != 4:
            raise ValueError(f"Expected 4 elements, got {a.shape[0]}")
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]))

    # -- Basic properties ---------------------------------------------------

    def norm(self) -> float:
        """Return the L2 norm (magnitude) of the quaternion.

        For a valid rotation quaternion this should be very close to 1.0.
        """
        return math.sqrt(
            self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
        )

    def normalize(self) -> Quaternion:
        """Return a unit quaternion (||q|| = 1) pointing in the same direction.

        Normalisation is essential after repeated Hamilton products to
        counteract floating-point drift.

        Raises
        ------
        ValueError
            If the quaternion has near-zero norm (cannot be normalised).
        """
        n = self.norm()
        if n < 1e-12:
            raise ValueError("Cannot normalise a near-zero quaternion")
        return Quaternion(self.x / n, self.y / n, self.z / n, self.w / n)

    def conjugate(self) -> Quaternion:
        """Return the conjugate q* = (-x, -y, -z, w).

        For a **unit** quaternion the conjugate equals the inverse, because
        ``q * q* = ||q||^2 = 1``.  Geometrically, the conjugate represents
        the reverse rotation.
        """
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def inverse(self) -> Quaternion:
        """Return the multiplicative inverse q^{-1} = q* / ||q||^2.

        For unit quaternions this is identical to :meth:`conjugate`, but
        this method is safe for non-unit quaternions as well.

        Raises
        ------
        ValueError
            If the quaternion has near-zero norm.
        """
        n_sq = (
            self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
        )
        if n_sq < 1e-24:
            raise ValueError("Cannot invert a near-zero quaternion")
        return Quaternion(
            -self.x / n_sq, -self.y / n_sq, -self.z / n_sq, self.w / n_sq
        )

    # -- Rotation matrix conversion -----------------------------------------

    def to_rotation_matrix(self) -> NDArray[np.float64]:
        """Convert this quaternion to a 3x3 rotation matrix.

        The rotation matrix R satisfies ``R @ v == q * (v, 0) * q*`` for
        every 3-D vector v (expressed as a pure quaternion).

        The formula (for a unit quaternion) is:

        .. math::

            R = \\begin{pmatrix}
            1-2(y^2+z^2) & 2(xy-zw)     & 2(xz+yw) \\\\
            2(xy+zw)     & 1-2(x^2+z^2) & 2(yz-xw) \\\\
            2(xz-yw)     & 2(yz+xw)     & 1-2(x^2+y^2)
            \\end{pmatrix}

        Returns
        -------
        np.ndarray
            A 3x3 orthogonal matrix with determinant +1.
        """
        # Normalise to guard against drift.
        q = self.normalize()
        x, y, z, w = q.x, q.y, q.z, q.w

        # Pre-compute repeated products.
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_rotation_matrix(cls, R: NDArray[np.float64]) -> Quaternion:
        """Recover a quaternion from a 3x3 rotation matrix via Shepperd's method.

        Shepperd's method (1978) avoids the numerical instability of the
        naive trace-based formula by choosing the largest diagonal element to
        compute the quaternion components.  There are four cases, one for
        each of (w, x, y, z) being the largest component, ensuring we never
        divide by a near-zero value.

        Parameters
        ----------
        R : np.ndarray
            A 3x3 rotation matrix (should be orthogonal with det = +1).

        Returns
        -------
        Quaternion
            A unit quaternion corresponding to the rotation.

        Notes
        -----
        The algorithm computes::

            trace = R[0,0] + R[1,1] + R[2,2]

        and then picks the case where the denominator is largest:

        - **Case 0 (w largest):** ``trace > 0`` — compute ``w`` first, then
          derive ``x, y, z`` from the off-diagonal differences.
        - **Case 1 (x largest):** ``R[0,0]`` is the biggest diagonal element —
          compute ``x`` first.
        - **Case 2 (y largest):** ``R[1,1]`` is the biggest — compute ``y`` first.
        - **Case 3 (z largest):** ``R[2,2]`` is the biggest — compute ``z`` first.
        """
        R = np.asarray(R, dtype=np.float64)
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0.0:
            # Case 0: w is the largest component.
            s = 2.0 * math.sqrt(trace + 1.0)  # s = 4w
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            # Case 1: x is the largest component.
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])  # s = 4x
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            # Case 2: y is the largest component.
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])  # s = 4y
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            # Case 3: z is the largest component.
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])  # s = 4z
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return cls(x, y, z, w).normalize()

    # -- Hamilton product (composition) -------------------------------------

    def __mul__(self, other: Quaternion) -> Quaternion:
        """Hamilton product: compose two rotations.

        Given quaternions ``q1`` and ``q2``, the product ``q1 * q2`` yields
        a quaternion whose rotation matrix is ``R(q1) @ R(q2)``.

        The explicit formula is::

            (q1 * q2).w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
            (q1 * q2).x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
            (q1 * q2).y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
            (q1 * q2).z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w

        Parameters
        ----------
        other : Quaternion
            The right-hand quaternion.

        Returns
        -------
        Quaternion
            The composed rotation (not automatically re-normalised).
        """
        if not isinstance(other, Quaternion):
            return NotImplemented
        # Unpack for readability.
        ax, ay, az, aw = self.x, self.y, self.z, self.w
        bx, by, bz, bw = other.x, other.y, other.z, other.w
        return Quaternion(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    # -- Angular distance ---------------------------------------------------

    def angular_distance(self, other: Quaternion) -> float:
        """Return the geodesic (angular) distance to *other* in radians.

        Because quaternions double-cover SO(3), we use::

            theta = 2 * arccos(|q1 . q2|)

        where the dot product is the 4-D inner product and we take the
        absolute value to account for the q / -q equivalence.  The result
        is in [0, pi].

        Parameters
        ----------
        other : Quaternion
            The target quaternion.

        Returns
        -------
        float
            Angle in radians in [0, pi].
        """
        dot = abs(
            self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
        )
        # Clamp for numerical safety.
        dot = min(dot, 1.0)
        return 2.0 * math.acos(dot)

    def __repr__(self) -> str:
        return (
            f"Quaternion(x={self.x:.6f}, y={self.y:.6f}, "
            f"z={self.z:.6f}, w={self.w:.6f})"
        )


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Transform:
    """An immutable rigid-body transform (rotation + translation).

    A transform T maps points from one coordinate frame to another:

        p_A = T_AB.apply(p_B)

    where ``T_AB`` transforms points expressed in frame B into frame A.

    Composition follows the TF2 convention::

        T_AC = T_AB * T_BC

    which reads: "to go from A to C, first go A -> B, then B -> C."
    Internally::

        R_AC = R_AB @ R_BC
        t_AC = R_AB @ t_BC + t_AB

    Parameters
    ----------
    translation : Vector3
        The translational component.
    rotation : Quaternion
        The rotational component (should be a unit quaternion).
    """

    translation: Vector3
    rotation: Quaternion

    # -- Class methods ------------------------------------------------------

    @classmethod
    def identity(cls) -> Transform:
        """Return the identity transform (no rotation, no translation)."""
        return cls(Vector3(0.0, 0.0, 0.0), Quaternion.identity())

    # -- Matrix conversion --------------------------------------------------

    def to_matrix(self) -> NDArray[np.float64]:
        """Convert to a 4x4 homogeneous transformation matrix.

        The matrix has the form::

            | R  t |
            | 0  1 |

        where R is the 3x3 rotation matrix and t is the 3x1 translation.

        Returns
        -------
        np.ndarray
            A 4x4 matrix.
        """
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = self.rotation.to_rotation_matrix()
        m[:3, 3] = self.translation.to_numpy()
        return m

    @classmethod
    def from_matrix(cls, m: NDArray[np.float64]) -> Transform:
        """Create a Transform from a 4x4 homogeneous matrix.

        Parameters
        ----------
        m : np.ndarray
            A 4x4 matrix of the form ``[[R, t], [0, 1]]``.
        """
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"Expected (4, 4) matrix, got {m.shape}")
        rotation = Quaternion.from_rotation_matrix(m[:3, :3])
        translation = Vector3.from_numpy(m[:3, 3])
        return cls(translation, rotation)

    # -- Inverse ------------------------------------------------------------

    def inverse(self) -> Transform:
        """Return the inverse transform T^{-1}.

        If ``T_AB`` maps B -> A, then ``T_AB.inverse()`` maps A -> B.

        Mathematically::

            R_inv = R^T
            t_inv = -R^T @ t

        Returns
        -------
        Transform
            The inverse transform.
        """
        q_inv = self.rotation.conjugate()
        # Rotate the negated translation by the inverse rotation.
        R_inv = q_inv.to_rotation_matrix()
        t_inv = R_inv @ (-self.translation.to_numpy())
        return Transform(Vector3.from_numpy(t_inv), q_inv)

    # -- Composition --------------------------------------------------------

    def __mul__(self, other: Transform) -> Transform:
        """Compose two transforms: ``T_AC = T_AB * T_BC``.

        The rotation is composed via Hamilton product and the translation
        via::

            t_AC = R_AB @ t_BC + t_AB

        Parameters
        ----------
        other : Transform
            The right-hand transform (T_BC).

        Returns
        -------
        Transform
            The composed transform (T_AC).
        """
        if not isinstance(other, Transform):
            return NotImplemented
        new_rotation = self.rotation * other.rotation
        R_self = self.rotation.to_rotation_matrix()
        new_translation = Vector3.from_numpy(
            R_self @ other.translation.to_numpy() + self.translation.to_numpy()
        )
        return Transform(new_translation, new_rotation.normalize())

    # -- Point application --------------------------------------------------

    def apply(self, point: Vector3) -> Vector3:
        """Apply this transform to a 3-D point.

        Computes ``R @ p + t``.

        Parameters
        ----------
        point : Vector3
            A point in the source frame.

        Returns
        -------
        Vector3
            The point expressed in the target frame.
        """
        R = self.rotation.to_rotation_matrix()
        result = R @ point.to_numpy() + self.translation.to_numpy()
        return Vector3.from_numpy(result)

    def __repr__(self) -> str:
        return f"Transform(translation={self.translation}, rotation={self.rotation})"


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
    """Spherical Linear Interpolation between two quaternions.

    SLERP traces the shortest great-circle arc on the unit quaternion
    4-sphere at **constant angular velocity**, which is essential for
    physically meaningful rotation interpolation.

    The formula is::

        slerp(q1, q2, t) = q1 * sin((1-t)*theta) / sin(theta)
                          + q2 * sin(t*theta)     / sin(theta)

    where ``theta = arccos(q1 . q2)`` is the angle between the quaternions.

    Short-path handling
    ~~~~~~~~~~~~~~~~~~~
    Because ``q`` and ``-q`` represent the same rotation, the dot product
    can be negative, which would cause SLERP to take the *long* path around
    the sphere (> 180 degrees).  We negate one quaternion when the dot
    product is negative to always take the short path.

    Degenerate case
    ~~~~~~~~~~~~~~~
    When the two quaternions are nearly identical (``theta ~ 0``),
    ``sin(theta)`` approaches zero and the SLERP formula becomes
    numerically unstable.  We fall back to normalised linear interpolation
    (NLERP) in this case, which is equivalent to SLERP for very small
    angles.

    Parameters
    ----------
    q1 : Quaternion
        Start quaternion (returned when t = 0).
    q2 : Quaternion
        End quaternion (returned when t = 1).
    t : float
        Interpolation parameter, typically in [0, 1].

    Returns
    -------
    Quaternion
        The interpolated unit quaternion.
    """
    # Compute 4-D dot product.
    dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w

    # Short-path: if dot < 0, negate q2 so we interpolate the short way.
    if dot < 0.0:
        q2 = Quaternion(-q2.x, -q2.y, -q2.z, -q2.w)
        dot = -dot

    # Clamp for numerical safety.
    dot = min(dot, 1.0)

    # Degenerate case: quaternions are nearly identical — fall back to NLERP.
    _SLERP_THRESHOLD = 1.0 - 1e-6
    if dot > _SLERP_THRESHOLD:
        # Linear interpolation + normalisation.
        result = Quaternion(
            q1.x + t * (q2.x - q1.x),
            q1.y + t * (q2.y - q1.y),
            q1.z + t * (q2.z - q1.z),
            q1.w + t * (q2.w - q1.w),
        )
        return result.normalize()

    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    w1 = math.sin((1.0 - t) * theta) / sin_theta
    w2 = math.sin(t * theta) / sin_theta

    return Quaternion(
        w1 * q1.x + w2 * q2.x,
        w1 * q1.y + w2 * q2.y,
        w1 * q1.z + w2 * q2.z,
        w1 * q1.w + w2 * q2.w,
    ).normalize()


def lerp_vector(v1: Vector3, v2: Vector3, t: float) -> Vector3:
    """Linearly interpolate between two 3-D vectors.

    Parameters
    ----------
    v1 : Vector3
        Start vector (returned when t = 0).
    v2 : Vector3
        End vector (returned when t = 1).
    t : float
        Interpolation parameter, typically in [0, 1].

    Returns
    -------
    Vector3
        ``(1 - t) * v1 + t * v2``.
    """
    return Vector3(
        v1.x + t * (v2.x - v1.x),
        v1.y + t * (v2.y - v1.y),
        v1.z + t * (v2.z - v1.z),
    )


def interpolate_transform(
    t1: Transform, t2: Transform, alpha: float
) -> Transform:
    """Interpolate between two rigid-body transforms.

    Translation is linearly interpolated (LERP) and rotation is
    spherically interpolated (SLERP) to produce smooth, constant-speed
    motion.

    Parameters
    ----------
    t1 : Transform
        Start transform (returned when alpha = 0).
    t2 : Transform
        End transform (returned when alpha = 1).
    alpha : float
        Interpolation parameter, typically in [0, 1].

    Returns
    -------
    Transform
        The interpolated transform.
    """
    translation = lerp_vector(t1.translation, t2.translation, alpha)
    rotation = slerp(t1.rotation, t2.rotation, alpha)
    return Transform(translation, rotation)
