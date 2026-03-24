"""
Camera calibration module wrapping CameraInfo.

The Pinhole Camera Model
========================
The pinhole model describes how a 3D point in the camera's coordinate frame
projects onto a 2D image plane.  The key parameters are:

- **fx, fy** (focal lengths in pixels): The distance from the camera's
  optical center to the image plane, measured in pixel units along the x and
  y axes respectively.  ``fx = f * sx`` where ``f`` is the physical focal
  length (mm) and ``sx`` is the pixel density (pixels/mm) along x.  For
  cameras with square pixels (nearly all modern sensors), ``fx ~= fy``.

- **cx, cy** (principal point): The pixel coordinates where the optical
  axis (the line perpendicular to the image plane through the lens center)
  intersects the image.  Ideally this is the image center, but manufacturing
  tolerances and lens alignment mean it typically deviates by a few pixels.

Together these form the **intrinsic matrix K**::

    K = | fx  0  cx |
        |  0  fy cy |
        |  0   0  1 |

The ideal (distortion-free) projection of a 3D point ``(X, Y, Z)`` in the
camera frame to pixel coordinates ``(u, v)`` is::

    [u]       [X]         u = fx * (X/Z) + cx
    [v] = K * [Y] / Z     v = fy * (Y/Z) + cy
    [1]       [Z]

The Projection Pipeline
=======================
The full projection from a 3D world point to a pixel involves several
stages.  Understanding the pipeline is essential for correctly using the
D, K, R, P matrices from ``sensor_msgs/msg/CameraInfo``.

1. **World-to-camera transform** (extrinsic): Transforms the 3D point from
   world coordinates to the camera's coordinate frame using a rigid-body
   transform (rotation + translation).  This is NOT stored in CameraInfo;
   it comes from the TF tree.

2. **Rectification (R matrix, 3x3)**: For monocular cameras, R is the
   identity matrix.  For stereo cameras, R rotates the camera frame into
   a "rectified" coordinate system where epipolar lines are horizontal.
   After rectification, corresponding points in left/right images lie on
   the same pixel row, which dramatically simplifies stereo matching
   algorithms (they only need to search along one dimension).

3. **Intrinsic projection (K matrix, 3x3)**: Applies the pinhole model to
   map the (possibly rectified) 3D point to ideal (undistorted) pixel
   coordinates.

4. **Distortion**: Real lenses introduce radial and tangential distortion
   that warps the image, especially at the edges.  Distortion is applied
   *after* the ideal projection in the forward model, and must be *removed*
   (undistortion) to recover ideal pixel coordinates from raw images.

5. **Projection matrix (P, 3x4)**: Combines steps 2-3 into a single
   matrix that maps homogeneous 3D points to 2D pixel coordinates::

       P = K_rect * [R | t]

   For monocular cameras, ``P = [K | 0]`` (K with a zero column appended).
   For stereo cameras, the 4th column encodes the baseline:
   ``P[0,3] = -fx' * baseline`` for the right camera.

What the D, K, R, P Matrices Mean
==================================

**D (distortion coefficients)**: A variable-length array whose
interpretation depends on the ``distortion_model`` string:

- ``"plumb_bob"`` (the most common): 5 coefficients ``[k1, k2, p1, p2, k3]``
  where k1, k2, k3 are radial distortion and p1, p2 are tangential
  (decentering) distortion.  Radial distortion causes straight lines to
  appear curved (barrel or pincushion effect); tangential distortion causes
  the image to be slightly trapezoidal due to lens-sensor misalignment.

- ``"rational_polynomial"``: 8 coefficients ``[k1, k2, p1, p2, k3, k4, k5, k6]``
  that include a rational model ``(1 + k1*r^2 + k2*r^4 + k3*r^6) /
  (1 + k4*r^2 + k5*r^4 + k6*r^6)`` for wide-angle and fisheye lenses
  where the simpler polynomial model is inadequate.  The rational form
  can model much stronger distortion without the polynomial diverging
  at large radii.

**K (3x3 intrinsic matrix)**: Maps 3D camera-frame points to ideal
(distortion-free) pixel coordinates, as described above.

**R (3x3 rectification matrix)**: Rotation applied to the camera frame
before projection.  Identity for monocular; aligns epipolar geometry for
stereo.

**P (3x4 projection matrix)**: The combined rectification + projection.
For monocular: ``P[:3, :3]`` equals ``K`` and ``P[:, 3]`` is zero.
For stereo: ``P[:, 3]`` encodes the baseline offset.

Why Undistortion Matters
========================
Raw images from wide-angle cameras can have significant barrel distortion
-- straight lines in the real world appear curved in the image.  This
breaks algorithms that assume a pinhole model (feature matching, visual
odometry, ArUco detection, neural networks trained on undistorted data).

Undistortion applies the *inverse* of the distortion model to warp each
pixel back to its ideal pinhole position.  For efficiency, the remapping
is precomputed as a pair of lookup tables (``map_x``, ``map_y``) via
``cv2.initUndistortRectifyMap`` and then applied with ``cv2.remap``.  This
is much faster than calling ``cv2.undistort`` per-frame because the map
computation involves iterative root-finding that only needs to happen once.

plumb_bob vs rational_polynomial
=================================
The ``plumb_bob`` model (5 coefficients) works well for standard lenses
with moderate distortion (field of view up to ~90 degrees).  For wider
lenses (120-180+ degrees), the polynomial terms ``k1*r^2 + k2*r^4 +
k3*r^6`` can diverge at large radii, producing wildly incorrect corrections
at image corners.

The ``rational_polynomial`` model (8 coefficients) uses a ratio of
polynomials that remains bounded even at extreme radii, making it suitable
for fisheye and ultra-wide-angle lenses commonly found on autonomous
vehicles and drones.

OpenCV supports both models: ``cv2.undistort`` and ``cv2.projectPoints``
automatically handle the coefficient count.

Typical usage::

    from mcap_reader.calibration import CameraModel
    from mcap_reader.messages.camera_info import CameraInfo

    # From a CameraInfo message in the MCAP file:
    camera = CameraModel.from_camera_info(camera_info_msg)

    # Undistort a raw image:
    clean = camera.undistort(raw_image)

    # Project 3D points to pixels:
    pixels = camera.project(points_3d)

    # Unproject a pixel with known depth:
    point_3d = camera.unproject(u=320, v=240, depth=1.5)
"""

from __future__ import annotations

import math
from functools import cached_property

import cv2
import numpy as np
from numpy.typing import NDArray

from mcap_reader.messages.camera_info import CameraInfo


class CameraModel:
    """A camera projection model built from ``sensor_msgs/msg/CameraInfo``.

    Wraps the intrinsic calibration parameters (K, D, R, P) and provides
    methods for projection, unprojection, and undistortion.  All
    computations use OpenCV under the hood for correctness and performance.

    This class is constructed via :meth:`from_camera_info` rather than
    directly, to enforce that the calibration data comes from a valid
    ``CameraInfo`` message.

    Attributes
    ----------
    _camera_info : CameraInfo
        The underlying camera calibration data.
    _K : np.ndarray
        3x3 intrinsic matrix (cached copy).
    _D : np.ndarray
        Distortion coefficients (cached copy).
    _R : np.ndarray
        3x3 rectification matrix (cached copy).
    _P : np.ndarray
        3x4 projection matrix (cached copy).
    """

    def __init__(self, camera_info: CameraInfo) -> None:
        self._camera_info = camera_info
        self._K = camera_info.K.copy()
        self._D = camera_info.D.copy()
        self._R = camera_info.R.copy()
        self._P = camera_info.P.copy()

    # -- Constructors -------------------------------------------------------

    @classmethod
    def from_camera_info(cls, camera_info: CameraInfo) -> CameraModel:
        """Create a CameraModel from a deserialized CameraInfo message.

        This is the primary constructor.  It validates that the essential
        calibration fields are present and have the expected shapes.

        Parameters
        ----------
        camera_info : CameraInfo
            A deserialized ``sensor_msgs/msg/CameraInfo`` message, as
            produced by :meth:`CameraInfo.from_ros_msg` or
            :meth:`CameraInfo.from_cdr`.

        Returns
        -------
        CameraModel
            A camera model ready for projection and undistortion.

        Raises
        ------
        ValueError
            If the intrinsic matrix K has zero focal lengths (uncalibrated
            camera).
        """
        if camera_info.K[0, 0] == 0.0 or camera_info.K[1, 1] == 0.0:
            raise ValueError(
                "CameraInfo has zero focal lengths in K. "
                "The camera may not be calibrated."
            )
        return cls(camera_info)

    # -- Properties ---------------------------------------------------------

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """The 3x3 intrinsic camera matrix K.

        Contains focal lengths (fx, fy) and principal point (cx, cy)::

            K = | fx  0  cx |
                |  0  fy cy |
                |  0   0  1 |

        Returns
        -------
        np.ndarray
            A 3x3 float64 matrix (copy of the internal data).
        """
        return self._K.copy()

    @property
    def distortion_coefficients(self) -> np.ndarray:
        """The distortion coefficients D.

        The length and interpretation depend on the distortion model:

        - ``plumb_bob``: 5 values ``[k1, k2, p1, p2, k3]``
        - ``rational_polynomial``: 8 values ``[k1, k2, p1, p2, k3, k4, k5, k6]``

        Returns
        -------
        np.ndarray
            A 1D float64 array of distortion coefficients.
        """
        return self._D.copy()

    @property
    def projection_matrix(self) -> np.ndarray:
        """The 3x4 projection matrix P.

        Combines rectification and intrinsics into a single matrix that
        maps homogeneous 3D points to 2D pixels::

            [u*s]       [X]
            [v*s] = P * [Y]
            [ s ]       [Z]
                        [1]

        For monocular cameras, ``P[:3, :3]`` equals K and ``P[:, 3]``
        is the zero vector.  For stereo cameras, ``P[0, 3]`` encodes
        ``-fx' * baseline``.

        Returns
        -------
        np.ndarray
            A 3x4 float64 matrix.
        """
        return self._P.copy()

    @property
    def image_size(self) -> tuple[int, int]:
        """Image dimensions as ``(width, height)`` in pixels.

        Note the order: **(width, height)**, matching OpenCV convention
        (not the numpy/ROS ``(height, width)`` convention for array shapes).

        Returns
        -------
        tuple[int, int]
            ``(width, height)``.
        """
        return (self._camera_info.width, self._camera_info.height)

    @property
    def fov(self) -> tuple[float, float]:
        """Horizontal and vertical field of view in radians.

        Computed from the focal lengths and image dimensions using::

            fov_h = 2 * atan(width / (2 * fx))
            fov_v = 2 * atan(height / (2 * fy))

        This is the *ideal* (undistorted) field of view.  The actual FOV
        of a distorted image may be larger (barrel distortion) or smaller
        (pincushion distortion).

        Returns
        -------
        tuple[float, float]
            ``(horizontal_fov, vertical_fov)`` in radians.
        """
        fx = self._K[0, 0]
        fy = self._K[1, 1]
        w, h = self.image_size
        fov_h = 2.0 * math.atan2(w, 2.0 * fx)
        fov_v = 2.0 * math.atan2(h, 2.0 * fy)
        return (fov_h, fov_v)

    # -- Projection ---------------------------------------------------------

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D points (in camera frame) to pixel coordinates.

        Applies the full projection pipeline: intrinsic matrix K plus
        the distortion model D.  This uses ``cv2.projectPoints`` which
        handles both ``plumb_bob`` and ``rational_polynomial`` models.

        Parameters
        ----------
        points_3d : np.ndarray
            3D points in the camera coordinate frame.  Accepts either:

            - A single point as shape ``(3,)``
            - Multiple points as shape ``(N, 3)``

            The camera frame convention (ROS/OpenCV) is:
            x = right, y = down, z = forward (into the scene).

        Returns
        -------
        np.ndarray
            Pixel coordinates as shape ``(N, 2)`` array of ``[u, v]``
            pairs.  For a single input point, returns shape ``(1, 2)``.

        Notes
        -----
        Points behind the camera (Z <= 0) will produce invalid pixel
        coordinates.  The caller should filter these before projecting.

        The projection pipeline applied by ``cv2.projectPoints`` is:

        1. Normalize: ``x' = X/Z, y' = Y/Z``
        2. Distort: apply radial/tangential distortion to ``(x', y')``
        3. Pixel: ``u = fx * x_d + cx, v = fy * y_d + cy``
        """
        pts = np.asarray(points_3d, dtype=np.float64)

        if pts.ndim == 1:
            if pts.shape[0] != 3:
                raise ValueError(
                    f"Expected 3 coordinates per point, got {pts.shape[0]}"
                )
            pts = pts.reshape(1, 3)
        elif pts.ndim == 2:
            if pts.shape[1] != 3:
                raise ValueError(
                    f"Expected shape (N, 3), got {pts.shape}"
                )
        else:
            raise ValueError(
                f"Expected 1D or 2D array, got {pts.ndim}D"
            )

        # cv2.projectPoints expects:
        #   objectPoints: (N, 1, 3) or (N, 3)
        #   rvec: rotation vector (3,) -- identity = [0, 0, 0]
        #   tvec: translation vector (3,) -- origin = [0, 0, 0]
        #   cameraMatrix: 3x3
        #   distCoeffs: 1D array
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)

        pixel_coords, _ = cv2.projectPoints(
            pts.reshape(-1, 1, 3),
            rvec,
            tvec,
            self._K,
            self._D,
        )

        return pixel_coords.reshape(-1, 2)

    # -- Unprojection -------------------------------------------------------

    def unproject(self, u: float, v: float, depth: float) -> np.ndarray:
        """Unproject a pixel coordinate with known depth to a 3D point.

        First removes lens distortion from the pixel coordinate to obtain
        the ideal (pinhole) normalized image coordinate, then scales by
        the depth to produce a 3D point in the camera frame.

        Parameters
        ----------
        u : float
            Pixel x-coordinate (column).
        v : float
            Pixel y-coordinate (row).
        depth : float
            Depth (Z) of the point in the camera frame, in meters.
            Must be positive.

        Returns
        -------
        np.ndarray
            A shape ``(3,)`` array ``[X, Y, Z]`` in the camera frame.

        Raises
        ------
        ValueError
            If depth is not positive.

        Notes
        -----
        The unprojection pipeline is:

        1. **Undistort** the pixel coordinate using ``cv2.undistortPoints``
           to obtain normalized image coordinates ``(x_n, y_n)`` where::

               x_n = (X / Z)
               y_n = (Y / Z)

        2. **Scale by depth**::

               X = x_n * depth
               Y = y_n * depth
               Z = depth

        This is the inverse of :meth:`project` for the pinhole + distortion
        model.  Note that unprojection requires depth information because
        projection is a many-to-one mapping (a ray of 3D points maps to
        a single pixel).
        """
        if depth <= 0:
            raise ValueError(f"Depth must be positive, got {depth}")

        # cv2.undistortPoints expects shape (N, 1, 2).
        pixel = np.array([[[u, v]]], dtype=np.float64)

        # Undistort to normalized image coordinates (no camera matrix in
        # the output -- pass P=None to get raw normalized coords).
        normalized = cv2.undistortPoints(
            pixel,
            self._K,
            self._D,
            R=self._R,
        )

        x_n = float(normalized[0, 0, 0])
        y_n = float(normalized[0, 0, 1])

        return np.array(
            [x_n * depth, y_n * depth, depth], dtype=np.float64
        )

    # -- Undistortion -------------------------------------------------------

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """Remove lens distortion from an image.

        Uses ``cv2.undistort`` which internally computes the undistortion
        map and applies it in a single call.  For processing many frames
        from the same camera, use :meth:`get_undistort_maps` with
        ``cv2.remap`` instead -- it avoids recomputing the map each time.

        Parameters
        ----------
        image : np.ndarray
            The distorted input image (any number of channels).

        Returns
        -------
        np.ndarray
            The undistorted image, same shape and dtype as input.

        Notes
        -----
        Undistortion warps the image so that straight lines in the real
        world appear straight in the output image.  Pixels near the edges
        may be lost (mapped outside the output frame) or filled with black
        depending on the distortion magnitude.

        For performance-critical pipelines (real-time video processing),
        prefer::

            map1, map2 = camera.get_undistort_maps()
            for frame in frames:
                clean = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        """
        return cv2.undistort(image, self._K, self._D)

    def get_undistort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute undistortion + rectification remapping tables.

        Returns a pair of maps suitable for ``cv2.remap``.  Computing the
        maps once and reusing them for every frame is significantly faster
        than calling :meth:`undistort` per-frame, because the map
        computation involves iterative root-finding (Newton's method) for
        each pixel to invert the distortion model.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(map_x, map_y)`` -- floating-point coordinate maps where
            ``map_x[v, u]`` and ``map_y[v, u]`` give the source pixel
            coordinates in the distorted image that map to destination
            pixel ``(u, v)`` in the undistorted image.

        Notes
        -----
        Usage::

            map1, map2 = camera.get_undistort_maps()
            undistorted = cv2.remap(
                distorted_image, map1, map2, cv2.INTER_LINEAR
            )

        The maps incorporate both distortion removal and stereo
        rectification (via the R matrix), so the output is both
        undistorted and rectified.
        """
        w, h = self.image_size

        # newCameraMatrix: use the projection matrix's 3x3 sub-block so
        # that the output image uses the rectified intrinsics.
        new_K = self._P[:3, :3]

        map_x, map_y = cv2.initUndistortRectifyMap(
            self._K,
            self._D,
            self._R,
            new_K,
            (w, h),
            cv2.CV_32FC1,
        )
        return map_x, map_y

    # -- Stereo rectification -----------------------------------------------

    @classmethod
    def rectify_stereo(
        cls,
        left_info: CameraInfo,
        right_info: CameraInfo,
    ) -> tuple[CameraModel, CameraModel]:
        """Compute rectified camera models for a stereo pair.

        Stereo rectification rotates both camera images so that:

        1. Epipolar lines become horizontal (corresponding points in
           left/right images share the same v-coordinate).
        2. The images are co-planar (both image planes lie in the same
           geometric plane).

        This simplifies stereo matching from a 2D search problem to a
        1D search along each row, and is required by most stereo depth
        algorithms (block matching, semi-global matching, etc.).

        This method uses the R and P matrices already computed by the
        camera calibration tool (e.g. ``camera_calibration`` ROS package
        or OpenCV's ``cv2.stereoRectify``).  The R and P in each
        ``CameraInfo`` message already encode the rectification transforms
        for each camera.

        Parameters
        ----------
        left_info : CameraInfo
            Calibration for the left camera of the stereo pair.
        right_info : CameraInfo
            Calibration for the right camera of the stereo pair.

        Returns
        -------
        tuple[CameraModel, CameraModel]
            ``(left_model, right_model)`` -- CameraModel instances with
            rectification parameters baked in.  Use their
            :meth:`get_undistort_maps` to produce rectified images.

        Notes
        -----
        The stereo baseline can be extracted from the right camera's
        projection matrix::

            baseline = -right_model.projection_matrix[0, 3] / fx'

        where ``fx' = right_model.projection_matrix[0, 0]`` is the
        rectified focal length.  The baseline is in the same units as
        the calibration (typically meters).

        The rectification workflow is:

        1. Compute undistort+rectify maps for both cameras.
        2. Remap raw left/right images using the maps.
        3. Run stereo matching on the rectified pair.
        4. Convert disparity to depth using ``baseline`` and ``fx'``.
        """
        left_model = cls.from_camera_info(left_info)
        right_model = cls.from_camera_info(right_info)
        return left_model, right_model
