"""
TF2-style transform buffer with timestamp lookup and interpolation.

This module provides a ``TransformBuffer`` that mirrors the functionality
of ROS 2's ``tf2_ros::Buffer``: it stores timestamped transforms between
coordinate frames and can look up the transform between any two frames
at any point in time.

Static vs dynamic transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ROS distinguishes two kinds of transforms:

- **Dynamic transforms** (``/tf`` topic) are published at a fixed rate
  (commonly 10--100 Hz) and change over time.  Examples: ``odom ->
  base_link`` (updated by wheel odometry), ``map -> odom`` (updated by
  a localisation node).

- **Static transforms** (``/tf_static`` topic) are published once with
  a latched publisher and never change.  Examples: ``base_link ->
  camera_link`` (a sensor bolted to the chassis), ``camera_link ->
  camera_optical_frame`` (a fixed axis rotation defined by REP-103).

The buffer stores static transforms separately and always returns the
single stored value regardless of the requested timestamp.  Dynamic
transforms are stored in a time-sorted list and looked up via binary
search, with optional interpolation.

Why interpolation is needed
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Transforms are published at discrete rates.  A camera image arriving
at ``t = 1.005 s`` may need the ``odom -> base_link`` transform, but
the most recent published values might be at ``t = 1.00 s`` and
``t = 1.01 s``.  Interpolation (LERP for translation, SLERP for
rotation) produces a smooth estimate at the exact requested time.

Graph search for frame paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To look up the transform between two frames that are not directly
connected (e.g. ``map`` to ``camera_optical_frame``), we find the
shortest path through the frame graph using BFS, then compose the
transforms along each edge.  When the path traverses an edge
*backwards* (child -> parent instead of parent -> child), we use the
inverse transform.

Transform composition order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Following TF2 conventions, if the path is::

    target <- A <- B <- C <- source

then the composed transform is::

    T_target_source = T_target_A * T_A_B * T_B_C * T_C_source

Each ``T_X_Y`` is "the transform that maps points in frame Y into
frame X".  When we traverse an edge in the stored (parent -> child)
direction, we use it directly; when we traverse it in reverse, we
invert it.
"""

from __future__ import annotations

import bisect
from typing import Any

from mcap_reader.transforms.frames import (
    FrameGraph,
    FrameNotFoundError,
    NoPathError,
)
from mcap_reader.transforms.math import (
    Quaternion,
    Transform,
    Vector3,
    interpolate_transform,
    slerp,
)


class TransformBuffer:
    """A TF2-style buffer that stores timestamped transforms and supports
    lookup between arbitrary frames at arbitrary times.

    Internal storage
    ~~~~~~~~~~~~~~~~
    Dynamic transforms are stored in a dict keyed by ``(parent, child)``
    tuples.  Each value is a list of ``(timestamp, Transform)`` pairs,
    kept sorted by timestamp so that binary search can efficiently find
    the two nearest entries for interpolation.

    Static transforms are stored in a separate dict with the same key
    structure but hold only a single Transform (no timestamp).

    A :class:`~mcap_reader.transforms.frames.FrameGraph` is maintained
    alongside the transform storage to enable path finding between
    non-adjacent frames.

    Buffer duration
    ~~~~~~~~~~~~~~~
    To bound memory usage, the buffer can be configured with a maximum
    duration.  When set, transforms older than
    ``newest_timestamp - duration`` are pruned on insertion.  This
    mirrors the ``buffer_length`` parameter of ``tf2_ros::Buffer``.

    Example
    -------
    ::

        buf = TransformBuffer()
        buf.add_transform("odom", "base_link", tf, timestamp=1.0)
        buf.add_transform("odom", "base_link", tf, timestamp=2.0)
        result = buf.lookup_transform("odom", "base_link", timestamp=1.5)
    """

    def __init__(self) -> None:
        # Dynamic transforms: (parent, child) -> sorted list of (time, Transform)
        self._dynamic: dict[tuple[str, str], list[tuple[float, Transform]]] = {}

        # Static transforms: (parent, child) -> Transform
        self._static: dict[tuple[str, str], Transform] = {}

        # Frame connectivity graph for BFS path finding.
        self._graph = FrameGraph()

        # Set of (parent, child) pairs that are static.
        self._static_keys: set[tuple[str, str]] = set()

        # Maximum buffer duration in seconds (None = unlimited).
        self._buffer_duration: float | None = None

    # -- Adding transforms --------------------------------------------------

    def add_transform(
        self,
        parent: str,
        child: str,
        transform: Transform,
        timestamp: float,
        is_static: bool = False,
    ) -> None:
        """Add a transform between two frames at a given timestamp.

        For **static** transforms, the timestamp is ignored and the
        transform is stored as a single value that is always returned
        regardless of the requested lookup time.

        For **dynamic** transforms, the ``(timestamp, transform)`` pair
        is inserted into a sorted list.  If a buffer duration has been
        set via :meth:`set_buffer_duration`, old entries beyond the
        duration window are pruned.

        Parameters
        ----------
        parent : str
            The parent frame identifier.
        child : str
            The child frame identifier.
        transform : Transform
            The rigid-body transform from *child* to *parent*.
        timestamp : float
            The time at which this transform is valid (seconds).
        is_static : bool
            If ``True``, treat this as a static (time-invariant) transform.
        """
        key = (parent, child)
        self._graph.add_edge(parent, child)

        if is_static:
            self._static[key] = transform
            self._static_keys.add(key)
            return

        # Dynamic transform: maintain a sorted list by timestamp.
        if key not in self._dynamic:
            self._dynamic[key] = []

        entries = self._dynamic[key]

        # Use bisect to find the insertion point.
        timestamps = [e[0] for e in entries]
        idx = bisect.bisect_left(timestamps, timestamp)

        # Avoid duplicate timestamps — overwrite if exact match.
        if idx < len(entries) and entries[idx][0] == timestamp:
            entries[idx] = (timestamp, transform)
        else:
            entries.insert(idx, (timestamp, transform))

        # Prune old entries if a buffer duration is configured.
        if self._buffer_duration is not None and entries:
            newest = entries[-1][0]
            cutoff = newest - self._buffer_duration
            # Find the first entry that is within the window.
            cut_idx = bisect.bisect_left(timestamps, cutoff)
            if cut_idx > 0:
                del entries[:cut_idx]

    def add_tf_message(self, tf_msg: Any, is_static: bool = False) -> None:
        """Convenience method to unpack a TFMessage and add all transforms.

        Expects *tf_msg* to have a ``transforms`` attribute that is an
        iterable of objects with:

        - ``header.stamp`` — a timestamp (float, or object with a
          ``to_sec()`` method, or object with ``sec`` and ``nanosec``
          attributes)
        - ``header.frame_id`` — the parent frame
        - ``child_frame_id`` — the child frame
        - ``transform.translation`` — object with ``x``, ``y``, ``z``
        - ``transform.rotation`` — object with ``x``, ``y``, ``z``, ``w``

        This covers both ROS 1 ``tf2_msgs/TFMessage`` and ROS 2
        ``tf2_msgs/msg/TFMessage`` formats.

        Parameters
        ----------
        tf_msg : object
            A TFMessage or compatible object.
        is_static : bool
            If ``True``, all transforms in the message are treated as
            static.
        """
        for stamped_tf in tf_msg.transforms:
            # Extract timestamp.
            stamp = stamped_tf.header.stamp
            if isinstance(stamp, (int, float)):
                timestamp = float(stamp)
            elif hasattr(stamp, "to_sec"):
                timestamp = stamp.to_sec()
            elif hasattr(stamp, "sec"):
                timestamp = stamp.sec + stamp.nanosec * 1e-9
            else:
                timestamp = float(stamp)

            parent = stamped_tf.header.frame_id
            child = stamped_tf.child_frame_id

            t = stamped_tf.transform
            translation = Vector3(t.translation.x, t.translation.y, t.translation.z)
            rotation = Quaternion(
                t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
            )
            transform = Transform(translation, rotation)

            self.add_transform(parent, child, transform, timestamp, is_static)

    # -- Lookup -------------------------------------------------------------

    def lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        timestamp: float,
        interpolate: bool = True,
    ) -> Transform:
        """Look up the transform that maps points in *source_frame* into
        *target_frame* at the given *timestamp*.

        The method works as follows:

        1. **Find path:** Use the :class:`FrameGraph` to find the
           shortest path from *target_frame* to *source_frame*.

        2. **Collect edge transforms:** For each consecutive pair of
           frames ``(A, B)`` in the path, retrieve the transform at
           the requested timestamp.  If the stored edge direction is
           ``(A, B)`` (A is parent of B), we use the transform directly.
           If it is ``(B, A)`` (stored in reverse), we invert it.

        3. **Compose:** Multiply all edge transforms together in order
           to get the final ``T_target_source``.

        Parameters
        ----------
        target_frame : str
            The frame to express the result in.
        source_frame : str
            The frame to transform from.
        timestamp : float
            The time at which to evaluate the transform (seconds).
        interpolate : bool
            If ``True`` (default), interpolate between the two nearest
            stored transforms when no exact match exists.

        Returns
        -------
        Transform
            The composed transform from *source_frame* to *target_frame*.

        Raises
        ------
        FrameNotFoundError
            If either frame does not exist in the graph.
        NoPathError
            If no path exists between the two frames.
        LookupError
            If no transform data is available at the requested time.
        """
        if target_frame == source_frame:
            return Transform.identity()

        # Step 1: find the path from target to source through the graph.
        path = self._graph.get_chain(target_frame, source_frame)

        # Step 2 & 3: walk the path, look up each edge, compose.
        result = Transform.identity()
        for i in range(len(path) - 1):
            frame_a = path[i]
            frame_b = path[i + 1]

            # Determine which direction this edge is stored.
            forward_key = (frame_a, frame_b)  # A is parent of B
            reverse_key = (frame_b, frame_a)  # B is parent of A

            if forward_key in self._static_keys or forward_key in self._dynamic:
                # Edge is stored as (A, B): T maps B -> A.
                # We want T_A_B which maps points in B into A.
                edge_tf = self._get_transform_at_time(
                    frame_a, frame_b, timestamp, interpolate
                )
            elif reverse_key in self._static_keys or reverse_key in self._dynamic:
                # Edge is stored as (B, A): T maps A -> B.
                # We want T_A_B, so invert: T_A_B = (T_B_A)^{-1}
                edge_tf = self._get_transform_at_time(
                    frame_b, frame_a, timestamp, interpolate
                ).inverse()
            else:
                raise LookupError(
                    f"No transform data between '{frame_a}' and '{frame_b}'"
                )

            result = result * edge_tf

        return result

    def can_transform(
        self, target: str, source: str, timestamp: float
    ) -> bool:
        """Check whether a transform lookup would succeed.

        This is a non-throwing version of :meth:`lookup_transform`.
        Returns ``True`` if the lookup would succeed, ``False``
        otherwise.

        Parameters
        ----------
        target : str
            The target frame.
        source : str
            The source frame.
        timestamp : float
            The query timestamp.

        Returns
        -------
        bool
            ``True`` if the transform can be looked up.
        """
        try:
            self.lookup_transform(target, source, timestamp)
            return True
        except (FrameNotFoundError, NoPathError, LookupError):
            return False

    # -- Accessors ----------------------------------------------------------

    def get_frames(self) -> set[str]:
        """Return the set of all known frame identifiers.

        Returns
        -------
        set[str]
            All frame names that have been added to the buffer.
        """
        return self._graph.all_frames()

    def get_frame_graph(self) -> FrameGraph:
        """Return the underlying :class:`FrameGraph`.

        This can be used for visualisation (e.g. :meth:`FrameGraph.to_ascii_tree`)
        or to inspect connectivity.

        Returns
        -------
        FrameGraph
            The frame graph instance.
        """
        return self._graph

    # -- Maintenance --------------------------------------------------------

    def clear(self) -> None:
        """Remove all transforms and reset the buffer to its initial state.

        This clears both dynamic and static transforms and resets the
        frame graph.
        """
        self._dynamic.clear()
        self._static.clear()
        self._static_keys.clear()
        self._graph = FrameGraph()

    def set_buffer_duration(self, duration: float) -> None:
        """Set the maximum age of dynamic transforms to retain.

        Transforms older than ``newest_timestamp - duration`` (per edge)
        are pruned on the next insertion.  This bounds memory usage for
        long-running applications.

        Calling with ``duration <= 0`` effectively disables buffering
        (only the latest transform is kept).

        Parameters
        ----------
        duration : float
            Maximum buffer duration in seconds.
        """
        self._buffer_duration = duration

    # -- Private helpers ----------------------------------------------------

    def _get_transform_at_time(
        self,
        parent: str,
        child: str,
        timestamp: float,
        interpolate: bool,
    ) -> Transform:
        """Retrieve the transform for a specific edge at a given time.

        For static transforms, the stored value is returned directly.
        For dynamic transforms, binary search finds the two nearest
        entries and (optionally) interpolates between them.

        Binary search rationale
        ~~~~~~~~~~~~~~~~~~~~~~~
        Dynamic transforms are stored in a time-sorted list.  A naive
        linear scan is O(n) per lookup; with thousands of transforms
        accumulated over minutes of recording, this becomes expensive
        when composing multi-edge paths.  Binary search (via
        ``bisect``) reduces this to O(log n).

        Parameters
        ----------
        parent : str
            The parent frame.
        child : str
            The child frame.
        timestamp : float
            The query time (seconds).
        interpolate : bool
            Whether to interpolate between bracketing entries.

        Returns
        -------
        Transform
            The transform at the requested time.

        Raises
        ------
        LookupError
            If no transform data is available for this edge, or if
            the requested time is outside the stored range and
            interpolation cannot be performed.
        """
        key = (parent, child)

        # Static transforms are time-invariant.
        if key in self._static_keys:
            return self._static[key]

        entries = self._dynamic.get(key)
        if not entries:
            raise LookupError(
                f"No transform data for edge '{parent}' -> '{child}'"
            )

        # Extract timestamps for binary search.
        timestamps = [e[0] for e in entries]

        # Exact match check via bisect.
        idx = bisect.bisect_left(timestamps, timestamp)
        if idx < len(entries) and entries[idx][0] == timestamp:
            return entries[idx][1]

        # Single entry — return it (nearest-neighbour).
        if len(entries) == 1:
            return entries[0][1]

        # Clamp to bounds if outside range.
        if idx == 0:
            # Before the earliest entry — return earliest.
            return entries[0][1]
        if idx >= len(entries):
            # After the latest entry — return latest.
            return entries[-1][1]

        # We have bracketing entries: entries[idx-1] and entries[idx].
        if not interpolate:
            # Return the nearest entry.
            t_before = entries[idx - 1][0]
            t_after = entries[idx][0]
            if abs(timestamp - t_before) <= abs(timestamp - t_after):
                return entries[idx - 1][1]
            return entries[idx][1]

        # Interpolate between the two bracketing transforms.
        return self._interpolate(
            entries[idx - 1][0],
            entries[idx - 1][1],
            entries[idx][0],
            entries[idx][1],
            timestamp,
        )

    @staticmethod
    def _interpolate(
        t1: float,
        tf1: Transform,
        t2: float,
        tf2: Transform,
        target_time: float,
    ) -> Transform:
        """Interpolate between two timestamped transforms.

        Translation is linearly interpolated (LERP) and rotation is
        spherically interpolated (SLERP).

        Why LERP for translation?
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        Linear interpolation of position is appropriate when the motion
        between samples is approximately linear (which it is for small
        time intervals at typical publication rates of 50--100 Hz).

        Why SLERP for rotation?
        ~~~~~~~~~~~~~~~~~~~~~~~
        Linearly interpolating quaternions and re-normalising (NLERP)
        does not produce constant angular velocity — the rotation speeds
        up in the middle and slows at the endpoints.  SLERP traverses
        the great-circle arc at constant speed, which is essential for
        physically meaningful motion.

        Parameters
        ----------
        t1 : float
            Timestamp of the first (earlier) transform.
        tf1 : Transform
            The earlier transform.
        t2 : float
            Timestamp of the second (later) transform.
        tf2 : Transform
            The later transform.
        target_time : float
            The desired timestamp, must satisfy ``t1 <= target_time <= t2``.

        Returns
        -------
        Transform
            The interpolated transform at *target_time*.
        """
        if t2 == t1:
            return tf1

        # Compute interpolation parameter alpha in [0, 1].
        alpha = (target_time - t1) / (t2 - t1)
        alpha = max(0.0, min(1.0, alpha))

        return interpolate_transform(tf1, tf2, alpha)

    def __repr__(self) -> str:
        n_dynamic = sum(len(v) for v in self._dynamic.values())
        n_static = len(self._static)
        n_frames = len(self.get_frames())
        return (
            f"TransformBuffer(frames={n_frames}, "
            f"dynamic_entries={n_dynamic}, static_entries={n_static})"
        )
