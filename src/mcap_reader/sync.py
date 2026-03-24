"""
Time synchronization across multi-rate sensor streams.

The Clock Problem
=================
A typical robot carries sensors that sample at different rates:

  - IMU: 200-1000 Hz
  - Joint encoders: 100-500 Hz
  - Cameras: 15-60 Hz
  - LiDAR: 10-20 Hz (rotation rate)

Each sensor has its own hardware clock that drifts independently.  Commodity
crystal oscillators drift at roughly **20 parts per million (ppm)**, which
means two clocks that were perfectly synchronized at power-on will diverge by
about 2 ms after 100 seconds.  Over a 10-minute recording that is 12 ms --
enough to misalign a camera frame with the IMU sample that was *actually*
captured at the same physical instant.

Even when sensors are synchronized to a shared clock (e.g. PTP / IEEE 1588
or ``chrony`` disciplining the system clock), residual jitter from the
software stack (kernel scheduling, DDS transport, rosbag2 writer thread)
introduces additional timing noise.  The result is that messages from
different topics arrive with slightly different timestamps, and downstream
algorithms (sensor fusion, SLAM, imitation learning) need a principled way
to align them.

header.stamp vs log_time
~~~~~~~~~~~~~~~~~~~~~~~~~
MCAP messages carry two timestamps:

* **log_time** -- when the rosbag2 writer thread committed the message to
  disk.  This is monotonically increasing but includes variable latency from
  the DDS middleware and the recording pipeline.
* **header.stamp** -- the sensor-side timestamp embedded inside the ROS 2
  message payload.  For a properly written driver this is the *capture time*
  (camera shutter, IMU sample-and-hold, LiDAR firing).

For cross-sensor alignment you almost always want ``header.stamp`` because
it reflects when the physical measurement was taken, not when the software
happened to record it.  The :class:`~mcap_reader.reader.RawMessage` dataclass
exposes ``timestamp`` (which prefers ``header.stamp``) and ``log_time``
separately for this reason.

Synchronization Strategies
==========================

Nearest-neighbor (``"nearest"``)
    For each reference message at time *t_ref*, find the message on each
    secondary topic whose ``timestamp`` is closest to *t_ref*.  Simple and
    lossless -- no assumptions about the signal shape.  The trade-off is
    that the matched message may be up to half a period old (e.g. 16 ms
    for a 30 Hz camera).

Interpolation (``"interpolate"``)
    For each reference message at time *t_ref*, find the two bracketing
    messages on each secondary topic (one before, one after) and linearly
    interpolate scalar fields or SLERP quaternion fields.  Produces a
    time-aligned "virtual sample" at *t_ref* exactly.  The trade-off is
    that the interpolated value is an *approximation* -- it assumes the
    signal varies smoothly between samples.

Why SLERP for orientation interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orientations live on the rotation manifold SO(3), not in Euclidean space.
Linearly interpolating two quaternions and re-normalizing (NLERP) does not
produce constant angular velocity -- the rotation speeds up in the middle
and slows down at the endpoints.  Spherical Linear Interpolation (SLERP)
traverses the geodesic (great-circle arc) on the unit quaternion 4-sphere
at constant angular velocity, giving physically meaningful intermediate
orientations.  See :func:`~mcap_reader.transforms.math.slerp` for the
implementation.

Why per-topic max_delay matters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A single global ``max_delay`` threshold does not work well when sensor
rates differ by an order of magnitude.  Consider:

  - IMU at 1000 Hz: samples are 1 ms apart, so a sync delay > 2 ms is
    suspicious.
  - Camera at 30 Hz: samples are 33 ms apart, so a sync delay of 16 ms
    (half a period) is perfectly normal for nearest-neighbor matching.

The ``per_topic_max_delay`` field in :class:`SyncConfig` lets you set
tight thresholds on high-rate sensors while being lenient on low-rate ones.
If a per-topic override is not provided, the global ``max_delay`` is used.

Typical usage::

    from mcap_reader.reader import McapReader
    from mcap_reader.sync import SyncConfig, TimeSynchronizer

    config = SyncConfig(
        reference_topic="/imu/data",
        topics=["/camera/image_raw", "/joint_states"],
        strategy="nearest",
        max_delay=0.05,
        per_topic_max_delay={"/camera/image_raw": 0.020},
    )

    with McapReader("recording.mcap") as reader:
        synchronizer = TimeSynchronizer(reader, config)
        for result in synchronizer.iter_synchronized():
            imu_time = result.reference_timestamp
            camera_msg = result.messages["/camera/image_raw"]
            if camera_msg is not None:
                process(imu_time, camera_msg)

        quality = synchronizer.get_quality()
        print(quality)
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass, field
from typing import Iterator

from mcap_reader.reader import McapReader, RawMessage
from mcap_reader.transforms.math import (
    Quaternion,
    Vector3,
    interpolate_transform,
    lerp_vector,
    slerp,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration and result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyncConfig:
    """Configuration for multi-topic time synchronization.

    Attributes
    ----------
    reference_topic : str
        The topic whose timestamps define the synchronization "clock".
        Every output :class:`SyncResult` corresponds to one message on
        this topic.  Typically the highest-rate sensor (e.g. IMU) so
        that slower modalities are up-sampled rather than down-sampled.
    topics : list[str]
        The secondary topics to synchronize against the reference.  Each
        will appear as a key in :attr:`SyncResult.messages`.
    strategy : str
        Synchronization strategy.  One of:

        - ``"nearest"`` -- nearest-neighbor matching (default).
        - ``"interpolate"`` -- linear/SLERP interpolation between
          bracketing messages.
    max_delay : float
        Global maximum allowable time delay (in seconds) between a
        reference timestamp and the matched secondary message.  If the
        closest match exceeds this threshold the secondary message is
        set to ``None`` in the result (a "drop").  Default: 50 ms.
    per_topic_max_delay : dict[str, float] | None
        Optional per-topic override for ``max_delay``.  Keys are topic
        names from ``topics``; any topic not listed falls back to the
        global ``max_delay``.
    """

    reference_topic: str
    topics: list[str]
    strategy: str = "nearest"
    max_delay: float = 0.05
    per_topic_max_delay: dict[str, float] | None = None

    def get_max_delay(self, topic: str) -> float:
        """Return the effective max_delay for a given topic.

        Checks ``per_topic_max_delay`` first, falling back to the global
        ``max_delay``.

        Parameters
        ----------
        topic : str
            The topic name to look up.

        Returns
        -------
        float
            Maximum allowable sync delay in seconds.
        """
        if self.per_topic_max_delay and topic in self.per_topic_max_delay:
            return self.per_topic_max_delay[topic]
        return self.max_delay


@dataclass(frozen=True, slots=True)
class SyncResult:
    """A single synchronized observation across all topics.

    Each ``SyncResult`` corresponds to one reference message.  The
    ``messages`` dict maps each secondary topic to the matched (or
    interpolated) message, or ``None`` if no match was found within the
    allowable delay.

    Attributes
    ----------
    reference_timestamp : float
        The ``header.stamp`` (or ``log_time`` fallback) of the reference
        message, in seconds.
    messages : dict[str, RawMessage | None]
        Matched messages keyed by topic name.  ``None`` means the topic
        had no message within ``max_delay`` of the reference timestamp
        (a "dropped" observation).
    sync_delays : dict[str, float | None]
        The signed time difference ``t_matched - t_reference`` for each
        topic, in seconds.  ``None`` when the message was dropped.
        Positive means the matched message is *newer* than the reference;
        negative means it is *older*.
    interpolation_alphas : dict[str, float | None]
        For the ``"interpolate"`` strategy, the interpolation parameter
        alpha in [0, 1] between the two bracketing messages.  ``None``
        for ``"nearest"`` strategy or when the message was dropped.
    """

    reference_timestamp: float
    messages: dict[str, RawMessage | None]
    sync_delays: dict[str, float | None]
    interpolation_alphas: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class SyncQuality:
    """Aggregate quality metrics for a completed synchronization pass.

    These metrics help diagnose timing problems in a recording:

    - High ``mean_delay`` on a topic suggests the sensor clock is
      systematically offset from the reference clock.
    - High ``max_delay`` suggests occasional timing glitches (e.g.
      kernel scheduling hiccups, USB bus contention).
    - High ``dropped_count`` means the sensor was often unavailable
      within the allowable delay window, which can happen when a
      camera driver skips frames under CPU load.

    Attributes
    ----------
    mean_delay : dict[str, float]
        Mean absolute sync delay per topic (in seconds), computed only
        over non-dropped observations.
    max_delay : dict[str, float]
        Maximum absolute sync delay per topic (in seconds).
    dropped_count : dict[str, int]
        Number of reference messages for which each topic had no match
        within the allowable delay.
    total_synced : int
        Total number of reference messages processed.
    """

    mean_delay: dict[str, float]
    max_delay: dict[str, float]
    dropped_count: dict[str, int]
    total_synced: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _TopicTimeline:
    """Sorted timeline of messages for a single topic.

    Stores messages sorted by timestamp for efficient binary-search
    lookups used by both nearest-neighbor and interpolation strategies.

    Attributes
    ----------
    timestamps : list[float]
        Sorted sensor timestamps (seconds).
    messages : list[RawMessage]
        Messages in the same order as ``timestamps``.
    """

    timestamps: list[float] = field(default_factory=list)
    messages: list[RawMessage] = field(default_factory=list)

    def find_nearest(self, t: float) -> tuple[RawMessage, float] | None:
        """Find the message whose timestamp is closest to *t*.

        Uses :func:`bisect.bisect_left` for O(log n) lookup in the
        sorted timestamp list.

        Parameters
        ----------
        t : float
            Target timestamp in seconds.

        Returns
        -------
        tuple[RawMessage, float] | None
            The nearest message and the signed delay (t_msg - t), or
            ``None`` if the timeline is empty.
        """
        if not self.timestamps:
            return None

        idx = bisect.bisect_left(self.timestamps, t)

        # Consider the candidate at idx and idx-1 (the two neighbors
        # of the insertion point).
        best_idx = None
        best_dist = float("inf")

        for candidate in (idx - 1, idx):
            if 0 <= candidate < len(self.timestamps):
                dist = abs(self.timestamps[candidate] - t)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = candidate

        if best_idx is None:
            return None

        delay = self.timestamps[best_idx] - t
        return self.messages[best_idx], delay

    def find_bracket(
        self, t: float
    ) -> tuple[tuple[RawMessage, RawMessage, float], None] | tuple[None, str]:
        """Find the two messages that bracket timestamp *t*.

        The bracketing messages satisfy ``t_before <= t <= t_after``.
        The interpolation alpha is::

            alpha = (t - t_before) / (t_after - t_before)

        Parameters
        ----------
        t : float
            Target timestamp in seconds.

        Returns
        -------
        tuple
            On success: ``((msg_before, msg_after, alpha), None)``.
            On failure: ``(None, reason_string)`` explaining why
            bracketing failed (e.g. target is before the first message
            or after the last).
        """
        if len(self.timestamps) < 2:
            return None, "fewer than 2 messages on topic"

        idx = bisect.bisect_left(self.timestamps, t)

        # t is before the first message.
        if idx == 0:
            return None, "target time is before first message"

        # t is after the last message.
        if idx >= len(self.timestamps):
            return None, "target time is after last message"

        before_idx = idx - 1
        after_idx = idx

        t_before = self.timestamps[before_idx]
        t_after = self.timestamps[after_idx]

        dt = t_after - t_before
        if dt < 1e-12:
            # Two messages at essentially the same timestamp -- alpha = 0.
            alpha = 0.0
        else:
            alpha = (t - t_before) / dt

        return (
            (self.messages[before_idx], self.messages[after_idx], alpha),
            None,
        )


# ---------------------------------------------------------------------------
# TimeSynchronizer
# ---------------------------------------------------------------------------


class TimeSynchronizer:
    """Align messages from multiple topics to a common reference timeline.

    This class reads all messages from the configured topics, indexes them
    by timestamp, and provides iterators that yield time-aligned
    :class:`SyncResult` bundles -- one per reference message.

    The synchronizer is **eager** in that it reads and indexes all messages
    up front (in ``__init__``).  This enables efficient binary-search
    lookups during iteration and makes it possible to compute quality
    metrics after a single pass.  For very large recordings where memory
    is a concern, consider filtering by time range when constructing the
    :class:`~mcap_reader.reader.McapReader`.

    Parameters
    ----------
    reader : McapReader
        An open MCAP reader to pull messages from.
    config : SyncConfig
        Synchronization parameters (reference topic, secondary topics,
        strategy, delay thresholds).

    Notes
    -----
    **Why eager indexing?**  Synchronization requires random access to
    messages from each topic (binary search for nearest neighbor or
    bracket finding).  Streaming one-by-one would require either O(n)
    scans per reference message or maintaining rolling buffers with
    careful bookkeeping.  Since the number of messages per topic is
    typically manageable (a 10-minute recording at 1000 Hz produces
    600 000 messages -- about 50 MB of timestamps+references), the
    eager approach is simpler and fast enough in practice.
    """

    def __init__(self, reader: McapReader, config: SyncConfig) -> None:
        self._config = config
        self._reader = reader

        # Validate strategy.
        valid_strategies = ("nearest", "interpolate")
        if config.strategy not in valid_strategies:
            raise ValueError(
                f"Unknown sync strategy {config.strategy!r}. "
                f"Expected one of {valid_strategies}."
            )

        # Collect messages from all topics (reference + secondary) into
        # sorted timelines.  We request all topics in a single pass through
        # the MCAP file to avoid re-reading.
        all_topics = [config.reference_topic] + list(config.topics)
        timelines: dict[str, _TopicTimeline] = {
            topic: _TopicTimeline() for topic in all_topics
        }

        for msg in reader.iter_messages(topics=all_topics):
            tl = timelines.get(msg.topic)
            if tl is not None:
                tl.timestamps.append(msg.timestamp)
                tl.messages.append(msg)

        # Verify chronological order (header.stamp should be monotonic for
        # a well-behaved driver, but defensive check).
        for topic, tl in timelines.items():
            for i in range(1, len(tl.timestamps)):
                if tl.timestamps[i] < tl.timestamps[i - 1]:
                    logger.warning(
                        "Non-monotonic timestamps detected on topic %s "
                        "(message %d: %.6f < %.6f). Results may be degraded.",
                        topic,
                        i,
                        tl.timestamps[i],
                        tl.timestamps[i - 1],
                    )
                    # Sort to recover.  Stable sort preserves order of
                    # equal-timestamp messages.
                    pairs = sorted(
                        zip(tl.timestamps, tl.messages), key=lambda p: p[0]
                    )
                    tl.timestamps = [p[0] for p in pairs]
                    tl.messages = [p[1] for p in pairs]
                    break

        self._reference_timeline = timelines[config.reference_topic]
        self._secondary_timelines: dict[str, _TopicTimeline] = {
            topic: timelines[topic] for topic in config.topics
        }

        # Accumulators for quality metrics -- populated during iteration.
        self._delay_accum: dict[str, list[float]] = {
            topic: [] for topic in config.topics
        }
        self._dropped_count: dict[str, int] = {
            topic: 0 for topic in config.topics
        }
        self._total_synced: int = 0

        logger.info(
            "TimeSynchronizer initialized: reference=%s (%d msgs), "
            "secondary topics=%s, strategy=%s",
            config.reference_topic,
            len(self._reference_timeline.timestamps),
            [
                f"{t} ({len(self._secondary_timelines[t].timestamps)})"
                for t in config.topics
            ],
            config.strategy,
        )

    # -- Nearest-neighbor synchronization -----------------------------------

    def sync_nearest(self) -> Iterator[SyncResult]:
        """Yield time-aligned results using nearest-neighbor matching.

        For each message on the reference topic (in chronological order),
        finds the temporally closest message on each secondary topic via
        binary search.  If the absolute delay exceeds the configured
        ``max_delay`` for that topic, the secondary message is set to
        ``None`` (a "drop").

        Yields
        ------
        SyncResult
            One result per reference message containing the matched
            secondary messages and delay metadata.

        Notes
        -----
        Nearest-neighbor matching is appropriate when:

        - The secondary signal is piecewise-constant or changes slowly
          between samples (e.g. robot joint positions at 500 Hz when the
          reference is a 30 Hz camera).
        - You need the *actual* recorded message (not an interpolated
          approximation) for downstream processing.

        The worst-case sync error for a periodic signal at frequency f
        is ``1 / (2f)`` (half the sampling period), which occurs when the
        reference timestamp falls exactly between two secondary samples.
        """
        for ref_msg in self._reference_timeline.messages:
            t_ref = ref_msg.timestamp
            messages: dict[str, RawMessage | None] = {}
            delays: dict[str, float | None] = {}
            alphas: dict[str, float | None] = {}

            for topic, tl in self._secondary_timelines.items():
                result = tl.find_nearest(t_ref)
                max_d = self._config.get_max_delay(topic)

                if result is None:
                    messages[topic] = None
                    delays[topic] = None
                    alphas[topic] = None
                    self._dropped_count[topic] += 1
                else:
                    matched_msg, delay = result
                    if abs(delay) > max_d:
                        messages[topic] = None
                        delays[topic] = None
                        alphas[topic] = None
                        self._dropped_count[topic] += 1
                    else:
                        messages[topic] = matched_msg
                        delays[topic] = delay
                        alphas[topic] = None
                        self._delay_accum[topic].append(abs(delay))

            self._total_synced += 1
            yield SyncResult(
                reference_timestamp=t_ref,
                messages=messages,
                sync_delays=delays,
                interpolation_alphas=alphas,
            )

    # -- Interpolation synchronization --------------------------------------

    def sync_interpolate(self) -> Iterator[SyncResult]:
        """Yield time-aligned results using linear/SLERP interpolation.

        For each message on the reference topic, finds the two bracketing
        messages on each secondary topic (the one immediately before and
        immediately after the reference timestamp) and computes the
        interpolation parameter::

            alpha = (t_ref - t_before) / (t_after - t_before)

        The caller can use ``alpha`` with the appropriate interpolation
        function (:func:`~mcap_reader.transforms.math.lerp_vector` for
        positions, :func:`~mcap_reader.transforms.math.slerp` for
        orientations, or :func:`~mcap_reader.transforms.math.interpolate_transform`
        for full rigid-body transforms).

        This method stores the *earlier* (before) message in the result's
        ``messages`` dict together with the ``alpha`` value so the caller
        can perform type-aware interpolation.  We return the raw message
        rather than attempting to interpolate arbitrary ROS message types,
        because the interpolation semantics are type-dependent:

        - Scalar fields (positions, velocities): linear interpolation
        - Orientation quaternions: SLERP
        - Images: not interpolatable (use nearest instead)
        - Discrete states (e.g. button presses): not interpolatable

        Yields
        ------
        SyncResult
            One result per reference message.  The ``messages`` dict
            contains the *before* bracketing message; ``interpolation_alphas``
            contains the alpha for computing the interpolated value.

        Notes
        -----
        If the reference timestamp falls exactly on a secondary message's
        timestamp, ``alpha`` will be 0.0 and the "before" message is the
        exact match.

        If the reference timestamp falls outside the range of a secondary
        topic's messages (before the first or after the last), that topic
        is dropped for that reference message.
        """
        for ref_msg in self._reference_timeline.messages:
            t_ref = ref_msg.timestamp
            messages: dict[str, RawMessage | None] = {}
            delays: dict[str, float | None] = {}
            alphas: dict[str, float | None] = {}

            for topic, tl in self._secondary_timelines.items():
                max_d = self._config.get_max_delay(topic)
                bracket_result, reason = tl.find_bracket(t_ref)

                if bracket_result is None:
                    messages[topic] = None
                    delays[topic] = None
                    alphas[topic] = None
                    self._dropped_count[topic] += 1
                else:
                    msg_before, msg_after, alpha = bracket_result

                    # Check that both bracketing messages are within
                    # the allowable delay window.
                    delay_before = abs(t_ref - msg_before.timestamp)
                    delay_after = abs(t_ref - msg_after.timestamp)
                    max_bracket_delay = max(delay_before, delay_after)

                    if max_bracket_delay > max_d:
                        messages[topic] = None
                        delays[topic] = None
                        alphas[topic] = None
                        self._dropped_count[topic] += 1
                    else:
                        # Store the before-message; the caller uses alpha
                        # to interpolate between before and after.
                        messages[topic] = msg_before
                        delays[topic] = msg_before.timestamp - t_ref
                        alphas[topic] = alpha
                        self._delay_accum[topic].append(
                            min(delay_before, delay_after)
                        )

            self._total_synced += 1
            yield SyncResult(
                reference_timestamp=t_ref,
                messages=messages,
                sync_delays=delays,
                interpolation_alphas=alphas,
            )

    # -- Dispatch -----------------------------------------------------------

    def iter_synchronized(self) -> Iterator[SyncResult]:
        """Iterate over synchronized results using the configured strategy.

        Dispatches to :meth:`sync_nearest` or :meth:`sync_interpolate`
        based on :attr:`SyncConfig.strategy`.

        Yields
        ------
        SyncResult
            One result per reference message.

        Raises
        ------
        ValueError
            If the configured strategy is not recognized (should be caught
            in ``__init__``, but guarded here as well).
        """
        if self._config.strategy == "nearest":
            yield from self.sync_nearest()
        elif self._config.strategy == "interpolate":
            yield from self.sync_interpolate()
        else:
            raise ValueError(
                f"Unknown strategy: {self._config.strategy!r}"
            )

    # -- Quality metrics ----------------------------------------------------

    def get_quality(self) -> SyncQuality:
        """Compute synchronization quality metrics after iteration.

        Call this after fully consuming :meth:`iter_synchronized` (or
        one of the strategy-specific iterators).  The metrics summarize
        timing alignment across the entire recording.

        Returns
        -------
        SyncQuality
            Aggregate delay statistics and drop counts per topic.

        Notes
        -----
        Metrics interpretation:

        - **mean_delay** close to zero indicates good clock alignment.
          A consistent non-zero bias suggests a fixed clock offset
          between the reference and secondary sensor.
        - **max_delay** close to ``max_delay`` threshold indicates the
          sensors are barely in sync; consider increasing the threshold
          or investigating clock discipline.
        - **dropped_count / total_synced** gives the drop rate.  Rates
          above 5% may indicate a driver issue (frame drops) or a
          ``max_delay`` threshold that is too tight for the sensor rate.
        """
        mean_delays: dict[str, float] = {}
        max_delays: dict[str, float] = {}

        for topic in self._config.topics:
            accum = self._delay_accum[topic]
            if accum:
                mean_delays[topic] = sum(accum) / len(accum)
                max_delays[topic] = max(accum)
            else:
                mean_delays[topic] = 0.0
                max_delays[topic] = 0.0

        return SyncQuality(
            mean_delay=mean_delays,
            max_delay=max_delays,
            dropped_count=dict(self._dropped_count),
            total_synced=self._total_synced,
        )

    # -- DataFrame export ---------------------------------------------------

    def to_pandas(self) -> "pd.DataFrame":
        """Export synchronized data as a pandas DataFrame.

        Runs :meth:`iter_synchronized` (if not already consumed) and
        collects the results into a tabular format with one row per
        reference message.

        Columns:

        - ``reference_timestamp``: float, the reference message time.
        - ``{topic}_timestamp``: float or NaN, the matched message time.
        - ``{topic}_delay``: float or NaN, the sync delay in seconds.
        - ``{topic}_alpha``: float or NaN, the interpolation alpha.

        Returns
        -------
        pd.DataFrame
            A DataFrame with one row per synchronized observation.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_pandas(). "
                "Install it with: pip install pandas"
            ) from None

        rows: list[dict] = []
        for result in self.iter_synchronized():
            row: dict = {"reference_timestamp": result.reference_timestamp}
            for topic in self._config.topics:
                msg = result.messages.get(topic)
                delay = result.sync_delays.get(topic)
                alpha = result.interpolation_alphas.get(topic)

                # Use a sanitized column prefix (replace '/' with '_',
                # strip leading underscore).
                prefix = topic.replace("/", "_").lstrip("_")

                row[f"{prefix}_timestamp"] = (
                    msg.timestamp if msg is not None else float("nan")
                )
                row[f"{prefix}_delay"] = (
                    delay if delay is not None else float("nan")
                )
                row[f"{prefix}_alpha"] = (
                    alpha if alpha is not None else float("nan")
                )

            rows.append(row)

        return pd.DataFrame(rows)
