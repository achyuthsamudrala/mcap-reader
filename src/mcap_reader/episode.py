"""Episode boundary detection and metadata extraction for robot recordings.

What is an "episode" in robot learning?
========================================
In imitation learning and reinforcement learning, a robot recording is
typically divided into **episodes** -- discrete chunks of time where the
robot attempts a single task (e.g. "pick up the red block" or "navigate
to the charging station"). Between episodes there may be:

  - A human resetting the scene (moving objects back to starting positions)
  - The robot returning to a home configuration
  - A pause while the operator presses "start" again
  - An explicit software signal (e.g. publishing to ``/episode_start``)

Knowing where episodes start and end is critical for downstream use:

  - **Training data loaders** need to split recordings into fixed-length
    windows *within* episodes -- crossing an episode boundary mid-window
    would mix unrelated behaviour and confuse the policy.
  - **Evaluation metrics** are computed per-episode (success rate, task
    completion time, cumulative reward).
  - **Data filtering** lets you discard failed episodes or outliers.

How real robot datasets are structured
--------------------------------------
Real-world robot datasets (DROID, BridgeData V2, RT-1, Aloha) record
continuously and rely on one of two conventions to mark episode boundaries:

1. **Gap-based:** The operator stops recording, resets the scene, and
   starts recording again. This creates a gap of several seconds (or
   minutes) in the message timestamps. The recording software may
   produce one MCAP file per episode, or one long file with gaps.

2. **Marker-based:** The recording runs continuously and a supervisory
   node publishes special messages to mark transitions:

     - ``/episode_start`` or ``/reset`` — signals the beginning of a new episode
     - ``/episode_end`` or ``/done`` — signals the end of the current episode
     - ``/episode_success`` (``std_msgs/Bool``) — whether the episode succeeded

   These marker topics typically carry minimal payloads (empty messages
   or a single boolean). Some setups use ``std_msgs/msg/Int32`` on an
   ``/episode_id`` topic instead.

Why gap-based detection works in practice
-----------------------------------------
Even without explicit markers, gap-based detection is surprisingly
reliable because:

  - **Sensor streams are continuous within an episode.** A 30 Hz camera
    produces a frame every ~33 ms, and an IMU at 200 Hz produces a sample
    every 5 ms. Any gap significantly larger than the sensor period
    indicates a pause in data collection.

  - **Reset periods are long relative to sensor periods.** A human resetting
    objects takes at least a few seconds. Even a quick "press button to
    restart" introduces a gap of 0.5-2 seconds -- orders of magnitude
    larger than inter-message intervals.

  - **Network hiccups are rare and short.** Occasional dropped messages
    create gaps of 1-2 sensor periods (tens of milliseconds), well below
    typical episode gap thresholds (2-10 seconds).

The main failure mode is when recording is paused for an unusually short
reset (< threshold) or when the robot idles within an episode for longer
than the threshold. Per-topic thresholds help here.

Why per-topic thresholds matter
-------------------------------
Different sensor modalities have vastly different publication rates:

  - IMU: 100-400 Hz (gap > 0.1s is suspicious)
  - Camera: 15-60 Hz (gap > 0.5s is suspicious)
  - Joint states: 50-500 Hz
  - LiDAR: 10-20 Hz
  - GPS: 1-10 Hz (gap > 5s may be normal)

A global 5-second threshold works well for cameras and joint states but
would miss brief pauses visible in the IMU stream. Conversely, a tight
threshold tuned for IMU would trigger false positives on slow topics
like GPS. Per-topic thresholds let you set ``{"/imu/data": 0.5,
"/camera/image_raw": 2.0, "/gps/fix": 30.0}`` for robust detection.

Common marker topic conventions
-------------------------------
Different research groups use different topic names:

  - **DROID / Berkeley:** ``/episode_start``, ``/episode_end``
  - **RT-X / Google:** ``/reset``, ``/done``
  - **Custom setups:** ``/task_start``, ``/task_end``, ``/trial_start``
  - **Boolean success:** ``/episode_success`` or ``/success`` carrying
    ``std_msgs/msg/Bool``

The marker-based detector watches a configurable set of topic names
and falls back to sensible defaults (the union of common conventions).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcap_reader.reader import McapReader

logger = logging.getLogger(__name__)

# Default topics to watch for episode boundary markers.
# These cover the most common conventions across major imitation learning
# datasets and research labs.
DEFAULT_START_TOPICS = ["/episode_start", "/reset", "/trial_start", "/task_start"]
DEFAULT_END_TOPICS = ["/episode_end", "/done", "/trial_end", "/task_end"]
DEFAULT_SUCCESS_TOPICS = ["/episode_success", "/success"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """Metadata for a single detected episode within a recording.

    An episode is a contiguous period of robot activity bounded by either
    temporal gaps, explicit marker messages, or user-specified timestamps.

    Attributes
    ----------
    index : int
        Zero-based episode index within the recording.
    start_time : float
        Timestamp (in seconds) of the first message in this episode.
    end_time : float
        Timestamp (in seconds) of the last message in this episode.
    duration : float
        ``end_time - start_time`` in seconds.
    topics : list[str]
        Topics that had messages during this episode.
    message_counts : dict[str, int]
        Number of messages per topic within this episode.
    sync_quality : dict[str, float] | None
        Optional per-topic synchronization quality scores (0.0 to 1.0).
        Computed as the fraction of messages on each topic that have a
        matching message on the reference topic within a tolerance window.
        ``None`` if sync quality was not evaluated.
    success : bool | None
        Whether this episode was marked as successful. ``None`` if no
        success marker was found (i.e., the dataset does not annotate
        success/failure, or this episode had no success signal).
    """

    index: int
    start_time: float
    end_time: float
    duration: float
    topics: list[str] = field(default_factory=list)
    message_counts: dict[str, int] = field(default_factory=dict)
    sync_quality: dict[str, float] | None = None
    success: bool | None = None

    def __str__(self) -> str:
        """Human-readable summary of this episode.

        Designed for CLI output and quick inspection. Shows the episode
        index, time range, duration, message counts, and success status.

        Example output::

            Episode 3: 45.200s - 67.800s (22.600s)
              Topics: /camera/image_raw (678 msgs), /imu/data (4520 msgs)
              Success: True
        """
        lines = [
            f"Episode {self.index}: "
            f"{self.start_time:.3f}s - {self.end_time:.3f}s "
            f"({self.duration:.3f}s)"
        ]
        if self.message_counts:
            topic_parts = []
            for topic in sorted(self.message_counts):
                count = self.message_counts[topic]
                topic_parts.append(f"{topic} ({count} msgs)")
            lines.append("  Topics: " + ", ".join(topic_parts))
        if self.success is not None:
            lines.append(f"  Success: {self.success}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary.

        Returns
        -------
        dict
            All episode metadata as plain Python types.
        """
        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "topics": self.topics,
            "message_counts": self.message_counts,
            "sync_quality": self.sync_quality,
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Episode detector
# ---------------------------------------------------------------------------


class EpisodeDetector:
    """Detects episode boundaries in an MCAP recording.

    Provides multiple detection strategies that can be selected based on
    the conventions used during data collection. The three strategies are:

    1. **Gap-based** (``detect_by_gaps``) -- looks for temporal gaps in
       message streams. Best for recordings where the operator paused
       between episodes.

    2. **Marker-based** (``detect_by_markers``) -- looks for explicit
       topic signals like ``/episode_start`` or ``/done``. Best for
       continuous recordings with supervisory software.

    3. **Manual** (``detect_manual``) -- user provides explicit
       ``(start, end)`` timestamp pairs. Best for post-hoc annotation.

    Parameters
    ----------
    reader : McapReader
        An open reader for the MCAP file to analyse. The reader must
        remain open for the lifetime of detection calls, as episode
        detection iterates over messages.

    Example
    -------
    ::

        with McapReader("recording.mcap") as reader:
            detector = EpisodeDetector(reader)

            # Automatic gap-based detection
            episodes = detector.detect_by_gaps(gap_threshold=5.0)

            # Or use the dispatcher
            episodes = detector.detect(method="gap", gap_threshold=3.0)

            for ep in episodes:
                print(ep)
    """

    def __init__(self, reader: McapReader) -> None:
        self._reader = reader

    # ------------------------------------------------------------------
    # Gap-based detection
    # ------------------------------------------------------------------

    def detect_by_gaps(
        self,
        gap_threshold: float = 5.0,
        per_topic_threshold: dict[str, float] | None = None,
    ) -> list[Episode]:
        """Detect episodes by finding temporal gaps in message streams.

        Iterates through all messages in chronological order and tracks
        the last seen timestamp per topic. When the gap between consecutive
        messages on *any* topic exceeds the threshold for that topic, a
        new episode boundary is declared.

        How it works step by step:

        1. Iterate all messages in ``log_time`` order.
        2. For each message, check if the gap since the last message
           *on the same topic* exceeds that topic's threshold.
        3. If a gap is detected, finalize the current episode and start
           a new one.
        4. After all messages are processed, finalize the last episode.

        The per-topic approach is more robust than a global "any message"
        gap because different topics have different natural cadences.
        A camera at 30 Hz has inter-message intervals of ~33 ms, while
        GPS at 1 Hz has intervals of ~1000 ms. A global gap detector
        would either set the threshold too high (missing camera gaps) or
        too low (triggering on normal GPS spacing).

        Parameters
        ----------
        gap_threshold : float
            Default gap threshold in seconds. If the time between two
            consecutive messages on any single topic exceeds this value,
            an episode boundary is declared. Default is 5.0 seconds,
            which works well for most camera and joint state topics.
        per_topic_threshold : dict[str, float] | None
            Optional per-topic overrides for the gap threshold. Keys are
            topic names, values are thresholds in seconds. Topics not
            listed use ``gap_threshold``. Example::

                {"/imu/data": 0.5, "/camera/image_raw": 2.0}

        Returns
        -------
        list[Episode]
            Detected episodes sorted by start time. If the recording has
            no gaps exceeding the threshold, a single episode spanning
            the entire recording is returned.

        Notes
        -----
        Gap-based detection examines ``log_time`` (wall-clock recording
        time), not ``header.stamp`` (sensor time). This is because gaps
        in recording correspond to pauses in the data collection process,
        which are reflected in ``log_time``. Sensor timestamps may have
        independent discontinuities (e.g., clock corrections) that do
        not correspond to episode boundaries.
        """
        if per_topic_threshold is None:
            per_topic_threshold = {}

        # State tracking across the iteration.
        last_time_per_topic: dict[str, float] = {}
        episode_messages: list[tuple[str, float]] = []  # (topic, log_time)
        episodes: list[Episode] = []

        for msg in self._reader.iter_messages():
            topic = msg.topic
            log_time = msg.log_time
            threshold = per_topic_threshold.get(topic, gap_threshold)

            # Check for a gap on this topic.
            if topic in last_time_per_topic:
                gap = log_time - last_time_per_topic[topic]
                if gap > threshold:
                    # Finalize the current episode.
                    if episode_messages:
                        ep = self._build_episode(
                            index=len(episodes),
                            messages=episode_messages,
                        )
                        episodes.append(ep)
                        logger.debug(
                            "Gap of %.2fs on %s at %.3f -> new episode %d",
                            gap, topic, log_time, len(episodes),
                        )
                    episode_messages = []

            last_time_per_topic[topic] = log_time
            episode_messages.append((topic, log_time))

        # Finalize the last episode.
        if episode_messages:
            ep = self._build_episode(
                index=len(episodes),
                messages=episode_messages,
            )
            episodes.append(ep)

        logger.info(
            "Gap-based detection found %d episode(s) with threshold=%.1fs",
            len(episodes), gap_threshold,
        )
        return episodes

    # ------------------------------------------------------------------
    # Marker-based detection
    # ------------------------------------------------------------------

    def detect_by_markers(
        self,
        start_topics: list[str] | None = None,
        end_topics: list[str] | None = None,
    ) -> list[Episode]:
        """Detect episodes using explicit marker topics.

        Watches for messages on designated "start" and "end" topics to
        delineate episode boundaries. This is the preferred method for
        recordings made with supervisory software that publishes episode
        lifecycle signals.

        How marker detection works:

        1. Intersect the requested marker topics with topics actually
           present in the recording. Warn if none are found.
        2. Iterate all messages in ``log_time`` order.
        3. When a start-marker message arrives, begin a new episode
           (finalizing any in-progress episode).
        4. When an end-marker message arrives, finalize the current episode.
        5. If a ``/episode_success`` (or similar) message carrying a
           ``std_msgs/msg/Bool`` arrives, record its ``data`` field as
           the success status of the current episode.
        6. After iteration, finalize any episode that was started but
           never ended (common when the recording was stopped mid-episode).

        Marker topic conventions across datasets:

        - **DROID / Berkeley:** ``/episode_start``, ``/episode_end``
        - **RT-X / Google:** ``/reset``, ``/done``
        - **Custom rigs:** ``/trial_start``, ``/trial_end``, ``/task_start``

        If neither ``start_topics`` nor ``end_topics`` is provided, the
        detector watches all of the above defaults plus
        ``/episode_success`` and ``/success`` for boolean success signals.

        Parameters
        ----------
        start_topics : list[str] | None
            Topic names that signal the start of a new episode. If
            ``None``, uses the built-in defaults (see
            ``DEFAULT_START_TOPICS``).
        end_topics : list[str] | None
            Topic names that signal the end of the current episode. If
            ``None``, uses the built-in defaults (see
            ``DEFAULT_END_TOPICS``).

        Returns
        -------
        list[Episode]
            Detected episodes sorted by start time. May be empty if no
            marker topics are present in the recording.

        Notes
        -----
        If only end-markers are present (no start-markers), the detector
        treats each end-marker as a boundary: the period between the
        previous end-marker (or recording start) and the current
        end-marker becomes one episode.
        """
        if start_topics is None:
            start_topics = DEFAULT_START_TOPICS
        if end_topics is None:
            end_topics = DEFAULT_END_TOPICS

        available_topics = set(self._reader.topic_names)
        start_set = set(start_topics) & available_topics
        end_set = set(end_topics) & available_topics
        success_set = set(DEFAULT_SUCCESS_TOPICS) & available_topics

        all_marker_topics = start_set | end_set | success_set

        if not all_marker_topics:
            logger.warning(
                "No marker topics found in recording. "
                "Looked for start=%s, end=%s, success=%s. "
                "Available topics: %s",
                start_topics, end_topics, DEFAULT_SUCCESS_TOPICS,
                sorted(available_topics),
            )
            return []

        # State tracking.
        episodes: list[Episode] = []
        current_messages: list[tuple[str, float]] = []
        current_success: bool | None = None
        in_episode = False

        for msg in self._reader.iter_messages():
            topic = msg.topic

            # Check for success marker (Bool message with a `data` field).
            if topic in success_set:
                try:
                    current_success = bool(msg.ros_msg.data)
                except AttributeError:
                    pass
                continue

            # Check for start marker.
            if topic in start_set:
                # Finalize any in-progress episode.
                if in_episode and current_messages:
                    ep = self._build_episode(
                        index=len(episodes),
                        messages=current_messages,
                        success=current_success,
                    )
                    episodes.append(ep)
                # Start new episode.
                current_messages = []
                current_success = None
                in_episode = True
                continue

            # Check for end marker.
            if topic in end_set:
                if in_episode and current_messages:
                    ep = self._build_episode(
                        index=len(episodes),
                        messages=current_messages,
                        success=current_success,
                    )
                    episodes.append(ep)
                current_messages = []
                current_success = None
                in_episode = False
                continue

            # Regular data message -- accumulate if inside an episode.
            # If no start markers exist but end markers do, treat all
            # messages as belonging to an episode until an end marker.
            if in_episode or (not start_set and end_set):
                if not in_episode:
                    in_episode = True
                current_messages.append((topic, msg.log_time))

        # Finalize any episode that was started but never ended.
        if in_episode and current_messages:
            ep = self._build_episode(
                index=len(episodes),
                messages=current_messages,
                success=current_success,
            )
            episodes.append(ep)

        logger.info(
            "Marker-based detection found %d episode(s) using "
            "start_topics=%s, end_topics=%s",
            len(episodes), sorted(start_set), sorted(end_set),
        )
        return episodes

    # ------------------------------------------------------------------
    # Manual detection
    # ------------------------------------------------------------------

    def detect_manual(
        self,
        boundaries: list[tuple[float, float]],
    ) -> list[Episode]:
        """Create episodes from user-specified time boundaries.

        This is useful when episode boundaries are known externally
        (e.g., from a separate annotation file, a CSV of timestamps,
        or visual inspection in Foxglove Studio).

        For each ``(start_ts, end_ts)`` pair, the detector iterates
        messages within that time window and builds an Episode with
        accurate topic lists and message counts.

        Parameters
        ----------
        boundaries : list[tuple[float, float]]
            A list of ``(start_time, end_time)`` pairs in seconds.
            Pairs should be non-overlapping and sorted by start time
            for best results, though overlapping ranges are handled
            (messages may appear in multiple episodes).

        Returns
        -------
        list[Episode]
            One Episode per boundary pair, in the order given.

        Raises
        ------
        ValueError
            If any boundary has ``start_time >= end_time``.
        """
        for i, (start, end) in enumerate(boundaries):
            if start >= end:
                raise ValueError(
                    f"Boundary {i} has start_time ({start}) >= end_time ({end}). "
                    f"Each boundary must have start_time < end_time."
                )

        episodes: list[Episode] = []
        for i, (start_ts, end_ts) in enumerate(boundaries):
            messages: list[tuple[str, float]] = []
            for msg in self._reader.iter_messages(
                start_time=start_ts, end_time=end_ts
            ):
                messages.append((msg.topic, msg.log_time))

            ep = self._build_episode(index=i, messages=messages)
            episodes.append(ep)

        logger.info(
            "Manual detection created %d episode(s) from user boundaries",
            len(episodes),
        )
        return episodes

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    def detect(self, method: str = "gap", **kwargs) -> list[Episode]:
        """Dispatch to the appropriate detection method.

        A convenience method that routes to ``detect_by_gaps``,
        ``detect_by_markers``, or ``detect_manual`` based on the
        ``method`` string. All additional keyword arguments are forwarded
        to the selected method.

        Parameters
        ----------
        method : str
            Detection strategy to use. One of:

            - ``"gap"`` -- gap-based detection (default). Accepts
              ``gap_threshold`` and ``per_topic_threshold``.
            - ``"marker"`` -- marker-based detection. Accepts
              ``start_topics`` and ``end_topics``.
            - ``"manual"`` -- manual boundaries. Requires
              ``boundaries`` kwarg.

        **kwargs
            Keyword arguments forwarded to the selected method.

        Returns
        -------
        list[Episode]
            Detected episodes from the selected method.

        Raises
        ------
        ValueError
            If ``method`` is not one of the recognized strategies.

        Example
        -------
        ::

            episodes = detector.detect("gap", gap_threshold=3.0)
            episodes = detector.detect("marker", start_topics=["/start"])
            episodes = detector.detect("manual", boundaries=[(0, 10), (15, 25)])
        """
        dispatch = {
            "gap": self.detect_by_gaps,
            "marker": self.detect_by_markers,
            "manual": self.detect_manual,
        }

        if method not in dispatch:
            raise ValueError(
                f"Unknown detection method {method!r}. "
                f"Choose from: {sorted(dispatch.keys())}"
            )

        return dispatch[method](**kwargs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_episode(
        index: int,
        messages: list[tuple[str, float]],
        success: bool | None = None,
    ) -> Episode:
        """Build an Episode dataclass from a list of (topic, timestamp) pairs.

        Computes the time range, duration, per-topic message counts, and
        the list of active topics from the raw message list.

        Parameters
        ----------
        index : int
            Episode index.
        messages : list[tuple[str, float]]
            List of ``(topic_name, log_time)`` pairs belonging to this
            episode.
        success : bool | None
            Optional success flag from a marker message.

        Returns
        -------
        Episode
            A fully populated Episode dataclass.
        """
        if not messages:
            return Episode(
                index=index,
                start_time=0.0,
                end_time=0.0,
                duration=0.0,
                topics=[],
                message_counts={},
                success=success,
            )

        timestamps = [t for _, t in messages]
        start_time = min(timestamps)
        end_time = max(timestamps)

        counts: dict[str, int] = defaultdict(int)
        for topic, _ in messages:
            counts[topic] += 1

        topics = sorted(counts.keys())

        return Episode(
            index=index,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            topics=topics,
            message_counts=dict(counts),
            success=success,
        )
