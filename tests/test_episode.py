"""Tests for episode detection (EpisodeDetector).

Uses the generated multi-episode MCAP files with known gap structure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcap_reader.reader import McapReader
from mcap_reader.episode import Episode, EpisodeDetector


# ---------------------------------------------------------------------------
# Gap-based detection
# ---------------------------------------------------------------------------


class TestEpisodeDetectionByGaps:
    """Test gap-based episode boundary detection."""

    def test_detects_correct_number_of_episodes(self, multi_episode_mcap: Path):
        """Should detect 3 episodes with 10s gaps between them.

        We use per_topic_threshold on a single topic to avoid
        double-splitting when both topics trigger the gap detector
        at the same boundary.
        """
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            # Use only IMU topic for gap detection to avoid double-splits
            episodes = detector.detect_by_gaps(
                gap_threshold=100.0,  # high default = no splits
                per_topic_threshold={"/imu/data": 5.0},
            )

            assert len(episodes) == 3

    def test_episode_indices(self, multi_episode_mcap: Path):
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            episodes = detector.detect_by_gaps(
                gap_threshold=100.0,
                per_topic_threshold={"/imu/data": 5.0},
            )

            for i, ep in enumerate(episodes):
                assert ep.index == i

    def test_episode_duration(self, multi_episode_mcap: Path):
        """Each episode should be approximately 2 seconds long."""
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            episodes = detector.detect_by_gaps(
                gap_threshold=100.0,
                per_topic_threshold={"/imu/data": 5.0},
            )

            for ep in episodes:
                # Episode duration should be close to 2.0s
                # (may be slightly less due to discrete sampling)
                assert 1.5 < ep.duration < 2.5

    def test_episode_has_topics(self, multi_episode_mcap: Path):
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            episodes = detector.detect_by_gaps(
                gap_threshold=100.0,
                per_topic_threshold={"/imu/data": 5.0},
            )

            for ep in episodes:
                assert len(ep.topics) > 0
                assert "/imu/data" in ep.topics
                assert "/joint_states" in ep.topics

    def test_episode_message_counts(self, multi_episode_mcap: Path):
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            episodes = detector.detect_by_gaps(
                gap_threshold=100.0,
                per_topic_threshold={"/imu/data": 5.0},
            )

            for ep in episodes:
                assert ep.message_counts["/imu/data"] > 0
                assert ep.message_counts["/joint_states"] > 0

    def test_high_threshold_single_episode(self, multi_episode_mcap: Path):
        """With a very high gap threshold, everything is one episode."""
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            episodes = detector.detect_by_gaps(gap_threshold=100.0)

            assert len(episodes) == 1

    def test_low_threshold_more_episodes(self, multi_episode_mcap: Path):
        """With a very low threshold, even normal inter-message gaps split episodes."""
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            # IMU at 100 Hz has dt=0.01s, JointState at 50 Hz has dt=0.02s
            # Threshold of 0.015 should split on JointState gaps
            episodes = detector.detect_by_gaps(gap_threshold=0.015)

            assert len(episodes) > 3

    def test_per_topic_threshold(self, multi_episode_mcap: Path):
        """Per-topic threshold can override the global one."""
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            episodes = detector.detect_by_gaps(
                gap_threshold=100.0,  # Very high global (no splits)
                per_topic_threshold={"/imu/data": 5.0},  # 5s for IMU only
            )
            # The IMU gap between episodes is 10s, so IMU threshold of 5s should split
            assert len(episodes) == 3

    def test_single_episode_no_gaps(self, imu_mcap: Path):
        """A continuous recording should produce a single episode."""
        with McapReader(imu_mcap) as reader:
            detector = EpisodeDetector(reader)
            episodes = detector.detect_by_gaps(gap_threshold=5.0)

            assert len(episodes) == 1


# ---------------------------------------------------------------------------
# Manual boundaries
# ---------------------------------------------------------------------------


class TestEpisodeDetectionManual:
    """Test manual episode boundary detection."""

    def test_manual_boundaries(self, multi_episode_mcap: Path):
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)

            # Get approximate start/end from reader
            start = reader.start_time
            end = reader.end_time
            mid = (start + end) / 2

            episodes = detector.detect_manual(
                boundaries=[(start, mid), (mid, end)]
            )
            assert len(episodes) == 2

    def test_manual_invalid_boundary_raises(self, multi_episode_mcap: Path):
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            with pytest.raises(ValueError, match="start_time.*end_time"):
                detector.detect_manual(boundaries=[(10.0, 5.0)])

    def test_manual_empty_boundary(self, multi_episode_mcap: Path):
        """Boundary outside recording range should produce empty episode."""
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            episodes = detector.detect_manual(
                boundaries=[(0.0, 1.0)]  # Way before recording starts
            )
            assert len(episodes) == 1
            assert episodes[0].duration == 0.0


# ---------------------------------------------------------------------------
# Dispatch method
# ---------------------------------------------------------------------------


class TestEpisodeDetectionDispatch:
    """Test the detect() dispatch method."""

    def test_dispatch_gap(self, multi_episode_mcap: Path):
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            episodes = detector.detect(
                method="gap",
                gap_threshold=100.0,
                per_topic_threshold={"/imu/data": 5.0},
            )
            assert len(episodes) == 3

    def test_dispatch_manual(self, multi_episode_mcap: Path):
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            start = reader.start_time
            end = reader.end_time
            episodes = detector.detect(
                method="manual",
                boundaries=[(start, end)],
            )
            assert len(episodes) == 1

    def test_dispatch_invalid_method_raises(self, multi_episode_mcap: Path):
        with McapReader(multi_episode_mcap) as reader:
            detector = EpisodeDetector(reader)
            with pytest.raises(ValueError, match="Unknown detection method"):
                detector.detect(method="invalid")


# ---------------------------------------------------------------------------
# Episode dataclass
# ---------------------------------------------------------------------------


class TestEpisodeDataclass:
    """Test Episode dataclass methods."""

    def test_str_representation(self):
        ep = Episode(
            index=0,
            start_time=10.0,
            end_time=20.0,
            duration=10.0,
            topics=["/imu/data", "/camera"],
            message_counts={"/imu/data": 100, "/camera": 30},
        )
        s = str(ep)
        assert "Episode 0" in s
        assert "10.000s" in s
        assert "20.000s" in s

    def test_to_dict(self):
        ep = Episode(
            index=1,
            start_time=5.0,
            end_time=15.0,
            duration=10.0,
            topics=["/test"],
            message_counts={"/test": 50},
            success=True,
        )
        d = ep.to_dict()
        assert d["index"] == 1
        assert d["start_time"] == 5.0
        assert d["duration"] == 10.0
        assert d["success"] is True

    def test_empty_episode(self):
        ep = Episode(
            index=0,
            start_time=0.0,
            end_time=0.0,
            duration=0.0,
        )
        assert ep.duration == 0.0
        assert ep.topics == []
        assert ep.message_counts == {}
        assert ep.success is None
