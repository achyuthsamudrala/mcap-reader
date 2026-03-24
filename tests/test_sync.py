"""Tests for time synchronization (TimeSynchronizer).

Uses the generated multi-topic MCAP files and also tests the internal
_TopicTimeline helpers directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcap_reader.reader import McapReader, RawMessage
from mcap_reader.sync import (
    SyncConfig,
    SyncQuality,
    SyncResult,
    TimeSynchronizer,
    _TopicTimeline,
)


# ---------------------------------------------------------------------------
# _TopicTimeline unit tests (no MCAP needed)
# ---------------------------------------------------------------------------


class TestTopicTimeline:
    """Test the internal _TopicTimeline helper class."""

    def _make_timeline(self, timestamps: list[float]) -> _TopicTimeline:
        """Build a timeline with fake messages at given timestamps."""
        tl = _TopicTimeline()
        for t in timestamps:
            msg = RawMessage(
                topic="/test",
                timestamp=t,
                log_time=t,
                data=b"",
                ros_msg=None,
                schema_name="test",
            )
            tl.timestamps.append(t)
            tl.messages.append(msg)
        return tl

    def test_find_nearest_exact(self):
        tl = self._make_timeline([1.0, 2.0, 3.0])
        result = tl.find_nearest(2.0)
        assert result is not None
        msg, delay = result
        assert msg.timestamp == 2.0
        assert abs(delay) < 1e-9

    def test_find_nearest_between(self):
        tl = self._make_timeline([1.0, 3.0])
        result = tl.find_nearest(2.2)
        assert result is not None
        msg, delay = result
        # Closer to 3.0 (distance 0.8) than 1.0 (distance 1.2)
        assert msg.timestamp == 3.0

    def test_find_nearest_before_all(self):
        tl = self._make_timeline([5.0, 6.0])
        result = tl.find_nearest(1.0)
        assert result is not None
        msg, _ = result
        assert msg.timestamp == 5.0

    def test_find_nearest_after_all(self):
        tl = self._make_timeline([1.0, 2.0])
        result = tl.find_nearest(10.0)
        assert result is not None
        msg, _ = result
        assert msg.timestamp == 2.0

    def test_find_nearest_empty(self):
        tl = _TopicTimeline()
        assert tl.find_nearest(1.0) is None

    def test_find_bracket_success(self):
        tl = self._make_timeline([1.0, 2.0, 3.0])
        result, reason = tl.find_bracket(1.5)
        assert result is not None
        msg_before, msg_after, alpha = result
        assert msg_before.timestamp == 1.0
        assert msg_after.timestamp == 2.0
        assert abs(alpha - 0.5) < 1e-9

    def test_find_bracket_before_first(self):
        tl = self._make_timeline([1.0, 2.0])
        result, reason = tl.find_bracket(0.5)
        assert result is None
        assert "before first" in reason

    def test_find_bracket_after_last(self):
        tl = self._make_timeline([1.0, 2.0])
        result, reason = tl.find_bracket(3.0)
        assert result is None
        assert "after last" in reason

    def test_find_bracket_too_few_messages(self):
        tl = self._make_timeline([1.0])
        result, reason = tl.find_bracket(1.0)
        assert result is None
        assert "fewer than 2" in reason


# ---------------------------------------------------------------------------
# SyncConfig
# ---------------------------------------------------------------------------


class TestSyncConfig:
    """Test SyncConfig methods."""

    def test_get_max_delay_global(self):
        config = SyncConfig(
            reference_topic="/imu",
            topics=["/camera"],
            max_delay=0.05,
        )
        assert config.get_max_delay("/camera") == 0.05

    def test_get_max_delay_per_topic(self):
        config = SyncConfig(
            reference_topic="/imu",
            topics=["/camera", "/joints"],
            max_delay=0.05,
            per_topic_max_delay={"/camera": 0.02},
        )
        assert config.get_max_delay("/camera") == 0.02
        assert config.get_max_delay("/joints") == 0.05


# ---------------------------------------------------------------------------
# TimeSynchronizer integration tests (requires MCAP files)
# ---------------------------------------------------------------------------


class TestTimeSynchronizerNearest:
    """Test nearest-neighbor synchronization with generated MCAP files."""

    def test_basic_sync(self, multi_topic_mcap: Path):
        with McapReader(multi_topic_mcap) as reader:
            config = SyncConfig(
                reference_topic="/imu/data",
                topics=["/joint_states", "/tf"],
                strategy="nearest",
                max_delay=0.1,
            )
            synchronizer = TimeSynchronizer(reader, config)
            results = list(synchronizer.iter_synchronized())

            # Should have one result per IMU message
            assert len(results) > 0

            # Each result should have messages for secondary topics
            for r in results:
                assert isinstance(r, SyncResult)
                assert "/joint_states" in r.messages
                assert "/tf" in r.messages

    def test_max_delay_filtering(self, multi_topic_mcap: Path):
        """With very tight max_delay, many messages should be dropped."""
        with McapReader(multi_topic_mcap) as reader:
            config = SyncConfig(
                reference_topic="/imu/data",
                topics=["/joint_states"],
                strategy="nearest",
                max_delay=0.0001,  # very tight: 0.1 ms
            )
            synchronizer = TimeSynchronizer(reader, config)
            results = list(synchronizer.iter_synchronized())

            # Many joint_states should be dropped
            dropped = sum(1 for r in results if r.messages["/joint_states"] is None)
            # With drift and rate difference, most should be dropped
            assert dropped > 0

    def test_sync_quality_metrics(self, multi_topic_mcap: Path):
        with McapReader(multi_topic_mcap) as reader:
            config = SyncConfig(
                reference_topic="/imu/data",
                topics=["/joint_states"],
                strategy="nearest",
                max_delay=0.1,
            )
            synchronizer = TimeSynchronizer(reader, config)
            # Must consume the iterator before getting quality
            _ = list(synchronizer.iter_synchronized())
            quality = synchronizer.get_quality()

            assert isinstance(quality, SyncQuality)
            assert quality.total_synced > 0
            assert "/joint_states" in quality.mean_delay
            assert "/joint_states" in quality.max_delay
            assert "/joint_states" in quality.dropped_count

    def test_invalid_strategy_raises(self, multi_topic_mcap: Path):
        with McapReader(multi_topic_mcap) as reader:
            config = SyncConfig(
                reference_topic="/imu/data",
                topics=["/joint_states"],
                strategy="invalid_strategy",
            )
            with pytest.raises(ValueError, match="Unknown sync strategy"):
                TimeSynchronizer(reader, config)


class TestTimeSynchronizerInterpolate:
    """Test interpolation synchronization."""

    def test_interpolation_sync(self, multi_topic_mcap: Path):
        with McapReader(multi_topic_mcap) as reader:
            config = SyncConfig(
                reference_topic="/imu/data",
                topics=["/joint_states"],
                strategy="interpolate",
                max_delay=0.1,
            )
            synchronizer = TimeSynchronizer(reader, config)
            results = list(synchronizer.iter_synchronized())

            assert len(results) > 0

            # Check that interpolation alphas are present for matched messages
            for r in results:
                if r.messages["/joint_states"] is not None:
                    alpha = r.interpolation_alphas["/joint_states"]
                    assert alpha is not None
                    assert 0.0 <= alpha <= 1.0


class TestTimeSynchronizerDispatch:
    """Test the iter_synchronized dispatch method."""

    def test_dispatch_nearest(self, multi_topic_mcap: Path):
        with McapReader(multi_topic_mcap) as reader:
            config = SyncConfig(
                reference_topic="/imu/data",
                topics=["/joint_states"],
                strategy="nearest",
                max_delay=0.1,
            )
            synchronizer = TimeSynchronizer(reader, config)
            results = list(synchronizer.iter_synchronized())
            assert len(results) > 0

    def test_dispatch_interpolate(self, multi_topic_mcap: Path):
        with McapReader(multi_topic_mcap) as reader:
            config = SyncConfig(
                reference_topic="/imu/data",
                topics=["/joint_states"],
                strategy="interpolate",
                max_delay=0.1,
            )
            synchronizer = TimeSynchronizer(reader, config)
            results = list(synchronizer.iter_synchronized())
            assert len(results) > 0
