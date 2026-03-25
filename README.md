# mcap-reader

A Python library and CLI for reading MCAP files (ROS 2 recording format). Reads robot sensor data and exposes it as structured Python objects, DataFrames, and numpy arrays with synchronized timestamps.

Built as a learning project for robotics data formats. See [docs/FUNDAMENTALS.md](docs/FUNDAMENTALS.md) for a standalone guide to the underlying concepts.

## Install

```bash
git clone <repo-url>
cd mcap-reader
pip install -e ".[dev]"
```

## 30-Second Overview

```python
from mcap_reader import McapReader

with McapReader("recording.mcap") as reader:
    # What's in the file? (instant — reads index only)
    print(f"{reader.duration:.0f}s, {reader.message_count:,} messages")
    for t in reader.topics:
        print(f"  {t.name}: {t.message_type} ({t.message_count} msgs)")

    # Iterate messages (lazy, constant memory)
    for msg in reader.iter_messages(topics=["/imu/data"], end_time=reader.start_time + 10):
        print(msg.timestamp, msg.ros_msg.linear_acceleration)
```

## CLI

All commands: `mcap-reader <command> <file> [options]`

```bash
# What's in this file?
mcap-reader summary recording.mcap

# Look at some messages
mcap-reader inspect recording.mcap --topic /imu/data --limit 5

# Check for corruption
mcap-reader validate recording.mcap

# Export a topic to Parquet
mcap-reader export recording.mcap --topic /imu/data -o imu.parquet

# Export a specific episode
mcap-reader export recording.mcap --topic /joint_states --episode 2 -o ep2.parquet

# Synchronize multiple topics to one timeline
mcap-reader sync recording.mcap \
    --reference /camera/image_raw \
    --topics /imu/data /joint_states \
    --max-delay 0.05 \
    -o synced.parquet

# Show the coordinate frame tree
mcap-reader frames recording.mcap

# Find episode boundaries
mcap-reader episodes recording.mcap --gap-threshold 5.0
```

## Common Tasks

### Synchronize sensors for training data

The core use case: align multi-rate sensor streams to produce (observation, action) pairs.

```python
from mcap_reader import McapReader
from mcap_reader.sync import SyncConfig, TimeSynchronizer

config = SyncConfig(
    reference_topic="/imu/data",           # align everything to IMU timestamps
    topics=["/camera/image_raw", "/joint_states"],
    strategy="nearest",                     # or "interpolate"
    max_delay=0.05,                         # 50ms global threshold
    per_topic_max_delay={
        "/camera/image_raw": 0.020,         # tighter for slow sensors
    },
)

with McapReader("recording.mcap") as reader:
    sync = TimeSynchronizer(reader, config)
    for result in sync.iter_synchronized():
        camera_msg = result.messages["/camera/image_raw"]  # RawMessage or None
        joint_msg = result.messages["/joint_states"]
        delay = result.sync_delays["/camera/image_raw"]    # float or None

    # After iteration: quality report
    q = sync.get_quality()
    print(f"Mean camera delay: {q.mean_delay['/camera/image_raw']*1000:.1f}ms")
    print(f"Dropped: {q.dropped_count}")
```

### Look up transforms between coordinate frames

```python
from mcap_reader import McapReader, TransformBuffer

buf = TransformBuffer()

with McapReader("recording.mcap") as reader:
    for msg in reader.iter_messages(topics=["/tf", "/tf_static"]):
        is_static = msg.topic == "/tf_static"
        buf.add_tf_message(msg.ros_msg, is_static=is_static)

# Query any transform at any time (BFS path finding + SLERP interpolation)
tf = buf.lookup_transform("base_link", "camera_optical_frame", timestamp=5.3)
print(tf.translation, tf.rotation)

# See the frame tree
print(buf.get_frame_graph().to_ascii_tree())
```

### Detect episodes in a recording

```python
from mcap_reader import McapReader
from mcap_reader.episode import EpisodeDetector

with McapReader("recording.mcap") as reader:
    detector = EpisodeDetector(reader)

    # Gap-based: pauses > 5s between messages = new episode
    episodes = detector.detect_by_gaps(gap_threshold=5.0)

    # Marker-based: looks for /episode_start, /episode_end topics
    episodes = detector.detect_by_markers()

    for ep in episodes:
        print(f"Episode {ep.index}: {ep.duration:.1f}s, "
              f"{sum(ep.message_counts.values())} msgs, "
              f"success={ep.success}")
```

### Project 3D points to camera pixels

```python
from mcap_reader.calibration import CameraModel

cam = CameraModel.from_camera_info(camera_info_msg)

pixels = cam.project(points_3d)              # (N,3) -> (N,2)
point_3d = cam.unproject(u=320, v=240, depth=1.5)  # pixel + depth -> 3D
undistorted = cam.undistort(raw_image)        # remove lens distortion
```

## Key Design Decisions

**`timestamp` vs `log_time`.** Every message has both. `timestamp` prefers `header.stamp` (when the sensor captured the data). `log_time` is when the recorder wrote it to disk. For cross-sensor alignment, always use `timestamp`. For seeking within the file, use `log_time`.

**Iterator pattern.** Recordings can be tens of gigabytes with millions of messages. `iter_messages()` yields one at a time with constant memory. The MCAP library skips entire compressed chunks that fall outside your topic/time filters.

**Per-topic sync thresholds.** IMU at 1000Hz has samples 1ms apart — a 2ms delay is suspicious. Camera at 30Hz has samples 33ms apart — a 16ms delay is normal. A single global threshold can't serve both.

**Frozen dataclasses.** `Vector3`, `Quaternion`, `Transform`, `RawMessage`, `SyncResult`, `Episode` are all immutable. Prevents accidental mutation of shared data.

## Architecture

```
mcap_reader/
    reader.py              # McapReader — open file, iterate messages
    deserializer.py        # CDR byte-level deserialization
    sync.py                # TimeSynchronizer — align multi-rate streams
    calibration.py         # CameraModel — pinhole + distortion
    episode.py             # EpisodeDetector — find episode boundaries
    cli.py                 # 7 CLI subcommands
    messages/              # Typed wrappers for ROS 2 sensor messages
        image.py           # sensor_msgs/Image → numpy HxWxC
        compressed_image.py # sensor_msgs/CompressedImage → numpy
        camera_info.py     # sensor_msgs/CameraInfo → K, D, R, P
        imu.py             # sensor_msgs/Imu → quaternion + vectors
        joint_state.py     # sensor_msgs/JointState → parallel arrays
        pointcloud.py      # sensor_msgs/PointCloud2 → structured numpy
        transform.py       # tf2_msgs/TFMessage → Transform objects
    transforms/            # Coordinate frame system
        math.py            # Quaternion, Transform, SLERP (from scratch)
        frames.py          # FrameGraph — BFS path finding
        buffer.py          # TransformBuffer — interpolating lookups
```

Dependencies flow downward: `deserializer` → `messages` → `reader` → `sync`/`episode`/`calibration` → `cli`. The `transforms/` module is independent (pure math + graph algorithms).

## Learning Path

This project is structured in six phases, each building on the previous. Work through them in order using the source files as reference implementations.

For the underlying concepts (CDR encoding, MCAP format, quaternions, camera models, clock drift), see [docs/FUNDAMENTALS.md](docs/FUNDAMENTALS.md).

| Phase | What to Build | Source Files | Key Concepts |
|-------|--------------|--------------|--------------|
| 1. Binary Deserialization | CDR decoder + MCAP reader | `deserializer.py`, `reader.py` | CDR alignment padding, encapsulation header, MCAP chunks/index, iterator pattern |
| 2. Message Types | Typed wrappers for sensor data | `messages/*.py` | numpy structured dtypes, image `step` vs `width*channels`, variable-length CDR arrays, covariance sentinel (-1) |
| 3. Camera Pipeline | CompressedImage + CameraModel | `compressed_image.py`, `camera_info.py`, `calibration.py` | Pinhole model (K matrix), distortion (D), projection/unprojection, topic auto-pairing |
| 4. Transform Tree | Quaternion math + frame graph + buffer | `transforms/*.py` | SO(3), Hamilton product, SLERP, BFS path finding, static vs dynamic transforms |
| 5. Time Sync | Multi-rate stream alignment | `sync.py` | Clock drift (20ppm), `header.stamp` vs `log_time`, nearest-neighbor vs interpolation, quality metrics |
| 6. Episodes + CLI | Episode detection + CLI integration | `episode.py`, `cli.py` | Gap/marker detection, Parquet schema design, dataset conventions |

### Exercises

Each phase has hands-on exercises to verify understanding:

**Phase 1:** Hex-dump the first `/imu/data` message and manually identify the CDR encapsulation header, alignment bytes, and float64 fields. Plot `header.stamp - log_time` across a topic to visualize pipeline latency.

**Phase 2:** Decode a `PointCloud2` from raw CDR bytes using the `fields` descriptor. Write an image converter that correctly handles `step` padding for non-standard widths.

**Phase 3:** Extract `CameraInfo`, compute the field of view, and verify that `project()` → `unproject()` round-trips a 3D point.

**Phase 4:** Compose 1000 random quaternions and measure norm drift. Build a frame tree from `/tf` + `/tf_static` and look up a transform between non-adjacent frames.

**Phase 5:** Synchronize IMU + camera, plot sync delays over time, and compare nearest-neighbor vs interpolation for joint state data.

**Phase 6:** Run gap-based episode detection on a multi-episode recording and verify boundaries against Foxglove Studio.

## Testing

```bash
pytest                    # 213 tests (181 functional + 32 performance)
pytest -k "transform"     # just transform math
pytest tests/test_performance.py  # performance regression tests
python -m tests.benchmarks.bench_all  # detailed timing report
```

Tests use synthetic MCAP generators (`tests/generators/`) that produce deterministic, mathematically verifiable test data — no real robot recordings required. Quaternion math is property-tested with [Hypothesis](https://hypothesis.readthedocs.io/) and cross-validated against SciPy.

## Out of Scope

| Excluded | Why |
|----------|-----|
| Real-time streaming | Offline analysis tool. Real-time needs async I/O and ring buffers. |
| ROS 1 bag support | Different container + serialization format, doubles code without new concepts. |
| Writing MCAP files | Read-only. Writing is simpler and less instructive. |
| GPU acceleration | Would obscure algorithms behind CUDA dependencies. |
| H.264/video codec decoding | CompressedImage (JPEG/PNG) covers the common case. See FUNDAMENTALS.md for why. |
| Multi-file datasets | One file at a time. Dataset-level management is a separate tool. |
| Sensor fusion / SLAM | That's a separate project built on top of this one. |
