# mcap-reader

A from-scratch Python library and CLI for reading, validating, and exporting [MCAP](https://mcap.dev/) files containing ROS 2 sensor data -- built as a learning project for robotics data formats.

## What This Project Is

This is an educational project designed to teach you how robot data actually works at every level: from individual bytes on disk to synchronized multi-sensor datasets ready for machine learning.

**What is MCAP?** MCAP is a modular, self-contained container format for heterogeneous timestamped data. It was created by [Foxglove](https://foxglove.dev/) as the successor to ROS 1 bag files and is the default recording format for ROS 2 (Iron+). Think of it as "a zip file for sensor data" -- it packages multiple streams of serialized messages (images, IMU readings, joint states, point clouds, transforms) into a single file with an index for fast random access.

An MCAP file is organized as:

```
Magic | Header | [ Data Section ] | [ Summary Section ] | Footer | Magic
```

- **Data Section** -- Chunks of compressed messages (lz4/zstd). Each chunk can be decompressed independently, so you only decompress what you need.
- **Summary Section** -- Written after recording finishes. Contains statistics, schemas, channel records, and chunk indices. This is what lets you query "what topics are in this file?" without scanning the entire data section.

**What you'll learn by working through this project:**

- Binary deserialization (CDR encoding, alignment padding, endianness)
- ROS 2 message type systems (schemas, channels, topics)
- Computer vision fundamentals (pinhole camera model, lens distortion)
- 3D rotation math (quaternions, SO(3), SLERP interpolation)
- Scene graph traversal (coordinate frame trees, BFS path finding)
- Time synchronization across multi-rate sensors (clock drift, nearest-neighbor matching)
- Robot learning dataset conventions (episodes, Parquet export)

## Installation

```bash
git clone <repo-url>
cd mcap-reader
pip install -e ".[dev]"
```

This installs the library in editable mode with all development dependencies (pytest, hypothesis, ruff, scipy).

The `mcap-reader` CLI is automatically available after installation:

```bash
mcap-reader --version
mcap-reader --help
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `mcap` | Low-level MCAP file reading |
| `mcap-ros2-support` | ROS 2 CDR decoding via DecoderFactory |
| `numpy` | Array operations, structured dtypes |
| `pandas` | DataFrame export |
| `pyarrow` | Parquet file I/O |
| `click` | CLI argument parsing |
| `opencv-python` | Camera calibration, undistortion |

## Quick Start

### Open a file and inspect topics

```python
from mcap_reader import McapReader

with McapReader("recording.mcap") as reader:
    print(f"Duration: {reader.duration:.1f}s")
    print(f"Messages: {reader.message_count:,}")

    for topic in reader.topics:
        print(f"  {topic.name}: {topic.message_type} "
              f"({topic.message_count} msgs, {topic.frequency} Hz)")
```

### Iterate messages

```python
with McapReader("recording.mcap") as reader:
    for msg in reader.iter_messages(topics=["/imu/data"]):
        imu = msg.ros_msg
        print(f"t={msg.timestamp:.6f}  "
              f"accel=({imu.linear_acceleration.x:.2f}, "
              f"{imu.linear_acceleration.y:.2f}, "
              f"{imu.linear_acceleration.z:.2f})")
```

Messages are yielded lazily in `log_time` order. You can filter by topic and time range, and the MCAP library skips entire chunks that fall outside your filters:

```python
# Read only camera images from the first 10 seconds
for msg in reader.iter_messages(
    topics=["/camera/image_raw"],
    start_time=reader.start_time,
    end_time=reader.start_time + 10.0,
):
    print(msg.ros_msg.width, msg.ros_msg.height, msg.ros_msg.encoding)
```

### Get the raw schema definition

```python
with McapReader("recording.mcap") as reader:
    schema_text = reader.get_schema("/imu/data")
    print(schema_text)  # Prints the .msg file contents
```

## CLI Reference

All commands follow the pattern `mcap-reader <command> <file.mcap> [options]`.

### `summary` -- File overview

```bash
mcap-reader summary recording.mcap
```

Prints duration, message count, time range, and a table of all topics with types, counts, and estimated frequencies. Reads only the summary section, so it is O(1) regardless of file size.

### `inspect` -- Peek at messages

```bash
mcap-reader inspect recording.mcap --topic /imu/data --limit 3
```

Decodes and pretty-prints the first N messages from a topic. Shows timestamps, frame IDs, and type-specific fields (orientation, angular velocity, joint positions, image dimensions, point cloud fields, TF parent/child relationships).

### `validate` -- Check for corruption

```bash
mcap-reader validate recording.mcap
```

Decodes every message in the file and reports per-topic success/failure counts with a progress bar. Exits with code 1 if any decoding errors are found. Useful for detecting truncated recordings, schema mismatches, or corrupted chunks.

### `export` -- Single-topic Parquet/CSV export

```bash
mcap-reader export recording.mcap --topic /imu/data -o imu.parquet
mcap-reader export recording.mcap --topic /joint_states -f csv -o joints.csv
mcap-reader export recording.mcap --topic /imu/data --episode 2 -o ep2_imu.parquet
```

Exports one topic to a structured file. The output schema is type-aware:

| Message Type | Columns |
|-------------|---------|
| `Imu` | `timestamp_ns`, `frame_id`, `orientation_x/y/z/w`, `angular_vel_x/y/z`, `linear_accel_x/y/z` |
| `JointState` | `timestamp_ns`, `frame_id`, `position_<name>`, `velocity_<name>`, `effort_<name>` per joint |
| `Image` | `timestamp_ns`, `frame_id`, `height`, `width`, `encoding`, `data` (zlib-compressed bytes) |
| `PointCloud2` | `timestamp_ns`, `frame_id`, `num_points`, `data` (raw bytes) |

The `--episode` flag filters to a single episode (detected via gap-based method).

### `sync` -- Multi-modal time synchronization

```bash
mcap-reader sync recording.mcap \
    --reference /camera/image_raw \
    --topics /imu/data /joint_states \
    --strategy nearest \
    --max-delay 0.05 \
    -o synced.parquet
```

Aligns messages from multiple topics to a reference topic's timeline. Outputs a Parquet/CSV file with one row per reference message, containing matched data from all target topics.

- `nearest` -- picks the closest message by timestamp (use for images, point clouds)
- `interpolate` -- linearly interpolates between bracketing messages (use for IMU, joint states)

### `frames` -- Coordinate frame tree

```bash
mcap-reader frames recording.mcap
```

Reads `/tf` and `/tf_static` topics, builds the coordinate frame graph, and displays it as an ASCII tree:

```
map
+-- odom
|   +-- base_link
|       +-- camera_link
|           +-- camera_optical_frame
```

### `episodes` -- Detect episode boundaries

```bash
mcap-reader episodes recording.mcap --gap-threshold 3.0
mcap-reader episodes recording.mcap --method marker
```

Detects episode boundaries using either gap-based detection (looks for temporal pauses between data collection runs) or marker-based detection (looks for explicit `/episode_start`, `/episode_end` topics). Prints time ranges, durations, and per-topic message counts for each detected episode.

## Library API

### McapReader

The central class. Opens an MCAP file, reads the summary section for O(1) metadata access, and provides lazy message iteration.

```python
from mcap_reader.reader import McapReader, RawMessage, TopicInfo

with McapReader("recording.mcap") as reader:
    # O(1) metadata (reads summary section only)
    reader.duration        # float, seconds
    reader.start_time      # float, seconds (wall-clock)
    reader.end_time        # float, seconds
    reader.message_count   # int, total across all topics
    reader.topic_names     # list[str], sorted
    reader.topics          # list[TopicInfo], with counts and frequencies

    # Lazy iteration with chunk-level filtering
    for msg in reader.iter_messages(topics=[...], start_time=..., end_time=...):
        msg.topic        # str
        msg.timestamp    # float -- prefers header.stamp, falls back to log_time
        msg.log_time     # float -- wall-clock recording time
        msg.ros_msg      # deserialized ROS 2 Python object
        msg.data         # bytes -- raw CDR payload
        msg.schema_name  # str -- e.g. "sensor_msgs/msg/Imu"

    # Find related topics by namespace
    reader.find_paired_topic("/camera/image_raw", "camera_info")
    # -> "/camera/camera_info"

    # Structured summary dict
    reader.summary()
```

**Design decision: iterator pattern.** Robot recordings routinely contain millions of messages totaling tens of gigabytes. The iterator pattern lets you filter at the chunk level (skipping compressed data you don't need), process messages with constant memory, and short-circuit early with `itertools.islice`.

**Design decision: `timestamp` vs `log_time`.** Every `RawMessage` exposes both. `timestamp` prefers `header.stamp` (the sensor capture time), falling back to `log_time` (the recorder's wall-clock). For cross-sensor alignment, you always want `header.stamp`. For seeking within the file, use `log_time`.

### Message Types

The `messages/` subpackage provides typed wrappers for common ROS 2 sensor messages, each with from-scratch CDR deserialization:

| Module | ROS 2 Type | Key Fields |
|--------|-----------|------------|
| `messages.image` | `sensor_msgs/msg/Image` | `height`, `width`, `encoding`, `step`, `data` |
| `messages.compressed_image` | `sensor_msgs/msg/CompressedImage` | `format`, `data` |
| `messages.camera_info` | `sensor_msgs/msg/CameraInfo` | `K` (3x3), `D`, `R` (3x3), `P` (3x4), `distortion_model` |
| `messages.imu` | `sensor_msgs/msg/Imu` | `orientation` (quaternion), `angular_velocity`, `linear_acceleration`, covariances |
| `messages.joint_state` | `sensor_msgs/msg/JointState` | `name[]`, `position[]`, `velocity[]`, `effort[]` |
| `messages.pointcloud` | `sensor_msgs/msg/PointCloud2` | `fields`, `point_step`, `data`, `is_dense` |
| `messages.transform` | `tf2_msgs/msg/TFMessage` | `transforms[]` with parent/child frame IDs and stamped transforms |

### Transform Tree

The `transforms/` subpackage implements a TF2-style coordinate frame system from scratch.

#### Quaternion math (`transforms.math`)

```python
from mcap_reader.transforms.math import Vector3, Quaternion, Transform, slerp

# Immutable value types
v = Vector3(1.0, 2.0, 3.0)
q = Quaternion(x=0.0, y=0.0, z=0.383, w=0.924)  # ~45 deg around Z

# Quaternion operations
q.norm()                    # should be ~1.0 for unit quaternions
q.normalize()               # re-normalize after repeated composition
q.conjugate()               # reverse rotation (== inverse for unit quaternions)
q.inverse()                 # safe for non-unit quaternions too
q.to_rotation_matrix()      # 3x3 numpy array
q.angular_distance(other)   # geodesic distance in radians

# Hamilton product composes rotations: R(q1) @ R(q2)
q_composed = q1 * q2

# Rigid-body transform (rotation + translation)
tf = Transform(translation=v, rotation=q)
tf.to_matrix()              # 4x4 homogeneous matrix
tf.inverse()                # T^{-1}
tf.apply(point)             # R @ p + t

# Composition follows TF2 convention: T_AC = T_AB * T_BC
tf_ac = tf_ab * tf_bc

# SLERP: constant-velocity rotation interpolation
q_mid = slerp(q1, q2, t=0.5)
```

**Why quaternions over rotation matrices?** Quaternions are more compact (4 floats vs 9), numerically stable under repeated composition (just re-normalize), and free of gimbal lock. The set of unit quaternions double-covers SO(3): both `q` and `-q` represent the same rotation.

**Why SLERP over LERP?** Linearly interpolating quaternions and re-normalizing (NLERP) does not produce constant angular velocity -- the rotation speeds up in the middle. SLERP traverses the great-circle arc on the unit 4-sphere at constant speed, which is essential for physically meaningful motion interpolation.

#### TransformBuffer (`transforms.buffer`)

```python
from mcap_reader.transforms.buffer import TransformBuffer

buf = TransformBuffer()

# Add transforms (typically from /tf and /tf_static messages)
buf.add_transform("odom", "base_link", tf, timestamp=1.0)
buf.add_transform("odom", "base_link", tf, timestamp=2.0)
buf.add_transform("base_link", "camera_link", tf_cam, timestamp=0.0, is_static=True)

# Or unpack TFMessage objects directly
buf.add_tf_message(tf_msg, is_static=False)

# Look up any transform at any time -- finds path via BFS, interpolates
tf_result = buf.lookup_transform("odom", "camera_link", timestamp=1.5)

# Non-throwing check
buf.can_transform("odom", "camera_link", timestamp=1.5)

# Inspect the frame graph
print(buf.get_frame_graph().to_ascii_tree())
buf.get_frames()  # set of all frame names
```

**Static vs dynamic transforms.** Static transforms (`/tf_static`) are published once and never change (e.g., `base_link -> camera_link` for a sensor bolted to the chassis). Dynamic transforms (`/tf`) change over time (e.g., `odom -> base_link` updated by wheel odometry at 50-100 Hz). The buffer stores them separately -- static transforms ignore the timestamp on lookup.

**Interpolation.** When you request a transform at `t=1.5s` but the nearest stored values are at `t=1.0s` and `t=2.0s`, the buffer interpolates: LERP for translation, SLERP for rotation. This uses binary search (O(log n)) over the time-sorted entry list.

**Graph search.** To look up `map -> camera_optical_frame`, the buffer finds the shortest path through the frame graph via BFS, then composes transforms along each edge (inverting edges traversed backwards).

### Time Synchronization

```python
from mcap_reader.sync import SyncConfig, TimeSynchronizer

config = SyncConfig(
    reference_topic="/imu/data",
    topics=["/camera/image_raw", "/joint_states"],
    strategy="nearest",          # or "interpolate"
    max_delay=0.05,              # 50ms global threshold
    per_topic_max_delay={
        "/camera/image_raw": 0.020,  # tighter for camera
    },
)

with McapReader("recording.mcap") as reader:
    synchronizer = TimeSynchronizer(reader, config)

    for result in synchronizer.iter_synchronized():
        result.reference_timestamp     # float
        result.messages["/camera/image_raw"]  # RawMessage or None
        result.sync_delays["/camera/image_raw"]  # float or None (signed)
        result.interpolation_alphas["/camera/image_raw"]  # float or None

    # Quality metrics after iteration
    quality = synchronizer.get_quality()
    quality.mean_delay    # dict[str, float] -- per topic
    quality.max_delay     # dict[str, float]
    quality.dropped_count # dict[str, int]
    quality.total_synced  # int

    # Export to pandas DataFrame
    df = synchronizer.to_pandas()
```

**Why per-topic `max_delay`?** An IMU at 1000 Hz has samples 1ms apart, so a sync delay > 2ms is suspicious. A camera at 30 Hz has samples 33ms apart, so 16ms delay is perfectly normal. A single global threshold cannot serve both.

### Camera Calibration

```python
from mcap_reader.calibration import CameraModel

camera = CameraModel.from_camera_info(camera_info_msg)

# Properties
camera.intrinsic_matrix       # 3x3 K matrix
camera.distortion_coefficients  # D vector
camera.projection_matrix      # 3x4 P matrix
camera.image_size             # (width, height)
camera.fov                    # (horizontal, vertical) in radians

# 3D -> 2D projection (applies distortion)
pixels = camera.project(points_3d)  # (N, 3) -> (N, 2)

# 2D -> 3D unprojection (requires known depth)
point_3d = camera.unproject(u=320, v=240, depth=1.5)  # -> (3,) array

# Undistort a single image
clean = camera.undistort(raw_image)

# Precompute maps for batch processing (much faster)
map1, map2 = camera.get_undistort_maps()
for frame in frames:
    clean = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

# Stereo rectification
left_model, right_model = CameraModel.rectify_stereo(left_info, right_info)
```

### Episode Detection

```python
from mcap_reader.episode import EpisodeDetector

with McapReader("recording.mcap") as reader:
    detector = EpisodeDetector(reader)

    # Gap-based: finds pauses in sensor streams
    episodes = detector.detect_by_gaps(
        gap_threshold=5.0,
        per_topic_threshold={"/imu/data": 0.5, "/camera/image_raw": 2.0},
    )

    # Marker-based: looks for /episode_start, /episode_end, etc.
    episodes = detector.detect_by_markers()

    # Manual: user-specified time boundaries
    episodes = detector.detect_manual(boundaries=[(0, 10), (15, 25)])

    # Dispatcher
    episodes = detector.detect(method="gap", gap_threshold=3.0)

    for ep in episodes:
        print(f"Episode {ep.index}: {ep.start_time:.3f}s - {ep.end_time:.3f}s "
              f"({ep.duration:.3f}s)")
        print(f"  Topics: {ep.topics}")
        print(f"  Message counts: {ep.message_counts}")
        print(f"  Success: {ep.success}")
```

## Learning Path

This project is structured in six phases. Each phase builds on the previous one, introducing new concepts and increasing complexity. You can use the corresponding source files as both reference implementations and starting points for your own exploration.

---

### Phase 1 -- Binary Deserialization

**What to build:** A CDR byte-level deserializer and an MCAP file reader.

**Source files:** `deserializer.py`, `reader.py`

**What you'll learn:**
- How bytes become structured data
- The MCAP container format (magic bytes, chunks, summary section)
- CDR (Common Data Representation) encoding from the DDS middleware
- Alignment padding rules and why they exist

**Key concepts:**

*CDR Alignment.* Every primitive value must be aligned to a byte boundary equal to its size: `uint8` at any offset, `uint16` at even offsets, `float64` at 8-byte boundaries. This is measured from byte 4 (after the 4-byte encapsulation header). The encapsulation header itself tells you the byte order:

```
Bytes 0-1: 0x00 0x01 = Big-Endian, 0x00 0x02 = Little-Endian (typical on x86/ARM)
Bytes 2-3: Options (unused, 0x00)
```

Why alignment? Modern CPUs access memory most efficiently at natural boundaries. Unaligned 4-byte reads on ARM can fault or incur a multi-cycle penalty. CDR's rules guarantee zero-copy-friendly decoding on every platform.

*Variable-length strings.* CDR encodes strings as a 4-byte `uint32` length (including the null terminator), followed by the UTF-8 bytes and one `\0`. After the string, you must re-align for the next field.

*The iterator pattern.* Robot recordings contain millions of messages spanning gigabytes. `McapReader.iter_messages()` yields one message at a time with constant memory, and the MCAP library applies topic/time filters at the chunk level -- skipping entire compressed blocks that fall outside your query.

**Suggested exercises:**
1. Write a hex dump of the first message on `/imu/data` and manually identify the encapsulation header, alignment padding bytes, and each float64 field.
2. Compare `msg.timestamp` (header.stamp) to `msg.log_time` across a topic. Plot the difference -- this is the recording pipeline latency.
3. Add an `iter_raw_messages()` method that yields only the raw CDR bytes without decoding, and measure the throughput difference.

**External resources:**
- [MCAP specification](https://mcap.dev/spec)
- [OMG CDR specification (formal)](https://www.omg.org/spec/CORBA/3.3/Interoperability/PDF)
- [Foxglove blog: Inside MCAP](https://foxglove.dev/blog/mcap)

---

### Phase 2 -- Message Type Deserializers

**What to build:** Typed wrappers for `Image`, `Imu`, `JointState`, `PointCloud2`, and `TFMessage`.

**Source files:** `messages/image.py`, `messages/imu.py`, `messages/joint_state.py`, `messages/pointcloud.py`, `messages/transform.py`

**What you'll learn:**
- How ROS 2 message schemas map to byte layouts
- numpy structured dtypes for efficient bulk decoding
- Variable-length arrays in CDR (prefixed with a `uint32` count)
- Image encoding conventions (`rgb8`, `bgr8`, `mono8`, `16UC1`, `32FC1`)

**Key concepts:**

*Fixed-size vs variable-size messages.* An `Imu` message has a fixed layout: always the same number of floats in the same order. A `JointState` message has variable-length arrays (`name[]`, `position[]`, `velocity[]`, `effort[]`) whose lengths are serialized as `uint32` prefixes. Image `data` is a variable-length byte array whose size depends on `height * step`.

*The `step` field in Image messages.* You might assume `step == width * bytes_per_pixel`, but camera drivers often pad each row to alignment boundaries (64 or 128 bytes) for SIMD/DMA efficiency. Always use `step` as the row stride when reshaping, then slice off padding columns.

*Covariance sentinel.* IMU covariance arrays use `-1` in the `[0,0]` position to signal "this measurement is not provided." An accelerometer-only IMU sets orientation covariance[0] to -1.

**Suggested exercises:**
1. Decode a `PointCloud2` message from raw CDR bytes manually, using the `fields` descriptor to interpret each point's bytes.
2. Write a function that converts `sensor_msgs/msg/Image` with encoding `bgr8` to a numpy RGB array, being careful about `step` vs `width * 3`.
3. Count how many padding bytes appear in a typical `Imu` CDR payload. What percentage of the payload is padding?

**External resources:**
- [ROS 2 common_interfaces (message definitions)](https://github.com/ros2/common_interfaces)
- [REP-118: Image encoding conventions](https://www.ros.org/reps/rep-0118.html)

---

### Phase 3 -- CompressedImage and CameraInfo

**What to build:** CompressedImage decoder and the `CameraModel` calibration wrapper.

**Source files:** `messages/compressed_image.py`, `messages/camera_info.py`, `calibration.py`

**What you'll learn:**
- How JPEG/PNG data lives inside ROS messages (it is just a byte blob with a `format` string)
- The pinhole camera model and intrinsic matrix K
- Lens distortion models (plumb_bob, rational_polynomial)
- The full projection pipeline: world -> camera -> normalized -> distorted -> pixel

**Key concepts:**

*Intrinsic matrix K.* Maps 3D points in the camera frame to ideal (undistorted) pixel coordinates:

```
K = | fx  0  cx |      u = fx * (X/Z) + cx
    |  0  fy cy |      v = fy * (Y/Z) + cy
    |  0   0  1 |
```

`fx`, `fy` are focal lengths in pixels. `cx`, `cy` is the principal point (where the optical axis hits the image plane -- ideally the center, but manufacturing tolerances cause a few pixels of offset).

*Distortion coefficients D.* Real lenses bend light non-uniformly. The `plumb_bob` model uses 5 coefficients `[k1, k2, p1, p2, k3]` for radial and tangential distortion. Wide-angle lenses (120-180+ degrees) need the `rational_polynomial` model (8 coefficients) because the polynomial diverges at large radii.

*Why undistortion matters.* Raw wide-angle images have barrel distortion -- straight lines appear curved. This breaks pinhole-assumption algorithms (feature matching, visual odometry, ArUco detection, neural networks trained on undistorted data). Undistortion warps each pixel back to its ideal position.

*The D, K, R, P matrices in CameraInfo.* `K` is intrinsics. `D` is distortion. `R` is the rectification rotation (identity for monocular, aligns epipolar lines for stereo). `P` is the combined projection matrix: for monocular, `P[:3,:3] == K` and `P[:,3] == 0`; for stereo, `P[0,3]` encodes `-fx' * baseline`.

**Suggested exercises:**
1. Extract `CameraInfo` from a recording, print K, and compute the horizontal/vertical field of view using `fov = 2 * atan(dimension / (2 * focal_length))`.
2. Undistort an image and overlay a grid of straight lines. Compare the distorted and undistorted results.
3. Project a known 3D point to pixel coordinates using `camera.project()`, then unproject it back with `camera.unproject()` and verify you recover the original point.

**External resources:**
- [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Multiple View Geometry (Hartley & Zisserman)](http://www.robots.ox.ac.uk/~vgg/hzbook/) -- the definitive reference
- [REP-104: CameraInfo](https://www.ros.org/reps/rep-0104.html)

---

### Phase 4 -- Transform Tree

**What to build:** Quaternion math, rigid-body transforms, a frame graph, and an interpolating transform buffer.

**Source files:** `transforms/math.py`, `transforms/frames.py`, `transforms/buffer.py`

**What you'll learn:**
- The rotation group SO(3) and its double cover by unit quaternions
- Hamilton product for composing rotations
- Scene graphs and kinematic chains in robotics
- BFS shortest-path for frame lookups
- Interpolation between transforms (LERP for translation, SLERP for rotation)
- Static vs dynamic transforms in ROS 2

**Key concepts:**

*Quaternion conventions.* ROS stores quaternions as `(x, y, z, w)`. SciPy uses `(w, x, y, z)`. This project follows the ROS convention everywhere. Both `q` and `-q` represent the same rotation (the double-cover property).

*Composition order.* Following TF2: `T_AC = T_AB * T_BC` reads "to go from A to C, first go A->B, then B->C." Internally: `R_AC = R_AB @ R_BC` and `t_AC = R_AB @ t_BC + t_AB`.

*Why BFS?* The frame tree can have dozens of frames. To look up `map -> camera_optical_frame`, you need the shortest path (fewest transforms to compose = least numerical error). BFS finds the shortest path in O(V+E), which is effectively free for robot-scale graphs (~100 frames).

*Why interpolation?* Transforms are published at discrete rates (50-100 Hz). A camera image at `t=1.005s` needs `odom -> base_link`, but the published values are at `t=1.00s` and `t=1.01s`. Interpolation produces a smooth estimate at the exact requested time.

**Suggested exercises:**
1. Construct a quaternion from an axis-angle representation, convert to a rotation matrix, and verify it is orthogonal with determinant +1.
2. Compose 1000 random unit quaternions via Hamilton product. Measure how far the norm drifts from 1.0 without re-normalization, then verify that `normalize()` corrects it.
3. Build a frame tree from a recording's `/tf` and `/tf_static` topics, render it as ASCII art, and look up a transform between two non-adjacent frames.
4. Implement Shepperd's method for matrix-to-quaternion conversion from scratch and test it against `Quaternion.from_rotation_matrix()`.

**External resources:**
- [Quaternion kinematics for the error-state Kalman filter (Sola)](https://arxiv.org/abs/1711.02508) -- excellent tutorial on quaternion math
- [ROS TF2 design](https://wiki.ros.org/tf2/Design)
- [REP-105: Coordinate frame conventions](https://www.ros.org/reps/rep-0105.html)
- [Visualizing quaternions (3Blue1Brown)](https://www.youtube.com/watch?v=zjMuIxRvygQ)

---

### Phase 5 -- Time Synchronization

**What to build:** A multi-topic time synchronizer with nearest-neighbor and interpolation strategies.

**Source files:** `sync.py`

**What you'll learn:**
- The clock drift problem (20 ppm crystal oscillator drift ~ 2ms per 100s)
- The difference between `header.stamp` (sensor time) and `log_time` (recording time)
- Nearest-neighbor matching vs interpolation
- SLERP for orientation interpolation on the SO(3) manifold
- Per-topic delay thresholds for heterogeneous sensor rates
- Quality metrics for diagnosing timing problems

**Key concepts:**

*The clock problem.* A typical robot carries sensors sampling at different rates: IMU at 200-1000 Hz, cameras at 15-60 Hz, LiDAR at 10-20 Hz. Even with PTP clock synchronization, residual jitter from the software stack (kernel scheduling, DDS transport, rosbag2 writer thread) introduces timing noise. Downstream algorithms (sensor fusion, SLAM, imitation learning) need principled alignment.

*header.stamp vs log_time.* `log_time` is when the rosbag2 writer committed the message to disk -- it is monotonically increasing but includes variable latency. `header.stamp` is the sensor-side timestamp -- when the camera shutter fired, the IMU sampled, the LiDAR fired. For cross-sensor alignment, you want `header.stamp`.

*Nearest-neighbor vs interpolation.* Nearest-neighbor is lossless (you get an actual recorded message) but has a worst-case sync error of half a sensor period. Interpolation creates a "virtual sample" at the exact reference time, but assumes the signal varies smoothly between samples and the semantics are type-dependent (LERP for positions, SLERP for orientations, not meaningful for images).

*Quality diagnostics.* After synchronization, `get_quality()` returns per-topic metrics: `mean_delay` (near zero = good clock alignment), `max_delay` (large values = timing glitches), `dropped_count / total_synced` (>5% = driver issue or threshold too tight).

**Suggested exercises:**
1. Synchronize an IMU and camera topic with `strategy="nearest"`. Plot the sync delays over time -- do you see systematic drift or random jitter?
2. Compare nearest-neighbor and interpolation for joint state data. How different are the interpolated values from the nearest actual sample?
3. Set `per_topic_max_delay` to half the sensor period for each topic and measure the drop rate. What is the tightest threshold that keeps drops below 1%?

**External resources:**
- [IEEE 1588 PTP (Precision Time Protocol)](https://en.wikipedia.org/wiki/Precision_Time_Protocol)
- [ROS 2 time synchronization](https://docs.ros.org/en/rolling/Concepts/About-Time.html)

---

### Phase 6 -- Episode Detection and CLI

**What to build:** Episode boundary detector (gap-based and marker-based) and a Click CLI that ties everything together.

**Source files:** `episode.py`, `cli.py`

**What you'll learn:**
- Dataset structure in robot learning (episodes, trials, success labels)
- Gap-based vs marker-based episode detection
- Parquet schema design for heterogeneous sensor data
- Building a CLI that composes library components

**Key concepts:**

*What is an episode?* In imitation learning and RL, a recording is divided into episodes -- discrete chunks where the robot attempts a single task. Between episodes, there is a reset period (human repositioning objects, robot returning to home configuration, operator pressing "start"). Training data loaders must not cross episode boundaries.

*Gap-based detection.* Sensor streams are continuous within an episode (30 Hz camera = frame every ~33ms). Any gap much larger than the sensor period indicates a pause. Gap-based detection iterates messages in `log_time` order and starts a new episode whenever any topic's gap exceeds its threshold.

*Marker-based detection.* Some data collection pipelines publish explicit signals: `/episode_start`, `/episode_end`, `/episode_success`. The detector watches configurable topic names and falls back to common conventions from major research labs (DROID, RT-X, BridgeData).

*Parquet for robot data.* The export command writes to Apache Parquet, a columnar format that supports nested types, efficient compression, and predicate pushdown. Timestamps are stored as `int64` nanoseconds for lossless precision. Variable-length data (images, point clouds) is stored as binary columns.

**Suggested exercises:**
1. Run gap-based detection on a recording with multiple episodes and verify the boundaries visually (compare against timestamps in Foxglove Studio).
2. Export two episodes of IMU data to separate Parquet files and compare their statistics.
3. Add a new CLI command that exports all episodes as separate Parquet files in one pass.
4. Implement a `detect_by_velocity()` method that detects episode boundaries based on the robot being stationary (joint velocities near zero).

**External resources:**
- [DROID dataset](https://droid-dataset.github.io/)
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/)
- [Apache Parquet format](https://parquet.apache.org/documentation/latest/)

---

## Concepts Reference

Quick reference for the key robotics, computer vision, and math concepts used in this project.

### CDR Encoding

Common Data Representation -- the binary wire format used by ROS 2 (via DDS). Key rules: 4-byte encapsulation header declares endianness, all primitives aligned to their own size, variable-length types prefixed with `uint32` length. After any variable-length field, re-align before the next field.

### SO(3) and Quaternions

SO(3) is the group of 3D rotations (3x3 orthogonal matrices with determinant +1). Unit quaternions `q = (x, y, z, w)` with `||q|| = 1` form a double cover of SO(3): every rotation corresponds to exactly two quaternions (`q` and `-q`). The Hamilton product composes rotations. Quaternions avoid gimbal lock and are numerically stable under repeated composition.

### Pinhole Camera Model

The ideal projection of a 3D point `(X, Y, Z)` in camera coordinates to pixel `(u, v)`:

```
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy
```

Real lenses add distortion (radial: barrel/pincushion; tangential: decentering). The `CameraInfo` message provides all parameters needed to correct for distortion and project/unproject points.

### Clock Synchronization

Robot sensors have independent hardware clocks that drift at ~20 ppm. `header.stamp` records the sensor capture time; `log_time` records when the recording software received it. The difference includes network latency, kernel scheduling jitter, and clock drift. Time synchronization aligns multiple sensor streams to a common timeline using nearest-neighbor matching or interpolation.

### Apache Parquet for Robot Data

A columnar storage format well-suited for robot datasets: supports nested types, per-column compression, predicate pushdown for filtered reads, and schema evolution. Timestamps stored as `int64` nanoseconds avoid floating-point precision loss. Binary columns hold image and point cloud data efficiently.

## Architecture

```
mcap_reader/
    __init__.py            # Lazy imports for top-level API
    reader.py              # McapReader, RawMessage, TopicInfo
    deserializer.py        # CDR byte-level deserialization
    sync.py                # TimeSynchronizer, SyncConfig, SyncQuality
    calibration.py         # CameraModel (pinhole + distortion)
    episode.py             # EpisodeDetector (gap/marker/manual)
    cli.py                 # Click CLI (7 subcommands)
    messages/
        image.py           # sensor_msgs/msg/Image
        compressed_image.py # sensor_msgs/msg/CompressedImage
        camera_info.py     # sensor_msgs/msg/CameraInfo
        imu.py             # sensor_msgs/msg/Imu
        joint_state.py     # sensor_msgs/msg/JointState
        pointcloud.py      # sensor_msgs/msg/PointCloud2
        transform.py       # tf2_msgs/msg/TFMessage
    transforms/
        math.py            # Vector3, Quaternion, Transform, slerp
        frames.py          # FrameGraph (BFS path finding)
        buffer.py          # TransformBuffer (interpolation, static/dynamic)
```

**Layered design.** Dependencies flow downward:

1. **`deserializer.py`** -- Pure byte manipulation, no domain knowledge. Reads CDR primitives (integers, floats, strings, arrays) with correct alignment.
2. **`messages/`** -- Uses the deserializer to decode specific ROS 2 types. Each module knows the field layout for one message type.
3. **`reader.py`** -- Uses `mcap` + `mcap-ros2-support` libraries for file I/O and CDR decoding. Provides the high-level iterator API.
4. **`transforms/`** -- Pure math (`math.py`), graph algorithms (`frames.py`), and the buffer that combines them (`buffer.py`). No dependency on the reader.
5. **`sync.py`, `calibration.py`, `episode.py`** -- Higher-level analysis that consumes `McapReader` output.
6. **`cli.py`** -- Integration layer that wires everything together behind Click commands. Uses lazy imports so `mcap-reader --help` is instant.

**Design decisions:**
- **Frozen dataclasses everywhere.** `Vector3`, `Quaternion`, `Transform`, `TopicInfo`, `RawMessage`, `SyncResult`, `Episode` are all immutable. This prevents accidental mutation of shared transform data and makes the code easier to reason about.
- **No tabulate dependency.** Tables in CLI output are formatted with simple column alignment to keep the dependency footprint small.
- **Lazy imports in CLI.** Heavy modules (pandas, pyarrow, numpy, opencv) are imported inside command functions, not at module level, so `mcap-reader --help` responds instantly.

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=mcap_reader --cov-report=term-missing

# Just the transform math tests
pytest tests/ -k "transform or quaternion"
```

### Synthetic data generators

The `tests/generators/` package provides utilities for creating synthetic MCAP data for testing without requiring real robot recordings. These generators produce deterministic, mathematically verifiable test data (known rotations, known camera intrinsics, known episode boundaries) so tests can assert exact values rather than relying on approximate comparisons.

### Property-based testing with Hypothesis

The `[dev]` dependencies include [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing. This is especially valuable for the transform math module, where you can test algebraic properties that must hold for any input:

- Quaternion normalization: `q.normalize().norm() == 1.0` for all non-zero `q`
- Composition associativity: `(q1 * q2) * q3 == q1 * (q2 * q3)`
- Inverse: `q * q.inverse() == identity`
- SLERP endpoints: `slerp(q1, q2, 0.0) == q1` and `slerp(q1, q2, 1.0) == q2`
- Transform round-trip: `T * T.inverse() == identity`

## Out of Scope

The following are intentionally excluded to keep the project focused:

| Excluded | Why |
|----------|-----|
| **Real-time streaming** | This is an offline analysis tool. Real-time would need async I/O, ring buffers, and a fundamentally different architecture. |
| **ROS 1 bag support** | ROS 1 uses a different container format (`.bag`) with a different serialization (not CDR). Supporting both would double the deserialization code without teaching new concepts. |
| **Full DDS/RTPS stack** | We read MCAP files after recording. The live DDS transport layer (discovery, QoS, RTPS protocol) is a separate concern. |
| **GPU acceleration** | Image undistortion and point cloud processing could be faster on GPU, but would add CUDA/OpenCL dependencies and obscure the algorithms. |
| **Protobuf/FlatBuffers schemas** | MCAP supports multiple serialization formats. We only handle ROS 2 CDR because it is the most common in robotics and the most instructive to deserialize by hand. |
| **Multi-file datasets** | Real robot datasets span many MCAP files. We handle one file at a time; dataset-level management (file discovery, sharding, indexing) is a separate tool. |
| **Writing MCAP files** | This is a read-only tool. Writing MCAP files is simpler (no index construction needed if you accept no-summary files) but would not add much educational value. |
