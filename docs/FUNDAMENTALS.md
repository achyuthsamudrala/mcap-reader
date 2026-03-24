# Robotics Data Fundamentals

A standalone guide to the concepts behind robot sensor data — how it's generated, encoded, stored, synchronized, and spatially referenced. Read this before (or alongside) the code.

---

## Table of Contents

1. [How Robots Produce Data](#1-how-robots-produce-data)
2. [Binary Serialization and CDR](#2-binary-serialization-and-cdr)
3. [The MCAP Container Format](#3-the-mcap-container-format)
4. [ROS 2 Message Types and Schemas](#4-ros-2-message-types-and-schemas)
5. [Images in Robotics](#5-images-in-robotics)
6. [3D Point Clouds](#6-3d-point-clouds)
7. [Inertial Measurement Units (IMUs)](#7-inertial-measurement-units-imus)
8. [Joint States and Robot Kinematics](#8-joint-states-and-robot-kinematics)
9. [Coordinate Frames and Transforms](#9-coordinate-frames-and-transforms)
10. [Quaternions and 3D Rotations](#10-quaternions-and-3d-rotations)
11. [The Pinhole Camera Model](#11-the-pinhole-camera-model)
12. [Time, Clocks, and Synchronization](#12-time-clocks-and-synchronization)
13. [Episodes and Dataset Structure](#13-episodes-and-dataset-structure)
14. [Columnar Storage with Parquet](#14-columnar-storage-with-parquet)
15. [Putting It All Together](#15-putting-it-all-together)
16. [Further Reading](#16-further-reading)

---

## 1. How Robots Produce Data

A robot is a collection of sensors and actuators connected by software. At every instant, each sensor independently captures a measurement and publishes it as a timestamped message. A typical mobile manipulator might produce:

```
Sensor            Rate      Message Size    Bandwidth
─────────────────────────────────────────────────────
IMU               200 Hz    ~300 bytes      60 KB/s
Joint encoders    100 Hz    ~500 bytes      50 KB/s
RGB camera         30 Hz    ~6 MB           180 MB/s
Depth camera       30 Hz    ~2.4 MB         72 MB/s
LiDAR              10 Hz    ~2 MB           20 MB/s
TF transforms      50 Hz    ~200 bytes      10 KB/s
```

### 🧩 Data Structure Examples

#### 1. IMU (Inertial Measurement Unit)
* **Description:** Measures linear acceleration and angular velocity.
* **Structure:**
    ```json
    {
      "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.81},
      "angular_velocity": {"x": 0.02, "y": -0.01, "z": 0.0},
      "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}
    }
    ```

#### 2. Joint Encoders
* **Description:** Reports the physical state of the robot's actuators.
* **Structure:**
    ```yaml
    joints: ["waist", "shoulder", "elbow"]
    position: [0.0, 1.57, 0.75] # Radians
    velocity: [0.0, 0.1, 0.2]   # Rad/s
    effort: [0.0, 10.5, 5.2]    # Nm (Torque)
    ```

#### 3. RGB Camera
* **Description:** Raw visual color stream.
* **Structure:**
    ```text
    Format: 1920x1080 (RGB8)
    Data: [ [R,G,B], [R,G,B], [R,G,B] ... ] 
    # Represents a 2D matrix of 2,073,600 color pixels
    ```

#### 4. Depth Camera
* **Description:** Per-pixel distance information.
* **Structure:**
    ```text
    Format: 640x480 (Uint16)
    Data: [ 1200, 1205, 1210, 0, 1215 ... ] 
    # Integer values represent distance in millimeters
    ```

#### 5. LiDAR (Light Detection and Ranging)
* **Description:** 2D or 3D point cloud of the environment.
* **Structure:**
    ```python
    # LaserScan Parameters
    range_min: 0.1
    range_max: 30.0
    ranges: [5.21, 5.22, 5.25, 10.1, "inf"] # Array of distances
    ```

#### 6. TF Transforms
* **Description:** Coordinate transformations between robot components.
* **Structure:**
    ```json
    {
      "parent_frame": "base_link",
      "child_frame": "camera_link",
      "translation": {"x": 0.1, "y": 0.0, "z": 0.5},
      "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}
    }
    ```

These streams are **asynchronous** — each sensor has its own clock, its own sample rate, and its own latency path through the software stack. There is no global "tick" that makes all sensors fire simultaneously. This is the fundamental challenge of robot data: **bringing independent, heterogeneous, asynchronous streams into alignment**.

### The recording pipeline

In ROS 2, sensor data flows through a publish-subscribe middleware called DDS (Data Distribution Service). Each sensor driver publishes messages to named **topics** (e.g., `/camera/color/image_raw`). A recording tool subscribes to all topics of interest and writes incoming messages into a file.

The standard file format for ROS 2 recordings is **MCAP**. Before MCAP, ROS 1 used `.bag` files (a simpler, less efficient format). MCAP adds features that matter at scale: chunk-based compression, a summary index for fast random access, and support for multiple serialization formats.

### Why not just use a video file?

A robot recording is fundamentally different from a video:

- **Multiple data modalities** — not just pixels, but also 3D points, inertial measurements, joint angles, and transform trees. No video codec handles this heterogeneity.
- **Per-message timestamps** — each message carries the exact time the physical measurement was taken, with nanosecond resolution. Video formats have frame rates, not per-frame timestamps.
- **Structured metadata** — each message has a schema describing its binary layout. A video frame is opaque bytes; a ROS message is self-describing.
- **Random access by topic and time** — you can read just the IMU data from minute 5 to minute 7 without touching camera frames. Video formats require sequential decoding.

---

## 2. Binary Serialization and CDR

### What is serialization?

When a sensor driver wants to publish an IMU reading, it has a structured object in memory — a C++ struct or Python dataclass with fields like `angular_velocity.x`, `angular_velocity.y`, etc. To send this over a network (or write it to disk), those fields must be packed into a contiguous byte sequence. This is **serialization**. The reverse — reconstructing the structured object from bytes — is **deserialization**.

### CDR: Common Data Representation

ROS 2 uses CDR (Common Data Representation) as its serialization format. CDR comes from CORBA (Common Object Request Broker Architecture), a 1990s distributed computing standard. ROS 2 adopted it through DDS, which inherited CORBA's wire format.

CDR is a **binary** format (not text like JSON or XML). A float64 value occupies exactly 8 bytes. There are no field names, no delimiters, no whitespace. The byte layout is fully determined by the message schema — if you know the schema, you can walk through the bytes field by field.

### Alignment padding

CDR requires every primitive value to be aligned to a byte boundary equal to its own size:

```
Type        Size    Alignment
─────────────────────────────
uint8       1       1 (always aligned)
uint16      2       2
uint32      4       4
float32     4       4
float64     8       8
string      varies  4 (for the length prefix)
```

**Why alignment?** Modern CPUs access memory in aligned chunks. Reading a 4-byte integer from address 0x1003 (not divisible by 4) is either:
- **Illegal** on some ARM processors (causes a bus fault)
- **Slow** on x86 (the CPU must perform two memory reads and stitch the bytes together)

CDR's alignment rules guarantee that a receiver can cast a pointer directly into the byte buffer and read native types without any unaligned access penalty. The cost is a few wasted padding bytes between fields.

### Example: how an IMU header is encoded

The ROS 2 `std_msgs/Header` message contains:

```
builtin_interfaces/Time stamp
    uint32 sec
    uint32 nanosec
string frame_id
```

In CDR bytes (little-endian):

```
Offset  Bytes           Field           Notes
──────────────────────────────────────────────────────
0x00    00 02 00 00     [encapsulation] LE CDR header
0x04    E8 03 00 00     stamp.sec       uint32 = 1000
0x08    00 65 CD 1D     stamp.nanosec   uint32 = 500000000
0x0C    0A 00 00 00     frame_id length uint32 = 10 (includes null)
0x10    62 61 73 65     "base"          4 ASCII bytes
0x14    5F 6C 69 6E     "_lin"          4 ASCII bytes
0x18    6B 00           "k\0"           1 char + null terminator
```

Notice: `sec` starts at offset 0x04 (4-byte aligned ✓), `nanosec` at 0x08 (4-byte aligned ✓), the string length at 0x0C (4-byte aligned ✓). No padding was needed here because consecutive uint32 fields are naturally aligned. But if you had a `uint8` followed by a `uint32`, there would be 3 padding bytes between them.

### The encapsulation header

Every CDR payload starts with 4 bytes:

```
Byte 0-1: Encoding identifier
    0x00 0x01 = Big-endian (CDR_BE)
    0x00 0x02 = Little-endian (CDR_LE) ← common on x86/ARM
Byte 2-3: Options (always 0x00 0x00 in ROS 2)
```

This lets the receiver know the byte order without any out-of-band negotiation. After these 4 bytes, all alignment offsets are measured relative to byte 4.

### Strings in CDR

CDR strings are both length-prefixed AND null-terminated — a quirk inherited from CORBA:

```
[uint32 length] [length bytes of data including trailing \0]
```

The `length` field includes the null terminator. So `"hello"` (5 characters) has length = 6. This redundancy exists because CORBA needed both C interoperability (null termination) and efficient skip-ahead (length prefix).

### Variable-length arrays (sequences)

```
[uint32 count] [count × element_size bytes of data]
```

The `count` is the number of elements, not the byte length. Each element follows its own alignment rules. For a `float64[]` sequence, each element is 8-byte aligned.

---

## 3. The MCAP Container Format

MCAP wraps serialized messages into an indexed, seekable container. Think of it as a database file optimized for time-series sensor data.

### File structure

```
┌─────────────────────────────────┐
│ Magic bytes (8 bytes)           │  "MCAP0\r\n"
├─────────────────────────────────┤
│ Header record                   │  Library name, profile
├─────────────────────────────────┤
│ Data Section                    │
│  ┌───────────────────────────┐  │
│  │ Schema records            │  │  Message type definitions
│  │ Channel records           │  │  Topic → schema mappings
│  │ Chunk 1                   │  │  Compressed block of messages
│  │ Chunk 2                   │  │
│  │ ...                       │  │
│  │ Message Index records     │  │  Per-chunk offset tables
│  └───────────────────────────┘  │
├─────────────────────────────────┤
│ Summary Section                 │
│  ┌───────────────────────────┐  │
│  │ Schema records (copies)   │  │  Duplicated for fast access
│  │ Channel records (copies)  │  │
│  │ Chunk Index records       │  │  Map: time range → chunk offset
│  │ Statistics record         │  │  Total counts, time range
│  └───────────────────────────┘  │
├─────────────────────────────────┤
│ Summary Offset record           │  Points to start of summary
├─────────────────────────────────┤
│ Footer record                   │  Summary offset, CRC
├─────────────────────────────────┤
│ Magic bytes (8 bytes)           │  "MCAP0\r\n"
└─────────────────────────────────┘
```

### Key concepts

**Schema** — the type definition for a message. For ROS 2, this is the `.msg` file content (e.g., the definition of `sensor_msgs/Imu`). A schema is registered once in the file and referenced by ID.

**Channel** — a named stream of messages with a specific schema. In ROS 2 terms, this is a topic: `/imu/data` is a channel whose schema is `sensor_msgs/Imu`. Multiple channels can share the same schema (e.g., `/left_camera/image` and `/right_camera/image` both use `sensor_msgs/Image`).

**Chunk** — a compressed block of sequential messages from potentially many channels. Chunks are the unit of compression (LZ4 or Zstd). To read a specific message, you decompress its entire chunk. Chunk sizes are tunable — larger chunks compress better but require decompressing more data for random access.

**Message Index** — an offset table that maps (channel_id, timestamp) → byte offset within a chunk. This is what makes random access fast: to find all IMU messages between t=10s and t=20s, the reader scans the message indices (cheap) rather than decompressing every chunk (expensive).

**Statistics** — a single record in the summary section containing the total message count, time range, and per-channel message counts. This is how `mcap-reader summary` can print file stats without reading any message data.

### Why chunks matter

Without chunks, reading a single message near the end of a 10 GB file would require decompressing everything before it. With chunks (typically 1-4 MB compressed), you decompress only the ~4 MB chunk containing your message. The chunk index in the summary section tells you exactly which chunk to read.

### CRC integrity

MCAP includes CRC-32 checksums in chunks and the footer. This catches data corruption from disk errors, incomplete writes (power loss during recording), or file truncation. When you see "CRC mismatch" errors, it means the data on disk doesn't match what was originally written.

---

## 4. ROS 2 Message Types and Schemas

### The type system

ROS 2 has a strongly-typed message system. Every topic has a fixed message type, and every message type has a schema defined in a `.msg` file. For example, `sensor_msgs/msg/Imu`:

```
std_msgs/Header header

geometry_msgs/Quaternion orientation
float64[9] orientation_covariance

geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance

geometry_msgs/Vector3 linear_acceleration
float64[9] linear_acceleration_covariance
```

This is a hierarchical definition — `Imu` contains a `Header` (which itself contains a `Time` and a `string`), two `Quaternion`s, etc. The CDR serializer flattens this hierarchy into a linear byte sequence using depth-first traversal.

### Type naming convention

```
package_name/msg/TypeName
```

Examples:
- `sensor_msgs/msg/Image` — raw camera images
- `sensor_msgs/msg/PointCloud2` — 3D point clouds
- `sensor_msgs/msg/Imu` — inertial measurements
- `sensor_msgs/msg/JointState` — robot joint angles/velocities/torques
- `sensor_msgs/msg/CameraInfo` — camera calibration parameters
- `tf2_msgs/msg/TFMessage` — coordinate frame transforms

### The Header pattern

Nearly every sensor message includes a `std_msgs/Header`:

```
builtin_interfaces/Time stamp
string frame_id
```

- **stamp** — when the measurement was taken (sensor time, not recording time)
- **frame_id** — which coordinate frame this data lives in (e.g., `"camera_optical_frame"`, `"imu_link"`)

The header is the bridge between the data stream (what was measured) and the spatial system (where it was measured). Without `frame_id`, you can't place the data in 3D space. Without `stamp`, you can't align it with other sensors.

---

## 5. Images in Robotics

### Raw vs compressed images

Robots typically publish camera data in one of two forms:

**`sensor_msgs/Image`** — raw, uncompressed pixel data. A 1920×1080 RGB8 image is exactly `1920 × 1080 × 3 = 6,220,800 bytes`. Fast to decode (just reshape the buffer) but enormous bandwidth. Rare in real recordings.

**`sensor_msgs/CompressedImage`** — JPEG or PNG compressed pixel data. A typical JPEG of the same scene is 100-500 KB — a 10-50x reduction. This is how most real-world datasets store camera data.

### Encodings

The `encoding` field describes the pixel format:

```
Encoding    Dtype      Channels  Use case
─────────────────────────────────────────────────
rgb8        uint8      3         Standard color (R, G, B order)
bgr8        uint8      3         OpenCV default (B, G, R order)
rgba8       uint8      4         Color with alpha channel
mono8       uint8      1         Grayscale
mono16      uint16     1         Depth (millimeters, typically)
16UC1       uint16     1         Depth (same as mono16, different convention)
32FC1       float32    1         Depth (meters, float precision)
bayer_rggb8 uint8      1         Raw Bayer mosaic (needs demosaicing)
```

### The `step` trap

The `Image` message has both `width` and `step` fields. You might assume `step == width * bytes_per_pixel`, but this is often false. GPU drivers and DMA engines pad each row to a specific alignment (64 or 128 bytes) for efficient SIMD processing.

```
Width = 641 pixels, 3 channels, uint8
Expected row bytes: 641 × 3 = 1923
Actual step: 1984 (padded to 64-byte boundary)
Padding per row: 61 bytes
```

If you reshape using `width * channels` instead of `step`, every row after the first reads the wrong bytes, producing a sheared image with diagonal artifacts. Always use `step` as the row stride.

### Depth images

Depth cameras (Intel RealSense, Microsoft Azure Kinect) publish depth as `mono16` (millimeters as uint16) or `32FC1` (meters as float32). These are NOT visual images — they're distance measurements per pixel. Never cast them to uint8 (which clips to 0-255), or you lose all depth information beyond 255mm.

A depth pixel value of `3000` in a `mono16` image means "the surface at this pixel is 3000mm = 3.0m from the camera." Combined with the camera's intrinsic parameters, you can convert this to a 3D point in space.

---

## 6. 3D Point Clouds

### What is PointCloud2?

`sensor_msgs/PointCloud2` is a universal container for 3D spatial data. Each "point" is a fixed-size binary blob, and the message carries an embedded schema (the `fields` list) describing what each blob contains.

Common configurations:

```
LiDAR:         x, y, z, intensity, ring, time
RGB-D camera:  x, y, z, rgb
Stereo camera: x, y, z
Semantic:      x, y, z, label, confidence
```

### The binary layout

Each point is `point_step` bytes. The `fields` list describes where within those bytes each named value lives:

```
PointField: name="x",         offset=0,  datatype=FLOAT32, count=1
PointField: name="y",         offset=4,  datatype=FLOAT32, count=1
PointField: name="z",         offset=8,  datatype=FLOAT32, count=1
PointField: name="intensity", offset=12, datatype=FLOAT32, count=1
→ point_step = 16 bytes
```

The data buffer is simply `num_points × point_step` bytes of these packed structs. The most efficient way to decode this is to build a numpy structured dtype from the fields list and call `np.frombuffer` — zero-copy, O(1) decoding regardless of point count.

### Organized vs unorganized clouds

- **Unorganized** (`height == 1`): a flat list of N points with no spatial structure. Typical output from a spinning LiDAR.
- **Organized** (`height > 1`): a 2D grid where each (row, col) corresponds to a specific angular or pixel coordinate. Typical from depth cameras or structured-light sensors. Invalid measurements are represented as NaN points (when `is_dense == False`).

---

## 7. Inertial Measurement Units (IMUs)

### What an IMU measures

An IMU contains:

- **Accelerometer** — measures linear acceleration along three axes (m/s²). At rest, it reads ~(0, 0, 9.81) due to gravity.
- **Gyroscope** — measures angular velocity around three axes (rad/s). At rest, it reads ~(0, 0, 0) plus noise.
- **Magnetometer** (sometimes) — measures magnetic field direction for heading estimation.

Many IMUs also fuse these measurements internally to estimate **orientation** as a quaternion, published in the `orientation` field.

### Why IMU data is tricky

**Frame dependence.** The orientation quaternion is in the sensor's local frame. Without a transform to a world frame (`map`, `odom`, or `base_link`), the orientation is not directly useful. The transform tree (Section 9) provides this connection.

**Covariance.** Each measurement comes with a 3×3 covariance matrix expressing uncertainty. A covariance matrix with `[0][0] == -1` is a sentinel meaning "this field is not provided" — the IMU driver doesn't estimate orientation, or the gyro is uncalibrated. Don't treat -1 as an actual covariance value.

**Rate vs accuracy tradeoff.** IMUs sample at 200-1000 Hz — much faster than cameras (30 Hz) or LiDAR (10 Hz). This makes them ideal as the "reference clock" for time synchronization. But individual measurements are noisy; their value comes from averaging over many samples (via Kalman filtering or complementary filtering).

---

## 8. Joint States and Robot Kinematics

### What JointState contains

`sensor_msgs/JointState` carries the state of every actuated joint on a robot:

```
string[]  name       ["shoulder_pan", "shoulder_lift", "elbow", ...]
float64[] position   [0.5, -1.2, 0.8, ...]
float64[] velocity   [0.01, -0.02, 0.0, ...]
float64[] effort     [12.5, 8.3, 5.1, ...]
```

### Parallel arrays

The four fields are **parallel arrays** — `position[i]` is the position of joint `name[i]`. This is important because:

1. **Joint ordering is not standardized.** Different robot drivers publish joints in different orders. Never assume `position[0]` is the first shoulder joint — always look up by name.
2. **Arrays may be empty.** A driver that doesn't measure velocity will publish an empty `velocity[]`. Always check the array length before accessing elements.
3. **Units vary.** Revolute joints are in radians, prismatic joints in meters. Effort is in Newton-meters (torque) for revolute joints and Newtons (force) for prismatic joints.

### Why this matters for robot learning

In imitation learning and reinforcement learning, the JointState is the **action space** (what the robot did) and part of the **observation space** (what the robot perceived about itself). Aligning joint states with camera images at the correct timestamps is critical — a 50ms misalignment at a joint velocity of 1 rad/s means a 0.05 radian position error, which at the end effector could be centimeters.

---

## 9. Coordinate Frames and Transforms

### The problem

Every sensor measures in its own coordinate frame. The IMU reports acceleration relative to its own axes. The camera sees the world through its own optical frame. The LiDAR produces points relative to its own center. To combine these measurements — for example, to project a 3D LiDAR point onto a camera image — you need to know the spatial relationship between each sensor.

### The transform tree

ROS represents spatial relationships as a tree of coordinate frames connected by transforms. Each transform describes the position and orientation of a child frame relative to its parent frame.

```
         map
          │
        odom
          │
      base_link
       /     \
  imu_link   camera_link
                  │
          camera_optical_frame
```

Each edge is a transform `T_parent_child` containing a translation (3D vector) and a rotation (quaternion). To find the relationship between any two frames, you find the path through the tree and compose the transforms along the path.

For example, to transform a point from `camera_optical_frame` to `base_link`:

```
T_base_camera = T_base_camera_link × T_camera_link_optical
```

If you need to go "up" the tree (child → parent direction), you use the inverse transform.

### Static vs dynamic transforms

**Static transforms** (`/tf_static`) don't change over time. Examples:
- `base_link → camera_link` — a camera bolted to the chassis
- `camera_link → camera_optical_frame` — a fixed axis rotation (Z-forward → Z-into-image)

**Dynamic transforms** (`/tf`) change continuously. Examples:
- `odom → base_link` — updated by wheel odometry as the robot moves
- `base_link → arm_link_1` — updated by joint encoders as the arm moves

Static transforms are published once and apply at all timestamps. Dynamic transforms are published at a fixed rate (10-100 Hz) and must be interpolated for timestamps that fall between publications.

### Why interpolation is needed

A camera image arrives at t = 1.005s. The most recent `odom → base_link` transforms were published at t = 1.00s and t = 1.01s. To get the transform at exactly t = 1.005s, you interpolate:
- **Translation**: linear interpolation (LERP)
- **Rotation**: spherical linear interpolation (SLERP) — see Section 10

---

## 10. Quaternions and 3D Rotations

### Why not Euler angles?

The intuitive way to describe rotation is three angles: roll, pitch, yaw. But Euler angles have a fatal flaw: **gimbal lock**. When the pitch reaches ±90°, the roll and yaw axes align, and you lose a degree of freedom. This isn't just a math curiosity — it causes real numerical failures in code that compounds rotations.

### What is a quaternion?

A quaternion is a 4-component number: q = w + xi + yj + zk, often written as (x, y, z, w). A **unit quaternion** (one with magnitude 1) represents a rotation in 3D space.

Geometrically: the vector part (x, y, z) points along the rotation axis, and the scalar part w encodes the rotation angle θ via:

```
q = (sin(θ/2) * axis_x, sin(θ/2) * axis_y, sin(θ/2) * axis_z, cos(θ/2))
```

So the identity rotation (no rotation, θ = 0) is q = (0, 0, 0, 1).

### The double-cover property

Both q and -q represent the **same rotation**. This is because rotating by angle θ around axis **n** is the same as rotating by (2π - θ) around -**n**. This matters for interpolation: when computing the "shortest path" between two rotations, you must check whether q₁·q₂ is negative and negate one quaternion if so.

### Quaternion composition (Hamilton product)

To compose two rotations — "first rotate by q₁, then by q₂" — you multiply the quaternions:

```
q_result = q1 * q2
```

The Hamilton product is:

```
(q1 * q2).w = q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
(q1 * q2).x = q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y
(q1 * q2).y = q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x
(q1 * q2).z = q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w
```

**Order matters.** q₁ * q₂ ≠ q₂ * q₁ (rotation is not commutative). In the TF2 convention: `T_AC = T_AB * T_BC` reads "to go from A to C, first go A→B, then B→C."

### Quaternion inverse

For a unit quaternion, the inverse is the conjugate:

```
q⁻¹ = (-x, -y, -z, w)
```

This reverses the rotation. `q * q⁻¹ = identity`.

### SLERP: Spherical Linear Interpolation

To interpolate between two rotations (e.g., for transform interpolation), you need SLERP:

```
slerp(q1, q2, t) = q1 * sin((1-t)θ) / sin(θ)  +  q2 * sin(tθ) / sin(θ)
```

where θ = arccos(q₁ · q₂).

**Why not just LERP and normalize?** Linear interpolation (NLERP) — `normalize((1-t)*q1 + t*q2)` — works but produces **non-uniform angular velocity**. The rotation speeds up in the middle and slows at the ends. SLERP traverses the great-circle arc on the unit 4-sphere at constant speed, which is physically correct for smooth motion.

For very small angles (θ ≈ 0), SLERP's formula becomes numerically unstable (dividing by sin(θ) ≈ 0). In this case, fall back to NLERP — the difference is negligible for small angles.

### Convention warning: (x,y,z,w) vs (w,x,y,z)

ROS uses `(x, y, z, w)` ordering. SciPy uses `(w, x, y, z)` (scalar-first). This is the single most common source of quaternion bugs. Pick one convention internally and convert at the boundaries.

---

## 11. The Pinhole Camera Model

### The basic model

A camera projects 3D points onto a 2D image plane. The simplest model for this is the **pinhole camera**:

```
         3D point (X, Y, Z)
              |
              |  focal length f
              |
    ──────────┼──────────  image plane
              |
         pixel (u, v)
```

The projection equations:

```
u = fx * (X/Z) + cx
v = fy * (Y/Z) + cy
```

Where:
- **(fx, fy)** — focal lengths in pixels. These convert from angular units to pixel units. Typically fx ≈ fy, but they differ for non-square pixels.
- **(cx, cy)** — principal point, where the optical axis intersects the image plane. Ideally (width/2, height/2) but usually slightly offset due to manufacturing.

### The intrinsic matrix K

These four parameters are arranged in a 3×3 matrix:

```
    ┌ fx  0  cx ┐
K = │  0 fy  cy │
    └  0  0   1 ┘
```

K is called the **camera intrinsic matrix** because it depends only on the camera's internal properties (lens, sensor), not on where the camera is in the world.

In `sensor_msgs/CameraInfo`, the K field is a 9-element row-major array:

```
K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
```

### Lens distortion

Real lenses bend light in ways the pinhole model doesn't capture. The two main types:

**Radial distortion** — straight lines in the world appear curved in the image. Barrel distortion (common in wide-angle lenses) pushes pixels outward. Pincushion distortion pushes them inward.

```
Undistorted:  ┌────────────┐     Barrel:  ╭────────────╮
              │            │              │            │
              │            │              │            │
              │            │              │            │
              └────────────┘              ╰────────────╯
```

**Tangential distortion** — the lens is not perfectly parallel to the image sensor, causing a slight tilt in the projected image.

The distortion coefficients are stored in the **D** vector:

```
plumb_bob model:       D = [k1, k2, p1, p2, k3]
rational_polynomial:   D = [k1, k2, p1, p2, k3, k4, k5, k6]
```

Where k1, k2, k3 are radial coefficients and p1, p2 are tangential coefficients.

### Why distortion matters

For a standard 60° FOV camera, distortion is minor — maybe 1-2 pixels at the edges. For a wide-angle (120°+) or fisheye lens, distortion can be 50+ pixels. If you project a 3D point to pixels without accounting for distortion, it will land in the wrong place. If you triangulate two camera views without undistorting, your 3D reconstruction will be warped.

**Undistortion** removes distortion from an image, making straight world lines appear straight in the image. This is done with `cv2.undistort()` using K and D.

### The R and P matrices

In `CameraInfo`:

- **R** (3×3) — the **rectification matrix**. For monocular cameras, this is identity. For stereo cameras, it rotates each camera's image plane so that epipolar lines are horizontal (essential for stereo matching).
- **P** (3×4) — the **projection matrix** for the rectified image. P = K' × [R | t], where K' is the new intrinsic matrix after rectification. For monocular cameras, P is just K with a zero column appended.

### Projection and unprojection

**Project** (3D → 2D): given a point (X, Y, Z) in the camera frame, find the pixel (u, v):

```
1. Normalize:    x' = X/Z,  y' = Y/Z
2. Distort:      apply radial and tangential distortion
3. Pixel coords: u = fx * x_distorted + cx
                  v = fy * y_distorted + cy
```

**Unproject** (2D + depth → 3D): given pixel (u, v) and depth d, find the 3D point:

```
1. Undistort:    (x', y') = undistort_point(u, v)
2. 3D point:     X = x' * d,  Y = y' * d,  Z = d
```

---

## 12. Time, Clocks, and Synchronization

### The clock problem

Every sensor has its own hardware clock. Commodity crystal oscillators drift at roughly **20 parts per million (ppm)**, which means:

```
After 1 second:     0.02 ms drift
After 100 seconds:  2 ms drift
After 10 minutes:   12 ms drift
After 1 hour:       72 ms drift
```

Even with software clock synchronization (NTP, PTP), residual jitter from the OS kernel, DDS middleware, and recording pipeline adds 0.1-5ms of timing noise.

### Two timestamps, different meanings

Every MCAP message carries two timestamps:

**`log_time`** — when the rosbag2 writer thread wrote the message to disk. This is monotonically increasing and includes variable latency from the software stack. Think of it as "when did the recording tool see this message?"

**`header.stamp`** — the sensor-side timestamp embedded in the ROS message payload. For a properly written driver, this is the **capture time** — when the camera shutter fired, when the IMU sample-and-hold circuit latched, when the LiDAR pulse returned.

**For cross-sensor alignment, always use `header.stamp`.** It reflects when the physical measurement was taken, not when the software happened to process it. The difference between the two (`log_time - header.stamp`) is the **pipeline latency**, which varies per message.

### Synchronization strategies

**Nearest-neighbor**: for each reference timestamp, find the closest message from each other topic. Simple and preserves original measurements, but the matched message can be up to half a sampling period old (e.g., 16ms for a 30Hz camera).

```
Reference (IMU, 200Hz):  ─────●─────●─────●─────●─────●─────●─────
Camera (30Hz):            ─────────────●─────────────────●─────────
                                       ↑                 ↑
                          nearest match for each IMU sample
```

**Interpolation**: find the two messages bracketing the reference timestamp and interpolate between them. Produces a "virtual sample" at the exact reference time.

- Scalar values (position, velocity): **linear interpolation**
- Orientations (quaternions): **SLERP** (see Section 10)
- Images: **not interpolatable** — use nearest-neighbor instead

### Max delay thresholds

A global "max delay" threshold doesn't work for mixed-rate sensors:

```
IMU at 1000 Hz:   samples are 1ms apart → 2ms delay is suspicious
Camera at 30 Hz:  samples are 33ms apart → 16ms delay is normal
```

Per-topic thresholds let you set tight bounds on high-rate sensors while being lenient on low-rate ones. If the closest match exceeds the threshold, that modality is "dropped" for that reference timestamp rather than silently using stale data.

### Sync quality metrics

After synchronization, compute:
- **Mean delay** per topic — systematic offset suggests a clock calibration issue
- **Max delay** per topic — occasional spikes suggest scheduling hiccups
- **Drop rate** per topic — frequent drops suggest a driver issue or too-tight threshold

---

## 13. Episodes and Dataset Structure

### What is an episode?

In robot learning, an **episode** is one contiguous task execution: pick up an object, navigate to a waypoint, assemble a part. A recording session typically contains many episodes with gaps between them (robot resetting, operator intervention, failed attempts).

Detecting episode boundaries is essential for training because:
- Data from different episodes should not be mixed within a single training sequence
- Episode success/failure labels determine the reward signal
- Episode duration and count are key dataset statistics

### Detection strategies

**Gap-based**: if the time gap between consecutive messages exceeds a threshold (commonly 5 seconds), declare an episode boundary. Simple and works well in practice because robots don't sit idle for 5 seconds mid-task.

**Marker-based**: look for special topics that explicitly signal episode boundaries:
- `/episode_start`, `/episode_end` — common in structured data collection
- `/reset` — the robot was reset to a starting configuration
- `/episode_success` (std_msgs/Bool) — whether the episode succeeded

**Manual**: the user specifies (start_time, end_time) pairs when automatic detection doesn't fit their recording structure.

### Common dataset formats

Research datasets in robot learning come in various formats:

- **RLDS** (Reinforcement Learning Datasets) — TFRecord-based, used by Google's RT-X project
- **HDF5** — hierarchical, used by robomimic and LIBERO
- **MCAP** — the ROS 2 native format, increasingly common in DROID and Aloha datasets
- **LeRobot** — Hugging Face's format, converting from all of the above

Understanding MCAP deeply gives you the foundation to convert to/from any of these.

---

## 14. Columnar Storage with Parquet

### Why Parquet for robot data?

When you export synchronized sensor data for machine learning, you need a format that supports:
- **Heterogeneous columns** — timestamps (int64), joint angles (float64), images (bytes)
- **Efficient filtering** — "give me only the joint positions, not the images" without reading the whole file
- **Compression** — reduce disk usage without manual zip/unzip

Apache Parquet is a columnar format that handles all of these. "Columnar" means data is stored column-by-column rather than row-by-row:

```
Row-oriented (CSV):           Columnar (Parquet):
───────────────────           ────────────────────
ts, joint1, joint2, img      ts:     [1.0, 1.01, 1.02, ...]
1.0, 0.5, -0.3, <bytes>      joint1: [0.5, 0.51, 0.52, ...]
1.01, 0.51, -0.29, <bytes>   joint2: [-0.3, -0.29, -0.28, ...]
1.02, 0.52, -0.28, <bytes>   img:    [<bytes>, <bytes>, <bytes>, ...]
```

### Column pushdown

When you query `SELECT joint1, joint2 FROM data`, a columnar format reads only the `joint1` and `joint2` columns — it never touches the (much larger) `img` column. For a dataset where images are 99% of the bytes, this is a 100x speedup over row-oriented formats.

### Images in Parquet

A key design decision: **store images as compressed bytes (JPEG/PNG), not as expanded pixel arrays**. A 1920×1080 RGB8 image is 6.2 MB as raw pixels but 100-500 KB as JPEG. Storing raw pixels in Parquet would make the file 10-50x larger for no benefit — you'd still need to convert to a numpy array at load time.

The Parquet schema for an image column is simply `BYTE_ARRAY` containing the JPEG/PNG bytes. The loader decompresses at read time.

### Parquet schema design for common types

```
IMU:        timestamp_ns (int64), frame_id (string),
            orientation_x/y/z/w (float64),
            angular_vel_x/y/z (float64),
            linear_accel_x/y/z (float64)

JointState: timestamp_ns (int64), frame_id (string),
            position_{name} (float64) for each joint,
            velocity_{name} (float64) for each joint

Image:      timestamp_ns (int64), frame_id (string),
            height (int32), width (int32), encoding (string),
            data (bytes, JPEG compressed)

PointCloud: timestamp_ns (int64), frame_id (string),
            num_points (int32),
            data (bytes, raw or compressed)
```

---

## 15. Putting It All Together

Here's how all these concepts connect when you process a real robot recording:

```
1. OPEN the MCAP file
   → Read the summary section (schemas, channels, statistics)
   → No message data touched yet

2. ITERATE messages
   → Decompress chunks on demand
   → Deserialize CDR bytes into typed Python objects
   → Each message has a topic, schema, timestamp, and frame_id

3. PARSE message types
   → Image: reshape bytes into numpy array using step and encoding
   → PointCloud2: build structured dtype, zero-copy np.frombuffer
   → IMU: extract quaternion, angular velocity, linear acceleration
   → JointState: parallel arrays keyed by joint name
   → CameraInfo: intrinsic matrix K, distortion D

4. BUILD the transform tree
   → Ingest TFMessage from /tf and /tf_static
   → Build frame graph, store timestamped transforms
   → Answer queries: "what's the transform from camera to base_link at t=5.3s?"

5. SYNCHRONIZE across sensors
   → Pick a reference topic (usually the fastest sensor)
   → For each reference timestamp, find nearest/interpolated match per topic
   → Track sync quality (delays, drops)

6. DETECT episodes
   → Find gaps in the message stream
   → Or use marker topics (/episode_start, /episode_end)
   → Compute per-episode metadata (duration, message counts, success)

7. EXPORT for downstream use
   → Write to Parquet with type-appropriate schema
   → Images as compressed bytes, scalars as native columns
   → One row per synchronized timestamp
```

Each of these steps builds on the concepts covered in this document. The code in `mcap-reader` implements all of them.

---

## 16. Further Reading

### Specifications and standards
- [MCAP format specification](https://mcap.dev/spec) — the official byte-level format definition
- [OMG CDR specification](https://www.omg.org/spec/CORBA/3.3/Interoperability/PDF) — Chapter 9 covers the CDR encoding rules
- [ROS 2 sensor_msgs](https://github.com/ros2/common_interfaces/tree/rolling/sensor_msgs) — the `.msg` definitions for all sensor types
- [REP-103: Standard Units of Measure](https://www.ros.org/reps/rep-0103.html) — why ROS uses meters, radians, and specific frame conventions

### Quaternions and rotations
- [3Blue1Brown: Quaternions and 3D Rotation](https://www.youtube.com/watch?v=zjMuIxRvygQ) — the best visual introduction
- [Wikipedia: SLERP](https://en.wikipedia.org/wiki/Slerp) — the math behind spherical interpolation
- [Euclidean Space: Quaternion to Matrix](https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/) — all conversion formulas with derivations
- [Shepperd's method (1978)](https://arc.aiaa.org/doi/10.2514/3.57311) — the numerically stable rotation matrix → quaternion algorithm

### Computer vision
- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) — practical tutorial with code
- Hartley & Zisserman, *Multiple View Geometry in Computer Vision* — the definitive textbook (Chapter 6 for camera models)

### Time synchronization
- [IEEE 1588 Precision Time Protocol (PTP)](https://en.wikipedia.org/wiki/Precision_Time_Protocol) — how hardware clock sync works
- [ROS 2 time concepts](https://design.ros2.org/articles/clock_and_time.html) — system time vs ROS time vs simulation time

### Robot learning datasets
- [Open X-Embodiment](https://robotics-transformer-x.github.io/) — the largest multi-robot dataset collection
- [DROID](https://droid-dataset.github.io/) — distributed robot interaction dataset, uses MCAP-compatible formats
- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face's robot learning framework and dataset format

### Tools
- [Foxglove Studio](https://foxglove.dev/) — the best tool for visualizing MCAP files (free, cross-platform)
- [mcap CLI](https://mcap.dev/guides/cli) — official command-line tool for inspecting MCAP files
