# Robotics Data Fundamentals

A standalone guide to the concepts behind robot sensor data — how it's generated, encoded, stored, synchronized, and spatially referenced. Read this before (or alongside) the code.

This guide is structured as a narrative, not a reference manual. It follows the journey of robot data from physical measurement to training-ready dataset, introducing each concept when you first need it.

---

## Table of Contents

### Part I — How Robot Data Gets to Disk
1. [How Robots Produce Data](#1-how-robots-produce-data) — sensors, the data pipeline, ML infra analogies
2. [ROS 2: The Robot Operating System](#2-ros-2-the-robot-operating-system) — pub-sub, message types, DDS
3. [Binary Serialization and CDR](#3-binary-serialization-and-cdr) — the wire format for ROS 2 messages
4. [The MCAP Container Format](#4-the-mcap-container-format) — chunks, indexes, the Parquet of robotics
5. [Why Not Just Use Video?](#5-why-not-just-use-video) — MP4/H.264 tradeoffs for robot data

### Part II — What's Inside a Recording
6. [Sensor Data Types](#6-sensor-data-types) — Images, point clouds, IMUs, joint states, and how to decode them
7. [Where Things Are: Coordinate Frames and Transforms](#7-where-things-are-coordinate-frames-and-transforms) — the spatial backbone
8. [Quaternions and 3D Rotations](#8-quaternions-and-3d-rotations) — the math behind transforms
9. [The Pinhole Camera Model](#9-the-pinhole-camera-model) — projecting 3D into 2D and back

### Part III — Making the Data Useful
10. [Time, Clocks, and Synchronization](#10-time-clocks-and-synchronization) — aligning asynchronous sensor streams
11. [Episodes and Dataset Structure](#11-episodes-and-dataset-structure) — from continuous recordings to training data
12. [Columnar Storage with Parquet](#12-columnar-storage-with-parquet) — the export format for ML

13. [Putting It All Together](#13-putting-it-all-together)
14. [Further Reading](#14-further-reading)

---

# Part I — How Robot Data Gets to Disk

*Sections 1-5 follow a single data point — an IMU reading — from the moment the physical measurement happens to the moment it's stored as bytes on disk. By the end, you'll understand every layer of the stack: hardware sensor → driver → ROS message → CDR bytes → DDS transport → MCAP file.*

---

## 1. How Robots Produce Data

### If you're coming from data/ML infrastructure

Robot data has a lot in common with the event-driven systems you already know, but the physical world adds constraints that software-only systems don't face. Here's a rough mapping to orient you:

```
ML Infra Concept              Robot Equivalent
──────────────────────────────────────────────────────────────
Microservice                  Sensor driver (a process that
                              talks to hardware and emits events)
Protobuf / Avro schema        ROS 2 .msg definition
Kafka topic                   ROS 2 topic (named pub-sub channel)
Kafka broker / event bus      DDS middleware (peer-to-peer, no broker)
Protobuf serialization        CDR serialization (binary, schema-driven)
Event (key + value + ts)      ROS 2 message (topic + CDR bytes + timestamp)
Kafka consumer writing to S3  rosbag2 recorder writing to MCAP file
Parquet file on S3            MCAP file on disk
Parquet row group             MCAP chunk (compressed block of messages)
Parquet footer / metadata     MCAP summary section (schemas, stats, index)
```

The fundamental difference: **in ML infra, events come from software and timing is mostly about ordering. In robotics, events come from physical sensors, and timing is about _when a physical measurement actually happened_ — because that determines whether two measurements can be meaningfully combined.**

A camera frame and an IMU reading are only useful together if they describe the same physical instant. If they're 50ms apart, the robot may have moved, and combining them produces garbage. This is why so much of the robotics data stack is about timestamps, clocks, and synchronization.

### The physical layer: sensors

A robot has hardware sensors that convert physical phenomena into electrical signals:

| Sensor | What it measures | Physical principle |
|---|---|---|
| Camera | Light intensity per pixel | Photons hit a CMOS/CCD sensor array |
| Depth camera | Distance per pixel | Structured light or time-of-flight |
| LiDAR | Distance + angle | Laser pulse round-trip time |
| IMU | Acceleration + angular velocity | MEMS accelerometer + gyroscope |
| Joint encoder | Motor shaft angle | Optical/magnetic position sensor |

Each sensor has its own **clock crystal** (a tiny quartz oscillator), its own **sample rate**, and its own **latency** (time between physical measurement and data availability). The camera doesn't know or care that the IMU exists. They operate independently.

### What each sensor actually produces

Before anything gets serialized or transmitted, here's what each sensor outputs as raw data:

**IMU** — 6 floating-point numbers, 200 times per second:
```json
{
  "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.81},
  "angular_velocity":    {"x": 0.02, "y": -0.01, "z": 0.0},
  "orientation":         {"x": 0, "y": 0, "z": 0, "w": 1}
}
```
The accelerometer reads ~9.81 m/s² on Z at rest (gravity). The gyroscope reads near-zero (no rotation). The orientation is a quaternion (explained in Section 8).

**Joint encoders** — parallel arrays of angles/velocities/torques, 100 times per second:
```yaml
joints:   ["waist", "shoulder", "elbow"]
position: [0.0, 1.57, 0.75]    # radians
velocity: [0.0, 0.1, 0.2]      # rad/s
effort:   [0.0, 10.5, 5.2]     # Nm (torque)
```

**RGB camera** — a grid of pixel values, 30 times per second:
```
1920 x 1080 pixels, 3 channels (R, G, B), 1 byte each
= 6,220,800 bytes per frame
```

**Depth camera** — a grid of distance values, 30 times per second:
```
640 x 480 pixels, 1 channel (distance), 2 bytes each (uint16, millimeters)
= 614,400 bytes per frame
Value 1200 means "the surface at this pixel is 1200mm = 1.2m away"
Value 0 means "no reading" (too close, too far, or transparent surface)
```

**LiDAR** — a list of 3D points, 10 times per second:
```python
# Each point: x, y, z (meters) + intensity + ring_id
# Typical scan: 30,000 - 300,000 points
points = [(5.21, 0.3, -0.1, 0.8, 12), (5.22, 0.31, -0.1, 0.7, 12), ...]
```

**Transforms** — the spatial relationship between two parts of the robot:
```json
{
  "parent_frame": "base_link",
  "child_frame": "camera_link",
  "translation": {"x": 0.1, "y": 0.0, "z": 0.5},
  "rotation": {"x": 0, "y": 0, "z": 0, "w": 1}
}
```
"The camera is 10cm forward and 50cm up from the robot's base, with no rotation."

### The data pipeline: sensor to disk

Here's the full path a single IMU reading takes from physical measurement to bytes on disk. This is the "event pipeline" of robotics:

```
Step 1: PHYSICAL MEASUREMENT
   MEMS accelerometer detects 9.81 m/s² on Z axis
   Gyroscope detects 0.02 rad/s on X axis
   Sensor's internal ADC converts analog signal to digital values
   Sensor's internal clock timestamps the sample
                    │
                    ▼
Step 2: DRIVER (a userspace process on the robot's computer)
   Reads raw values over I2C/SPI/USB from the sensor hardware
   Packages them into a structured object:
     Imu {
       header: { stamp: {sec: 1700000000, nanosec: 500000000},
                 frame_id: "imu_link" },
       linear_acceleration: {x: 0.0, y: 0.0, z: 9.81},
       angular_velocity: {x: 0.02, y: -0.01, z: 0.0},
       orientation: {x: 0, y: 0, z: 0, w: 1},
       orientation_covariance: [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01],
       ...
     }
                    │
                    ▼
Step 3: SERIALIZATION (CDR encoding)
   The DDS middleware serializes the struct into a flat byte array:
     [00 02 00 00]           ← CDR encapsulation header (little-endian)
     [65 A0 C8 03 00 00 ...] ← sec, nanosec, frame_id, then all fields
   Result: ~300 bytes of CDR-encoded data
                    │
                    ▼
Step 4: PUBLISH (DDS middleware)
   The driver calls rclcpp::Publisher::publish(imu_msg)
   DDS sends the CDR bytes to all subscribers on topic "/imu/data"
   This is peer-to-peer UDP multicast — no broker, unlike Kafka
                    │
                    ▼
Step 5: SUBSCRIBE + RECORD (rosbag2 recorder)
   The recording tool receives the CDR bytes from DDS
   It wraps them into an MCAP message record:
     {
       channel_id: 3,           ← maps to topic "/imu/data"
       log_time: 1700000000.501 ← recorder's wall-clock time (NOT sensor time)
       publish_time: ...,
       data: [the CDR bytes]
     }
   The message is appended to the current MCAP chunk
                    │
                    ▼
Step 6: MCAP FILE ON DISK
   When the chunk fills up (~1-4 MB), it's compressed (LZ4/Zstd)
   and flushed to the .mcap file.
   When recording stops, the summary section is written:
     - Schema records (what message types exist)
     - Channel records (what topics exist, which schema each uses)
     - Chunk index (which time ranges are in which chunks)
     - Statistics (total message counts, time range)
```

**The key insight:** there are TWO timestamps. `header.stamp` (Step 2) is when the sensor physically took the measurement. `log_time` (Step 5) is when the recorder wrote it to disk. The difference is the pipeline latency — typically 0.5-5ms, but it varies per message. For synchronizing sensors with each other, you always want `header.stamp`.

### Bandwidth reality check

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

Cameras dominate. A 10-minute recording with one RGB camera and one depth camera is ~150 GB uncompressed. With JPEG compression for RGB and LZ4 for the MCAP container, that drops to ~5-15 GB. This is why `CompressedImage` (JPEG bytes inside a ROS message) is far more common than raw `Image` in real datasets.

Now that you know what sensors produce, the next question is: how does all this data flow through the robot's software? That's where ROS 2 comes in.

---

## 2. ROS 2: The Robot Operating System

### What ROS 2 is (and isn't)

ROS 2 (Robot Operating System 2) is **not an operating system**. It's a middleware framework — a set of libraries, tools, and conventions that standardize how robot software is written. It runs on top of Linux (or macOS/Windows).

Think of it as the "Spring Boot of robotics": it provides dependency injection (launch files), inter-process communication (topics/services), a type system (`.msg` definitions), and an ecosystem of reusable components (sensor drivers, SLAM algorithms, motion planners).

### The publish-subscribe model

ROS 2 applications are composed of **nodes** — independent processes that communicate through **topics** using publish-subscribe:

```
                        Topic: /imu/data
   ┌──────────┐        (sensor_msgs/Imu)         ┌──────────┐
   │ IMU      │ ─────────────────────────────────▶│ SLAM     │
   │ Driver   │                                   │ Node     │
   └──────────┘                                   └──────────┘
                                                       ▲
   ┌──────────┐     Topic: /camera/image_raw           │
   │ Camera   │ ─────────────────────────────────▶     │
   │ Driver   │     (sensor_msgs/Image)                │
   └──────────┘                                        │
                                                       │
   ┌──────────┐     Topic: /joint_states               │
   │ Joint    │ ─────────────────────────────────▶     │
   │ Driver   │     (sensor_msgs/JointState)     ┌─────┴────┐
   └──────────┘                                  │ rosbag2  │
                                                 │ recorder │
         All topics are also received by ────────▶ (writes  │
         the recorder for logging to disk         │  MCAP)  │
                                                 └──────────┘
```

Every topic has a **name** (like `/camera/color/image_raw`) and a **message type** (like `sensor_msgs/msg/Image`). The message type is fixed for a topic — you can't publish an IMU reading on a camera topic.

### How ROS 2 relates to DDS

Under the hood, ROS 2 topics are implemented using **DDS** (Data Distribution Service), an industry-standard pub-sub middleware. DDS handles:

- **Discovery** — nodes automatically find each other on the network (no broker needed, unlike Kafka)
- **Transport** — messages are sent over UDP multicast or shared memory
- **QoS** — configurable reliability, durability, and deadline policies (similar to Kafka retention/ack settings)
- **Serialization** — messages are encoded as CDR bytes (explained in Section 3)

You don't need to know DDS to use ROS 2 data — just know that it's the reason messages are CDR-encoded and that topics are the core data transport abstraction.

### Message types: the schema system

Every ROS 2 message type is defined in a `.msg` file — a simple DSL that's conceptually similar to Protobuf `.proto` files:

```protobuf
// For comparison — this is Protobuf syntax
message Imu {
  Header header = 1;
  Quaternion orientation = 2;
  repeated double orientation_covariance = 3;
  Vector3 angular_velocity = 4;
  ...
}
```

```
# This is the actual ROS 2 .msg syntax
std_msgs/Header header

geometry_msgs/Quaternion orientation
float64[9] orientation_covariance

geometry_msgs/Vector3 angular_velocity
float64[9] angular_velocity_covariance

geometry_msgs/Vector3 linear_acceleration
float64[9] linear_acceleration_covariance
```

Key differences from Protobuf:
- **No field numbers** — fields are identified by position, not by tag. This means you can't add/remove fields without breaking the schema (no backwards compatibility).
- **No optional/required** — all fields are always present.
- **Fixed-size arrays** — `float64[9]` is always exactly 9 elements. Variable-length arrays use `float64[]` (no size bound in the schema).

### The Header: every message's metadata

Nearly every sensor message includes a `std_msgs/Header`:

```
builtin_interfaces/Time stamp
string frame_id
```

- **stamp** — when the measurement was taken (sensor time, not recording time)
- **frame_id** — which coordinate frame this data lives in (e.g., `"camera_optical_frame"`, `"imu_link"`)

The header is the bridge between the data stream (what was measured) and the spatial system (where it was measured). Without `frame_id`, you can't place the data in 3D space. Without `stamp`, you can't align it with other sensors.

### Why this matters for reading MCAP files

When you open an MCAP file, you're looking at the output of this entire system:
- **Schemas** in the MCAP correspond to `.msg` definitions
- **Channels** correspond to topics (name + message type)
- **Messages** are CDR-encoded instances of those types
- **Timestamps** come from both the message header (sensor time) and the recording system (log time)

Understanding this lineage — physical sensor → driver → message type → CDR bytes → DDS transport → MCAP chunk — is what makes the rest of this document make sense. The next section zooms into Step 3 of the pipeline: how exactly are those structured messages packed into bytes?

---

## 3. Binary Serialization and CDR

Now that you know where messages come from (sensors → drivers → topics), let's look at how they're encoded into bytes for transport and storage.

### What is serialization?

When a sensor driver publishes an IMU reading, it has a structured object in memory — a C++ struct or Python dataclass with fields like `angular_velocity.x`, `angular_velocity.y`, etc. To send this over a network (or write it to disk), those fields must be packed into a contiguous byte sequence. This is **serialization**. The reverse — reconstructing the structured object from bytes — is **deserialization**.

If you've worked with Protobuf, this is the same idea — just a different wire format.

### CDR: Common Data Representation

ROS 2 uses CDR (Common Data Representation) as its serialization format. CDR comes from CORBA (Common Object Request Broker Architecture), a 1990s distributed computing standard. ROS 2 adopted it through DDS, which inherited CORBA's wire format.

**How CDR compares to formats you know:**

```
Format      Field IDs?   Self-describing?   Alignment?   Typical use
──────────────────────────────────────────────────────────────────────
JSON        By name      Yes                No           REST APIs
Protobuf    By tag #     Partially          No           gRPC, storage
Avro        By position  Schema in header   No           Kafka, Spark
CDR         By position  Schema out-of-band Yes          DDS / ROS 2
```

CDR is a **positional binary** format. A float64 value occupies exactly 8 bytes. There are no field names, no delimiters, no whitespace. The byte layout is fully determined by the message schema — if you know the schema, you can walk through the bytes field by field. This is why schemas and messages are stored together in the MCAP file.

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

If you've ever dealt with Arrow or Parquet's internal alignment requirements for SIMD processing, this is the same principle applied at the serialization level.

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

Notice: `sec` starts at offset 0x04 (4-byte aligned), `nanosec` at 0x08 (4-byte aligned), the string length at 0x0C (4-byte aligned). No padding was needed here because consecutive uint32 fields are naturally aligned. But if you had a `uint8` followed by a `uint32`, there would be 3 padding bytes between them.

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

At this point, you understand how a single message is serialized. But a recording contains millions of messages from dozens of topics. How are they organized into a file? That's the MCAP container format.

---

## 4. The MCAP Container Format

Now you know: sensors produce structured data (Section 1), ROS 2 defines the types and pub-sub transport (Section 2), and CDR is the wire encoding (Section 3). The last piece of the pipeline is **storage**: how do all those CDR-encoded messages end up in a file you can read offline?

### MCAP is the Parquet of robotics

If you know Parquet, you already understand the key ideas behind MCAP:

```
Parquet                         MCAP
──────────────────────────────────────────────────────────
Row group                       Chunk (compressed block)
Column chunk                    Messages from one channel in a chunk
Page                            Individual message
Schema / footer metadata        Summary section (schemas, channels, stats)
Row group offset index          Chunk index (time range → file offset)
Column statistics (min/max)     Statistics record (counts, time range)
Compression (Snappy/Zstd)       Compression (LZ4/Zstd)
```

The core design principle is the same: **write data in compressed blocks, then write an index at the end so readers can skip to exactly what they need without scanning the whole file.**

### File structure

```
┌─────────────────────────────────┐
│ Magic bytes (8 bytes)           │  "MCAP0\r\n"
├─────────────────────────────────┤
│ Header record                   │  Library name, profile
├─────────────────────────────────┤
│ Data Section                    │
│  ┌───────────────────────────┐  │
│  │ Schema records            │  │  Message type definitions (.msg content)
│  │ Channel records           │  │  Topic name → schema ID mappings
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

**Schema** — the type definition for a message. For ROS 2, this is the `.msg` file content (e.g., the definition of `sensor_msgs/Imu`). A schema is registered once in the file and referenced by ID. Think of it like a Protobuf descriptor embedded in the file.

**Channel** — a named stream of messages with a specific schema. In ROS 2 terms, this is a topic: `/imu/data` is a channel whose schema is `sensor_msgs/Imu`. Multiple channels can share the same schema (e.g., `/left_camera/image` and `/right_camera/image` both use `sensor_msgs/Image`). A channel is like a Kafka topic partition — a single ordered stream of typed events.

**Chunk** — a compressed block of sequential messages from potentially many channels. Chunks are the unit of compression (LZ4 or Zstd). To read a specific message, you decompress its entire chunk. Chunk sizes are tunable — larger chunks compress better but require decompressing more data for random access. This is the same tradeoff as Parquet row group sizing.

**Message Index** — an offset table that maps (channel_id, timestamp) → byte offset within a chunk. This is what makes random access fast: to find all IMU messages between t=10s and t=20s, the reader scans the message indices (cheap) rather than decompressing every chunk (expensive). This is like Parquet's page offset index.

**Statistics** — a single record in the summary section containing the total message count, time range, and per-channel message counts. This is how `mcap-reader summary` can print file stats without reading any message data — just like reading Parquet metadata without scanning row groups.

### Why chunks matter

Without chunks, reading a single message near the end of a 10 GB file would require decompressing everything before it. With chunks (typically 1-4 MB compressed), you decompress only the ~4 MB chunk containing your message. The chunk index in the summary section tells you exactly which chunk to read.

### CRC integrity

MCAP includes CRC-32 checksums in chunks and the footer. This catches data corruption from disk errors, incomplete writes (power loss during recording), or file truncation. When you see "CRC mismatch" errors, it means the data on disk doesn't match what was originally written.

At this point you understand the full pipeline from sensor to disk. But one question from Section 1 is still dangling: why not just record a video? The answer matters because it shapes how you think about robot data for the rest of this guide.

---

## 5. Why Not Just Use Video?

### How video codecs work (the 60-second version)

**MP4** is a container format (like MCAP). **H.264** (also called AVC) is a compression codec that goes inside MP4 (like CDR goes inside MCAP). H.265/HEVC and VP9/AV1 are newer codecs with better compression.

Video codecs achieve 100-1000x compression (vs raw pixels) through two key tricks:

1. **Spatial compression (intra-frame)** — within a single frame, nearby pixels are similar. Divide the frame into 16x16 blocks, predict each block from its neighbors, and encode only the prediction error. This is similar to JPEG.

2. **Temporal compression (inter-frame)** — consecutive frames are nearly identical. Instead of encoding every frame independently, encode most frames as "the difference from the previous frame." This is where the big wins come from: a static background contributes zero bytes across hundreds of frames.

This creates three types of frames:
```
I-frame (keyframe)    — fully self-contained, like a JPEG. Large.
P-frame (predicted)   — encoded as delta from a previous frame. Small.
B-frame (bidirectional) — encoded as delta from both past AND future frames. Smallest.

Typical GOP (Group of Pictures):
I  B  B  P  B  B  P  B  B  P  B  B  P  B  B  I  B  B  P ...
│                                              │
└──── keyframe every 30-60 frames ─────────────┘
```

**The critical consequence:** to decode frame N, you must first decode the nearest preceding I-frame, then every P-frame and B-frame between it and frame N. You can't jump to an arbitrary frame without this chain. This is fundamentally different from MCAP, where every message is independently decodable.

### What robotics needs vs. what video provides

```
Requirement                     Video (MP4/H.264)       MCAP + CompressedImage
─────────────────────────────────────────────────────────────────────────────────
Per-frame timestamp (ns)        No (fixed frame rate)    Yes (per-message)
Dropped frame handling          Breaks decode chain      Just a missing message
Random access to frame N        Decode from last I-frame Direct seek via index
Pixel-exact values              No (lossy transforms)    Yes (JPEG/PNG are standard)
Depth images (uint16/float32)   No (designed for 8-bit)  Yes (any dtype)
Non-image data alongside        Separate file or hack    Same file, same timeline
Multiple cameras in one file    Separate tracks           Separate topics
Variable frame rate             Poorly supported          Natural (async messages)
Compression ratio (RGB)         ~200-500x (H.264)        ~10-50x (per-frame JPEG)
```

### Where video codecs break down for robotics

**1. Lossy temporal compression destroys pixel-exact reproducibility.** H.264 doesn't store your actual pixels — it stores a compressed approximation that looks good to human eyes. The exact decoded pixel values depend on the decoder implementation. For training a robot policy on pixel observations, you need bit-exact reproducibility.

**2. Depth images are not 8-bit color.** A depth image is `uint16` (millimeters) or `float32` (meters). Feeding this through H.264 would clamp to 8-bit, destroying all depth beyond 2.55 meters.

**3. The inter-frame dependency chain is fragile.** If a camera driver drops frame 47 (common under CPU load), frames 48-59 are corrupted until the next I-frame. In MCAP with per-frame JPEG, a dropped frame is just a gap.

**4. Per-frame timestamps are essential, not optional.** A camera running at "30 fps" actually fires every 31-36ms due to USB scheduling and OS jitter. Video formats either assume fixed rate (wrong) or encode timestamps with limited precision.

### When video formats ARE useful

- **Visualization and review** — convert to MP4 for watching, store as MCAP for processing
- **Network streaming** — H.264/H.265 is essential for teleoperation over limited bandwidth
- **Long-term archival** — for visual-only data at scale, H.264's 200-500x compression beats per-frame JPEG's 10-50x
- **Video prediction models** — some world model architectures ingest MP4 directly

### The bottom line

Video codecs are optimized for a different problem: compressing a single continuous stream of 8-bit color frames for human viewing at a fixed rate. Robotics data is multi-modal, multi-rate, variable-timing, and needs pixel-exact values across diverse dtypes. MCAP with per-frame compression hits the right tradeoff.

That concludes Part I — you now understand the full journey from physical sensor to bytes on disk. Part II opens the file and looks at what's inside.

---

# Part II — What's Inside a Recording

*You have an MCAP file. You can list its topics, read its schemas, and iterate its messages. But what do those messages actually contain? Part II walks through the sensor data types you'll encounter, then explains the spatial and geometric concepts needed to make sense of them.*

*The sections here are ordered by complexity: simple scalar data first (IMU, joints), then rich spatial data (images, point clouds), then the geometric framework that ties it all together (coordinate frames, quaternions, camera models).*

---

## 6. Sensor Data Types

Each ROS 2 message type has a specific binary layout defined by its `.msg` schema. This section covers the six sensor types you'll encounter most often, in order from simplest to most complex.

### IMU (sensor_msgs/Imu)

The simplest sensor message — all scalar fields, no binary blobs.

An IMU contains:
- **Accelerometer** — measures linear acceleration along three axes (m/s²). At rest, it reads ~(0, 0, 9.81) due to gravity.
- **Gyroscope** — measures angular velocity around three axes (rad/s). At rest, it reads ~(0, 0, 0) plus noise.
- **Magnetometer** (sometimes) — measures magnetic field direction for heading estimation.

Many IMUs also fuse these measurements internally to estimate **orientation** as a quaternion, published in the `orientation` field.

**Why IMU data is tricky:**

**Frame dependence.** The orientation quaternion is in the sensor's local frame. Without a transform to a world frame (`map`, `odom`, or `base_link`), the orientation is not directly useful. The transform tree (Section 7) provides this connection.

**Covariance.** Each measurement comes with a 3×3 covariance matrix expressing uncertainty. A covariance matrix with `[0][0] == -1` is a sentinel meaning "this field is not provided" — the IMU driver doesn't estimate orientation, or the gyro is uncalibrated. Don't treat -1 as an actual covariance value.

**Rate vs accuracy tradeoff.** IMUs sample at 200-1000 Hz — much faster than cameras (30 Hz) or LiDAR (10 Hz). This makes them ideal as the "reference clock" for time synchronization (Section 10). But individual measurements are noisy; their value comes from averaging over many samples.

### Joint States (sensor_msgs/JointState)

The next simplest — variable-length parallel arrays instead of fixed fields.

`sensor_msgs/JointState` carries the state of every actuated joint on a robot:

```
string[]  name       ["shoulder_pan", "shoulder_lift", "elbow", ...]
float64[] position   [0.5, -1.2, 0.8, ...]
float64[] velocity   [0.01, -0.02, 0.0, ...]
float64[] effort     [12.5, 8.3, 5.1, ...]
```

**Parallel arrays** — `position[i]` is the position of joint `name[i]`. This is important because:

1. **Joint ordering is not standardized.** Different robot drivers publish joints in different orders. Never assume `position[0]` is the first shoulder joint — always look up by name.
2. **Arrays may be empty.** A driver that doesn't measure velocity will publish an empty `velocity[]`. Always check the array length before accessing elements.
3. **Units vary.** Revolute joints are in radians, prismatic joints in meters. Effort is in Newton-meters (torque) for revolute joints and Newtons (force) for prismatic joints.

**Why this matters for robot learning:** In imitation learning and reinforcement learning, the JointState is the **action space** (what the robot did) and part of the **observation space** (what the robot perceived about itself). Aligning joint states with camera images at the correct timestamps is critical — a 50ms misalignment at a joint velocity of 1 rad/s means a 0.05 radian position error, which at the end effector could be centimeters.

### Images (sensor_msgs/Image and CompressedImage)

Now we move from scalar data to binary blobs — the most bandwidth-intensive messages.

Robots typically publish camera data in one of two forms:

**`sensor_msgs/Image`** — raw, uncompressed pixel data. A 1920×1080 RGB8 image is exactly `1920 × 1080 × 3 = 6,220,800 bytes`. Fast to decode (just reshape the buffer) but enormous bandwidth. Rare in real recordings.

**`sensor_msgs/CompressedImage`** — JPEG or PNG compressed pixel data. A typical JPEG of the same scene is 100-500 KB — a 10-50x reduction. This is how most real-world datasets store camera data.

**Encodings** — the `encoding` field describes the pixel format:

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

**The `step` trap:** The `Image` message has both `width` and `step` fields. You might assume `step == width * bytes_per_pixel`, but this is often false. GPU drivers pad each row to a specific alignment (64 or 128 bytes) for efficient SIMD processing. If you reshape using `width * channels` instead of `step`, every row after the first reads the wrong bytes, producing a sheared image.

**Depth images** are NOT visual images — they're distance measurements per pixel. Never cast them to uint8. A depth pixel value of `3000` in a `mono16` image means "3000mm = 3.0m from the camera." Combined with the camera's intrinsic parameters (Section 9), you can convert this to a 3D point in space.

### Point Clouds (sensor_msgs/PointCloud2)

The most complex message type — a self-describing binary container with an embedded schema.

`sensor_msgs/PointCloud2` is a universal container for 3D spatial data. Each "point" is a fixed-size binary blob, and the message carries a `fields` list describing what each blob contains:

```
PointField: name="x",         offset=0,  datatype=FLOAT32, count=1
PointField: name="y",         offset=4,  datatype=FLOAT32, count=1
PointField: name="z",         offset=8,  datatype=FLOAT32, count=1
PointField: name="intensity", offset=12, datatype=FLOAT32, count=1
→ point_step = 16 bytes
```

The data buffer is `num_points × point_step` bytes. The most efficient decode: build a numpy structured dtype from the fields list and call `np.frombuffer` — zero-copy, O(1) regardless of point count.

**Organized vs unorganized clouds:**
- **Unorganized** (`height == 1`): a flat list of N points. Typical from spinning LiDAR.
- **Organized** (`height > 1`): a 2D grid. Typical from depth cameras. Invalid measurements are NaN (when `is_dense == False`).

### Camera Info (sensor_msgs/CameraInfo)

Not sensor data itself, but the metadata needed to interpret camera data geometrically. `CameraInfo` is almost always published on a sibling topic — `/camera/color/image_raw` pairs with `/camera/color/camera_info`.

It contains the camera's **intrinsic matrix K**, **distortion coefficients D**, **rectification matrix R**, and **projection matrix P**. We'll explain what these mean in Section 9 (The Pinhole Camera Model).

### Transforms (tf2_msgs/TFMessage)

The spatial glue that connects everything — explained fully in the next section.

A `TFMessage` contains a list of `TransformStamped` messages, each describing the position and orientation of one coordinate frame relative to another. These build the **transform tree** — the spatial backbone of the robot.

All six sensor types above include a `frame_id` in their header, saying "this data was measured in frame X." But what does that mean spatially? How do you combine an IMU reading in `imu_link` with a camera image in `camera_optical_frame`? You need the transform tree. That's where we go next.

---

## 7. Where Things Are: Coordinate Frames and Transforms

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
- **Rotation**: spherical linear interpolation (SLERP)

But what is SLERP, and why can't you just linearly interpolate a rotation? That requires understanding how rotations are represented — which brings us to quaternions.

---

## 8. Quaternions and 3D Rotations

*This section is the mathematical foundation for transforms (Section 7) and camera projection (Section 9). If you're comfortable with rotation matrices and quaternions, skip to the convention warning at the end — it'll save you the most common bug.*

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

This is why we're here — Section 7 said transforms need interpolation, and rotations can't be interpolated linearly.

```
slerp(q1, q2, t) = q1 * sin((1-t)θ) / sin(θ)  +  q2 * sin(tθ) / sin(θ)
```

where θ = arccos(q₁ · q₂).

**Why not just LERP and normalize?** Linear interpolation (NLERP) — `normalize((1-t)*q1 + t*q2)` — works but produces **non-uniform angular velocity**. The rotation speeds up in the middle and slows at the ends. SLERP traverses the great-circle arc at constant speed, which is physically correct for smooth motion.

For very small angles (θ ≈ 0), SLERP's formula becomes numerically unstable (dividing by sin(θ) ≈ 0). In this case, fall back to NLERP — the difference is negligible for small angles.

### Convention warning: (x,y,z,w) vs (w,x,y,z)

ROS uses `(x, y, z, w)` ordering. SciPy uses `(w, x, y, z)` (scalar-first). This is the single most common source of quaternion bugs. Pick one convention internally and convert at the boundaries.

Now you understand both the transform tree (Section 7) and the math behind it. There's one more geometric concept needed before we can combine all our sensor data: the camera model, which connects the 3D world to 2D images.

---

## 9. The Pinhole Camera Model

*Images (Section 6) give you pixels. Transforms (Section 7) give you 3D positions. The camera model is the bridge between them: it defines how a 3D point in the world maps to a pixel in the image, and vice versa. This is where CameraInfo (from Section 6) gets used.*

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

**Radial distortion** — straight lines in the world appear curved in the image. Barrel distortion (common in wide-angle lenses) pushes pixels outward.

```
Undistorted:  ┌────────────┐     Barrel:  ╭────────────╮
              │            │              │            │
              │            │              │            │
              │            │              │            │
              └────────────┘              ╰────────────╯
```

**Tangential distortion** — the lens is not perfectly parallel to the image sensor.

The distortion coefficients are stored in the **D** vector:

```
plumb_bob model:       D = [k1, k2, p1, p2, k3]
rational_polynomial:   D = [k1, k2, p1, p2, k3, k4, k5, k6]
```

### Why distortion matters

For a standard 60° FOV camera, distortion is minor — maybe 1-2 pixels at the edges. For a wide-angle (120°+) or fisheye lens, distortion can be 50+ pixels. If you project a 3D point to pixels without accounting for distortion, it will land in the wrong place.

**Undistortion** removes distortion from an image using `cv2.undistort()` with K and D.

### The R and P matrices

In `CameraInfo`:

- **R** (3×3) — the **rectification matrix**. For monocular cameras, this is identity. For stereo cameras, it rotates each camera's image plane so that epipolar lines are horizontal.
- **P** (3×4) — the **projection matrix** for the rectified image. For monocular cameras, P is just K with a zero column appended.

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

That wraps up Part II. You now know what every sensor message contains, how frames relate to each other in 3D space, how rotations work, and how cameras project the 3D world onto 2D images. But all of this assumes the data is from the same instant in time. In reality, each sensor fires independently at its own rate. Part III tackles the hard problem: making asynchronous sensor data work together.

---

# Part III — Making the Data Useful

*You can open an MCAP file, decode every message type, and understand the spatial relationships between sensors. But the data is still raw: asynchronous streams at different rates, potentially hours long, with no structure beyond chronological order. Part III covers the three transformations needed to turn raw recordings into training-ready datasets: temporal alignment, episode segmentation, and structured export.*

---

## 10. Time, Clocks, and Synchronization

This is the payoff section. Everything in Parts I and II builds toward this: **how do you combine data from different sensors that fire at different rates with different clocks?**

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

**For cross-sensor alignment, always use `header.stamp`.** The difference between the two (`log_time - header.stamp`) is the **pipeline latency**, which varies per message.

### Synchronization strategies

**Nearest-neighbor**: for each reference timestamp, find the closest message from each other topic. Simple and preserves original measurements, but the matched message can be up to half a sampling period old.

```
Reference (IMU, 200Hz):  ─────●─────●─────●─────●─────●─────●─────
Camera (30Hz):            ─────────────●─────────────────●─────────
                                       ↑                 ↑
                          nearest match for each IMU sample
```

**Interpolation**: find the two messages bracketing the reference timestamp and interpolate. Produces a "virtual sample" at the exact reference time.

- Scalar values (position, velocity): **linear interpolation**
- Orientations (quaternions): **SLERP** (Section 8 — this is why we covered it)
- Images: **not interpolatable** — use nearest-neighbor

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

## 11. Episodes and Dataset Structure

Now that you can synchronize sensor streams, the next question is: where does one task end and the next begin?

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

## 12. Columnar Storage with Parquet

The final step: exporting synchronized, episodic data into a format optimized for ML consumption.

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

A key design decision: **store images as compressed bytes (JPEG/PNG), not as expanded pixel arrays**. A 1920×1080 RGB8 image is 6.2 MB as raw pixels but 100-500 KB as JPEG. The Parquet schema for an image column is simply `BYTE_ARRAY` containing the JPEG/PNG bytes. The loader decompresses at read time.

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

### Why Parquet and not Lance, TFRecord, or HDF5?

If you come from ML infrastructure, you've probably encountered several data formats that seem like they'd fit robot data. Here's how they compare and why Parquet is the current default for export — even though none of them are perfect.

**Lance** is the most interesting alternative. Built by LanceDB, it's a columnar format designed specifically for ML workloads — with native support for vector embeddings, fast random access to individual rows (O(1) via row IDs), versioned datasets, and efficient append/update operations. Parquet was designed for analytics (scan large ranges of rows); Lance was designed for training (random-access individual samples).

```
                    Parquet             Lance               TFRecord            HDF5
────────────────────────────────────────────────────────────────────────────────────────────
Designed for        Analytics           ML training          TF pipelines        Scientific data
Access pattern      Sequential scan     Random row access    Sequential scan     Random access
Append/update       Rewrite file        In-place versioned   Append-only         In-place update
Column pushdown     Yes                 Yes                  No                  Partial
Nested types        Yes (struct/list)   Yes                  Yes (protobuf)      Yes (groups)
Binary blobs        BYTE_ARRAY          BLOB (ext storage)   bytes_list          opaque dtype
Vector search       No                  Native (ANN index)   No                  No
Row-level access    Slow (scan to row)  O(1) by row ID       Slow (scan)         O(1) by index
Ecosystem           Universal           Growing              TensorFlow-only     Broad (science)
Compression         Zstd/Snappy/LZ4     Zstd                 Gzip/Zstd           Gzip/LZF/Zstd
Mutability          Immutable           Versioned (MVCC)     Immutable           Mutable
```

**Where Lance would be better than Parquet for robot data:**

- **Random-access training.** PyTorch `DataLoader` with `shuffle=True` needs random access to individual samples. With Parquet, this means either loading the entire file into memory or accepting slow seeks. Lance gives O(1) row access by ID — exactly what `__getitem__(idx)` needs. For large robot datasets (10k+ episodes), this is the difference between a fast and painfully slow training loop.

- **Versioned datasets.** You discover that episodes 500-520 had bad calibration and want to mark them as excluded. With Parquet, you rewrite the entire file. Lance supports zero-copy versioning — tag a new version with those rows filtered out, and the old version is still accessible. For iterative dataset curation (which is where the field is heading), this is transformative.

- **Embedding search across episodes.** "Find me episodes similar to this one" requires computing embeddings over observations and searching nearest neighbors. Lance has a built-in ANN vector index. With Parquet, you'd need a separate vector database (Pinecone, Milvus, etc.) alongside your data.

- **Append without rewrite.** New recordings come in daily from a robot fleet. With Parquet, you either maintain one massive file (expensive to rewrite) or a directory of small files (fragmented reads). Lance handles appends natively with automatic compaction.

**Where Parquet is still the right choice:**

- **Universality.** Every tool reads Parquet — pandas, Spark, DuckDB, Polars, Arrow, BigQuery, Snowflake. Lance is growing but not yet universally supported. If your downstream consumer is unknown, Parquet is the safe bet.

- **Analytics and aggregation.** "What's the mean sync delay across all episodes?" is a scan query that Parquet is optimized for. Lance optimizes for point lookups, not full-table aggregations.

- **Maturity and trust.** Parquet has been battle-tested for a decade at petabyte scale. Lance is newer and still stabilizing its format spec. For archival data that needs to be readable in 5 years, Parquet is lower risk.

- **Interoperability with existing robotics tools.** LeRobot uses Parquet. Most ROS 2 data analysis tools export to Parquet. The ecosystem expects it.

**Where TFRecord falls short:**

TFRecord is a sequential format — it has no index, no column pushdown, and no random access without scanning. It exists because TensorFlow's `tf.data` pipeline was designed around sequential reads with prefetching. For any access pattern beyond "iterate from start to end," it's painful. The robotics community has largely moved away from TFRecord except in Google-adjacent projects (RLDS, RT-X).

**Where HDF5 falls short:**

HDF5 supports random access and nested data, which is why robomimic and LIBERO used it. But it has terrible concurrent read performance (the global lock problem), poor compression compared to Parquet/Lance, and no column pushdown. It also doesn't compose well with distributed training — multiple workers reading the same HDF5 file will contend on the lock.

**The practical answer today:**

Use Parquet for export and interchange — it's the lingua franca. Watch Lance closely — its random-access and versioning properties are exactly what robot learning datasets will need as they scale from "one researcher, one dataset" to "fleet of robots, continuously curated data." The transition will likely happen in 2026-2027 as Lance's ecosystem matures and robot datasets grow large enough that Parquet's sequential-scan model becomes the bottleneck.

---

## 13. Putting It All Together

Here's how all these concepts connect when you process a real robot recording:

```
1. OPEN the MCAP file                               [Section 4]
   → Read the summary section (schemas, channels, statistics)
   → No message data touched yet

2. ITERATE messages                                  [Sections 3-4]
   → Decompress chunks on demand
   → Deserialize CDR bytes into typed Python objects
   → Each message has a topic, schema, timestamp, and frame_id

3. PARSE message types                               [Section 6]
   → Image: reshape bytes into numpy array using step and encoding
   → PointCloud2: build structured dtype, zero-copy np.frombuffer
   → IMU: extract quaternion, angular velocity, linear acceleration
   → JointState: parallel arrays keyed by joint name
   → CameraInfo: intrinsic matrix K, distortion D

4. BUILD the transform tree                          [Sections 7-8]
   → Ingest TFMessage from /tf and /tf_static
   → Build frame graph, store timestamped transforms
   → Answer queries: "what's the transform from camera to
     base_link at t=5.3s?" (using SLERP for rotation interpolation)

5. SYNCHRONIZE across sensors                        [Section 10]
   → Pick a reference topic (usually the fastest sensor)
   → For each reference timestamp, find nearest/interpolated
     match per topic
   → Track sync quality (delays, drops)

6. DETECT episodes                                   [Section 11]
   → Find gaps in the message stream
   → Or use marker topics (/episode_start, /episode_end)
   → Compute per-episode metadata (duration, message counts, success)

7. EXPORT for downstream use                         [Section 12]
   → Write to Parquet with type-appropriate schema
   → Images as compressed bytes, scalars as native columns
   → One row per synchronized timestamp
```

Each of these steps builds on the concepts covered in this document. The code in `mcap-reader` implements all of them.

---

## 14. Further Reading

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
