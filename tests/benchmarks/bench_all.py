"""
Benchmark suite for mcap-reader.

Produces detailed timing reports for all performance-critical operations.
Run with: python -m tests.benchmarks.bench_all

Each benchmark reports:
  - ops/sec (throughput)
  - mean time per operation
  - min/max/stddev across iterations

The benchmarks are designed to test realistic workloads:
  - CDR deserialization: bulk primitive reads, string reads, full message decode
  - Quaternion math: hamilton product chains, SLERP interpolation, matrix conversion
  - Transform buffer: lookup with varying buffer sizes and chain lengths
  - Point cloud: to_numpy (zero-copy) vs to_xyz (data extraction) at various sizes
  - Image: decode at various resolutions and encodings
  - Sync: nearest-neighbor and bracket finding at various data sizes
"""

import math
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np

# Ensure the package is importable
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from mcap_reader.deserializer import CdrDeserializer, Header, deserialize_header
from mcap_reader.messages.image import ENCODING_DTYPE_MAP, Image
from mcap_reader.messages.pointcloud import PointCloud2, PointField
from mcap_reader.sync import _TopicTimeline
from mcap_reader.transforms.buffer import TransformBuffer
from mcap_reader.transforms.frames import FrameGraph
from mcap_reader.transforms.math import (
    Quaternion,
    Transform,
    Vector3,
    interpolate_transform,
    slerp,
)


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    name: str
    iterations: int
    total_seconds: float
    times: list[float]

    @property
    def ops_per_sec(self) -> float:
        return self.iterations / self.total_seconds if self.total_seconds > 0 else float("inf")

    @property
    def mean_ms(self) -> float:
        return (self.total_seconds / self.iterations) * 1000

    @property
    def min_ms(self) -> float:
        return min(self.times) * 1000

    @property
    def max_ms(self) -> float:
        return max(self.times) * 1000

    @property
    def stddev_ms(self) -> float:
        if len(self.times) < 2:
            return 0.0
        mean = sum(self.times) / len(self.times)
        variance = sum((t - mean) ** 2 for t in self.times) / (len(self.times) - 1)
        return math.sqrt(variance) * 1000

    def __str__(self) -> str:
        return (
            f"  {self.name:<50s} "
            f"{self.mean_ms:>8.3f}ms avg  "
            f"{self.min_ms:>8.3f}ms min  "
            f"{self.max_ms:>8.3f}ms max  "
            f"±{self.stddev_ms:>6.3f}ms  "
            f"({self.ops_per_sec:>10,.0f} ops/sec)"
        )


def bench(name: str, fn, iterations: int = 100, warmup: int = 5) -> BenchResult:
    """Run a benchmark and return timing results."""
    # Warmup
    for _ in range(warmup):
        fn()

    times = []
    total = 0.0
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total += elapsed

    return BenchResult(name=name, iterations=iterations, total_seconds=total, times=times)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_unit_quat():
    v = np.random.randn(4)
    v /= np.linalg.norm(v)
    return Quaternion(float(v[0]), float(v[1]), float(v[2]), float(v[3]))


def _make_cdr_float64_payload(n):
    header = b"\x00\x02\x00\x00"
    body = struct.pack(f"<{n}d", *range(n))
    return header + body


def _make_pointcloud(n):
    fields = [
        PointField("x", 0, 7, 1), PointField("y", 4, 7, 1),
        PointField("z", 8, 7, 1), PointField("intensity", 12, 7, 1),
    ]
    data = np.random.randn(n * 4).astype(np.float32).tobytes()
    return PointCloud2(
        header=Header(0, 0, "lidar"), height=1, width=n, fields=fields,
        is_bigendian=False, point_step=16, row_step=n * 16, data=data, is_dense=True,
    )


def _make_image(h, w, encoding="rgb8"):
    dtype, ch = ENCODING_DTYPE_MAP[encoding]
    step = w * dtype.itemsize * ch
    data = np.random.randint(0, 255, h * step, dtype=np.uint8).tobytes()
    return Image(Header(0, 0, "cam"), h, w, encoding, False, step, data)


def _make_tf_buffer(n_frames, n_ts):
    buf = TransformBuffer()
    for i in range(n_frames - 1):
        for t in range(n_ts):
            ts = float(t) * 0.01
            a = ts * 0.1
            tf = Transform(Vector3(t * 0.001, 0, 0), Quaternion(0, 0, math.sin(a / 2), math.cos(a / 2)))
            buf.add_transform(f"f{i}", f"f{i+1}", tf, ts)
    return buf


def _make_timeline(n):
    tl = _TopicTimeline()
    tl.timestamps = [float(i) * 0.001 for i in range(n)]
    tl.messages = [None] * n
    return tl


# ---------------------------------------------------------------------------
# Benchmark groups
# ---------------------------------------------------------------------------

def bench_cdr_deserialization():
    print("\n=== CDR Deserialization ===")

    # Float64 bulk read
    for n in [100, 1_000, 10_000]:
        payload = _make_cdr_float64_payload(n)
        def read_all(p=payload, count=n):
            cdr = CdrDeserializer(p)
            for _ in range(count):
                cdr.read_float64()
        result = bench(f"read_float64 x{n:,}", read_all, iterations=50)
        print(result)

    # String read
    header = b"\x00\x02\x00\x00"
    body = b""
    s = b"base_link\x00"
    for _ in range(1000):
        pad = (4 - (len(body) % 4)) % 4
        body += b"\x00" * pad + struct.pack("<I", len(s)) + s
    payload = header + body
    def read_strings():
        cdr = CdrDeserializer(payload)
        for _ in range(1000):
            cdr.read_string()
    print(bench("read_string x1,000", read_strings, iterations=50))

    # Header deserialization
    hdr_body = struct.pack("<II", 1000, 500000000) + struct.pack("<I", 10) + b"base_link\x00"
    hdr_payload = b"\x00\x02\x00\x00" + hdr_body
    def read_header():
        cdr = CdrDeserializer(hdr_payload)
        deserialize_header(cdr)
    print(bench("deserialize_header", read_header, iterations=1000))


def bench_quaternion_math():
    print("\n=== Quaternion Math ===")

    quats = [_random_unit_quat() for _ in range(1001)]

    # Hamilton product
    def hamilton_1000():
        for i in range(1000):
            _ = quats[i] * quats[i+1]
    print(bench("hamilton_product x1,000", hamilton_1000, iterations=50))

    # SLERP
    q1, q2 = quats[0], quats[1]
    ts = np.linspace(0, 1, 1000).tolist()
    def slerp_1000():
        for t in ts:
            slerp(q1, q2, t)
    print(bench("slerp x1,000", slerp_1000, iterations=50))

    # to_rotation_matrix
    def rot_mat_1000():
        for i in range(1000):
            quats[i].to_rotation_matrix()
    print(bench("to_rotation_matrix x1,000", rot_mat_1000, iterations=50))

    # from_rotation_matrix
    matrices = [q.to_rotation_matrix() for q in quats[:1000]]
    def from_mat_1000():
        for m in matrices:
            Quaternion.from_rotation_matrix(m)
    print(bench("from_rotation_matrix x1,000", from_mat_1000, iterations=50))

    # Transform composition
    transforms = [
        Transform(Vector3(*np.random.randn(3)), _random_unit_quat())
        for _ in range(1001)
    ]
    def compose_1000():
        for i in range(1000):
            _ = transforms[i] * transforms[i+1]
    print(bench("transform_compose x1,000", compose_1000, iterations=50))

    # Transform.apply
    tf = transforms[0]
    points = [Vector3(*np.random.randn(3)) for _ in range(1000)]
    def apply_1000():
        for p in points:
            tf.apply(p)
    print(bench("transform_apply x1,000", apply_1000, iterations=50))

    # Interpolate transform
    t1, t2 = transforms[0], transforms[1]
    def interp_1000():
        for t in ts:
            interpolate_transform(t1, t2, t)
    print(bench("interpolate_transform x1,000", interp_1000, iterations=50))


def bench_transform_buffer():
    print("\n=== Transform Buffer ===")

    for n_ts in [100, 1_000, 10_000]:
        buf = _make_tf_buffer(3, n_ts)
        mid = float(n_ts // 2) * 0.01
        def lookup(b=buf, t=mid):
            b.lookup_transform("f0", "f2", t)
        result = bench(f"lookup (2-edge chain, {n_ts:,} timestamps)", lookup, iterations=1000)
        print(result)

    # Chain length scaling
    for n_frames in [3, 5, 10, 20]:
        buf = _make_tf_buffer(n_frames, 100)
        def lookup(b=buf, target=f"f{n_frames-1}"):
            b.lookup_transform("f0", target, 0.5)
        print(bench(f"lookup ({n_frames-1}-edge chain, 100 timestamps)", lookup, iterations=500))

    # Static transform lookup
    buf = TransformBuffer()
    buf.add_transform("a", "b", Transform(Vector3(1, 0, 0), Quaternion.identity()), 0.0, is_static=True)
    buf.add_transform("b", "c", Transform(Vector3(0, 1, 0), Quaternion.identity()), 0.0, is_static=True)
    def lookup_static():
        buf.lookup_transform("a", "c", 999.0)
    print(bench("lookup (2-edge static chain)", lookup_static, iterations=5000))


def bench_frame_graph():
    print("\n=== Frame Graph ===")

    for depth in [10, 50, 100]:
        g = FrameGraph()
        for i in range(depth - 1):
            g.add_edge(f"f{i}", f"f{i+1}")
        def find_path(graph=g, target=f"f{depth-1}"):
            graph.get_chain("f0", target)
        print(bench(f"BFS chain depth={depth}", find_path, iterations=1000))

    # Wide tree
    g = FrameGraph()
    for i in range(100):
        g.add_edge("root", f"child_{i}")
        for j in range(10):
            g.add_edge(f"child_{i}", f"gc_{i}_{j}")
    def find_wide():
        g.get_chain("gc_0_0", "gc_99_9")
    print(bench("BFS wide tree (1100 nodes, cross-tree)", find_wide, iterations=500))


def bench_pointcloud():
    print("\n=== Point Cloud Decoding ===")

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        pc = _make_pointcloud(n)

        def to_numpy(p=pc):
            p.to_numpy()
        print(bench(f"to_numpy {n:>10,} points (zero-copy)", to_numpy, iterations=100))

        def to_xyz(p=pc):
            p.to_xyz()
        iters = 100 if n <= 100_000 else 10
        print(bench(f"to_xyz  {n:>10,} points", to_xyz, iterations=iters))


def bench_image():
    print("\n=== Image Decoding ===")

    resolutions = [
        ("VGA", 480, 640),
        ("720p", 720, 1280),
        ("1080p", 1080, 1920),
        ("4K", 2160, 3840),
    ]

    for name, h, w in resolutions:
        img = _make_image(h, w, "rgb8")
        def decode(i=img):
            i.to_numpy()
        print(bench(f"to_numpy {name} rgb8", decode, iterations=100))

    # Depth image (32FC1)
    for name, h, w in resolutions[:3]:
        img = _make_image(h, w, "32FC1")
        def decode(i=img):
            i.to_numpy()
        print(bench(f"to_numpy {name} 32FC1 (depth)", decode, iterations=100))


def bench_sync():
    print("\n=== Time Synchronization ===")

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        tl = _make_timeline(n)
        mid = float(n // 2) * 0.001

        def nearest(t=tl, target=mid):
            t.find_nearest(target)
        print(bench(f"find_nearest ({n:>10,} messages)", nearest, iterations=5000))

        def bracket(t=tl, target=mid):
            t.find_bracket(target)
        print(bench(f"find_bracket ({n:>10,} messages)", bracket, iterations=5000))

    # Full sync simulation
    print("\n  --- Full sync simulation (200Hz ref, 3 secondaries, 50s recording) ---")
    ref = _make_timeline(10_000)
    secondaries = {
        "cam_30hz": _make_timeline(1_500),
        "joints_50hz": _make_timeline(2_500),
        "tf_50hz": _make_timeline(2_500),
    }

    def sync_nearest():
        for i in range(len(ref.timestamps)):
            t = ref.timestamps[i]
            for tl in secondaries.values():
                tl.find_nearest(t)

    print(bench("sync_nearest 10k ref x 3 topics", sync_nearest, iterations=5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    np.random.seed(42)  # Reproducibility

    print("=" * 100)
    print("mcap-reader Performance Benchmarks")
    print("=" * 100)

    bench_cdr_deserialization()
    bench_quaternion_math()
    bench_transform_buffer()
    bench_frame_graph()
    bench_pointcloud()
    bench_image()
    bench_sync()

    print("\n" + "=" * 100)
    print("Benchmarks complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
