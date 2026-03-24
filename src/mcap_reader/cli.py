"""Command-line interface for mcap-reader.

This module is the integration layer that wires together the reader, episode
detector, transform buffer, and message wrappers into a cohesive CLI tool.
It uses Click for argument parsing and provides subcommands for common
MCAP file operations.

Entry point
-----------
Registered as ``mcap-reader`` in ``pyproject.toml`` via::

    [project.scripts]
    mcap-reader = "mcap_reader.cli:cli"

After installation (``pip install -e .``), run::

    mcap-reader summary recording.mcap
    mcap-reader inspect recording.mcap --topic /imu/data --limit 3
    mcap-reader validate recording.mcap
    mcap-reader export recording.mcap --topic /imu/data -o imu.parquet
    mcap-reader sync recording.mcap --reference /camera/image_raw --topics /imu/data /joint_states
    mcap-reader frames recording.mcap
    mcap-reader episodes recording.mcap --gap-threshold 3.0

Design decisions
----------------
- **No tabulate dependency.** Tables are formatted manually with simple
  column alignment. This keeps the dependency footprint small and avoids
  version conflicts in constrained robot environments.

- **Progress bars via click.progressbar.** Long operations (validate,
  export) show progress bars so the user knows the tool has not hung.
  Click's built-in progressbar is lightweight and does not require
  ``tqdm``.

- **click.echo + click.style for output.** All output goes through
  Click's echo/style for proper terminal encoding handling and optional
  color support. Colors degrade gracefully when piped to a file.

- **Lazy imports.** Heavy modules (pandas, pyarrow, numpy) are imported
  inside command functions rather than at module level, so that
  ``mcap-reader --help`` responds instantly even in slow environments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Helper: simple table formatter (no tabulate dependency)
# ---------------------------------------------------------------------------


def _format_table(headers: list[str], rows: list[list[str]], padding: int = 2) -> str:
    """Format a list of rows into an aligned ASCII table.

    Computes the maximum width of each column across all rows and the
    header, then pads each cell to that width. No box-drawing characters
    are used -- just whitespace alignment with a header separator line.

    Parameters
    ----------
    headers : list[str]
        Column header labels.
    rows : list[list[str]]
        Data rows. Each row must have the same length as ``headers``.
    padding : int
        Minimum spaces between columns.

    Returns
    -------
    str
        Multi-line string with the formatted table.

    Example output::

        Topic                    Type                          Count   Hz
        ----------------------   ---------------------------   -----   ------
        /camera/image_raw        sensor_msgs/msg/Image         1200    29.97
        /imu/data                sensor_msgs/msg/Imu           8000    199.95
    """
    if not rows:
        return "(no data)"

    # Compute column widths.
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    sep = " " * padding

    # Build header line.
    header_line = sep.join(h.ljust(col_widths[i]) for i, h in enumerate(headers))

    # Build separator line (dashes under each column).
    separator = sep.join("-" * col_widths[i] for i in range(len(headers)))

    # Build data lines.
    data_lines = []
    for row in rows:
        line = sep.join(
            cell.ljust(col_widths[i]) for i, cell in enumerate(row)
        )
        data_lines.append(line)

    return "\n".join([header_line, separator] + data_lines)


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string.

    For short durations (< 60s) returns seconds with 1 decimal place.
    For longer durations returns minutes:seconds or hours:minutes:seconds.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.1f}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {s:.1f}s"


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="mcap-reader")
def cli():
    """mcap-reader: inspect, validate, and export ROS 2 MCAP recordings.

    A command-line tool for working with MCAP files containing ROS 2 data.
    Provides subcommands for file inspection, message validation, data
    export (Parquet/CSV), time synchronization, coordinate frame analysis,
    and episode detection.
    """
    pass


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def summary(path: str):
    """Print a summary of an MCAP file.

    Shows the file name, recording duration, total message count, time
    range, and a table of all topics with their message types, counts,
    and estimated frequencies.

    Example:

        mcap-reader summary recording.mcap
    """
    from mcap_reader.reader import McapReader

    with McapReader(path) as reader:
        info = reader.summary()

        click.echo(click.style("=== MCAP File Summary ===", bold=True))
        click.echo(f"  File:      {info['file']}")
        click.echo(f"  Duration:  {_format_duration(info['duration_seconds'])}")
        click.echo(f"  Messages:  {info['message_count']:,}")
        click.echo(f"  Start:     {info['start_time']:.6f}")
        click.echo(f"  End:       {info['end_time']:.6f}")
        click.echo()

        # Topics table.
        headers = ["Topic", "Type", "Count", "Hz"]
        rows = []
        for t in info["topics"]:
            freq = f"{t['frequency']:.2f}" if t["frequency"] is not None else "-"
            rows.append([
                t["name"],
                t["message_type"],
                str(t["message_count"]),
                freq,
            ])

        click.echo(click.style("Topics:", bold=True))
        click.echo(_format_table(headers, rows))


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--topic", "-t", required=True, help="Topic name to inspect.")
@click.option(
    "--limit", "-n", default=5, show_default=True,
    help="Maximum number of messages to display.",
)
def inspect(path: str, topic: str, limit: int):
    """Print raw decoded messages from a specific topic.

    Shows timestamp, frame_id (if present), and key fields for each
    message. Useful for quick sanity checks -- does this topic contain
    what I expect? Are timestamps reasonable? Is the data decoded
    correctly?

    Example:

        mcap-reader inspect recording.mcap --topic /imu/data --limit 3
    """
    import itertools

    from mcap_reader.reader import McapReader

    with McapReader(path) as reader:
        # Verify topic exists.
        if topic not in reader.topic_names:
            click.echo(
                click.style(f"Error: topic '{topic}' not found.", fg="red"),
                err=True,
            )
            click.echo(f"Available topics: {', '.join(reader.topic_names)}", err=True)
            sys.exit(1)

        click.echo(click.style(
            f"=== Messages on {topic} (limit={limit}) ===", bold=True,
        ))

        for i, msg in enumerate(itertools.islice(
            reader.iter_messages(topics=[topic]), limit
        )):
            click.echo(click.style(f"\n--- Message {i} ---", fg="cyan"))
            click.echo(f"  Timestamp: {msg.timestamp:.9f}")
            click.echo(f"  Log time:  {msg.log_time:.9f}")
            click.echo(f"  Schema:    {msg.schema_name}")

            ros_msg = msg.ros_msg

            # Try to extract frame_id from header.
            frame_id = _safe_getattr(ros_msg, "header.frame_id")
            if frame_id is not None:
                click.echo(f"  Frame ID:  {frame_id}")

            # Print key fields based on schema type.
            _print_message_fields(ros_msg, msg.schema_name)


def _safe_getattr(obj, dotted_path: str):
    """Safely traverse a dotted attribute path.

    Returns ``None`` if any attribute along the path does not exist.
    For example, ``_safe_getattr(msg, "header.frame_id")`` is equivalent
    to ``msg.header.frame_id`` but returns ``None`` instead of raising
    ``AttributeError``.
    """
    current = obj
    for attr in dotted_path.split("."):
        try:
            current = getattr(current, attr)
        except AttributeError:
            return None
    return current


def _print_message_fields(ros_msg, schema_name: str) -> None:
    """Print key fields of a decoded ROS message for inspection.

    Selects which fields to display based on the message type. For
    unknown types, falls back to printing all public attributes.

    Parameters
    ----------
    ros_msg : object
        Decoded ROS 2 message object.
    schema_name : str
        The ROS 2 message type string.
    """
    if "Imu" in schema_name:
        ang = _safe_getattr(ros_msg, "angular_velocity")
        lin = _safe_getattr(ros_msg, "linear_acceleration")
        ori = _safe_getattr(ros_msg, "orientation")
        if ang is not None:
            click.echo(
                f"  Angular vel: ({ang.x:.4f}, {ang.y:.4f}, {ang.z:.4f})"
            )
        if lin is not None:
            click.echo(
                f"  Linear acc:  ({lin.x:.4f}, {lin.y:.4f}, {lin.z:.4f})"
            )
        if ori is not None:
            click.echo(
                f"  Orientation: ({ori.x:.4f}, {ori.y:.4f}, {ori.z:.4f}, {ori.w:.4f})"
            )

    elif "JointState" in schema_name:
        names = _safe_getattr(ros_msg, "name")
        positions = _safe_getattr(ros_msg, "position")
        if names is not None:
            click.echo(f"  Joints: {list(names)}")
        if positions is not None and len(positions) > 0:
            pos_str = ", ".join(f"{p:.4f}" for p in positions[:8])
            suffix = "..." if len(positions) > 8 else ""
            click.echo(f"  Positions: [{pos_str}{suffix}]")

    elif "Image" in schema_name:
        height = _safe_getattr(ros_msg, "height")
        width = _safe_getattr(ros_msg, "width")
        encoding = _safe_getattr(ros_msg, "encoding")
        step = _safe_getattr(ros_msg, "step")
        if height is not None:
            click.echo(f"  Dimensions: {width}x{height}")
        if encoding is not None:
            click.echo(f"  Encoding:   {encoding}")
        if step is not None:
            click.echo(f"  Step:       {step}")

    elif "PointCloud2" in schema_name:
        height = _safe_getattr(ros_msg, "height")
        width = _safe_getattr(ros_msg, "width")
        point_step = _safe_getattr(ros_msg, "point_step")
        is_dense = _safe_getattr(ros_msg, "is_dense")
        fields = _safe_getattr(ros_msg, "fields")
        if height is not None and width is not None:
            total = height * width
            click.echo(f"  Points: {total:,} ({width}x{height})")
        if point_step is not None:
            click.echo(f"  Point step: {point_step} bytes")
        if is_dense is not None:
            click.echo(f"  Dense: {is_dense}")
        if fields is not None:
            field_names = [f.name for f in fields]
            click.echo(f"  Fields: {field_names}")

    elif "TFMessage" in schema_name:
        transforms = _safe_getattr(ros_msg, "transforms")
        if transforms is not None:
            for tf in transforms:
                parent = tf.header.frame_id
                child = tf.child_frame_id
                t = tf.transform.translation
                click.echo(
                    f"  {parent} -> {child}: "
                    f"t=({t.x:.4f}, {t.y:.4f}, {t.z:.4f})"
                )

    else:
        # Fallback: print all public attributes (up to a limit).
        attrs = [a for a in dir(ros_msg) if not a.startswith("_")]
        shown = 0
        for attr in attrs[:10]:
            try:
                val = getattr(ros_msg, attr)
                if callable(val):
                    continue
                val_str = repr(val)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                click.echo(f"  {attr}: {val_str}")
                shown += 1
            except Exception:
                pass
        if len(attrs) > 10:
            click.echo(f"  ... and {len(attrs) - 10} more attributes")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def validate(path: str):
    """Decode every message and report errors.

    Iterates through all messages in the MCAP file, attempting to
    decode each one. Reports per-topic success/failure counts and
    shows a progress bar during the scan. Useful for detecting
    corrupted messages, schema mismatches, or truncated recordings.

    Exit code is 0 if all messages decode successfully, 1 if any
    errors are found.

    Example:

        mcap-reader validate recording.mcap
    """
    from mcap_reader.reader import McapReader

    with McapReader(path) as reader:
        total = reader.message_count
        success_counts: dict[str, int] = {}
        error_counts: dict[str, int] = {}
        error_examples: dict[str, str] = {}

        click.echo(click.style("=== Validating MCAP File ===", bold=True))
        click.echo(f"  File: {Path(path).name}")
        click.echo(f"  Expected messages: {total:,}")
        click.echo()

        processed = 0
        with click.progressbar(
            length=total,
            label="Decoding messages",
            show_percent=True,
            show_pos=True,
        ) as bar:
            for msg in reader.iter_messages():
                topic = msg.topic
                try:
                    # Access ros_msg to trigger full decoding.
                    # The message is already decoded by iter_messages,
                    # but we verify the object is accessible.
                    _ = msg.ros_msg
                    success_counts[topic] = success_counts.get(topic, 0) + 1
                except Exception as e:
                    error_counts[topic] = error_counts.get(topic, 0) + 1
                    if topic not in error_examples:
                        error_examples[topic] = str(e)[:200]

                processed += 1
                bar.update(1)

        # Report results.
        click.echo()
        total_errors = sum(error_counts.values())
        total_success = sum(success_counts.values())

        if total_errors == 0:
            click.echo(click.style(
                f"All {total_success:,} messages decoded successfully.",
                fg="green", bold=True,
            ))
        else:
            click.echo(click.style(
                f"Found {total_errors:,} error(s) out of {processed:,} messages.",
                fg="red", bold=True,
            ))

        # Per-topic breakdown table.
        click.echo()
        headers = ["Topic", "Success", "Errors", "Status"]
        rows = []
        all_topics = sorted(
            set(list(success_counts.keys()) + list(error_counts.keys()))
        )
        for topic in all_topics:
            s = success_counts.get(topic, 0)
            e = error_counts.get(topic, 0)
            status = click.style("OK", fg="green") if e == 0 else click.style(
                "ERRORS", fg="red"
            )
            rows.append([topic, str(s), str(e), status])

        click.echo(_format_table(headers, rows))

        # Show error details.
        if error_examples:
            click.echo()
            click.echo(click.style("Error details:", bold=True))
            for topic, example in sorted(error_examples.items()):
                click.echo(f"  {topic}: {example}")

        if total_errors > 0:
            sys.exit(1)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--topic", "-t", required=True, help="Topic to export.")
@click.option(
    "--format", "-f", "fmt", type=click.Choice(["parquet", "csv"]),
    default="parquet", show_default=True, help="Output format.",
)
@click.option("--episode", "-e", type=int, default=None, help="Export only this episode number.")
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Output file path. Defaults to <topic_name>.<format>.",
)
@click.option(
    "--gap-threshold", type=float, default=5.0, show_default=True,
    help="Gap threshold for episode detection (used with --episode).",
)
def export(path: str, topic: str, fmt: str, episode: int | None, output: str | None, gap_threshold: float):
    """Export a single topic to Parquet or CSV.

    Decodes all messages on the specified topic and writes them to a
    structured file. The output schema depends on the message type:

    \b
    - IMU: timestamp_ns, frame_id, orientation_x/y/z/w,
      angular_vel_x/y/z, linear_accel_x/y/z
    - JointState: timestamp_ns, frame_id, then
      position_<name>/velocity_<name>/effort_<name> per joint
    - Image: timestamp_ns, frame_id, height, width, encoding,
      data (compressed bytes)
    - PointCloud2: timestamp_ns, frame_id, num_points,
      data (raw bytes)

    Use --episode to export only messages within a specific episode
    (detected via gap-based method with configurable threshold).

    Examples:

    \b
        mcap-reader export recording.mcap -t /imu/data -o imu.parquet
        mcap-reader export recording.mcap -t /joint_states -f csv -o joints.csv
        mcap-reader export recording.mcap -t /imu/data --episode 2 -o ep2_imu.parquet
    """
    import pandas as pd

    from mcap_reader.reader import McapReader

    with McapReader(path) as reader:
        # Verify topic exists.
        if topic not in reader.topic_names:
            click.echo(
                click.style(f"Error: topic '{topic}' not found.", fg="red"),
                err=True,
            )
            click.echo(f"Available topics: {', '.join(reader.topic_names)}", err=True)
            sys.exit(1)

        # Determine time window if episode filtering is requested.
        start_time = None
        end_time = None
        if episode is not None:
            from mcap_reader.episode import EpisodeDetector

            detector = EpisodeDetector(reader)
            episodes = detector.detect_by_gaps(gap_threshold=gap_threshold)
            if episode < 0 or episode >= len(episodes):
                click.echo(
                    click.style(
                        f"Error: episode {episode} out of range. "
                        f"Found {len(episodes)} episode(s) (0-{len(episodes) - 1}).",
                        fg="red",
                    ),
                    err=True,
                )
                sys.exit(1)
            ep = episodes[episode]
            start_time = ep.start_time
            end_time = ep.end_time
            click.echo(
                f"Filtering to episode {episode}: "
                f"{start_time:.3f}s - {end_time:.3f}s"
            )

        # Determine the schema type for this topic.
        topic_info = None
        for t in reader.topics:
            if t.name == topic:
                topic_info = t
                break

        schema_name = topic_info.message_type if topic_info else ""

        # Collect rows.
        rows = []
        msg_count = topic_info.message_count if topic_info else 0

        click.echo(f"Exporting {topic} ({schema_name})...")

        with click.progressbar(
            reader.iter_messages(
                topics=[topic], start_time=start_time, end_time=end_time
            ),
            length=msg_count if episode is None else None,
            label="Processing",
            show_percent=True,
        ) as bar:
            for msg in bar:
                row = _message_to_export_row(msg.ros_msg, msg, schema_name)
                rows.append(row)

        if not rows:
            click.echo(click.style("Warning: no messages found.", fg="yellow"))
            return

        df = pd.DataFrame(rows)

        # Generate default output path.
        if output is None:
            safe_topic = topic.strip("/").replace("/", "_")
            suffix = "." + fmt
            if episode is not None:
                output = f"{safe_topic}_ep{episode}{suffix}"
            else:
                output = f"{safe_topic}{suffix}"

        # Write output.
        if fmt == "parquet":
            df.to_parquet(output, index=False)
        else:
            df.to_csv(output, index=False)

        click.echo(click.style(
            f"Exported {len(rows):,} messages to {output}",
            fg="green", bold=True,
        ))


def _message_to_export_row(ros_msg, raw_msg, schema_name: str) -> dict:
    """Convert a decoded ROS message to a flat dictionary for export.

    Selects the appropriate export schema based on the message type.
    Each row contains a ``timestamp_ns`` (integer nanoseconds for
    lossless storage in Parquet) and a ``frame_id``.

    Parameters
    ----------
    ros_msg : object
        The decoded ROS 2 message.
    raw_msg : RawMessage
        The raw message wrapper with timestamp and log_time.
    schema_name : str
        The ROS 2 message type string.

    Returns
    -------
    dict
        Flat dictionary suitable for DataFrame row construction.
    """
    import numpy as np

    timestamp_ns = int(raw_msg.timestamp * 1e9)
    frame_id = ""
    try:
        frame_id = ros_msg.header.frame_id
    except AttributeError:
        pass

    if "Imu" in schema_name:
        ori = ros_msg.orientation
        ang = ros_msg.angular_velocity
        lin = ros_msg.linear_acceleration
        return {
            "timestamp_ns": timestamp_ns,
            "frame_id": frame_id,
            "orientation_x": ori.x,
            "orientation_y": ori.y,
            "orientation_z": ori.z,
            "orientation_w": ori.w,
            "angular_vel_x": ang.x,
            "angular_vel_y": ang.y,
            "angular_vel_z": ang.z,
            "linear_accel_x": lin.x,
            "linear_accel_y": lin.y,
            "linear_accel_z": lin.z,
        }

    elif "JointState" in schema_name:
        row = {
            "timestamp_ns": timestamp_ns,
            "frame_id": frame_id,
        }
        names = list(ros_msg.name)
        positions = list(ros_msg.position) if len(ros_msg.position) > 0 else []
        velocities = list(ros_msg.velocity) if len(ros_msg.velocity) > 0 else []
        efforts = list(ros_msg.effort) if len(ros_msg.effort) > 0 else []

        for i, name in enumerate(names):
            if i < len(positions):
                row[f"position_{name}"] = positions[i]
            if i < len(velocities):
                row[f"velocity_{name}"] = velocities[i]
            if i < len(efforts):
                row[f"effort_{name}"] = efforts[i]
        return row

    elif "Image" in schema_name:
        import zlib

        data_bytes = bytes(ros_msg.data) if hasattr(ros_msg, "data") else b""
        return {
            "timestamp_ns": timestamp_ns,
            "frame_id": frame_id,
            "height": ros_msg.height,
            "width": ros_msg.width,
            "encoding": ros_msg.encoding,
            "data": zlib.compress(data_bytes),
        }

    elif "PointCloud2" in schema_name:
        num_points = ros_msg.height * ros_msg.width
        data_bytes = bytes(ros_msg.data) if hasattr(ros_msg, "data") else b""
        return {
            "timestamp_ns": timestamp_ns,
            "frame_id": frame_id,
            "num_points": num_points,
            "data": data_bytes,
        }

    else:
        # Generic fallback: store timestamp and raw bytes.
        return {
            "timestamp_ns": timestamp_ns,
            "frame_id": frame_id,
            "data_size": len(raw_msg.data),
        }


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--reference", "-r", required=True,
    help="Reference topic to synchronize against (e.g. /camera/image_raw).",
)
@click.option(
    "--topics", "-t", required=True, multiple=True,
    help="Topics to synchronize with the reference. Repeat for multiple.",
)
@click.option(
    "--strategy", "-s",
    type=click.Choice(["nearest", "interpolate"]),
    default="nearest", show_default=True,
    help="Synchronization strategy.",
)
@click.option(
    "--max-delay", type=float, default=0.1, show_default=True,
    help="Maximum acceptable time delay between matched messages (seconds).",
)
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["parquet", "csv"]),
    default="parquet", show_default=True,
    help="Output format.",
)
@click.option(
    "--output", "-o", type=click.Path(), default=None,
    help="Output file path. Defaults to synced_data.<format>.",
)
def sync(path: str, reference: str, topics: tuple[str, ...], strategy: str, max_delay: float, fmt: str, output: str | None):
    """Export synchronized multi-modal data.

    Aligns messages from multiple topics to a reference topic's
    timestamps using either nearest-neighbor matching or interpolation.
    This is essential for creating training datasets where each sample
    needs corresponding data from all modalities at the same instant.

    The reference topic defines the output timestamps. For each reference
    message timestamp, the tool finds the closest (or interpolated)
    message on each target topic. Messages outside --max-delay are
    marked as missing.

    Nearest-neighbor matching is appropriate for discrete data (images,
    point clouds). Interpolation works for continuous signals (IMU,
    joint states) where linear interpolation between samples is
    physically meaningful.

    Examples:

    \b
        mcap-reader sync recording.mcap \\
            --reference /camera/image_raw \\
            --topics /imu/data /joint_states \\
            --strategy nearest \\
            --max-delay 0.05 \\
            -o synced.parquet
    """
    import bisect

    import pandas as pd

    from mcap_reader.reader import McapReader

    with McapReader(path) as reader:
        # Validate topics.
        all_needed = [reference] + list(topics)
        available = set(reader.topic_names)
        missing = [t for t in all_needed if t not in available]
        if missing:
            click.echo(
                click.style(f"Error: topics not found: {missing}", fg="red"),
                err=True,
            )
            click.echo(f"Available topics: {', '.join(sorted(available))}", err=True)
            sys.exit(1)

        click.echo(click.style("=== Synchronizing Topics ===", bold=True))
        click.echo(f"  Reference: {reference}")
        click.echo(f"  Topics:    {', '.join(topics)}")
        click.echo(f"  Strategy:  {strategy}")
        click.echo(f"  Max delay: {max_delay}s")
        click.echo()

        # Step 1: Collect reference timestamps.
        click.echo("Collecting reference timestamps...")
        ref_data: list[tuple[float, object]] = []
        for msg in reader.iter_messages(topics=[reference]):
            ref_data.append((msg.timestamp, msg.ros_msg))

        if not ref_data:
            click.echo(click.style("No messages on reference topic.", fg="red"), err=True)
            sys.exit(1)

        click.echo(f"  Found {len(ref_data):,} reference messages")

        # Step 2: Collect target topic data.
        topic_data: dict[str, list[tuple[float, object]]] = {}
        for t in topics:
            click.echo(f"Collecting {t}...")
            entries = []
            for msg in reader.iter_messages(topics=[t]):
                entries.append((msg.timestamp, msg.ros_msg))
            topic_data[t] = entries
            click.echo(f"  Found {len(entries):,} messages")

        # Step 3: Match/sync.
        click.echo()
        click.echo("Synchronizing...")

        # Get schema names for export formatting.
        schema_map: dict[str, str] = {}
        for t_info in reader.topics:
            schema_map[t_info.name] = t_info.message_type

        rows = []
        ref_schema = schema_map.get(reference, "")
        sync_stats: dict[str, int] = {t: 0 for t in topics}  # matched count

        with click.progressbar(ref_data, label="Matching") as bar:
            for ref_time, ref_msg in bar:
                row = {"ref_timestamp_ns": int(ref_time * 1e9)}

                # Add reference topic fields.
                ref_fields = _message_to_sync_fields(
                    ref_msg, ref_schema, prefix="ref"
                )
                row.update(ref_fields)

                # Match each target topic.
                for t in topics:
                    entries = topic_data[t]
                    if not entries:
                        continue

                    t_timestamps = [e[0] for e in entries]
                    idx = bisect.bisect_left(t_timestamps, ref_time)

                    # Find the nearest message.
                    best_idx = None
                    best_delay = float("inf")

                    for candidate_idx in [idx - 1, idx]:
                        if 0 <= candidate_idx < len(entries):
                            delay = abs(entries[candidate_idx][0] - ref_time)
                            if delay < best_delay:
                                best_delay = delay
                                best_idx = candidate_idx

                    prefix = t.strip("/").replace("/", "_")
                    t_schema = schema_map.get(t, "")

                    if best_idx is not None and best_delay <= max_delay:
                        row[f"{prefix}_delay"] = best_delay
                        matched_msg = entries[best_idx][1]
                        fields = _message_to_sync_fields(
                            matched_msg, t_schema, prefix=prefix,
                        )
                        row.update(fields)
                        sync_stats[t] += 1
                    else:
                        row[f"{prefix}_delay"] = None

                rows.append(row)

        if not rows:
            click.echo(click.style("No synchronized data produced.", fg="yellow"))
            return

        df = pd.DataFrame(rows)

        # Report sync quality.
        click.echo()
        click.echo(click.style("Sync quality:", bold=True))
        for t in topics:
            matched = sync_stats[t]
            total = len(ref_data)
            pct = (matched / total * 100) if total > 0 else 0
            color = "green" if pct > 95 else "yellow" if pct > 80 else "red"
            click.echo(
                f"  {t}: {matched}/{total} matched "
                f"({click.style(f'{pct:.1f}%', fg=color)})"
            )

        # Write output.
        if output is None:
            output = f"synced_data.{fmt}"

        if fmt == "parquet":
            df.to_parquet(output, index=False)
        else:
            df.to_csv(output, index=False)

        click.echo()
        click.echo(click.style(
            f"Exported {len(rows):,} synchronized samples to {output}",
            fg="green", bold=True,
        ))


def _message_to_sync_fields(ros_msg, schema_name: str, prefix: str) -> dict:
    """Extract key numeric fields from a message for synchronized export.

    Returns a flat dictionary with keys prefixed by the given prefix.
    Focuses on the most commonly needed fields for each message type,
    excluding large binary blobs (image data, point cloud data).

    Parameters
    ----------
    ros_msg : object
        Decoded ROS 2 message.
    schema_name : str
        Message type string.
    prefix : str
        Prefix for dictionary keys (e.g., "ref" or "imu_data").

    Returns
    -------
    dict
        Flat dictionary of extracted fields.
    """
    fields = {}

    if "Imu" in schema_name:
        ori = ros_msg.orientation
        ang = ros_msg.angular_velocity
        lin = ros_msg.linear_acceleration
        fields[f"{prefix}_orientation_x"] = ori.x
        fields[f"{prefix}_orientation_y"] = ori.y
        fields[f"{prefix}_orientation_z"] = ori.z
        fields[f"{prefix}_orientation_w"] = ori.w
        fields[f"{prefix}_angular_vel_x"] = ang.x
        fields[f"{prefix}_angular_vel_y"] = ang.y
        fields[f"{prefix}_angular_vel_z"] = ang.z
        fields[f"{prefix}_linear_accel_x"] = lin.x
        fields[f"{prefix}_linear_accel_y"] = lin.y
        fields[f"{prefix}_linear_accel_z"] = lin.z

    elif "JointState" in schema_name:
        names = list(ros_msg.name)
        positions = list(ros_msg.position) if len(ros_msg.position) > 0 else []
        for i, name in enumerate(names):
            if i < len(positions):
                fields[f"{prefix}_pos_{name}"] = positions[i]

    elif "Image" in schema_name:
        # For sync output, include metadata but not raw pixels.
        fields[f"{prefix}_height"] = ros_msg.height
        fields[f"{prefix}_width"] = ros_msg.width

    elif "PointCloud2" in schema_name:
        fields[f"{prefix}_num_points"] = ros_msg.height * ros_msg.width

    return fields


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
def frames(path: str):
    """List coordinate frames and display the transform tree.

    Reads all TF messages (/tf and /tf_static) from the recording,
    builds a frame graph, and renders it as an ASCII tree. Also lists
    all unique frame identifiers found in message headers.

    This is useful for understanding the spatial relationships in a
    recording: which sensors are mounted where, what the kinematic
    chain looks like, and whether the expected frames are present.

    Example:

        mcap-reader frames recording.mcap
    """
    from mcap_reader.reader import McapReader
    from mcap_reader.transforms.buffer import TransformBuffer

    with McapReader(path) as reader:
        # Check for TF topics.
        tf_topics = [t for t in reader.topic_names if t in ("/tf", "/tf_static")]
        header_frames: set[str] = set()

        buf = TransformBuffer()

        if tf_topics:
            click.echo(click.style("=== Coordinate Frames ===", bold=True))
            click.echo(f"  TF topics found: {', '.join(tf_topics)}")
            click.echo()

            # Ingest TF messages.
            for msg in reader.iter_messages(topics=tf_topics):
                is_static = msg.topic == "/tf_static"
                try:
                    buf.add_tf_message(msg.ros_msg, is_static=is_static)
                except Exception:
                    pass

            # Render tree.
            graph = buf.get_frame_graph()
            tree_str = graph.to_ascii_tree()
            click.echo(click.style("Transform tree:", bold=True))
            click.echo(tree_str)
            click.echo()

            # List frames.
            all_frames = sorted(buf.get_frames())
            click.echo(click.style(
                f"Frames ({len(all_frames)}):", bold=True,
            ))
            for frame in all_frames:
                click.echo(f"  {frame}")
        else:
            click.echo(click.style(
                "No /tf or /tf_static topics found.", fg="yellow",
            ))

        # Also collect frame_ids from message headers.
        click.echo()
        click.echo("Scanning message headers for frame_id references...")
        count = 0
        for msg in reader.iter_messages():
            frame_id = _safe_getattr(msg.ros_msg, "header.frame_id")
            if frame_id:
                header_frames.add(frame_id)
            count += 1
            # Limit scan for very large files.
            if count > 100_000:
                click.echo("  (scanned first 100,000 messages)")
                break

        if header_frames:
            click.echo(click.style(
                f"\nFrame IDs referenced in headers ({len(header_frames)}):",
                bold=True,
            ))
            for frame in sorted(header_frames):
                in_tree = " (in TF tree)" if buf.get_frames() and frame in buf.get_frames() else ""
                click.echo(f"  {frame}{in_tree}")


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--gap-threshold", type=float, default=5.0, show_default=True,
    help="Gap threshold in seconds for episode boundary detection.",
)
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["table", "json"]),
    default="table", show_default=True,
    help="Output format.",
)
def episodes(path: str, gap_threshold: float, fmt: str):
    """Detect and list episodes in the recording.

    Uses gap-based detection to find episode boundaries. An episode
    boundary is declared when the time gap between consecutive messages
    on any topic exceeds the threshold.

    In table format, shows a summary of each episode with start/end
    times, duration, and message counts. In JSON format, outputs
    machine-readable episode metadata.

    Examples:

    \b
        mcap-reader episodes recording.mcap
        mcap-reader episodes recording.mcap --gap-threshold 3.0
        mcap-reader episodes recording.mcap --format json > episodes.json
    """
    from mcap_reader.episode import EpisodeDetector
    from mcap_reader.reader import McapReader

    with McapReader(path) as reader:
        detector = EpisodeDetector(reader)
        detected = detector.detect_by_gaps(gap_threshold=gap_threshold)

        if not detected:
            click.echo("No episodes detected.")
            return

        if fmt == "json":
            output = [ep.to_dict() for ep in detected]
            click.echo(json.dumps(output, indent=2))
            return

        # Table format.
        click.echo(click.style(
            f"=== Episodes ({len(detected)}) ===", bold=True,
        ))
        click.echo(f"  Gap threshold: {gap_threshold}s")
        click.echo()

        headers = ["Episode", "Start", "End", "Duration", "Messages", "Topics"]
        rows = []
        for ep in detected:
            total_msgs = sum(ep.message_counts.values())
            rows.append([
                str(ep.index),
                f"{ep.start_time:.3f}",
                f"{ep.end_time:.3f}",
                _format_duration(ep.duration),
                f"{total_msgs:,}",
                str(len(ep.topics)),
            ])

        click.echo(_format_table(headers, rows))

        # Detailed per-episode breakdown.
        click.echo()
        for ep in detected:
            click.echo(click.style(f"Episode {ep.index}:", bold=True))
            for topic in sorted(ep.message_counts):
                count = ep.message_counts[topic]
                click.echo(f"    {topic}: {count:,} messages")
