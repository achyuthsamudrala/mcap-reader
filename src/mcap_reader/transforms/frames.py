"""
Frame graph for finding paths between coordinate frames.

Scene graphs and kinematic chains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A robot's body is a tree of rigid links connected by joints.  Each link
lives in its own coordinate frame, and the relationship between adjacent
frames is described by a rigid-body transform.  This tree of frames is
called a **scene graph** (in graphics) or a **kinematic chain** (in
robotics).

For example, a typical mobile robot with a camera might have::

    map -> odom -> base_link -> camera_link -> camera_optical_frame

The root frame is usually one of:

- **world** or **map** — a fixed, global reference frame.
- **odom** — a local frame whose origin is where the robot started.
  Odometry is smooth but drifts over time, so ``map -> odom`` is
  periodically corrected by a localisation system (AMCL, SLAM, etc.).
- **base_link** — the robot's own body frame, centred between the
  drive wheels or at the IMU.

Why an undirected graph?
~~~~~~~~~~~~~~~~~~~~~~~~
TF2 transforms are published as directed parent -> child edges, but
lookups can go in *either* direction (the transform is simply inverted
when traversed backwards).  We therefore store the graph as an
**undirected** adjacency dict so that ``get_chain("camera_link",
"base_link")`` works just as well as the forward direction.

Why BFS for path finding?
~~~~~~~~~~~~~~~~~~~~~~~~~
The frame tree can have dozens of frames.  A chain from ``base_link`` to
``camera_optical_frame`` may traverse 4+ intermediate frames (e.g.
``base_link -> shoulder -> upper_arm -> wrist -> camera_mount ->
camera_optical_frame``).  Breadth-first search (BFS) finds the
**shortest** path in an unweighted graph, which is exactly what we want:
the fewest intermediate transforms to compose.

BFS runs in O(V + E) time, which is effectively free for the small
graphs found in robotics (rarely more than ~100 frames).
"""

from __future__ import annotations

from collections import deque


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class FrameNotFoundError(KeyError):
    """Raised when a requested frame does not exist in the graph.

    This is a subclass of ``KeyError`` so callers that already catch
    ``KeyError`` will handle it transparently, while callers that want
    finer-grained control can catch ``FrameNotFoundError`` specifically.

    Parameters
    ----------
    frame : str
        The name of the frame that was not found.
    """

    def __init__(self, frame: str) -> None:
        self.frame = frame
        super().__init__(f"Frame '{frame}' does not exist in the frame graph")


class NoPathError(RuntimeError):
    """Raised when no path exists between two frames.

    This can happen when the frame graph has disconnected components —
    for example, if transforms for two independent robots are loaded
    into the same buffer but share no common ancestor frame.

    Parameters
    ----------
    source : str
        The starting frame.
    target : str
        The destination frame.
    """

    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        super().__init__(
            f"No path from '{source}' to '{target}' in the frame graph"
        )


# ---------------------------------------------------------------------------
# FrameGraph
# ---------------------------------------------------------------------------

class FrameGraph:
    """An undirected graph of coordinate frames with BFS path finding.

    The graph is stored as an adjacency dict::

        {
            "map":       {"odom"},
            "odom":      {"map", "base_link"},
            "base_link": {"odom", "camera_link"},
            ...
        }

    Each call to :meth:`add_edge` records both directions, so lookups
    work regardless of whether the edge was added as ``(parent, child)``
    or ``(child, parent)``.

    We additionally maintain a directed ``_parent`` mapping so that
    :meth:`get_parent` and :meth:`get_children` can answer tree-style
    queries.  This reflects the original parent -> child direction
    specified in the TF messages.

    Thread safety
    ~~~~~~~~~~~~~
    This class is **not** thread-safe.  If multiple threads add edges
    and query paths concurrently, the caller must provide external
    synchronisation.
    """

    def __init__(self) -> None:
        # Undirected adjacency dict for BFS path finding.
        self._adj: dict[str, set[str]] = {}

        # Directed parent mapping for tree-style queries.
        # _parent[child] = parent — records the original TF direction.
        self._parent: dict[str, str] = {}

    # -- Mutators -----------------------------------------------------------

    def add_edge(self, parent: str, child: str) -> None:
        """Add a frame relationship (parent -> child).

        Both frames are added to the graph if they do not already exist.
        The undirected adjacency dict is updated in both directions, and
        the directed parent mapping records ``parent`` as the parent of
        ``child``.

        Parameters
        ----------
        parent : str
            The parent frame identifier (e.g. ``"base_link"``).
        child : str
            The child frame identifier (e.g. ``"camera_link"``).
        """
        self._adj.setdefault(parent, set()).add(child)
        self._adj.setdefault(child, set()).add(parent)
        self._parent[child] = parent

    # -- Queries ------------------------------------------------------------

    def has_frame(self, frame: str) -> bool:
        """Return ``True`` if *frame* exists in the graph.

        Parameters
        ----------
        frame : str
            The frame identifier to look up.
        """
        return frame in self._adj

    def get_chain(self, source: str, target: str) -> list[str]:
        """Find the shortest path from *source* to *target* using BFS.

        The returned list starts with *source* and ends with *target*,
        with all intermediate frames in order.  For example::

            >>> graph.get_chain("map", "camera_optical_frame")
            ["map", "odom", "base_link", "camera_link", "camera_optical_frame"]

        Why BFS?
        ~~~~~~~~
        In an unweighted graph, BFS is guaranteed to find the shortest
        path (fewest edges).  This is important because each edge in the
        path corresponds to a transform that must be looked up and
        composed.  Fewer edges means fewer lookups, fewer matrix
        multiplications, and less accumulated numerical error.

        Parameters
        ----------
        source : str
            The starting frame.
        target : str
            The destination frame.

        Returns
        -------
        list[str]
            Ordered list of frames from *source* to *target*, inclusive.

        Raises
        ------
        FrameNotFoundError
            If *source* or *target* is not in the graph.
        NoPathError
            If no path exists between the two frames (disconnected
            components).
        """
        if not self.has_frame(source):
            raise FrameNotFoundError(source)
        if not self.has_frame(target):
            raise FrameNotFoundError(target)

        if source == target:
            return [source]

        # Standard BFS with parent tracking for path reconstruction.
        visited: dict[str, str | None] = {source: None}
        queue: deque[str] = deque([source])

        while queue:
            current = queue.popleft()
            for neighbour in self._adj.get(current, set()):
                if neighbour not in visited:
                    visited[neighbour] = current
                    if neighbour == target:
                        # Reconstruct path by walking the parent chain.
                        path: list[str] = []
                        node: str | None = target
                        while node is not None:
                            path.append(node)
                            node = visited[node]
                        path.reverse()
                        return path
                    queue.append(neighbour)

        raise NoPathError(source, target)

    def get_parent(self, frame: str) -> str | None:
        """Return the parent of *frame*, or ``None`` if it is a root.

        The parent relationship reflects the original direction given in
        :meth:`add_edge`.  Root frames (like ``"map"`` or ``"world"``)
        have no parent.

        Parameters
        ----------
        frame : str
            The frame to query.

        Returns
        -------
        str or None
            The parent frame identifier, or ``None`` if *frame* is a
            root or is not in the graph.
        """
        return self._parent.get(frame)

    def get_children(self, frame: str) -> set[str]:
        """Return the set of direct children of *frame*.

        Children are frames for which *frame* was specified as the
        parent in :meth:`add_edge`.

        Parameters
        ----------
        frame : str
            The frame to query.

        Returns
        -------
        set[str]
            The child frame identifiers.  May be empty.
        """
        return {child for child, parent in self._parent.items() if parent == frame}

    def all_frames(self) -> set[str]:
        """Return the set of all frame identifiers in the graph.

        Returns
        -------
        set[str]
            All known frame names.
        """
        return set(self._adj.keys())

    # -- Visualisation ------------------------------------------------------

    def to_ascii_tree(self, root: str | None = None) -> str:
        """Render the frame graph as an ASCII tree for CLI display.

        If *root* is ``None``, the method auto-detects root frames
        (frames with no parent in the directed parent mapping).  If
        there are multiple roots, each sub-tree is rendered separately.

        The output looks like::

            map
            +-- odom
            |   +-- base_link
            |       +-- camera_link
            |           +-- camera_optical_frame
            +-- gps_frame

        Parameters
        ----------
        root : str or None
            The frame to use as the tree root.  If ``None``, all root
            frames are discovered automatically.

        Returns
        -------
        str
            A multi-line ASCII tree string.

        Raises
        ------
        FrameNotFoundError
            If *root* is specified but does not exist in the graph.
        """
        if root is not None:
            if not self.has_frame(root):
                raise FrameNotFoundError(root)
            return self._render_subtree(root, prefix="")

        # Auto-detect roots: frames that have no parent.
        roots = sorted(
            frame for frame in self._adj if frame not in self._parent
        )
        if not roots:
            # All frames have parents — pick the first alphabetically
            # as a fallback (cycle or fully connected).
            roots = sorted(self._adj.keys())[:1]

        if not roots:
            return "(empty frame graph)"

        parts: list[str] = []
        for r in roots:
            parts.append(self._render_subtree(r, prefix=""))
        return "\n".join(parts)

    def _render_subtree(self, frame: str, prefix: str) -> str:
        """Recursively render a subtree rooted at *frame*.

        Parameters
        ----------
        frame : str
            The current node.
        prefix : str
            Indentation prefix accumulated from parent levels.

        Returns
        -------
        str
            Multi-line ASCII art for this subtree.
        """
        lines: list[str] = [f"{prefix}{frame}"]
        children = sorted(self.get_children(frame))
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            connector = "+-- " if not is_last else "+-- "
            child_prefix = prefix + ("|   " if not is_last else "    ")
            child_tree = self._render_subtree(child, prefix="")
            # Indent the child subtree.
            child_lines = child_tree.split("\n")
            lines.append(f"{prefix}{connector}{child_lines[0]}")
            for cl in child_lines[1:]:
                lines.append(f"{child_prefix}{cl}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        n_frames = len(self._adj)
        n_edges = len(self._parent)
        return f"FrameGraph(frames={n_frames}, edges={n_edges})"
