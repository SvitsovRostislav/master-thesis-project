"""Editable mesh panel for the digital-twin editor.

Four interaction modes, chosen by the host window via `set_mode()`:

- ``"orbit"`` (default): plain camera orbit/zoom/pan, no editing.
- ``"sculpt"``: click-drag smoothly deforms a radius-weighted neighborhood
  of vertices, with a cosine falloff -- a small sculpting-style brush,
  rather than single raw-vertex editing (impractical on the dense meshes
  SuGaR produces).
- ``"select"``: drag a screen-space rectangle (or, holding Shift, a
  freeform lasso) to select a region of vertices, then click-drag *inside*
  the highlighted selection to move it rigidly (no falloff -- the whole
  selection translates together). The host window reads
  `selected_vertices` to enable/disable a "Delete Selected" action; actual
  deletion happens one level up (`digital_twin.py`), since it also has to
  touch the Gaussian bindings.
- ``"pick"``: click a vertex to select its whole mesh-connectivity
  component (flood-fill along mesh edges, stopping at real gaps in the
  surface) in one click, instead of manually drawing a rectangle/lasso
  around it -- e.g. a cup sitting on (not fused to) a table is a separate
  connected component, so one click grabs exactly the cup. Once something
  is selected, click-drag on it to move it rigidly -- this reuses
  "select"'s own move handling verbatim (`_try_start_move`), since moving
  a selection works identically no matter which tool made it.

Vertex picking and dragging are done entirely from the *screen-space
projection* of the mesh (`node.transforms.get_transform(..., map_to="canvas")`),
not by reasoning about the camera's internal axes -- this keeps the code
correct regardless of the current camera orientation/zoom. Both the sculpt
brush and the rigid move solve for the world-space movement whose
*projected* screen displacement matches the mouse, via the pseudo-inverse
of a small numerical Jacobian (2x3): this is exactly "move within the
screen-facing plane", without hand-deriving that plane from camera
internals.

Rectangle/lasso selection is a *screen-space* test (every vertex whose
projection falls inside the drawn region is selected, regardless of
depth/occlusion) -- there is no z-buffer/visible-surface check. That is a
deliberate simplification, consistent with how the sculpt brush already
picks in screen space.
"""

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from vispy import scene
from vispy.util import keys

# Brush radius as a fraction of the mesh's bounding-box diagonal; retuned per
# mesh in load_mesh() so it scales with whatever scene is loaded.
DEFAULT_BRUSH_RADIUS_FRACTION = 0.08
PICK_RADIUS_PIXELS = 25.0
# Minimum screen-space spacing between consecutive recorded lasso points --
# keeps the accumulated path (and the point-in-polygon test) small even
# during a slow, jittery drag.
LASSO_MIN_POINT_SPACING_PIXELS = 4.0

BASE_MESH_COLOR = (0.65, 0.68, 0.75, 1.0)
SELECTED_VERTEX_COLOR = (1.0, 0.55, 0.1, 1.0)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray-casting point-in-polygon test.

    `points` is (N, 2), `polygon` is (M, 2) (an open ring -- the edge from
    the last vertex back to the first is implied). Returns an (N,) bool
    mask. Plain numpy, no new dependency for what is otherwise a two-line
    algorithm.
    """
    x, y = points[:, 0], points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    n = polygon.shape[0]
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Guard against a horizontal edge (yj == yi) causing a division by zero.
        denom = (yj - yi) if (yj - yi) != 0 else 1e-12
        crosses = (yi > y) != (yj > y)
        x_intersect = (xj - xi) * (y - yi) / denom + xi
        inside ^= crosses & (x < x_intersect)
        j = i
    return inside


def _build_vertex_adjacency(faces: np.ndarray, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR-style vertex adjacency from a triangle mesh: vertex `i`'s
    neighbors are `adj_indices[adj_ptr[i]:adj_ptr[i+1]]`. Plain numpy (no
    scipy dependency) -- built once per mesh load, reused by every "pick"
    click via `_connected_component`.
    """
    e = faces
    edges = np.concatenate(
        [e[:, [0, 1]], e[:, [1, 0]], e[:, [1, 2]], e[:, [2, 1]], e[:, [2, 0]], e[:, [0, 2]]], axis=0
    )
    order = np.argsort(edges[:, 0], kind="stable")
    edges = edges[order]
    src = edges[:, 0]
    adj_indices = edges[:, 1].astype(np.int64)
    adj_ptr = np.searchsorted(src, np.arange(n_vertices + 1)).astype(np.int64)
    return adj_ptr, adj_indices


def _neighbors_of(vertices: np.ndarray, adj_ptr: np.ndarray, adj_indices: np.ndarray) -> np.ndarray:
    """All neighbors of every vertex in `vertices` (a whole BFS frontier at
    once), via the standard numpy idiom for gathering variable-length CSR
    row slices without a Python-level loop over vertices."""
    starts = adj_ptr[vertices]
    ends = adj_ptr[vertices + 1]
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=adj_indices.dtype)
    offsets = np.repeat(starts, counts) + (np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts))
    return adj_indices[offsets]


def _connected_component(
    start_vertex: int, adj_ptr: np.ndarray, adj_indices: np.ndarray, n_vertices: int
) -> np.ndarray:
    """Every vertex reachable from `start_vertex` via mesh edges --
    breadth-first search, vectorized one whole frontier (BFS level) at a
    time rather than one vertex at a time, so it stays fast even on
    meshes with hundreds of thousands of vertices."""
    visited = np.zeros(n_vertices, dtype=bool)
    visited[start_vertex] = True
    frontier = np.array([start_vertex], dtype=np.int64)
    while frontier.size > 0:
        neighbors = _neighbors_of(frontier, adj_ptr, adj_indices)
        neighbors = np.unique(neighbors[~visited[neighbors]])
        visited[neighbors] = True
        frontier = neighbors
    return np.nonzero(visited)[0]


class _SelectionOverlay(QtWidgets.QWidget):
    """Transparent widget drawing the in-progress rectangle/lasso outline.

    A plain Qt child of the vispy canvas's native widget rather than a
    vispy visual: this sidesteps any ambiguity about the scenegraph's
    pixel-coordinate conventions for a 2D overlay atop a 3D view, since a
    Qt widget's own paint coordinates are, by construction, the same pixel
    space as the mouse events this module already works in. vispy's PyQt5
    backend renders into a QOpenGLWidget, which composites normally with a
    translucent child widget on top of it.
    """

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.kind: str | None = None  # "rect" | "lasso" | None
        self.path: list[tuple[float, float]] = []
        self.hide()

    def set_path(self, kind: str, path: list[tuple[float, float]]) -> None:
        self.kind = kind
        self.path = path
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self.update()

    def clear(self) -> None:
        self.kind = None
        self.path = []
        self.hide()

    def paintEvent(self, _event) -> None:
        if not self.kind or len(self.path) < 2:
            return
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 140, 25), 2))
        if self.kind == "rect":
            (x0, y0), (x1, y1) = self.path[0], self.path[-1]
            painter.drawRect(QtCore.QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))
        else:
            painter.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(x, y) for x, y in self.path]))


class MeshEditorPanel:
    """A vispy canvas showing an editable mesh."""

    def __init__(self, on_edited=None, on_selection_changed=None):
        self.canvas = scene.SceneCanvas(
            keys="interactive", size=(700, 720), bgcolor="#1c1c22"
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45.0)
        self.mesh = scene.visuals.Mesh(
            color=BASE_MESH_COLOR, shading="smooth", parent=self.view.scene
        )
        self._overlay = _SelectionOverlay(self.canvas.native)

        self.vertices: np.ndarray | None = None
        self.faces: np.ndarray | None = None
        self._vertex_colors: np.ndarray | None = None  # (V, 4), kept in sync so
        # the selection highlight survives every set_data() call (set_data
        # replaces the mesh's whole color state, it does not merge).
        self.mode = "orbit"  # "orbit" | "sculpt" | "select" | "pick"
        self.brush_radius = 0.2
        self.selected_vertices: np.ndarray | None = None  # (K,) int, or None
        self.on_edited = on_edited
        self.on_selection_changed = on_selection_changed
        self._adjacency: tuple[np.ndarray, np.ndarray] | None = None  # (adj_ptr, adj_indices), lazy/cached

        # Sculpt-drag state.
        self._drag_affected = None  # (K,) indices of vertices moving with this drag
        self._drag_weights = None  # (K,) falloff weight per affected vertex
        self._drag_original_positions = None  # (K, 3) positions at drag start
        self._drag_start_mouse = None  # (2,) screen position at drag start
        self._drag_jacobian_pinv = None  # (3, 2) pseudo-inverse, screen delta -> world delta

        # Select-mode drag state.
        self._select_drag_kind: str | None = None  # "rect" | "lasso" | "move" | None
        self._select_path: list[tuple[float, float]] = []  # accumulated screen points (rect/lasso)
        self._move_start_mouse = None
        self._move_original_positions = None  # (K, 3) selection's positions at drag start
        self._move_jacobian_pinv = None

        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)

    def load_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """Load (or replace) the displayed/editable mesh."""
        self.vertices = np.asarray(vertices, dtype=np.float32).copy()
        self.faces = np.asarray(faces, dtype=np.int32)
        self._vertex_colors = np.tile(
            np.array(BASE_MESH_COLOR, dtype=np.float32), (self.vertices.shape[0], 1)
        )
        self.selected_vertices = None
        self._adjacency = None
        self._overlay.clear()
        self._refresh_mesh()
        self.view.camera.set_range()

        extent = float(np.linalg.norm(self.vertices.max(axis=0) - self.vertices.min(axis=0)))
        self.brush_radius = max(extent * DEFAULT_BRUSH_RADIUS_FRACTION, 1e-6)

    def set_mode(self, mode: str) -> None:
        """Switch between camera-orbit (default), sculpt, and select.

        Plain left-drag already orbits the camera (TurntableCamera), so
        only one interaction can own the left mouse button at a time --
        `camera.interactive` is only True in "orbit" mode.
        """
        assert mode in ("orbit", "sculpt", "select", "pick")
        self.mode = mode
        self.view.camera.interactive = mode == "orbit"
        # Cancel anything in flight when switching tools mid-gesture.
        self._drag_affected = None
        self._select_drag_kind = None
        self._overlay.clear()

    def clear_selection(self) -> None:
        """Drop the current selection (called by the host window after a
        delete/undo/reload, whose new vertex indices make the old
        selection meaningless)."""
        self.selected_vertices = None
        if self._vertex_colors is not None:
            self._update_highlight()
        if self.on_selection_changed is not None:
            self.on_selection_changed(None)

    def _refresh_mesh(self) -> None:
        self.mesh.set_data(vertices=self.vertices, faces=self.faces, vertex_colors=self._vertex_colors)

    def _update_highlight(self) -> None:
        self._vertex_colors[:] = BASE_MESH_COLOR
        if self.selected_vertices is not None:
            self._vertex_colors[self.selected_vertices] = SELECTED_VERTEX_COLOR
        self._refresh_mesh()

    def _project_to_canvas(self, points: np.ndarray) -> np.ndarray:
        """Project Nx3 mesh-local points to Nx2 canvas (pixel) coordinates."""
        transform = self.mesh.transforms.get_transform(map_from="visual", map_to="canvas")
        mapped = np.asarray(transform.map(points))
        if mapped.shape[-1] == 4:
            w = mapped[:, 3:4]
            w = np.where(w == 0, 1e-9, w)
            mapped = mapped[:, :2] / w
        else:
            mapped = mapped[:, :2]
        return mapped

    def _pick_nearest_vertex(self, screen_pos: np.ndarray) -> tuple[int, float]:
        """Return (index, distance_px) of the vertex closest to screen_pos."""
        screen_verts = self._project_to_canvas(self.vertices)
        dists_px = np.linalg.norm(screen_verts - screen_pos[None, :], axis=1)
        nearest = int(np.argmin(dists_px))
        return nearest, float(dists_px[nearest])

    def _numerical_jacobian(self, center: np.ndarray) -> np.ndarray:
        """Numerical screen-space Jacobian at `center`: how the projected
        position changes per unit world-space movement along each axis.
        Its pseudo-inverse maps a desired screen displacement to the
        (minimum-norm, i.e. screen-facing-plane) world displacement."""
        eps = max(self.brush_radius * 0.01, 1e-5)
        probe_points = np.stack(
            [center, center + [eps, 0, 0], center + [0, eps, 0], center + [0, 0, eps]]
        )
        screen_probe = self._project_to_canvas(probe_points)
        jacobian = (screen_probe[1:] - screen_probe[0]).T / eps  # (2, 3)
        return np.linalg.pinv(jacobian)  # (3, 2)

    # ----- mouse dispatch -----

    def _on_mouse_press(self, event) -> None:
        if self.vertices is None or event.button != 1:
            return
        if self.mode == "sculpt":
            self._sculpt_press(event)
        elif self.mode == "select":
            self._select_press(event)
        elif self.mode == "pick":
            self._pick_press(event)

    def _on_mouse_move(self, event) -> None:
        if self.mode == "sculpt":
            self._sculpt_move(event)
        elif self.mode in ("select", "pick"):
            # Shared: dragging an existing selection works identically no
            # matter which tool made it (rect/lasso only ever apply to
            # "select" -- _select_drag_kind is never "rect"/"lasso" here
            # when the active mode is "pick").
            self._select_move(event)

    def _on_mouse_release(self, event) -> None:
        if self.mode == "sculpt":
            self._sculpt_release()
        elif self.mode in ("select", "pick"):
            self._select_release()

    # ----- sculpt mode -----

    def _sculpt_press(self, event) -> None:
        screen_pos = np.array(event.pos, dtype=np.float64)
        nearest, dist = self._pick_nearest_vertex(screen_pos)
        if dist > PICK_RADIUS_PIXELS:
            return  # clicked too far from any vertex; ignore

        center = self.vertices[nearest].astype(np.float64)
        dists_3d = np.linalg.norm(self.vertices.astype(np.float64) - center[None, :], axis=1)
        affected = np.nonzero(dists_3d <= self.brush_radius)[0]
        # Smooth cosine falloff: 1.0 at the picked vertex, 0.0 at the brush edge.
        weights = 0.5 * (1.0 + np.cos(np.pi * dists_3d[affected] / self.brush_radius))

        self._drag_affected = affected
        self._drag_weights = weights
        self._drag_original_positions = self.vertices[affected].copy()
        self._drag_start_mouse = screen_pos
        self._drag_jacobian_pinv = self._numerical_jacobian(center)

    def _sculpt_move(self, event) -> None:
        if self._drag_affected is None or 1 not in event.buttons:
            return

        screen_delta = np.array(event.pos, dtype=np.float64) - self._drag_start_mouse
        world_delta = self._drag_jacobian_pinv @ screen_delta  # (3,)

        moved = self._drag_original_positions + self._drag_weights[:, None] * world_delta[None, :]
        self.vertices[self._drag_affected] = moved.astype(np.float32)

        self._refresh_mesh()
        if self.on_edited is not None:
            self.on_edited(self.vertices)

    def _sculpt_release(self) -> None:
        self._drag_affected = None
        self._drag_weights = None
        self._drag_original_positions = None
        self._drag_start_mouse = None
        self._drag_jacobian_pinv = None

    # ----- select mode -----

    def _try_start_move(self, screen_pos: np.ndarray) -> bool:
        """If `screen_pos` lands on the current selection, start a rigid
        move drag (as "select" mode's own click-drag-to-move already did)
        and return True. Shared by "select" and "pick" modes -- once
        something is selected, dragging it works the same regardless of
        how the selection was made."""
        if self.selected_vertices is None:
            return False
        screen_sel = self._project_to_canvas(self.vertices[self.selected_vertices])
        dists_px = np.linalg.norm(screen_sel - screen_pos[None, :], axis=1)
        if dists_px.min() > PICK_RADIUS_PIXELS:
            return False
        self._select_drag_kind = "move"
        center = self.vertices[self.selected_vertices].astype(np.float64).mean(axis=0)
        self._move_original_positions = self.vertices[self.selected_vertices].copy()
        self._move_start_mouse = screen_pos
        self._move_jacobian_pinv = self._numerical_jacobian(center)
        return True

    def _select_press(self, event) -> None:
        screen_pos = np.array(event.pos, dtype=np.float64)
        if self._try_start_move(screen_pos):
            return

        self._select_drag_kind = "lasso" if keys.SHIFT in event.modifiers else "rect"
        self._select_path = [(float(screen_pos[0]), float(screen_pos[1]))]

    def _select_move(self, event) -> None:
        if self._select_drag_kind in ("rect", "lasso"):
            if 1 not in event.buttons:
                return
            pos = (float(event.pos[0]), float(event.pos[1]))
            if self._select_drag_kind == "rect":
                self._select_path = [self._select_path[0], pos]
            else:
                last = self._select_path[-1]
                if (pos[0] - last[0]) ** 2 + (pos[1] - last[1]) ** 2 >= LASSO_MIN_POINT_SPACING_PIXELS ** 2:
                    self._select_path.append(pos)
            self._overlay.set_path(self._select_drag_kind, self._select_path)

        elif self._select_drag_kind == "move":
            if 1 not in event.buttons:
                return
            screen_delta = np.array(event.pos, dtype=np.float64) - self._move_start_mouse
            world_delta = self._move_jacobian_pinv @ screen_delta  # (3,) -- uniform, no falloff
            moved = self._move_original_positions + world_delta[None, :]
            self.vertices[self.selected_vertices] = moved.astype(np.float32)
            self._refresh_mesh()
            if self.on_edited is not None:
                self.on_edited(self.vertices)

    def _select_release(self) -> None:
        if self._select_drag_kind in ("rect", "lasso") and len(self._select_path) >= 2:
            polygon = self._polygon_from_path(self._select_drag_kind, self._select_path)
            screen_verts = self._project_to_canvas(self.vertices)
            inside = _points_in_polygon(screen_verts, polygon)
            selected = np.nonzero(inside)[0]
            self.selected_vertices = selected if selected.size > 0 else None
            self._update_highlight()
            if self.on_selection_changed is not None:
                self.on_selection_changed(self.selected_vertices)

        self._select_drag_kind = None
        self._select_path = []
        self._move_original_positions = None
        self._move_jacobian_pinv = None
        self._move_start_mouse = None
        self._overlay.clear()

    @staticmethod
    def _polygon_from_path(kind: str, path: list[tuple[float, float]]) -> np.ndarray:
        if kind == "rect":
            (x0, y0), (x1, y1) = path[0], path[-1]
            return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
        return np.array(path, dtype=np.float64)

    # ----- pick mode -----

    def _ensure_adjacency(self) -> None:
        if self._adjacency is None:
            self._adjacency = _build_vertex_adjacency(self.faces, self.vertices.shape[0])

    def _pick_press(self, event) -> None:
        screen_pos = np.array(event.pos, dtype=np.float64)
        if self._try_start_move(screen_pos):
            return

        nearest, dist = self._pick_nearest_vertex(screen_pos)
        if dist > PICK_RADIUS_PIXELS:
            return  # clicked empty space; leave any existing selection as-is

        self._ensure_adjacency()
        adj_ptr, adj_indices = self._adjacency
        component = _connected_component(nearest, adj_ptr, adj_indices, self.vertices.shape[0])
        self.selected_vertices = component if component.size > 0 else None
        self._update_highlight()
        if self.on_selection_changed is not None:
            self.on_selection_changed(self.selected_vertices)
