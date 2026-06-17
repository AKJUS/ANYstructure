"""
Fast dependency-free 3D drawing on a Tkinter Canvas.

Improvements in this version
----------------------------
* Static 3D geometry is cached and is not rebuilt during camera movement.
* Camera basis and projection constants are calculated once per frame.
* Shared vertices are projected only once per frame.
* Back-facing cylinder shell patches are culled.
* A lower-detail interactive representation is used while orbiting/zooming.
* Redraws are throttled during mouse movement.
* Cylinder and longitudinal stiffeners use adaptive vertical subdivision around
  ring girder elevations, improving painter-order occlusion.
* Longitudinal stiffeners are vertically subdivided so their upper parts do not
  disappear incorrectly behind the cylinder top cap.
* Longitudinal and ring stiffeners can be placed on either the outside or the
  inside of the shell.
* Open-ended cylinders and stippled semi-transparent shell rendering are
  supported for viewing internal structure.
* Transparent shells render both front- and back-facing surface patches so the
  complete cylindrical shell remains visible.
* Shell plates can be colour-coded by thickness, with a fixed legend rendered
  in a reserved panel on the right side of the canvas.

The cylinder axis is the global Z axis.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_EPS = 1.0e-12

# Blue -> cyan -> green -> yellow -> orange -> red.  The interpolation is
# implemented locally so the canvas keeps its zero-dependency design.
_THICKNESS_COLOR_STOPS: Tuple[Tuple[float, str], ...] = (
    (0.00, "#313695"),
    (0.18, "#4575b4"),
    (0.36, "#74add1"),
    (0.52, "#abd9e9"),
    (0.66, "#e0f3f8"),
    (0.78, "#fee090"),
    (0.88, "#fdae61"),
    (0.95, "#f46d43"),
    (1.00, "#a50026"),
)


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB colour, got {color!r}")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(red: float, green: float, blue: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, round(red))),
        max(0, min(255, round(green))),
        max(0, min(255, round(blue))),
    )


def _interpolate_thickness_color(
    value: float,
    minimum: float,
    maximum: float,
) -> str:
    if maximum <= minimum + _EPS:
        position = 0.5
    else:
        position = (float(value) - minimum) / (maximum - minimum)
    position = max(0.0, min(1.0, position))

    for index in range(len(_THICKNESS_COLOR_STOPS) - 1):
        start_position, start_color = _THICKNESS_COLOR_STOPS[index]
        end_position, end_color = _THICKNESS_COLOR_STOPS[index + 1]
        if position <= end_position:
            span = max(_EPS, end_position - start_position)
            fraction = (position - start_position) / span
            start_rgb = _hex_to_rgb(start_color)
            end_rgb = _hex_to_rgb(end_color)
            return _rgb_to_hex(
                start_rgb[0] + fraction * (end_rgb[0] - start_rgb[0]),
                start_rgb[1] + fraction * (end_rgb[1] - start_rgb[1]),
                start_rgb[2] + fraction * (end_rgb[2] - start_rgb[2]),
            )

    return _THICKNESS_COLOR_STOPS[-1][1]


def _flatten_numeric_values(value: Any) -> List[float]:
    """Extract finite numeric values from scalar or nested list/tuple input."""
    if value is None or callable(value) or isinstance(value, (str, bytes)):
        return []
    if isinstance(value, (int, float)):
        number = float(value)
        return [number] if math.isfinite(number) else []
    try:
        items = list(value)
    except TypeError:
        return []

    result: List[float] = []
    for item in items:
        result.extend(_flatten_numeric_values(item))
    return result


class Point3D:
    """A lightweight three-dimensional vector/point."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Point3D({self.x:g}, {self.y:g}, {self.z:g})"

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z

    def __add__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Point3D":
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Point3D":
        return self * scalar

    def __truediv__(self, scalar: float) -> "Point3D":
        if abs(scalar) <= _EPS:
            raise ZeroDivisionError("Cannot divide Point3D by zero")
        return Point3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Point3D":
        magnitude = self.length()
        if magnitude <= _EPS:
            return Point3D(0.0, 0.0, 0.0)
        return self / magnitude

    def dot(self, other: "Point3D") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Point3D") -> "Point3D":
        return Point3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def rotate_x(self, angle: float) -> "Point3D":
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return Point3D(
            self.x,
            self.y * cosine - self.z * sine,
            self.y * sine + self.z * cosine,
        )

    def rotate_y(self, angle: float) -> "Point3D":
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return Point3D(
            self.x * cosine + self.z * sine,
            self.y,
            -self.x * sine + self.z * cosine,
        )

    def rotate_z(self, angle: float) -> "Point3D":
        cosine = math.cos(angle)
        sine = math.sin(angle)
        return Point3D(
            self.x * cosine - self.y * sine,
            self.x * sine + self.y * cosine,
            self.z,
        )


class Camera3D:
    """Orbit camera looking at a target point."""

    def __init__(self) -> None:
        self.target = Point3D(0.0, 0.0, 0.0)
        self.world_up = Point3D(0.0, 0.0, 1.0)

        self.fov = math.radians(45.0)
        self.near = 0.01
        self.far = 10000.0

        self.azimuth = math.radians(-45.0)
        self.elevation = math.radians(25.0)
        self.distance = 10.0

        self.position = Point3D(0.0, 0.0, 0.0)
        self._update_position()

    def _update_position(self) -> None:
        cosine_elevation = math.cos(self.elevation)
        offset = Point3D(
            self.distance * cosine_elevation * math.cos(self.azimuth),
            self.distance * cosine_elevation * math.sin(self.azimuth),
            self.distance * math.sin(self.elevation),
        )
        self.position = self.target + offset

    def set_orbit(
        self,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        distance: Optional[float] = None,
    ) -> None:
        if azimuth is not None:
            self.azimuth = float(azimuth)
        if elevation is not None:
            limit = math.radians(89.5)
            self.elevation = max(-limit, min(limit, float(elevation)))
        if distance is not None:
            self.distance = max(float(distance), self.near * 2.0)
        self._update_position()

    def orbit(
        self,
        delta_azimuth: float = 0.0,
        delta_elevation: float = 0.0,
        delta_distance: float = 0.0,
    ) -> None:
        self.set_orbit(
            azimuth=self.azimuth + delta_azimuth,
            elevation=self.elevation + delta_elevation,
            distance=self.distance + delta_distance,
        )

    def zoom(self, factor: float) -> None:
        if factor > 0.0:
            self.set_orbit(distance=max(self.near * 2.0, self.distance * factor))

    def set_target(self, target: Point3D) -> None:
        self.target = Point3D(target.x, target.y, target.z)
        self._update_position()

    def set_position(self, position: Point3D) -> None:
        offset = position - self.target
        distance = max(offset.length(), self.near * 2.0)
        self.distance = distance
        self.azimuth = math.atan2(offset.y, offset.x)
        self.elevation = math.asin(max(-1.0, min(1.0, offset.z / distance)))
        self._update_position()

    def basis(self) -> Tuple[Point3D, Point3D, Point3D]:
        """Return camera right, camera up and camera forward vectors."""
        forward = (self.target - self.position).normalized()
        right = forward.cross(self.world_up)
        if right.length() <= _EPS:
            right = forward.cross(Point3D(0.0, 1.0, 0.0))
        right = right.normalized()
        camera_up = right.cross(forward).normalized()
        return right, camera_up, forward

    def world_to_camera(self, point: Point3D) -> Tuple[float, float, float]:
        right, camera_up, forward = self.basis()
        relative = point - self.position
        return relative.dot(right), relative.dot(camera_up), -relative.dot(forward)

    def project_point(
        self,
        point: Point3D,
        width: int,
        height: int,
    ) -> Optional[Tuple[float, float]]:
        """Compatibility projection method; frame rendering uses a faster path."""
        width = max(1, int(width))
        height = max(1, int(height))
        camera_x, camera_y, camera_z = self.world_to_camera(point)
        depth = -camera_z
        if depth <= self.near or depth >= self.far:
            return None
        scale = 1.0 / math.tan(self.fov / 2.0)
        aspect = width / height
        return (
            (camera_x * scale / aspect / depth + 1.0) * 0.5 * width,
            (1.0 - camera_y * scale / depth) * 0.5 * height,
        )


class Tkinter3DCanvas(tk.Frame):
    """A fast pure-Tkinter 3D scene widget."""

    def __init__(
        self,
        master: tk.Misc,
        width: int = 800,
        height: int = 600,
        bg: str = "white",
        interactive_fps: int = 40,
        **canvas_kwargs: Any,
    ) -> None:
        super().__init__(master, background=bg)

        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.bg = bg
        self.camera = Camera3D()
        self.objects: List[Dict[str, Any]] = []

        # Optional fixed 2D legend.  The 3D projection reserves space for it,
        # so the legend never covers the model.
        self._thickness_legend: Optional[Dict[str, Any]] = None

        canvas_kwargs.setdefault("highlightthickness", 0)
        canvas_kwargs.setdefault("borderwidth", 0)
        self.canvas = tk.Canvas(
            self,
            width=self.width,
            height=self.height,
            background=bg,
            **canvas_kwargs,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.interactive_fps = max(10, min(120, int(interactive_fps)))
        self._interactive_delay_ms = max(1, round(1000 / self.interactive_fps))

        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._is_dragging = False
        self._interactive_render = False

        self._redraw_after_id: Optional[str] = None
        self._finish_interaction_after_id: Optional[str] = None

        # World-space primitive caches. Camera movement does not invalidate them.
        self._world_primitive_cache: Dict[str, List[Dict[str, Any]]] = {}

        self.canvas.bind("<Configure>", self._on_resize, add="+")
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down, add="+")
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag, add="+")
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up, add="+")
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel, add="+")
        self.canvas.bind("<Button-4>", self._on_mouse_wheel, add="+")
        self.canvas.bind("<Button-5>", self._on_mouse_wheel, add="+")

        self.after_idle(self._request_redraw)

    # ------------------------------------------------------------------
    # Thickness colour scale and fixed legend
    # ------------------------------------------------------------------

    def _plot_width(self) -> int:
        if self._thickness_legend is None:
            return max(1, self.width)
        legend_width = int(self._thickness_legend.get("width", 170))
        return max(120, self.width - legend_width)

    def set_thickness_legend(
        self,
        values: Sequence[float],
        unit: str = "mm",
        title: str = "Plate thickness",
        width: int = 170,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        clean_values = sorted(
            {float(value) for value in values if math.isfinite(float(value))}
        )
        if not clean_values and value_range is None:
            self._thickness_legend = None
            self._invalidate_geometry_cache()
            self._request_redraw()
            return

        if value_range is None:
            minimum = clean_values[0]
            maximum = clean_values[-1]
        else:
            minimum = float(value_range[0])
            maximum = float(value_range[1])
            if maximum < minimum:
                minimum, maximum = maximum, minimum

        self._thickness_legend = {
            "values": clean_values,
            "minimum": minimum,
            "maximum": maximum,
            "unit": str(unit),
            "title": str(title),
            "width": max(130, int(width)),
        }
        self._invalidate_geometry_cache()
        self._request_redraw()

    def clear_thickness_legend(self) -> None:
        self._thickness_legend = None
        self._invalidate_geometry_cache()
        self._request_redraw()

    def thickness_color(
        self,
        thickness: float,
        value_range: Optional[Tuple[float, float]] = None,
    ) -> str:
        if value_range is not None:
            minimum, maximum = value_range
        elif self._thickness_legend is not None:
            minimum = float(self._thickness_legend["minimum"])
            maximum = float(self._thickness_legend["maximum"])
        else:
            minimum = maximum = float(thickness)
        return _interpolate_thickness_color(float(thickness), minimum, maximum)

    @staticmethod
    def _format_legend_value(value: float) -> str:
        if abs(value) >= 100.0:
            return f"{value:.0f}"
        if abs(value) >= 10.0:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.2f}".rstrip("0").rstrip(".")

    def _draw_thickness_legend(self) -> None:
        legend = self._thickness_legend
        if legend is None:
            return

        panel_width = int(legend.get("width", 170))
        left = max(0, self.width - panel_width)
        right = self.width
        top = 0
        bottom = self.height

        self.canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            fill=self.bg,
            outline="#d0d0d0",
            width=1,
        )

        padding = 14
        title = str(legend.get("title", "Plate thickness"))
        unit = str(legend.get("unit", ""))
        title_text = f"{title} [{unit}]" if unit else title
        self.canvas.create_text(
            left + padding,
            18,
            text=title_text,
            anchor="nw",
            font=("TkDefaultFont", 10, "bold"),
            fill="#202020",
        )

        values = list(legend.get("values", []))
        minimum = float(legend["minimum"])
        maximum = float(legend["maximum"])
        available_height = max(80, self.height - 70)

        # A short set of distinct thicknesses is clearer as labelled swatches.
        if 1 <= len(values) <= 10:
            values = sorted(values, reverse=True)
            row_height = min(34, max(23, available_height // len(values)))
            swatch_width = 34
            y_coord = 54
            for value in values:
                color = _interpolate_thickness_color(value, minimum, maximum)
                self.canvas.create_rectangle(
                    left + padding,
                    y_coord,
                    left + padding + swatch_width,
                    y_coord + 17,
                    fill=color,
                    outline="#505050",
                    width=1,
                )
                self.canvas.create_text(
                    left + padding + swatch_width + 10,
                    y_coord + 8,
                    text=self._format_legend_value(value),
                    anchor="w",
                    fill="#202020",
                )
                y_coord += row_height
            return

        # Continuous or highly populated scales are rendered as a gradient.
        bar_top = 54
        bar_bottom = min(self.height - 28, bar_top + available_height)
        bar_left = left + padding
        bar_right = bar_left + 30
        steps = max(30, min(180, bar_bottom - bar_top))
        for index in range(steps):
            fraction_0 = index / steps
            fraction_1 = (index + 1) / steps
            value = maximum - fraction_0 * (maximum - minimum)
            color = _interpolate_thickness_color(value, minimum, maximum)
            y_0 = bar_top + fraction_0 * (bar_bottom - bar_top)
            y_1 = bar_top + fraction_1 * (bar_bottom - bar_top) + 1
            self.canvas.create_rectangle(
                bar_left,
                y_0,
                bar_right,
                y_1,
                fill=color,
                outline=color,
            )
        self.canvas.create_rectangle(
            bar_left,
            bar_top,
            bar_right,
            bar_bottom,
            fill="",
            outline="#505050",
            width=1,
        )

        tick_count = 6
        for index in range(tick_count):
            fraction = index / (tick_count - 1)
            value = maximum - fraction * (maximum - minimum)
            y_coord = bar_top + fraction * (bar_bottom - bar_top)
            self.canvas.create_line(bar_right, y_coord, bar_right + 5, y_coord, fill="#505050")
            self.canvas.create_text(
                bar_right + 10,
                y_coord,
                text=self._format_legend_value(value),
                anchor="w",
                fill="#202020",
            )

    # ------------------------------------------------------------------
    # Event handling and redraw scheduling
    # ------------------------------------------------------------------

    def _on_resize(self, event: tk.Event) -> None:
        new_width = max(1, int(event.width))
        new_height = max(1, int(event.height))
        if new_width == self.width and new_height == self.height:
            return
        self.width = new_width
        self.height = new_height
        self._request_redraw()

    def _on_mouse_down(self, event: tk.Event) -> None:
        self._last_mouse_x = int(event.x)
        self._last_mouse_y = int(event.y)
        self._is_dragging = True
        self._interactive_render = True
        self.canvas.focus_set()

    def _on_mouse_up(self, _event: tk.Event) -> None:
        self._is_dragging = False
        self._interactive_render = False
        self._cancel_scheduled_redraw()
        self._request_redraw()

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if not self._is_dragging:
            return

        dx = int(event.x) - self._last_mouse_x
        dy = int(event.y) - self._last_mouse_y
        self._last_mouse_x = int(event.x)
        self._last_mouse_y = int(event.y)

        self.camera.orbit(
            delta_azimuth=-dx * 0.008,
            delta_elevation=dy * 0.008,
        )
        self._interactive_render = True
        self._request_redraw(interactive=True)

    def _on_mouse_wheel(self, event: tk.Event) -> str:
        event_num = getattr(event, "num", None)
        event_delta = getattr(event, "delta", 0)

        if event_num == 4 or event_delta > 0:
            self.camera.zoom(0.90)
        elif event_num == 5 or event_delta < 0:
            self.camera.zoom(1.10)
        else:
            return "break"

        self._interactive_render = True
        self._request_redraw(interactive=True)

        if self._finish_interaction_after_id is not None:
            try:
                self.after_cancel(self._finish_interaction_after_id)
            except tk.TclError:
                pass
        self._finish_interaction_after_id = self.after(120, self._finish_interaction)
        return "break"

    def _finish_interaction(self) -> None:
        self._finish_interaction_after_id = None
        if self._is_dragging:
            return
        self._interactive_render = False
        self._cancel_scheduled_redraw()
        self._request_redraw()

    def _cancel_scheduled_redraw(self) -> None:
        if self._redraw_after_id is not None:
            try:
                self.after_cancel(self._redraw_after_id)
            except tk.TclError:
                pass
            self._redraw_after_id = None

    def _request_redraw(self, interactive: Optional[bool] = None) -> None:
        if interactive is None:
            interactive = self._interactive_render
        if self._redraw_after_id is not None:
            return

        if interactive:
            self._redraw_after_id = self.after(
                self._interactive_delay_ms,
                self._run_scheduled_redraw,
            )
        else:
            self._redraw_after_id = self.after_idle(self._run_scheduled_redraw)

    def _run_scheduled_redraw(self) -> None:
        self._redraw_after_id = None
        self.redraw()

    # ------------------------------------------------------------------
    # Scene lifecycle and cache management
    # ------------------------------------------------------------------

    def _invalidate_geometry_cache(self) -> None:
        self._world_primitive_cache.clear()

    def _clear_canvas_only(self) -> None:
        self.canvas.delete("all")

    def clear(self) -> None:
        self.objects.clear()
        self._invalidate_geometry_cache()
        self._clear_canvas_only()

    def redraw(self) -> None:
        """Render the scene; static world geometry is reused from cache."""
        if not self.winfo_exists() or not self.canvas.winfo_exists():
            return

        self.width = max(1, self.canvas.winfo_width())
        self.height = max(1, self.canvas.winfo_height())
        self._clear_canvas_only()

        quality = "fast" if self._interactive_render else "full"
        primitives = self._get_world_primitives(quality)

        right, camera_up, forward = self.camera.basis()
        position = self.camera.position
        scale = 1.0 / math.tan(self.camera.fov / 2.0)
        plot_width = self._plot_width()
        aspect = plot_width / self.height
        x_scale = scale / aspect
        half_width = 0.5 * plot_width
        half_height = 0.5 * self.height
        near = self.camera.near
        far = self.camera.far

        # Point object IDs are stable because world primitives are cached.
        projected_points: Dict[int, Optional[Tuple[float, float, float]]] = {}

        def project(point: Point3D) -> Optional[Tuple[float, float, float]]:
            key = id(point)
            if key in projected_points:
                return projected_points[key]

            rx = point.x - position.x
            ry = point.y - position.y
            rz = point.z - position.z

            camera_x = rx * right.x + ry * right.y + rz * right.z
            camera_y = rx * camera_up.x + ry * camera_up.y + rz * camera_up.z
            depth = rx * forward.x + ry * forward.y + rz * forward.z

            if depth <= near or depth >= far:
                projected_points[key] = None
                return None

            screen_x = (camera_x * x_scale / depth + 1.0) * half_width
            screen_y = (1.0 - camera_y * scale / depth) * half_height
            result = (screen_x, screen_y, depth)
            projected_points[key] = result
            return result

        render_items: List[Tuple[float, int, Dict[str, Any], Tuple[float, ...]]] = []

        for primitive in primitives:
            if primitive.get("cull_backface", False):
                normal = primitive["normal"]
                center = primitive["center"]
                to_camera_x = position.x - center.x
                to_camera_y = position.y - center.y
                to_camera_z = position.z - center.z
                facing = (
                    normal.x * to_camera_x
                    + normal.y * to_camera_y
                    + normal.z * to_camera_z
                )
                if facing <= 0.0:
                    continue

            if primitive["kind"] == "line":
                start = project(primitive["start"])
                end = project(primitive["end"])
                if start is None or end is None:
                    continue
                depth = 0.5 * (start[2] + end[2])
                coords = (start[0], start[1], end[0], end[1])
            else:
                projected: List[Tuple[float, float, float]] = []
                clipped = False
                for vertex in primitive["vertices"]:
                    point_2d = project(vertex)
                    if point_2d is None:
                        clipped = True
                        break
                    projected.append(point_2d)
                if clipped or len(projected) < 3:
                    continue

                depth = sum(item[2] for item in projected) / len(projected)
                flat: List[float] = []
                for screen_x, screen_y, _point_depth in projected:
                    flat.extend((screen_x, screen_y))
                coords = tuple(flat)

            render_items.append(
                (
                    depth,
                    int(primitive.get("layer", 0)),
                    primitive,
                    coords,
                )
            )

        depth_span = max(0.001, far - near)
        # Apply a layer bias to resolve depth sorting of overlapping/coplanar elements
        # (e.g. plate and stiffeners). Higher layers are drawn later (on top).
        layer_epsilon = depth_span * 0.025
        render_items.sort(key=lambda item: (-(item[0] - item[1] * layer_epsilon), item[1]))

        interactive = self._interactive_render
        for _depth, _layer, primitive, coords in render_items:
            if primitive["kind"] == "line":
                self.canvas.create_line(
                    *coords,
                    fill=primitive["color"],
                    width=primitive["width"],
                )
            else:
                outline = "" if interactive and primitive.get("fast_no_outline") else primitive["outline"]
                kwargs = {
                    "fill": primitive["color"],
                    "outline": outline,
                    "width": primitive["width"],
                    "stipple": primitive.get("stipple", ""),
                }
                if primitive.get("tags"):
                    kwargs["tags"] = primitive.get("tags")
                self.canvas.create_polygon(
                    *coords,
                    **kwargs
                )

        self._draw_thickness_legend()

    def _get_world_primitives(self, quality: str) -> List[Dict[str, Any]]:
        cached = self._world_primitive_cache.get(quality)
        if cached is not None:
            return cached

        primitives: List[Dict[str, Any]] = []
        for obj in self.objects:
            primitives.extend(self._object_to_primitives(obj, quality))

        self._world_primitive_cache[quality] = primitives
        return primitives

    # ------------------------------------------------------------------
    # Primitive construction
    # ------------------------------------------------------------------

    @staticmethod
    def _polygon_normal(vertices: Sequence[Point3D]) -> Point3D:
        if len(vertices) < 3:
            return Point3D(0.0, 0.0, 0.0)
        origin = vertices[0]
        for index in range(1, len(vertices) - 1):
            edge_1 = vertices[index] - origin
            edge_2 = vertices[index + 1] - origin
            normal = edge_1.cross(edge_2)
            if normal.length() > _EPS:
                return normal.normalized()
        return Point3D(0.0, 0.0, 0.0)

    def _polygon_primitive(
        self,
        vertices: Sequence[Point3D],
        color: str,
        outline: str,
        width: int = 1,
        layer: int = 0,
        cull_backface: bool = False,
        fast_no_outline: bool = True,
        stipple: str = "",
        tags: str = "",
    ) -> Optional[Dict[str, Any]]:
        if len(vertices) < 3:
            return None
        vertices_list = list(vertices)
        count = len(vertices_list)
        center = Point3D(
            sum(vertex.x for vertex in vertices_list) / count,
            sum(vertex.y for vertex in vertices_list) / count,
            sum(vertex.z for vertex in vertices_list) / count,
        )
        return {
            "kind": "polygon",
            "vertices": vertices_list,
            "color": color,
            "outline": outline,
            "width": width,
            "layer": layer,
            "center": center,
            "normal": self._polygon_normal(vertices_list),
            "cull_backface": cull_backface,
            "fast_no_outline": fast_no_outline,
            "stipple": stipple,
            "tags": tags,
        }

    @staticmethod
    def _line_primitive(
        start: Point3D,
        end: Point3D,
        color: str,
        width: int,
        layer: int = 30,
    ) -> Dict[str, Any]:
        return {
            "kind": "line",
            "start": start,
            "end": end,
            "color": color,
            "width": width,
            "layer": layer,
        }

    def _object_to_primitives(
        self,
        obj: Dict[str, Any],
        quality: str,
    ) -> List[Dict[str, Any]]:
        object_type = obj.get("type")
        if object_type == "line":
            return [
                self._line_primitive(
                    obj["start"],
                    obj["end"],
                    obj.get("color", "black"),
                    int(obj.get("width", 1)),
                )
            ]
        elif object_type == "polygon":
            primitive = self._polygon_primitive(
                vertices=obj["vertices"],
                color=obj["color"],
                outline=obj["outline"],
                width=obj["width"],
                layer=obj.get("layer", 5),
                cull_backface=obj.get("cull_backface", False),
                stipple=obj.get("stipple", ""),
                tags=obj.get("tags", ""),
            )
            return [primitive] if primitive else []
        if object_type == "cylinder":
            return self._cylinder_primitives(obj, quality)
        if object_type == "stiffener":
            if obj.get("stiffener_type") == "ring":
                return self._ring_stiffener_primitives(obj, quality)
            return self._longitudinal_stiffener_primitives(obj, quality)
        return []

    def _adaptive_z_breaks(
        self,
        z_bottom: float,
        z_top: float,
        requested_segments: int,
        quality: str,
    ) -> List[float]:
        """Build local vertical patches around ring girder elevations."""
        requested_segments = max(1, int(requested_segments))
        if quality == "fast":
            uniform_segments = max(3, min(5, round(requested_segments / 7)))
        else:
            uniform_segments = max(5, min(12, round(requested_segments / 4)))

        values = {
            z_bottom + (z_top - z_bottom) * index / uniform_segments
            for index in range(uniform_segments + 1)
        }

        for obj in self.objects:
            if obj.get("type") != "stiffener" or obj.get("stiffener_type") != "ring":
                continue
            z_position = float(obj.get("z_position", 0.0))
            half_width = 0.5 * max(
                float(obj.get("web_thickness", 0.0)),
                float(obj.get("flange_width", 0.0)),
            )
            if quality == "fast":
                candidates = (z_position,)
            else:
                candidates = (z_position - half_width, z_position, z_position + half_width)
            for value in candidates:
                if z_bottom + _EPS < value < z_top - _EPS:
                    values.add(value)

        return sorted(values)

    @staticmethod
    def _scaled_segments(segments: int, quality: str, minimum: int = 12) -> int:
        segments = max(3, int(segments))
        if quality == "fast":
            return max(minimum, segments // 2)
        return segments

    @staticmethod
    def _opacity_to_stipple(opacity: float) -> str:
        """Map requested opacity to Tk's built-in stipple densities."""
        opacity = max(0.0, min(1.0, float(opacity)))
        if opacity >= 0.90:
            return ""
        if opacity >= 0.65:
            return "gray75"
        if opacity >= 0.38:
            return "gray50"
        if opacity >= 0.20:
            return "gray25"
        return "gray12"

    @staticmethod
    def _resolve_plate_thickness(
        specification: Any,
        angle: float,
        z_coord: float,
        z_bottom: float,
        z_top: float,
    ) -> Optional[float]:
        """
        Resolve a shell patch thickness.

        Supported specifications:
        * scalar: one thickness for the whole shell;
        * 1D sequence: axial bands ordered bottom to top;
        * 2D sequence: axial rows, with circumferential columns;
        * callable: called as ``fn(angle, z)``.  If that signature is not
          accepted, ``fn(angle, z, angle_fraction, height_fraction)`` is used.
        """
        if specification is None:
            return None

        angle_fraction = (angle % (2.0 * math.pi)) / (2.0 * math.pi)
        height_span = max(_EPS, z_top - z_bottom)
        height_fraction = max(0.0, min(1.0, (z_coord - z_bottom) / height_span))

        if callable(specification):
            try:
                value = specification(angle, z_coord)
            except TypeError:
                value = specification(angle, z_coord, angle_fraction, height_fraction)
            number = float(value)
            return number if math.isfinite(number) else None

        if isinstance(specification, (int, float)):
            number = float(specification)
            return number if math.isfinite(number) else None

        if isinstance(specification, (str, bytes)):
            return None

        try:
            rows = list(specification)
        except TypeError:
            return None
        if not rows:
            return None

        row_index = min(len(rows) - 1, int(height_fraction * len(rows)))
        selected = rows[row_index]

        if isinstance(selected, (int, float)):
            number = float(selected)
            return number if math.isfinite(number) else None

        if isinstance(selected, (str, bytes)):
            return None
        try:
            columns = list(selected)
        except TypeError:
            return None
        if not columns:
            return None

        column_index = min(len(columns) - 1, int(angle_fraction * len(columns)))
        number = float(columns[column_index])
        return number if math.isfinite(number) else None

    def _cylinder_primitives(
        self,
        obj: Dict[str, Any],
        quality: str,
    ) -> List[Dict[str, Any]]:
        radius = max(0.0, float(obj.get("radius", 1.0)))
        height = max(0.0, float(obj.get("height", 1.0)))
        center = obj.get("center", Point3D(0.0, 0.0, 0.0))
        color = obj.get("color", "lightgray")
        outline = obj.get("outline", "black")
        plate_thickness = obj.get("plate_thickness")
        thickness_range = obj.get("thickness_range")
        if thickness_range is not None:
            thickness_minimum = float(thickness_range[0])
            thickness_maximum = float(thickness_range[1])
        elif self._thickness_legend is not None:
            thickness_minimum = float(self._thickness_legend["minimum"])
            thickness_maximum = float(self._thickness_legend["maximum"])
        else:
            thickness_values = _flatten_numeric_values(plate_thickness)
            thickness_minimum = min(thickness_values) if thickness_values else 0.0
            thickness_maximum = max(thickness_values) if thickness_values else 1.0
        segments = self._scaled_segments(int(obj.get("segments", 32)), quality)
        requested_height_segments = max(1, int(obj.get("height_segments", 24)))
        capped = bool(obj.get("capped", True))
        opacity = max(0.0, min(1.0, float(obj.get("opacity", 1.0))))
        shell_stipple = self._opacity_to_stipple(opacity)

        # An opaque shell only needs the camera-facing half. A transparent or
        # stippled shell must retain the back-facing half as well; otherwise the
        # cylinder looks cut away rather than semi-transparent. The public
        # show_backfaces option can override this automatic behaviour.
        show_backfaces = obj.get("show_backfaces")
        if show_backfaces is None:
            show_backfaces = opacity < 0.90
        cull_shell_backfaces = not bool(show_backfaces)

        z_bottom = center.z - height / 2.0
        z_top = center.z + height / 2.0
        z_breaks = self._adaptive_z_breaks(
            z_bottom,
            z_top,
            requested_height_segments,
            quality,
        )

        angles = [2.0 * math.pi * index / segments for index in range(segments)]
        cosines = [math.cos(angle) for angle in angles]
        sines = [math.sin(angle) for angle in angles]

        rings: List[List[Point3D]] = []
        for z_coord in z_breaks:
            rings.append(
                [
                    Point3D(
                        center.x + radius * cosines[index],
                        center.y + radius * sines[index],
                        z_coord,
                    )
                    for index in range(segments)
                ]
            )

        primitives: List[Dict[str, Any]] = []
        for z_index in range(len(rings) - 1):
            lower_ring = rings[z_index]
            upper_ring = rings[z_index + 1]
            for index in range(segments):
                next_index = (index + 1) % segments
                angle_mid = 2.0 * math.pi * (index + 0.5) / segments
                z_mid = 0.5 * (z_breaks[z_index] + z_breaks[z_index + 1])
                patch_thickness = self._resolve_plate_thickness(
                    plate_thickness,
                    angle_mid,
                    z_mid,
                    z_bottom,
                    z_top,
                )
                patch_color = (
                    _interpolate_thickness_color(
                        patch_thickness,
                        thickness_minimum,
                        thickness_maximum,
                    )
                    if patch_thickness is not None
                    else color
                )
                primitive = self._polygon_primitive(
                    [
                        lower_ring[index],
                        lower_ring[next_index],
                        upper_ring[next_index],
                        upper_ring[index],
                    ],
                    patch_color,
                    outline,
                    width=1,
                    layer=0,
                    cull_backface=cull_shell_backfaces,
                    stipple=shell_stipple,
                )
                if primitive:
                    primitives.append(primitive)

        if capped and rings:
            top_cap = self._polygon_primitive(
                rings[-1],
                color,
                outline,
                width=1,
                layer=1,
                cull_backface=cull_shell_backfaces,
                stipple=shell_stipple,
            )
            bottom_cap = self._polygon_primitive(
                list(reversed(rings[0])),
                color,
                outline,
                width=1,
                layer=1,
                cull_backface=cull_shell_backfaces,
                stipple=shell_stipple,
            )
            if top_cap:
                primitives.append(top_cap)
            if bottom_cap:
                primitives.append(bottom_cap)

        return primitives

    def _longitudinal_stiffener_primitives(
        self,
        obj: Dict[str, Any],
        quality: str,
    ) -> List[Dict[str, Any]]:
        radius = float(obj.get("radius", 1.0))
        height = float(obj.get("height", 1.0))
        angle = float(obj.get("angle", 0.0))
        web_height = max(0.0, float(obj.get("web_height", 0.1)))
        flange_width = max(0.0, float(obj.get("flange_width", 0.05)))
        flange_thickness = max(0.0, float(obj.get("flange_thickness", 0.01)))
        color = obj.get("color", "silver")
        outline = obj.get("outline", "black")
        width_segments = max(2, int(obj.get("segments", 4)))
        height_segments = max(4, int(obj.get("height_segments", 16)))
        inside = bool(obj.get("inside", False))
        radial_direction = -1.0 if inside else 1.0

        web_thickness = float(obj.get("web_thickness", 0.01))
        web_thickness_mm = web_thickness * 1000.0 if web_thickness < 1.0 else web_thickness
        flange_thickness_mm = flange_thickness * 1000.0 if flange_thickness < 1.0 else flange_thickness

        if self._thickness_legend is not None:
            web_color = self.thickness_color(web_thickness_mm)
            flange_color = self.thickness_color(flange_thickness_mm)
        else:
            web_color = flange_color = color

        if quality == "fast":
            width_segments = 2

        z_offset = float(obj.get("z_offset", 0.0))
        z_bottom = -height / 2.0 + z_offset
        z_top = height / 2.0 + z_offset
        # Longitudinal stiffeners need finer local vertical patches than the
        # shell.  In particular, a short patch immediately below the top cap
        # prevents the painter-order sort from hiding the visible outer part of
        # a front-side stiffener behind the large cap polygon.
        longitudinal_break_request = (
            height_segments if quality == "fast" else height_segments * 3
        )
        z_breaks = self._adaptive_z_breaks(
            z_bottom,
            z_top,
            longitudinal_break_request,
            quality,
        )

        cosine = math.cos(angle)
        sine = math.sin(angle)
        attachment_radius = max(_EPS, radius)
        web_tip_radius = max(_EPS, radius + radial_direction * web_height)
        attachment_points = [
            Point3D(attachment_radius * cosine, attachment_radius * sine, z)
            for z in z_breaks
        ]
        tip_points = [
            Point3D(web_tip_radius * cosine, web_tip_radius * sine, z)
            for z in z_breaks
        ]

        primitives: List[Dict[str, Any]] = []
        for z_index in range(len(z_breaks) - 1):
            web = self._polygon_primitive(
                [
                    attachment_points[z_index],
                    tip_points[z_index],
                    tip_points[z_index + 1],
                    attachment_points[z_index + 1],
                ],
                web_color,
                outline,
                width=1,
                layer=12,
                cull_backface=False,
            )
            if web:
                primitives.append(web)

        if flange_width > 0.0:
            flange_radius = max(
                _EPS,
                web_tip_radius + radial_direction * 0.5 * flange_thickness,
            )
            half_angle = 0.5 * flange_width / flange_radius
            flange_angles = [
                angle - half_angle + 2.0 * half_angle * index / (width_segments - 1)
                for index in range(width_segments)
            ]
            flange_grid: List[List[Point3D]] = []
            for z_coord in z_breaks:
                flange_grid.append(
                    [
                        Point3D(
                            flange_radius * math.cos(flange_angle),
                            flange_radius * math.sin(flange_angle),
                            z_coord,
                        )
                        for flange_angle in flange_angles
                    ]
                )

            for z_index in range(len(z_breaks) - 1):
                lower = flange_grid[z_index]
                upper = flange_grid[z_index + 1]
                for width_index in range(width_segments - 1):
                    flange = self._polygon_primitive(
                        [
                            lower[width_index],
                            lower[width_index + 1],
                            upper[width_index + 1],
                            upper[width_index],
                        ],
                        flange_color,
                        outline,
                        width=1,
                        layer=13,
                        cull_backface=False,
                    )
                    if flange:
                        primitives.append(flange)

        return primitives

    def _ring_stiffener_primitives(
        self,
        obj: Dict[str, Any],
        quality: str,
    ) -> List[Dict[str, Any]]:
        radius = float(obj.get("radius", 1.0))
        z_position = float(obj.get("z_position", 0.0))
        web_height = max(0.0, float(obj.get("web_height", 0.1)))
        web_thickness = max(0.0, float(obj.get("web_thickness", 0.01)))
        flange_width = max(0.0, float(obj.get("flange_width", 0.05)))
        flange_thickness = max(0.0, float(obj.get("flange_thickness", 0.01)))
        color = obj.get("color", "dimgray")
        outline = obj.get("outline", "black")
        segments = self._scaled_segments(int(obj.get("segments", 32)), quality)
        inside = bool(obj.get("inside", False))
        radial_direction = -1.0 if inside else 1.0

        web_thickness_mm = web_thickness * 1000.0 if web_thickness < 1.0 else web_thickness
        flange_thickness_mm = flange_thickness * 1000.0 if flange_thickness < 1.0 else flange_thickness

        if self._thickness_legend is not None:
            web_color = self.thickness_color(web_thickness_mm)
            flange_color = self.thickness_color(flange_thickness_mm)
        else:
            web_color = flange_color = color

        attachment_radius = max(_EPS, radius)
        tip_radius = max(_EPS, radius + radial_direction * web_height)
        z_lower = z_position - web_thickness / 2.0
        z_upper = z_position + web_thickness / 2.0

        angles = [2.0 * math.pi * index / segments for index in range(segments)]
        cosines = [math.cos(angle) for angle in angles]
        sines = [math.sin(angle) for angle in angles]

        attachment_lower = [
            Point3D(
                attachment_radius * cosines[index],
                attachment_radius * sines[index],
                z_lower,
            )
            for index in range(segments)
        ]
        tip_lower = [
            Point3D(tip_radius * cosines[index], tip_radius * sines[index], z_lower)
            for index in range(segments)
        ]
        attachment_upper = [
            Point3D(
                attachment_radius * cosines[index],
                attachment_radius * sines[index],
                z_upper,
            )
            for index in range(segments)
        ]
        tip_upper = [
            Point3D(tip_radius * cosines[index], tip_radius * sines[index], z_upper)
            for index in range(segments)
        ]

        primitives: List[Dict[str, Any]] = []
        for index in range(segments):
            next_index = (index + 1) % segments

            # During interaction use one web mid-surface instead of three solid
            # web surfaces. The full representation is restored on release.
            if quality == "fast":
                mid_attachment_0 = Point3D(
                    attachment_radius * cosines[index],
                    attachment_radius * sines[index],
                    z_position,
                )
                mid_tip_0 = Point3D(
                    tip_radius * cosines[index],
                    tip_radius * sines[index],
                    z_position,
                )
                mid_tip_1 = Point3D(
                    tip_radius * cosines[next_index],
                    tip_radius * sines[next_index],
                    z_position,
                )
                mid_attachment_1 = Point3D(
                    attachment_radius * cosines[next_index],
                    attachment_radius * sines[next_index],
                    z_position,
                )
                faces = [[mid_attachment_0, mid_tip_0, mid_tip_1, mid_attachment_1]]
            else:
                faces = [
                    [
                        attachment_lower[index],
                        tip_lower[index],
                        tip_lower[next_index],
                        attachment_lower[next_index],
                    ],
                    [
                        attachment_upper[next_index],
                        tip_upper[next_index],
                        tip_upper[index],
                        attachment_upper[index],
                    ],
                    [
                        tip_lower[index],
                        tip_upper[index],
                        tip_upper[next_index],
                        tip_lower[next_index],
                    ],
                ]

            for face in faces:
                primitive = self._polygon_primitive(
                    face,
                    web_color,
                    outline,
                    width=1,
                    layer=20,
                    cull_backface=False,
                )
                if primitive:
                    primitives.append(primitive)

        if flange_width > 0.0:
            flange_radius = max(
                _EPS,
                tip_radius + radial_direction * 0.5 * flange_thickness,
            )
            flange_z_lower = z_position - flange_width / 2.0
            flange_z_upper = z_position + flange_width / 2.0
            flange_lower = [
                Point3D(
                    flange_radius * cosines[index],
                    flange_radius * sines[index],
                    flange_z_lower,
                )
                for index in range(segments)
            ]
            flange_upper = [
                Point3D(
                    flange_radius * cosines[index],
                    flange_radius * sines[index],
                    flange_z_upper,
                )
                for index in range(segments)
            ]

            for index in range(segments):
                next_index = (index + 1) % segments
                primitive = self._polygon_primitive(
                    [
                        flange_lower[index],
                        flange_lower[next_index],
                        flange_upper[next_index],
                        flange_upper[index],
                    ],
                    flange_color,
                    outline,
                    width=1,
                    layer=21,
                    cull_backface=False,
                )
                if primitive:
                    primitives.append(primitive)

        return primitives

    # ------------------------------------------------------------------
    # Public scene API
    # ------------------------------------------------------------------

    def add_line(
        self,
        start: Point3D,
        end: Point3D,
        color: str = "black",
        width: int = 1,
    ) -> None:
        self.objects.append(
            {
                "type": "line",
                "start": start,
                "end": end,
                "color": color,
                "width": width,
            }
        )
        self._invalidate_geometry_cache()
        self._request_redraw()

    def add_polygon(
        self,
        vertices: Iterable[Point3D],
        color: str = "gray",
        outline: str = "black",
        width: int = 1,
        cull_backface: bool = False,
        layer: int = 5,
        stipple: str = "",
        tags: str = "",
    ) -> None:
        self.objects.append(
            {
                "type": "polygon",
                "vertices": list(vertices),
                "color": color,
                "outline": outline,
                "width": width,
                "cull_backface": cull_backface,
                "layer": layer,
                "stipple": stipple,
                "tags": tags,
            }
        )
        self._invalidate_geometry_cache()
        self._request_redraw()

    def add_cylinder(
        self,
        radius: float,
        height: float,
        center: Optional[Point3D] = None,
        color: str = "lightgray",
        outline: str = "black",
        segments: int = 32,
        height_segments: int = 24,
        capped: bool = True,
        opacity: float = 1.0,
        show_backfaces: Optional[bool] = None,
        plate_thickness: Any = None,
        thickness_range: Optional[Tuple[float, float]] = None,
        thickness_unit: str = "mm",
        thickness_legend_title: str = "Plate thickness",
        show_thickness_legend: bool = True,
    ) -> None:
        self.objects.append(
            {
                "type": "cylinder",
                "radius": radius,
                "height": height,
                "center": center if center is not None else Point3D(0.0, 0.0, 0.0),
                "color": color,
                "outline": outline,
                "segments": segments,
                "height_segments": height_segments,
                "capped": capped,
                "opacity": max(0.0, min(1.0, float(opacity))),
                "show_backfaces": show_backfaces,
                "plate_thickness": plate_thickness,
                "thickness_range": thickness_range,
            }
        )

        thickness_values = _flatten_numeric_values(plate_thickness)
        if show_thickness_legend and (thickness_values or thickness_range is not None):
            if thickness_range is not None and not thickness_values:
                thickness_values = [float(thickness_range[0]), float(thickness_range[1])]
            self.set_thickness_legend(
                thickness_values,
                unit=thickness_unit,
                title=thickness_legend_title,
                value_range=thickness_range,
            )
        else:
            self._invalidate_geometry_cache()
            self._request_redraw()

    def add_longitudinal_stiffener(
        self,
        radius: float,
        height: float,
        angle: float,
        web_height: float = 0.1,
        web_thickness: float = 0.01,
        flange_width: float = 0.05,
        flange_thickness: float = 0.01,
        color: str = "silver",
        outline: str = "black",
        segments: int = 4,
        height_segments: int = 16,
        inside: bool = False,
        z_offset: float = 0.0,
    ) -> None:
        self.objects.append(
            {
                "type": "stiffener",
                "stiffener_type": "longitudinal",
                "radius": radius,
                "height": height,
                "angle": angle,
                "web_height": web_height,
                "web_thickness": web_thickness,
                "flange_width": flange_width,
                "flange_thickness": flange_thickness,
                "color": color,
                "outline": outline,
                "segments": segments,
                "height_segments": height_segments,
                "inside": bool(inside),
                "z_offset": z_offset,
            }
        )
        self._invalidate_geometry_cache()
        self._request_redraw()

    def add_ring_stiffener(
        self,
        radius: float,
        z_position: float,
        web_height: float = 0.1,
        web_thickness: float = 0.01,
        flange_width: float = 0.05,
        flange_thickness: float = 0.01,
        color: str = "dimgray",
        outline: str = "black",
        segments: int = 32,
        inside: bool = False,
    ) -> None:
        self.objects.append(
            {
                "type": "stiffener",
                "stiffener_type": "ring",
                "radius": radius,
                "z_position": z_position,
                "web_height": web_height,
                "web_thickness": web_thickness,
                "flange_width": flange_width,
                "flange_thickness": flange_thickness,
                "color": color,
                "outline": outline,
                "segments": segments,
                "inside": bool(inside),
            }
        )
        self._invalidate_geometry_cache()
        self._request_redraw()

    # ------------------------------------------------------------------
    # Camera API
    # ------------------------------------------------------------------

    def set_camera_position(self, position: Point3D) -> None:
        self.camera.set_position(position)
        self._request_redraw()

    def set_camera_target(self, target: Point3D) -> None:
        self.camera.set_target(target)
        self._request_redraw()

    def set_view(self, azimuth_degrees: float, elevation_degrees: float) -> None:
        self._interactive_render = False
        self.camera.set_orbit(
            azimuth=math.radians(azimuth_degrees),
            elevation=math.radians(elevation_degrees),
        )
        self._request_redraw()

    def set_iso_view(self) -> None:
        self.set_view(-45.0, 25.0)

    def set_top_view(self) -> None:
        self.set_view(-90.0, 89.0)

    def set_side_view(self) -> None:
        self.set_view(0.0, 0.0)

    def set_front_view(self) -> None:
        self.set_view(-90.0, 0.0)

    def reset_camera(self) -> None:
        self._interactive_render = False
        self.camera = Camera3D()
        self.fit_to_scene(redraw=False)
        self._request_redraw()

    def fit_to_scene(self, padding: float = 1.25, redraw: bool = True) -> None:
        bounds = self._scene_bounds()
        if bounds is None:
            if redraw:
                self._request_redraw()
            return

        minimum, maximum = bounds
        center = Point3D(
            0.5 * (minimum.x + maximum.x),
            0.5 * (minimum.y + maximum.y),
            0.5 * (minimum.z + maximum.z),
        )
        diagonal = (maximum - minimum).length()
        radius = max(0.5 * diagonal, 0.1)

        width = max(1, self._plot_width())
        height = max(1, self.canvas.winfo_height())
        aspect = width / height
        vertical_half_fov = self.camera.fov / 2.0
        horizontal_half_fov = math.atan(math.tan(vertical_half_fov) * aspect)
        limiting_half_fov = max(
            math.radians(5.0),
            min(vertical_half_fov, horizontal_half_fov),
        )
        distance = padding * radius / math.sin(limiting_half_fov)

        self.camera.target = center
        self.camera.set_orbit(distance=distance)
        self.camera.near = max(distance - 3.0 * radius, 0.001)
        self.camera.far = max(distance + 3.0 * radius, self.camera.near + 1.0)

        if redraw:
            self._interactive_render = False
            self._request_redraw()

    def _scene_bounds(self) -> Optional[Tuple[Point3D, Point3D]]:
        points: List[Point3D] = []

        for obj in self.objects:
            object_type = obj.get("type")
            if object_type == "line":
                points.extend((obj["start"], obj["end"]))
            elif object_type == "polygon":
                points.extend(obj.get("vertices", []))
            elif object_type == "cylinder":
                center = obj.get("center", Point3D(0.0, 0.0, 0.0))
                radius = float(obj.get("radius", 1.0))
                half_height = 0.5 * float(obj.get("height", 1.0))
                points.extend(
                    (
                        Point3D(center.x - radius, center.y - radius, center.z - half_height),
                        Point3D(center.x + radius, center.y + radius, center.z + half_height),
                    )
                )
            elif object_type == "stiffener":
                radius = float(obj.get("radius", 1.0))
                web_height = float(obj.get("web_height", 0.0))
                flange_thickness = float(obj.get("flange_thickness", 0.0))
                inside = bool(obj.get("inside", False))
                outer_radius = (
                    radius
                    if inside
                    else radius + web_height + flange_thickness
                )
                if obj.get("stiffener_type") == "ring":
                    z_position = float(obj.get("z_position", 0.0))
                    half_width = 0.5 * max(
                        float(obj.get("web_thickness", 0.0)),
                        float(obj.get("flange_width", 0.0)),
                    )
                    points.extend(
                        (
                            Point3D(-outer_radius, -outer_radius, z_position - half_width),
                            Point3D(outer_radius, outer_radius, z_position + half_width),
                        )
                    )
                else:
                    half_height = 0.5 * float(obj.get("height", 1.0))
                    z_offset = float(obj.get("z_offset", 0.0))
                    points.extend(
                        (
                            Point3D(-outer_radius, -outer_radius, -half_height + z_offset),
                            Point3D(outer_radius, outer_radius, half_height + z_offset),
                        )
                    )

        if not points:
            return None

        minimum = Point3D(
            min(point.x for point in points),
            min(point.y for point in points),
            min(point.z for point in points),
        )
        maximum = Point3D(
            max(point.x for point in points),
            max(point.y for point in points),
            max(point.z for point in points),
        )
        return minimum, maximum


def populate_stiffened_cylinder(canvas_3d: Tkinter3DCanvas) -> None:
    """Populate a canvas with open-ended internal stiffening."""
    cylinder_radius = 2.0
    cylinder_height = 4.0

    canvas_3d.add_cylinder(
        radius=cylinder_radius,
        height=cylinder_height,
        center=Point3D(0.0, 0.0, 0.0),
        color="#d8e2ea",
        outline="#708090",
        segments=48,
        height_segments=24,
        capped=False,
        opacity=0.38,
        show_backfaces=True,
        # Four axial shell strakes, ordered from bottom to top.
        plate_thickness=[18.0, 16.0, 14.0, 12.0],
        thickness_unit="mm",
        thickness_legend_title="Plate thickness",
        show_thickness_legend=True,
    )

    number_of_longitudinals = 8
    for index in range(number_of_longitudinals):
        angle = 2.0 * math.pi * index / number_of_longitudinals
        canvas_3d.add_longitudinal_stiffener(
            radius=cylinder_radius,
            height=cylinder_height,
            angle=angle,
            web_height=0.15,
            web_thickness=0.01,
            flange_width=0.10,
            flange_thickness=0.02,
            color="#a0a0ff",
            outline="#404080",
            segments=4,
            height_segments=16,
            inside=True,
        )

    number_of_rings = 4
    for index in range(number_of_rings):
        z_position = (
            -cylinder_height / 2.0
            + (index + 1) * cylinder_height / (number_of_rings + 1)
        )
        canvas_3d.add_ring_stiffener(
            radius=cylinder_radius,
            z_position=z_position,
            web_height=0.12,
            web_thickness=0.02,
            flange_width=0.08,
            flange_thickness=0.015,
            color="#ffa0a0",
            outline="#804040",
            segments=48,
            inside=True,
        )

    canvas_3d.after_idle(canvas_3d.fit_to_scene)


def _add_controls(parent: tk.Misc, canvas_3d: Tkinter3DCanvas) -> tk.Frame:
    controls = tk.Frame(parent)
    controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

    tk.Button(controls, text="Fit", command=canvas_3d.fit_to_scene).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Reset", command=canvas_3d.reset_camera).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Top", command=canvas_3d.set_top_view).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Side", command=canvas_3d.set_side_view).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Front", command=canvas_3d.set_front_view).pack(side=tk.LEFT, padx=3)
    tk.Button(controls, text="Iso", command=canvas_3d.set_iso_view).pack(side=tk.LEFT, padx=3)

    tk.Label(
        controls,
        text="Drag with left mouse button to orbit; use the wheel to zoom.",
    ).pack(side=tk.RIGHT, padx=6)
    return controls


def create_stiffened_cylinder_demo(root: tk.Misc) -> tk.Toplevel:
    """Open the demonstration in a child window."""
    demo_window = tk.Toplevel(root)
    demo_window.title("Tkinter 3D - Stiffened Cylinder Demo")
    demo_window.geometry("1120x800")
    demo_window.minsize(500, 400)

    canvas_3d = Tkinter3DCanvas(demo_window, width=1120, height=720, bg="white")
    _add_controls(demo_window, canvas_3d)
    canvas_3d.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
    populate_stiffened_cylinder(canvas_3d)
    return demo_window


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Tkinter 3D - Stiffened Cylinder Demo")
    root.geometry("1120x800")
    root.minsize(500, 400)

    canvas_3d = Tkinter3DCanvas(root, width=1120, height=720, bg="white")
    _add_controls(root, canvas_3d)
    canvas_3d.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
    populate_stiffened_cylinder(canvas_3d)

    root.mainloop()
