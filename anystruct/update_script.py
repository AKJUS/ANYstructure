file_path = r'C:\Github\ANYstructure\anystruct\tkinter_3d_canvas_thickness_v6.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Modify __init__
content = content.replace(
    '''        self._animation_cache: List[List[Tuple[str, Tuple[float, ...], Dict[str, Any]]]] = []
        self._is_capturing_animation = False
        self._current_animation_frame: List[Tuple[str, Tuple[float, ...], Dict[str, Any]]] = []
        self._is_playing_animation = False
        self._animation_frame_index = 0
        self._animation_after_id: Optional[str] = None
        self._animation_fps = 30''',
    '''        self._animation_cache: List[Dict[str, Any]] = []
        self._is_capturing_animation = False
        self._is_playing_animation = False
        self._animation_frame_index = 0
        self._animation_after_id: Optional[str] = None
        self._animation_fps = 30
        self._polygon_pool: List[int] = []'''
)

# 2. Modify _clear_canvas_only
content = content.replace(
    '''    def _clear_canvas_only(self) -> None:
        self.canvas.delete("all")''',
    '''    def _clear_canvas_only(self) -> None:
        self.canvas.delete("all")
        self._polygon_pool.clear()'''
)

# 3. Modify mouse drag
content = content.replace(
    '''    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self._is_playing_animation:
            self.stop_animation()
            
        if not self._is_dragging:
            return''',
    '''    def _on_mouse_drag(self, event: tk.Event) -> None:
        if not self._is_dragging:
            return'''
)

# 4. Modify mouse wheel
content = content.replace(
    '''    def _on_mouse_wheel(self, event: tk.Event) -> str:
        if self._is_playing_animation:
            self.stop_animation()
            
        event_num = getattr(event, "num", None)''',
    '''    def _on_mouse_wheel(self, event: tk.Event) -> str:
        event_num = getattr(event, "num", None)'''
)

# 5. Modify begin_animation_cache to _animation_tick
old_anim_block = '''    def begin_animation_cache(self) -> None:
        self.stop_animation()
        self._animation_cache.clear()

    def capture_animation_frame(self) -> None:
        self._is_capturing_animation = True
        self._current_animation_frame = []
        self.redraw()
        self._animation_cache.append(self._current_animation_frame)
        self._current_animation_frame = []
        self._is_capturing_animation = False

    def play_animation(self, fps: int = 30) -> None:
        if not self._animation_cache:
            return
        self.stop_animation()
        self._animation_fps = max(1, fps)
        self._is_playing_animation = True
        self._animation_frame_index = 0
        self._animation_tick()

    def stop_animation(self) -> None:
        self._is_playing_animation = False
        if self._animation_after_id is not None:
            self.after_cancel(self._animation_after_id)
            self._animation_after_id = None
        # Restore normal view
        self._request_redraw(interactive=False)

    def _animation_tick(self) -> None:
        if not self._is_playing_animation or not self._animation_cache:
            return
        
        frame = self._animation_cache[self._animation_frame_index]
        self._clear_canvas_only()
        for kind, coords, kwargs in frame:
            if kind == "polygon":
                self.canvas.create_polygon(*coords, **kwargs)
            elif kind == "line":
                self.canvas.create_line(*coords, **kwargs)
            elif kind == "text":
                self.canvas.create_text(*coords, **kwargs)

        self._draw_thickness_legend()
        self._draw_axis_indicator()

        self._animation_frame_index = (self._animation_frame_index + 1) % len(self._animation_cache)
        delay_ms = max(1, int(1000.0 / self._animation_fps))
        self._animation_after_id = self.after(delay_ms, self._animation_tick)'''

new_anim_block = '''    def begin_animation_cache(self) -> None:
        self.stop_animation()
        self._animation_cache.clear()

    def capture_animation_frame(self) -> None:
        self._is_capturing_animation = True
        
        primitives_full = self._get_world_primitives("full")
        np_vertices_full = self._np_vertices_cache["full"]
        
        primitives_fast = self._get_world_primitives("fast")
        np_vertices_fast = self._np_vertices_cache.get("fast", np_vertices_full)
        
        self._animation_cache.append({
            "primitives_full": primitives_full,
            "np_vertices_full": np_vertices_full,
            "primitives_fast": primitives_fast,
            "np_vertices_fast": np_vertices_fast,
            "legend": self._thickness_legend,
        })
        self._is_capturing_animation = False

    def play_animation(self, fps: int = 30) -> None:
        if not self._animation_cache:
            return
        self.stop_animation()
        self._animation_fps = max(1, fps)
        self._is_playing_animation = True
        self._animation_frame_index = 0
        self._animation_tick()

    def stop_animation(self) -> None:
        self._is_playing_animation = False
        if self._animation_after_id is not None:
            self.after_cancel(self._animation_after_id)
            self._animation_after_id = None
        # Restore normal view
        self._request_redraw(interactive=False)

    def _animation_tick(self) -> None:
        if not self._is_playing_animation or not self._animation_cache:
            return
        
        frame_data = self._animation_cache[self._animation_frame_index]
        
        orig_primitives_full = self._world_primitive_cache.get("full")
        orig_vertices_full = self._np_vertices_cache.get("full")
        orig_primitives_fast = self._world_primitive_cache.get("fast")
        orig_vertices_fast = self._np_vertices_cache.get("fast")
        orig_legend = self._thickness_legend
        
        self._world_primitive_cache["full"] = frame_data["primitives_full"]
        self._np_vertices_cache["full"] = frame_data["np_vertices_full"]
        self._world_primitive_cache["fast"] = frame_data["primitives_fast"]
        self._np_vertices_cache["fast"] = frame_data["np_vertices_fast"]
        self._thickness_legend = frame_data["legend"]
        
        self.redraw()
        
        if orig_primitives_full is not None:
            self._world_primitive_cache["full"] = orig_primitives_full
        if orig_vertices_full is not None:
            self._np_vertices_cache["full"] = orig_vertices_full
        if orig_primitives_fast is not None:
            self._world_primitive_cache["fast"] = orig_primitives_fast
        if orig_vertices_fast is not None:
            self._np_vertices_cache["fast"] = orig_vertices_fast
        self._thickness_legend = orig_legend

        self._animation_frame_index = (self._animation_frame_index + 1) % len(self._animation_cache)
        delay_ms = max(1, int(1000.0 / self._animation_fps))
        self._animation_after_id = self.after(delay_ms, self._animation_tick)'''

content = content.replace(old_anim_block, new_anim_block)


# 6. Rewrite redraw
old_redraw_start = '''    def redraw(self) -> None:
        """Render the scene; static world geometry is reused from cache."""
        if not self.winfo_exists() or not self.canvas.winfo_exists():
            return

        self.width = max(1, self.canvas.winfo_width())
        self.height = max(1, self.canvas.winfo_height())
        if not self._is_capturing_animation:
            self._clear_canvas_only()

        interactive = self._interactive_render'''

new_redraw_start = '''    def redraw(self) -> None:
        """Render the scene; static world geometry is reused from cache."""
        if not self.winfo_exists() or not self.canvas.winfo_exists():
            return

        self.width = max(1, self.canvas.winfo_width())
        self.height = max(1, self.canvas.winfo_height())

        interactive = self._interactive_render'''
content = content.replace(old_redraw_start, new_redraw_start)

# Replace the drawing loop in redraw
old_redraw_loop = '''        target_list = self._current_animation_frame if self._is_capturing_animation else None

        for _phase, _depth, _layer, primitive, coords in render_items:
            kind = primitive.get("kind")
            if not kind:
                continue
            if kind == "line":
                kwargs = {
                    "fill": primitive["color"],
                    "width": primitive["width"],
                }
                if target_list is not None:
                    target_list.append(("line", coords, kwargs))
                else:
                    self.canvas.create_line(*coords, **kwargs)
            elif kind == "text":
                kwargs = {
                    "text": primitive["text"],
                    "fill": primitive["color"],
                    "font": primitive["font"],
                    "anchor": primitive["anchor"],
                }
                if target_list is not None:
                    target_list.append(("text", coords, kwargs))
                else:
                    self.canvas.create_text(*coords, **kwargs)
            else:
                outline = "" if (not self.show_mesh_lines) or (interactive and primitive.get("fast_no_outline")) else primitive["outline"]
                fill_color = primitive["color"]
                if not primitive.get("_front_facing", True):
                    fill_color = primitive.get("back_color") or fill_color
                stipple = "" if interactive else primitive.get("stipple", "")
                kwargs = {
                    "fill": fill_color,
                    "outline": outline,
                    "width": primitive["width"],
                    "stipple": stipple,
                }
                if primitive.get("tags"):
                    kwargs["tags"] = primitive.get("tags")
                
                if target_list is not None:
                    target_list.append(("polygon", coords, kwargs))
                else:
                    self.canvas.create_polygon(*coords, **kwargs)

        overlay_items.sort(key=lambda item: (
            item[0],
            -(item[1] - item[2] * layer_epsilon),
            item[2],
        ))
        for _phase, _depth, _layer, primitive, coords in overlay_items:
            if primitive["kind"] == "line":
                kwargs = {
                    "fill": primitive["color"],
                    "width": primitive["width"],
                }
                if target_list is not None:
                    target_list.append(("line", coords, kwargs))
                else:
                    self.canvas.create_line(*coords, **kwargs)
            elif primitive["kind"] == "text":
                kwargs = {
                    "text": primitive["text"],
                    "fill": primitive["color"],
                    "font": primitive["font"],
                    "anchor": primitive["anchor"],
                }
                if target_list is not None:
                    target_list.append(("text", coords, kwargs))
                else:
                    self.canvas.create_text(*coords, **kwargs)

        if not interactive and not self._is_capturing_animation:
            self._draw_thickness_legend()
        if not self._is_capturing_animation:
            self._draw_axis_indicator()'''

new_redraw_loop = '''        self.canvas.delete("overlay_item")

        polygon_items = []
        for item in render_items:
            primitive = item[3]
            if primitive.get("kind") == "polygon":
                polygon_items.append((primitive, item[4]))
            else:
                overlay_items.append(item)

        pool_size = len(self._polygon_pool)
        needed_polygons = len(polygon_items)
        if needed_polygons > pool_size:
            for _ in range(needed_polygons - pool_size):
                item_id = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, state="hidden")
                self._polygon_pool.append(item_id)

        for i, (primitive, coords) in enumerate(polygon_items):
            item_id = self._polygon_pool[i]
            outline = "" if (not self.show_mesh_lines) or (interactive and primitive.get("fast_no_outline")) else primitive["outline"]
            fill_color = primitive["color"]
            if not primitive.get("_front_facing", True):
                fill_color = primitive.get("back_color") or fill_color
            stipple = "" if interactive else primitive.get("stipple", "")
            
            kwargs = {
                "fill": fill_color,
                "outline": outline,
                "width": primitive["width"],
                "stipple": stipple,
                "state": "normal",
            }
            if primitive.get("tags"):
                kwargs["tags"] = primitive.get("tags")
                
            self.canvas.coords(item_id, *coords)
            self.canvas.itemconfig(item_id, **kwargs)

        for i in range(needed_polygons, len(self._polygon_pool)):
            self.canvas.itemconfig(self._polygon_pool[i], state="hidden")

        overlay_items.sort(key=lambda item: (
            item[0],
            -(item[1] - item[2] * layer_epsilon),
            item[2],
        ))
        
        for _phase, _depth, _layer, primitive, coords in overlay_items:
            if primitive["kind"] == "line":
                kwargs = {
                    "fill": primitive["color"],
                    "width": primitive["width"],
                    "tags": "overlay_item",
                }
                self.canvas.create_line(*coords, **kwargs)
            elif primitive["kind"] == "text":
                kwargs = {
                    "text": primitive["text"],
                    "fill": primitive["color"],
                    "font": primitive["font"],
                    "anchor": primitive["anchor"],
                    "tags": "overlay_item",
                }
                self.canvas.create_text(*coords, **kwargs)

        if not interactive and not self._is_capturing_animation:
            self._draw_thickness_legend()
        if not self._is_capturing_animation:
            self._draw_axis_indicator()'''
content = content.replace(old_redraw_loop, new_redraw_loop)

# 7. Update tags in _draw_thickness_legend and _draw_axis_indicator
old_legend_axis = '''    def _draw_thickness_legend(self) -> None:
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
        max_title_chars = max(12, int((panel_width - 2 * padding) / 7))
        title_lines = self._legend_text_lines(title_text, max_title_chars)
        title_y = 18
        for line in title_lines:
            self.canvas.create_text(
                left + padding,
                title_y,
                text=line,
                anchor="nw",
                font=("TkDefaultFont", 10, "bold"),
                fill="#202020",
            )
            title_y += 15

        values = list(legend.get("values", []))
        minimum = float(legend["minimum"])
        maximum = float(legend["maximum"])
        bar_top = max(54, title_y + 8)
        available_height = max(80, self.height - (bar_top + 16))

        # A short set of distinct thicknesses is clearer as labelled swatches.
        if 1 <= len(values) <= 10:
            values = sorted(values, reverse=True)
            row_height = min(34, max(23, available_height // len(values)))
            swatch_width = 34
            y_coord = bar_top
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

    def _draw_axis_indicator(self) -> None:
        if not self._show_axis_indicator:
            return

        plot_width = self._plot_width()
        if plot_width < 95 or self.height < 95:
            return

        origin_x = min(max(58.0, plot_width * 0.065), max(58.0, plot_width - 78.0))
        origin_y = max(58.0, self.height - 64.0)
        axis_length = min(58.0, max(34.0, min(plot_width, self.height) * 0.085))
        right, camera_up, forward = self.camera.basis()

        axes = [
            ("X", Point3D(1.0, 0.0, 0.0), "#9b111e"),
            ("Y", Point3D(0.0, 1.0, 0.0), "#159447"),
            ("Z", Point3D(0.0, 0.0, 1.0), "#0d47a1"),
        ]

        def screen_delta(vector: Point3D) -> Tuple[float, float]:
            return (
                vector.dot(right) * axis_length,
                -vector.dot(camera_up) * axis_length,
            )

        # Draw the visually deeper axis first so overlapping arrows read like a
        # small 3D triad instead of a flat logo.
        axes = sorted(axes, key=lambda item: item[1].dot(forward), reverse=True)

        self.canvas.create_oval(
            origin_x - 2,
            origin_y - 2,
            origin_x + 2,
            origin_y + 2,
            fill="#202020",
            outline="",
        )
        for label, vector, color in axes:
            dx, dy = screen_delta(vector)
            end_x = origin_x + dx
            end_y = origin_y + dy
            if abs(dx) + abs(dy) < 3.0:
                continue
            self.canvas.create_line(
                origin_x,
                origin_y,
                end_x,
                end_y,
                fill=color,
                width=2,
                arrow=tk.LAST,
                arrowshape=(9, 11, 4),
            )
            label_offset = 11.0
            length = max(math.hypot(dx, dy), 1.0)
            self.canvas.create_text(
                end_x + label_offset * dx / length,
                end_y + label_offset * dy / length,
                text=label,
                fill=color,
                font=("TkDefaultFont", 10, "bold"),
            )'''

new_legend_axis = '''    def _draw_thickness_legend(self) -> None:
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
            tags="overlay_item"
        )

        padding = 14
        title = str(legend.get("title", "Plate thickness"))
        unit = str(legend.get("unit", ""))
        title_text = f"{title} [{unit}]" if unit else title
        max_title_chars = max(12, int((panel_width - 2 * padding) / 7))
        title_lines = self._legend_text_lines(title_text, max_title_chars)
        title_y = 18
        for line in title_lines:
            self.canvas.create_text(
                left + padding,
                title_y,
                text=line,
                anchor="nw",
                font=("TkDefaultFont", 10, "bold"),
                fill="#202020",
                tags="overlay_item"
            )
            title_y += 15

        values = list(legend.get("values", []))
        minimum = float(legend["minimum"])
        maximum = float(legend["maximum"])
        bar_top = max(54, title_y + 8)
        available_height = max(80, self.height - (bar_top + 16))

        # A short set of distinct thicknesses is clearer as labelled swatches.
        if 1 <= len(values) <= 10:
            values = sorted(values, reverse=True)
            row_height = min(34, max(23, available_height // len(values)))
            swatch_width = 34
            y_coord = bar_top
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
                    tags="overlay_item"
                )
                self.canvas.create_text(
                    left + padding + swatch_width + 10,
                    y_coord + 8,
                    text=self._format_legend_value(value),
                    anchor="w",
                    fill="#202020",
                    tags="overlay_item"
                )
                y_coord += row_height
            return

        # Continuous or highly populated scales are rendered as a gradient.
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
                tags="overlay_item"
            )
        self.canvas.create_rectangle(
            bar_left,
            bar_top,
            bar_right,
            bar_bottom,
            fill="",
            outline="#505050",
            width=1,
            tags="overlay_item"
        )

        tick_count = 6
        for index in range(tick_count):
            fraction = index / (tick_count - 1)
            value = maximum - fraction * (maximum - minimum)
            y_coord = bar_top + fraction * (bar_bottom - bar_top)
            self.canvas.create_line(bar_right, y_coord, bar_right + 5, y_coord, fill="#505050", tags="overlay_item")
            self.canvas.create_text(
                bar_right + 10,
                y_coord,
                text=self._format_legend_value(value),
                anchor="w",
                fill="#202020",
                tags="overlay_item"
            )

    def _draw_axis_indicator(self) -> None:
        if not self._show_axis_indicator:
            return

        plot_width = self._plot_width()
        if plot_width < 95 or self.height < 95:
            return

        origin_x = min(max(58.0, plot_width * 0.065), max(58.0, plot_width - 78.0))
        origin_y = max(58.0, self.height - 64.0)
        axis_length = min(58.0, max(34.0, min(plot_width, self.height) * 0.085))
        right, camera_up, forward = self.camera.basis()

        axes = [
            ("X", Point3D(1.0, 0.0, 0.0), "#9b111e"),
            ("Y", Point3D(0.0, 1.0, 0.0), "#159447"),
            ("Z", Point3D(0.0, 0.0, 1.0), "#0d47a1"),
        ]

        def screen_delta(vector: Point3D) -> Tuple[float, float]:
            return (
                vector.dot(right) * axis_length,
                -vector.dot(camera_up) * axis_length,
            )

        # Draw the visually deeper axis first so overlapping arrows read like a
        # small 3D triad instead of a flat logo.
        axes = sorted(axes, key=lambda item: item[1].dot(forward), reverse=True)

        self.canvas.create_oval(
            origin_x - 2,
            origin_y - 2,
            origin_x + 2,
            origin_y + 2,
            fill="#202020",
            outline="",
            tags="overlay_item"
        )
        for label, vector, color in axes:
            dx, dy = screen_delta(vector)
            end_x = origin_x + dx
            end_y = origin_y + dy
            if abs(dx) + abs(dy) < 3.0:
                continue
            self.canvas.create_line(
                origin_x,
                origin_y,
                end_x,
                end_y,
                fill=color,
                width=2,
                arrow=tk.LAST,
                arrowshape=(9, 11, 4),
                tags="overlay_item"
            )
            label_offset = 11.0
            length = max(math.hypot(dx, dy), 1.0)
            self.canvas.create_text(
                end_x + label_offset * dx / length,
                end_y + label_offset * dy / length,
                text=label,
                fill=color,
                font=("TkDefaultFont", 10, "bold"),
                tags="overlay_item"
            )'''
content = content.replace(old_legend_axis, new_legend_axis)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated successfully")
