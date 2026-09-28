"""Live point-layer compatibility for Raman segmentation and tracking.

``RamanEngine.update_aim`` reads a point layer, performs comparatively slow
segmentation/tracking work, and later writes one result for every point it
read.  A user can edit the live napari layer during that interval.  Assigning
the stale result directly then either raises a shape mismatch or overwrites a
manual edit.

This module wraps an engine class without changing or monkeypatching the
installed ``raman-mda-engine`` package.  Each aiming update runs against a
private point-data proxy.  Its result is merged back into the live layer only
for rows that are still identical to the rows in the original snapshot.
Consequently, additions survive, deletions are not resurrected, and a manual
move wins over an in-flight tracking result.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from threading import RLock
from typing import Any, Callable

import numpy as np
from qtpy.QtCore import QCoreApplication, QObject, QThread, Qt, Signal, Slot

__all__ = ["make_live_cell_engine_type"]


@dataclass
class _Invocation:
    """One synchronous call dispatched to the Qt application thread."""

    callback: Callable[[], Any]
    result: Any = None
    error: BaseException | None = None


class _LayerCallBridge(QObject):
    """Serialize short napari layer operations on the GUI thread."""

    requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        app = QCoreApplication.instance()
        if app is not None and self.thread() != app.thread():
            self.moveToThread(app.thread())
        self.requested.connect(self._execute, Qt.BlockingQueuedConnection)

    @Slot(object)
    def _execute(self, invocation: _Invocation) -> None:
        try:
            invocation.result = invocation.callback()
        except BaseException as error:  # re-raised in the requesting thread
            invocation.error = error

    def call(self, callback: Callable[[], Any]) -> Any:
        app = QCoreApplication.instance()
        if app is None or QThread.currentThread() == self.thread():
            return callback()

        invocation = _Invocation(callback)
        self.requested.emit(invocation)
        if invocation.error is not None:
            raise invocation.error
        return invocation.result


class _PointDataProxy:
    """Expose frozen point data while delegating other layer attributes."""

    def __init__(self, layer: Any, data: np.ndarray) -> None:
        self._layer = layer
        self._data = np.array(data, copy=True)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._data = np.array(value, copy=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


@dataclass(frozen=True)
class _PointUpdateSession:
    source: Any
    layer: Any
    before: np.ndarray
    proxy: _PointDataProxy


@dataclass(frozen=True)
class _FrozenAnchorSource:
    """A source containing the transformed pattern for one cell anchor."""

    name: str
    anchor: np.ndarray
    transformer: Any

    def get_mda_points(self, event: Any, transform: bool = True) -> np.ndarray:
        del event
        points = np.array(self.anchor, dtype=float, copy=True).reshape(1, 2)
        if transform:
            return np.asarray(self.transformer.transform(points), dtype=float)
        return points


def _copy_point_data(layer: Any) -> np.ndarray:
    data = np.asarray(layer.data)
    if data.ndim != 2:
        raise ValueError("Point layer data must be a two-dimensional array.")
    return np.array(data, copy=True)


def _point_layer(source: Any) -> Any | None:
    layer = getattr(source, "_points", None)
    if layer is None or not hasattr(layer, "data"):
        return None
    if not callable(getattr(source, "get_mda_points", None)):
        return None
    return layer


def _rows_equal(left: np.ndarray, right: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left, right))


def _merge_tracked_rows(session: _PointUpdateSession) -> int:
    """Merge tracked coordinates into rows unchanged since the snapshot."""

    before = np.asarray(session.before)
    after = np.asarray(session.proxy.data)
    live = _copy_point_data(session.layer)

    if before.shape != after.shape or before.shape[1:] != live.shape[1:]:
        return 0

    used_live_rows: set[int] = set()
    changed = 0
    for original, tracked in zip(before, after, strict=True):
        if _rows_equal(original[-2:], tracked[-2:]):
            continue

        matching_row = None
        for live_index, live_row in enumerate(live):
            if live_index in used_live_rows:
                continue
            if _rows_equal(live_row, original):
                matching_row = live_index
                break

        if matching_row is None:
            # The row was deleted or manually changed while tracking ran.
            continue

        live[matching_row, -2:] = tracked[-2:]
        used_live_rows.add(matching_row)
        changed += 1

    if changed:
        session.layer.data = live
    return changed


def _is_splittable_cell_source(source: Any) -> bool:
    return (
        "cell" in str(getattr(source, "name", "")).lower()
        and _point_layer(source) is not None
        and hasattr(source, "_pos_idx")
        and hasattr(source, "_img_shape")
        and callable(getattr(getattr(source, "transformer", None), "transform", None))
    )


def _split_cell_sources(sources: list[Any], event: Any) -> list[Any]:
    """Freeze point-backed cell sources as one source per live anchor."""

    position = event.index.get("p")
    split_sources: list[Any] = []
    for source in sources:
        if not _is_splittable_cell_source(source):
            split_sources.append(source)
            continue

        data = _copy_point_data(source._points)
        position_index = int(source._pos_idx)
        image_shape = np.asarray(source._img_shape, dtype=float)
        if (
            position_index < 0
            or position_index >= data.shape[1]
            or image_shape.shape != (2,)
            or np.any(image_shape <= 0)
        ):
            split_sources.append(source)
            continue

        rows = data[data[:, position_index] == position]
        if len(rows) == 0:
            # Preserve the base engine's established empty-source behavior.
            split_sources.append(source)
            continue

        anchors = rows[:, -2:] / image_shape
        split_sources.extend(
            _FrozenAnchorSource(
                name=str(source.name),
                anchor=anchor,
                transformer=source.transformer,
            )
            for anchor in anchors
        )

    return split_sources


def _cell_points_and_target(
    sources: list[Any],
    position: int,
) -> tuple[np.ndarray, tuple[Any, Any, int] | None]:
    """Return live full-resolution cell points and the first writable layer."""

    point_sets: list[np.ndarray] = []
    target: tuple[Any, Any, int] | None = None
    for source in sources:
        if "cell" not in str(getattr(source, "name", "")).lower():
            continue
        layer = _point_layer(source)
        position_index = getattr(source, "_pos_idx", None)
        if layer is None or not isinstance(position_index, (int, np.integer)):
            continue

        data = _copy_point_data(layer)
        position_index = int(position_index)
        if position_index < 0 or position_index >= data.shape[1]:
            continue
        if target is None:
            target = (source, layer, position_index)
        rows = data[data[:, position_index] == position]
        if len(rows):
            point_sets.append(np.array(rows[:, -2:], dtype=float, copy=True))

    if not point_sets:
        return np.empty((0, 2), dtype=float), target
    return np.vstack(point_sets), target


def _represented_labels(
    labels: np.ndarray,
    points_yx: np.ndarray,
    scale: float,
) -> set[int]:
    """Return foreground labels containing full-resolution live points."""

    if len(points_yx) == 0:
        return set()
    points_yx = np.asarray(points_yx, dtype=float)
    finite = np.isfinite(points_yx).all(axis=1)
    pixels = np.zeros((len(points_yx), 2), dtype=int)
    pixels[finite] = np.floor(points_yx[finite] / scale).astype(int)
    in_bounds = (
        finite
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < labels.shape[0])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < labels.shape[1])
    )
    if not np.any(in_bounds):
        return set()
    return {
        int(value)
        for value in labels[pixels[in_bounds, 0], pixels[in_bounds, 1]]
        if value != 0
    }


def _append_new_targets(
    *,
    sources: list[Any],
    position: int,
    target: tuple[Any, Any, int],
    label_mask: np.ndarray,
    label_ids: np.ndarray,
    points_yx: np.ndarray,
    scale: float,
) -> int:
    """Atomically deduplicate and append proposed targets to a live layer."""

    live_points, _ = _cell_points_and_target(sources, position)
    represented = _represented_labels(label_mask, live_points, scale)
    keep = np.asarray(
        [int(label_id) not in represented for label_id in label_ids],
        dtype=bool,
    )
    points_yx = np.asarray(points_yx, dtype=float)[keep]
    if len(points_yx) == 0:
        return 0

    _, layer, position_index = target
    data = _copy_point_data(layer)
    dimensions = data.shape[1]
    if dimensions < 2 or position_index >= dimensions:
        return 0

    if len(data):
        template = np.array(data[0], dtype=float, copy=True)
    else:
        template = np.zeros(dimensions, dtype=float)
    rows = np.repeat(template[None, :], len(points_yx), axis=0)
    rows[:, position_index] = position
    rows[:, -2:] = points_yx

    add = getattr(layer, "add", None)
    if callable(add):
        add(rows)
    else:
        layer.data = np.vstack([data, rows])
    return len(rows)


@lru_cache(maxsize=None)
def make_live_cell_engine_type(base_engine_type: type) -> type:
    """Return an instance-local live-cell wrapper for ``base_engine_type``.

    The generated subclass accepts ``auto_add_new_cells`` in addition to the
    base constructor's arguments.  Automatic discovery is performed by an
    overridable post-update hook so its policy remains separate from the
    concurrency-safe point merge implemented here.
    """

    class LiveCellEngine(base_engine_type):
        def __init__(
            self,
            *args: Any,
            auto_add_new_cells: bool = False,
            **kwargs: Any,
        ) -> None:
            self._auto_add_new_cells = bool(auto_add_new_cells)
            self._live_cell_lock = RLock()
            super().__init__(*args, **kwargs)
            # Widgets construct engines on the GUI thread, so this object's
            # affinity is the correct one for subsequent layer operations.
            self._live_layer_bridge = _LayerCallBridge()

        @property
        def auto_add_new_cells(self) -> bool:
            return self._auto_add_new_cells

        def _capture_point_sessions(self) -> list[_PointUpdateSession]:
            source_layers = [
                (source, layer)
                for source in self.aiming_sources
                if (layer := _point_layer(source)) is not None
            ]
            snapshots = self._live_layer_bridge.call(
                lambda: [
                    _copy_point_data(layer)
                    for _, layer in source_layers
                ]
            )
            return [
                _PointUpdateSession(
                    source=source,
                    layer=layer,
                    before=snapshot,
                    proxy=_PointDataProxy(layer, snapshot),
                )
                for (source, layer), snapshot in zip(
                    source_layers,
                    snapshots,
                    strict=True,
                )
            ]

        def _after_live_aim_update(
            self,
            *,
            pos: int,
            event: Any,
            image: np.ndarray,
            use_same_img: bool,
            previous_mask: np.ndarray | None,
        ) -> None:
            """Run optional discovery after tracked rows reach live layers."""

            del image
            time_index = int(event.index.get("t", 0))
            if (
                not self._auto_add_new_cells
                or bool(getattr(self, "_batch", False))
                or time_index <= 0
                or use_same_img
                or previous_mask is None
            ):
                return

            current = getattr(self, "_last_segments", {}).get(pos)
            if current is None:
                return
            previous = np.asarray(previous_mask)
            current = np.asarray(current)
            if (
                previous.ndim != 2
                or current.ndim != 2
                or previous.shape != current.shape
                or not np.issubdtype(previous.dtype, np.integer)
                or not np.issubdtype(current.dtype, np.integer)
            ):
                return

            cell_points, target = self._live_layer_bridge.call(
                lambda: _cell_points_and_target(
                    list(self.aiming_sources),
                    pos,
                )
            )
            if target is None:
                return

            tracked_labels = None
            candidate = getattr(self, "_tracks", {}).get(pos)
            if candidate is not None:
                candidate = np.asarray(candidate)
                raw_pair = np.stack([previous, current])
                if (
                    candidate.shape == raw_pair.shape
                    and np.issubdtype(candidate.dtype, np.integer)
                    and not np.any(candidate < 0)
                    and not np.array_equal(candidate, raw_pair)
                ):
                    tracked_labels = np.array(candidate, copy=True)

            from .new_cell_targets import find_new_cell_targets

            label_ids, new_points = find_new_cell_targets(
                previous,
                current,
                cell_points,
                scale=float(getattr(self, "_scale", 1.0)),
                tracked_labels=tracked_labels,
                center_yx=getattr(self, "_circle_center", None),
            )
            if len(new_points) == 0:
                return

            label_mask = current if tracked_labels is None else tracked_labels[1]
            added_count = self._live_layer_bridge.call(
                lambda: _append_new_targets(
                    sources=list(self.aiming_sources),
                    position=pos,
                    target=target,
                    label_mask=label_mask,
                    label_ids=label_ids,
                    points_yx=new_points,
                    scale=float(getattr(self, "_scale", 1.0)),
                )
            )
            if added_count:
                print(
                    f"[auto-add] added {added_count} new cell target(s) "
                    f"at p={pos}, t={time_index}"
                )

        def update_aim(
            self,
            pos: int,
            event: Any,
            img: np.ndarray,
            use_same_img: bool = False,
        ) -> Any:
            with self._live_cell_lock:
                prior = getattr(self, "_last_segments", {}).get(pos)
                previous_mask = (
                    None if prior is None else np.array(prior, copy=True)
                )
                sessions = self._capture_point_sessions()
                for session in sessions:
                    session.source._points = session.proxy

                succeeded = False
                try:
                    result = super().update_aim(
                        pos,
                        event,
                        img,
                        use_same_img=use_same_img,
                    )
                    succeeded = True
                finally:
                    for session in sessions:
                        session.source._points = session.layer

                if succeeded:
                    self._live_layer_bridge.call(
                        lambda: [
                            _merge_tracked_rows(session)
                            for session in sessions
                        ]
                    )
                    self._after_live_aim_update(
                        pos=pos,
                        event=event,
                        image=img,
                        use_same_img=use_same_img,
                        previous_mask=previous_mask,
                    )
                return result

        def record_raman(self, event: Any) -> Any:
            with self._live_cell_lock:
                original_sources = self._sources
                split_sources = self._live_layer_bridge.call(
                    lambda: _split_cell_sources(list(original_sources), event)
                )
                self._sources = split_sources
                try:
                    return super().record_raman(event)
                finally:
                    self._sources = original_sources

    LiveCellEngine.__name__ = f"LiveCell{base_engine_type.__name__}"
    LiveCellEngine.__qualname__ = LiveCellEngine.__name__
    LiveCellEngine.__module__ = __name__
    return LiveCellEngine
