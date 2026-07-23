"""Interactive point selection for pixel-to-stage calibration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["StagePointPicker"]


class StagePointPicker:
    """Select one calibration point from each image frame.

    Controls
    --------
    Left click
        Select or replace the point in the current frame.
    Enter
        Advance to the next frame.
    Backspace
        Return to the previous frame.
    R
        Clear the selected point in the current frame.
    N
        Mark the current frame as unavailable and advance.
    Mouse wheel
        Zoom when ``mpl-interactions`` is installed.
    Middle-button drag
        Pan when ``mpl-interactions`` is installed.

    Attributes
    ----------
    points
        Array with shape ``(number_of_frames, 2)`` containing selected
        ``(x, y)`` pixel coordinates. Unselected frames contain NaN.
    """

    def __init__(
        self,
        images: np.ndarray,
        cmap: str = "gray",
    ) -> None:
        self.images = np.asarray(images)

        if self.images.ndim < 3:
            raise ValueError(
                "images must contain one or more two-dimensional frames."
            )

        if len(self.images) == 0:
            raise ValueError("images must contain at least one frame.")

        self.n_frames = len(self.images)
        self.points = np.full(
            (self.n_frames, 2),
            np.nan,
            dtype=float,
        )
        self.current_index = 0

        self.figure, self.axes = plt.subplots()
        self.image_artist = self.axes.imshow(
            self.images[0],
            cmap=cmap,
        )

        (self.marker,) = self.axes.plot(
            [],
            [],
            "r+",
            markersize=15,
            markeredgewidth=2,
        )

        self.title = self.axes.set_title("")

        self._zoom_disconnect = None
        self._pan_handler = None

        self._enable_optional_navigation()

        self.figure.canvas.mpl_connect(
            "button_press_event",
            self._on_click,
        )
        self.figure.canvas.mpl_connect(
            "key_press_event",
            self._on_key,
        )

        self._draw()

    @property
    def i(self) -> int:
        """Current frame index retained for compatibility."""
        return self.current_index

    @i.setter
    def i(self, value: int) -> None:
        self.current_index = int(value)

    @property
    def fig(self):
        """Matplotlib figure retained for compatibility."""
        return self.figure

    @property
    def ax(self):
        """Matplotlib axes retained for compatibility."""
        return self.axes

    @property
    def im(self):
        """Matplotlib image artist retained for compatibility."""
        return self.image_artist

    def _enable_optional_navigation(self) -> None:
        """Enable zooming and panning when mpl-interactions is installed."""
        try:
            from mpl_interactions import panhandler, zoom_factory
        except ImportError:
            return

        self._zoom_disconnect = zoom_factory(self.axes)
        self._pan_handler = panhandler(
            self.figure,
            button=2,
        )

    def _draw(self) -> None:
        """Display the current frame and its selected point."""
        image = self.images[self.current_index]

        self.image_artist.set_data(image)
        self.image_artist.set_clim(
            float(np.nanmin(image)),
            float(np.nanmax(image)),
        )

        current_point = self.points[self.current_index]

        if np.isnan(current_point).any():
            self.marker.set_data([], [])
        else:
            self.marker.set_data(
                [current_point[0]],
                [current_point[1]],
            )

        completed = int(
            (~np.isnan(self.points).any(axis=1)).sum()
        )

        self.title.set_text(
            f"Frame {self.current_index}/{self.n_frames - 1} "
            f"({completed} marked)\n"
            "Click point | Enter next | Backspace previous | "
            "R reset | N unavailable"
        )

        self.figure.canvas.draw_idle()

    def _on_click(self, event) -> None:
        """Record a left-click inside the calibration image."""
        if event.inaxes is not self.axes:
            return

        if event.button != 1:
            return

        if event.xdata is None or event.ydata is None:
            return

        self.points[self.current_index] = [
            event.xdata,
            event.ydata,
        ]

        self._draw()

    def _advance(self) -> None:
        """Advance to the next frame or finish point selection."""
        if self.current_index < self.n_frames - 1:
            self.current_index += 1
            self._draw()
            return

        print("Finished picking calibration points.")
        plt.close(self.figure)

    def _on_key(self, event) -> None:
        """Handle point-selection keyboard controls."""
        key = event.key

        if key == "enter":
            self._advance()

        elif key == "backspace":
            if self.current_index > 0:
                self.current_index -= 1
                self._draw()

        elif key and key.lower() == "r":
            self.points[self.current_index] = np.nan
            self._draw()

        elif key and key.lower() == "n":
            self.points[self.current_index] = np.nan
            self._advance()

    def close(self) -> None:
        """Close the point-selection window."""
        plt.close(self.figure)