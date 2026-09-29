"""Reusable Qt panel for embedded Matplotlib calibration figures."""

from __future__ import annotations

from collections.abc import Callable

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from qtpy.QtCore import QEvent, QObject, QTimer, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from qtpy.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget

__all__ = ["CalibrationFigurePanel", "FigurePanel"]


def _configure_matplotlib_figure(figure: Figure) -> None:
    """Keep axes decorations inside an embedded canvas as it is resized."""
    if hasattr(figure, "set_layout_engine"):
        figure.set_layout_engine("constrained")
    else:  # pragma: no cover - compatibility with old Matplotlib releases
        figure.set_constrained_layout(True)


def _luminance(color: QColor) -> float:
    """Return perceived sRGB luminance on a 0--255 scale."""
    return (
        0.2126 * color.red()
        + 0.7152 * color.green()
        + 0.0722 * color.blue()
    )


def _rendered_background_color(widget: QWidget) -> QColor:
    """Return the background Qt actually paints, including stylesheets."""
    if widget.isVisible() and widget.width() > 4 and widget.height() > 2:
        image = widget.grab().toImage()
        if not image.isNull():
            color = image.pixelColor(
                max(0, image.width() - 3), image.height() // 2
            )
            if color.alpha() > 0:
                return color
    return widget.palette().color(widget.backgroundRole())


class _ToolbarIconContrast(QObject):
    """Retint Matplotlib toolbar icons for the rendered Qt background.

    Matplotlib chooses black or light icons while the toolbar is constructed.
    Plot panels are top-level widgets at that point, before Napari reparents
    them into its dark-themed dock, so that one-time choice can be wrong.
    """

    _REFRESH_EVENTS = {
        QEvent.Show,
        QEvent.ParentChange,
        QEvent.PaletteChange,
        QEvent.StyleChange,
    }

    def __init__(self, toolbar) -> None:
        super().__init__(toolbar)
        self.toolbar = toolbar
        self._refresh_pending = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        self._source_icons = []
        for action in toolbar.actions():
            icon = action.icon()
            if not icon.isNull():
                self._source_icons.append((action, QIcon(icon)))

    def eventFilter(self, watched, event):
        if event.type() in self._REFRESH_EVENTS:
            self.schedule_refresh()
        return False

    def schedule_refresh(self) -> None:
        # Python state can already be cleared when Qt emits teardown events.
        timer = getattr(self, "_refresh_timer", None)
        if timer is None or getattr(self, "_refresh_pending", True):
            return
        self._refresh_pending = True
        try:
            timer.start(0)
        except RuntimeError:  # the owning toolbar has been deleted
            self._refresh_pending = False

    def _background_color(self) -> QColor:
        """Sample the toolbar after Qt stylesheets have painted it."""
        return _rendered_background_color(self.toolbar)

    def refresh(self) -> None:
        if not getattr(self, "_source_icons", None):
            return
        self._refresh_pending = False
        try:
            background = self._background_color()
            icon_color = (
                QColor("#f0f2f5")
                if _luminance(background) < 140
                else QColor("#202124")
            )
            icon_size = self.toolbar.iconSize()
            for action, source_icon in self._source_icons:
                source = source_icon.pixmap(icon_size)
                if source.isNull():
                    continue
                tinted = QPixmap(source)
                painter = QPainter(tinted)
                painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
                painter.fillRect(tinted.rect(), icon_color)
                painter.end()
                action.setIcon(QIcon(tinted))
        except RuntimeError:  # toolbar/actions may have been deleted by Qt
            return


def _style_matplotlib_toolbar(toolbar) -> None:
    """Make a Matplotlib toolbar respond to its eventual host theme."""
    if hasattr(toolbar, "_raman_icon_contrast"):
        return
    contrast = _ToolbarIconContrast(toolbar)
    toolbar._raman_icon_contrast = contrast
    toolbar.installEventFilter(contrast)
    contrast.schedule_refresh()


class _MatplotlibBackgroundTheme(QObject):
    """Keep an embedded Matplotlib canvas readable on its Qt host theme."""

    _REFRESH_EVENTS = {
        QEvent.Show,
        QEvent.ParentChange,
        QEvent.PaletteChange,
        QEvent.StyleChange,
    }

    def __init__(self, owner, figure, canvas, toolbar, checkbox) -> None:
        super().__init__(owner)
        self.owner = owner
        self.figure = figure
        self.canvas = canvas
        self.toolbar = toolbar
        self.checkbox = checkbox
        self._applying = False
        self._refresh_pending = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        for widget in (owner, canvas, toolbar):
            widget.installEventFilter(self)
        self._install_canvas_hooks()

    def eventFilter(self, watched, event):
        # Cyclic GC can clear this wrapper's Python attributes before Qt
        # finishes delivering events from the owner's destruction. Teardown
        # must be a no-op, not an uncaught exception from a Qt virtual method.
        if event.type() in self._REFRESH_EVENTS and not getattr(self, "_applying", True):
            self.schedule_refresh()
        return False

    def _install_canvas_hooks(self) -> None:
        """Apply theme immediately before on-screen draws and exports."""
        original_draw = self.canvas.draw
        original_print_figure = self.canvas.print_figure

        def themed_draw(*args, **kwargs):
            self.apply(redraw=False)
            return original_draw(*args, **kwargs)

        def themed_print_figure(*args, **kwargs):
            self.apply(redraw=False)
            return original_print_figure(*args, **kwargs)

        self.canvas.draw = themed_draw
        self.canvas.print_figure = themed_print_figure

    def schedule_refresh(self) -> None:
        timer = getattr(self, "_refresh_timer", None)
        if timer is None or getattr(self, "_refresh_pending", True):
            return
        self._refresh_pending = True
        try:
            timer.start(0)
        except RuntimeError:  # the owning plot has been deleted
            self._refresh_pending = False

    def _theme_colors(self):
        if self.checkbox.isChecked():
            return "white", "#202124"
        background = _rendered_background_color(self.toolbar)
        foreground = (
            "#f0f2f5" if _luminance(background) < 140 else "#202124"
        )
        return "none", foreground

    def _canvas_background_matches(self, white_background: bool) -> bool:
        expected = "white" if white_background else "transparent"
        return self.canvas.property("ramanBackgroundMode") == expected

    def _apply_canvas_background(self, white_background: bool) -> None:
        mode = "white" if white_background else "transparent"
        self.canvas.setProperty("ramanBackgroundMode", mode)
        self.canvas.setAttribute(Qt.WA_TranslucentBackground, True)
        self.canvas.setAttribute(Qt.WA_NoSystemBackground, not white_background)
        self.canvas.setAttribute(Qt.WA_OpaquePaintEvent, white_background)
        self.canvas.setAutoFillBackground(False)
        self.canvas.setStyleSheet(
            "background: white;" if white_background else "background: transparent;"
        )
        palette = self.canvas.palette()
        palette.setColor(
            QPalette.Window,
            QColor("white") if white_background else QColor(0, 0, 0, 0),
        )
        self.canvas.setPalette(palette)

    @staticmethod
    def _apply_axis_colors(ax, facecolor, foreground) -> None:
        ax.set_facecolor(facecolor)
        ax.tick_params(axis="both", which="both", colors=foreground)
        for label in (
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
            ax.xaxis.offsetText,
            ax.yaxis.offsetText,
            getattr(ax, "_left_title", None),
            getattr(ax, "_right_title", None),
        ):
            if label is not None:
                label.set_color(foreground)
        for spine in ax.spines.values():
            spine.set_color(foreground)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(foreground)
            legend.get_title().set_color(foreground)
            legend.get_frame().set_facecolor(facecolor)
            legend.get_frame().set_edgecolor(foreground)

    def apply(self, *, redraw=True) -> None:
        if getattr(self, "_applying", True):
            return
        self._applying = True
        try:
            facecolor, foreground = self._theme_colors()
            white_background = self.checkbox.isChecked()
            if not self._canvas_background_matches(white_background):
                self._apply_canvas_background(white_background)
            self.figure.set_facecolor(facecolor)
            for text in self.figure.texts:
                text.set_color(foreground)
            for ax in self.figure.axes:
                self._apply_axis_colors(ax, facecolor, foreground)
        finally:
            self._applying = False
        if redraw:
            self.canvas.draw_idle()

    def refresh(self) -> None:
        self._refresh_pending = False
        try:
            self.apply()
        except RuntimeError:
            pass


def _make_white_background_checkbox(owner) -> QCheckBox:
    """Create or return a plot owner's public background checkbox."""
    existing = getattr(owner, "white_background_check", None)
    if existing is not None:
        return existing
    checkbox = QCheckBox("White background")
    checkbox.setChecked(False)
    checkbox.setToolTip(
        "Use a white plot and saved-image background. Uncheck for a "
        "transparent plot that follows the Napari theme."
    )
    owner.white_background_check = checkbox
    return checkbox


def _add_matplotlib_background_control(
    owner,
    layout,
    figure,
    canvas,
    toolbar,
    *,
    add_to_layout=True,
) -> QCheckBox:
    """Bind the shared transparent/white Matplotlib background switch."""
    checkbox = _make_white_background_checkbox(owner)
    if add_to_layout:
        layout.addWidget(checkbox)
    manager = _MatplotlibBackgroundTheme(
        owner, figure, canvas, toolbar, checkbox
    )
    owner._matplotlib_background_theme = manager
    checkbox.toggled.connect(lambda _checked: manager.apply())
    manager.apply(redraw=False)
    return checkbox


class CalibrationFigurePanel(QWidget):
    """Matplotlib canvas with guidance suitable for calibration selectors.

    The canvas is installed before :attr:`figure` is exposed, so callers can
    safely pass the figure to a selector that immediately registers Matplotlib
    callbacks.  Keyboard focus is kept on the canvas for selector shortcuts.
    """

    def __init__(
        self,
        title: str = "Calibration",
        instructions: str = "",
        *,
        parent: QWidget | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._cleanup_callbacks: list[Callable[[], None]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.instructions = QLabel(instructions)
        self.instructions.setWordWrap(True)
        self.instructions.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.instructions.setVisible(bool(instructions))
        layout.addWidget(self.instructions)

        self.figure = Figure(figsize=figsize)
        _configure_matplotlib_figure(self.figure)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        _style_matplotlib_toolbar(self.toolbar)
        _add_matplotlib_background_control(
            self, layout, self.figure, self.canvas, self.toolbar
        )

        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.focus_canvas()

    def focus_canvas(self) -> None:
        """Give keyboard shortcuts to the Matplotlib canvas."""
        self.canvas.setFocus(Qt.FocusReason.OtherFocusReason)

    def add_cleanup(self, callback: Callable[[], None]) -> None:
        """Run *callback* once when the panel is closed.

        Selectors use this to disconnect Matplotlib callbacks while keeping
        their selected-point data available to the owning workflow.
        """
        self._cleanup_callbacks.append(callback)

    def _run_cleanup(self) -> None:
        callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
        for callback in callbacks:
            try:
                callback()
            except (ReferenceError, RuntimeError):
                # Qt may already be tearing down the canvas during shutdown.
                pass

    def closeEvent(self, event) -> None:
        """Release selector callbacks before the canvas is destroyed."""
        self._run_cleanup()
        super().closeEvent(event)

    def showEvent(self, event) -> None:
        """Restore canvas focus whenever the panel becomes visible."""
        super().showEvent(event)
        self.focus_canvas()


# Short name retained for concise construction at plot launch sites.
FigurePanel = CalibrationFigurePanel
