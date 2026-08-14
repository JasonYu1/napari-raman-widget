"""Pop-up matplotlib windows used to display calibration, spectra, and scans."""
import numpy as np
from napari_raman_widget.spectra import (
    filter_mean,
    subtract_spectral_bias,
    sum_detector_rows,
)
from napari_raman_widget.spectral_calibration import (
    PixelToWavenumberCalibration,
    save_pixel_to_wavenumber_calibration,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)


def _spectral_x(length, calibration, show_pixels):
    pixels = np.arange(length, dtype=float)
    if calibration is None or show_pixels:
        return pixels
    return calibration.transform(pixels)


def _set_spectral_line_axis(ax, lines, calibration, show_pixels):
    """Update spectrum line x data and label for the selected axis units."""
    for line in lines:
        line.set_xdata(
            _spectral_x(len(line.get_ydata()), calibration, show_pixels)
        )
    if calibration is not None and not show_pixels:
        ax.set_xlabel("Raman shift (cm⁻¹)")
    else:
        ax.set_xlabel("Pixel")
    ax.relim()
    ax.autoscale_view()


def _make_pixel_axis_checkbox(calibration, callback):
    checkbox = QCheckBox("Show pixels")
    checkbox.setChecked(calibration is None)
    checkbox.setEnabled(calibration is not None)
    if calibration is None:
        checkbox.setToolTip("Load a pixel-to-wavenumber calibration first")
    else:
        checkbox.setToolTip("Use detector pixels instead of Raman shift")
    checkbox.toggled.connect(callback)
    return checkbox


class CalibrationPlotWindow(QMainWindow):
    """Pop-up showing max projection of calibration images with point overlay."""

    def __init__(self, ds, title="Calibration result"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(700, 650)
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        central = QWidget()
        layout = QVBoxLayout(central)
        self.fig = Figure(figsize=(7, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        ax = self.fig.add_subplot(111)
        imgs = ds["imgs"].max(axis=0)
        ax.imshow(np.asarray(imgs))
        X = ds.dims["X"]
        Y = ds.dims["Y"]
        pix_BF = np.asarray(ds["rel_BF_pos"])
        ax.scatter(pix_BF[:, 0], pix_BF[:, 1], color="r", s=20)
        ax.set_title(title)
        self.fig.tight_layout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)


class DetectorImageWindow(QMainWindow):
    """Detector image with a switchable, row-summed spectrum view."""

    def __init__(
        self,
        frames,
        title="Detector image",
        spectral_calibration=None,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(800, 550)
        self.image = self._mean_detector_image(frames)
        self.spectral_calibration = spectral_calibration
        self._show_spectrum = False
        self._fixed_y_limits = None

        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )

        central = QWidget()
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()
        self.toggle_btn = QPushButton("Show row-sum spectrum")
        self.toggle_btn.clicked.connect(self._toggle_view)
        controls.addWidget(self.toggle_btn)
        controls.addWidget(QLabel("Rows:"))
        self.start_row_input = QSpinBox()
        self.start_row_input.setRange(0, self.image.shape[0] - 1)
        self.start_row_input.setValue(0)
        self.start_row_input.valueChanged.connect(self._on_start_row_changed)
        controls.addWidget(self.start_row_input)
        controls.addWidget(QLabel("to"))
        self.end_row_input = QSpinBox()
        self.end_row_input.setRange(0, self.image.shape[0] - 1)
        self.end_row_input.setValue(self.image.shape[0] - 1)
        self.end_row_input.valueChanged.connect(self._on_end_row_changed)
        controls.addWidget(self.end_row_input)
        self.pixel_axis_check = _make_pixel_axis_checkbox(
            spectral_calibration, self._redraw
        )
        self.pixel_axis_check.hide()
        controls.addWidget(self.pixel_axis_check)
        self.fix_y_scale_check = QCheckBox("Fix Y scale")
        self.fix_y_scale_check.setToolTip(
            "Keep the spectrum Y limits from the frame shown when checked"
        )
        self.fix_y_scale_check.toggled.connect(
            self._on_fix_y_scale_toggled
        )
        self.fix_y_scale_check.hide()
        controls.addWidget(self.fix_y_scale_check)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        self._colorbar = None
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)
        self._redraw()

    @staticmethod
    def _mean_detector_image(frames):
        frames = np.asarray(frames)
        if frames.ndim == 3:
            return np.mean(frames, axis=0)
        if frames.ndim == 2:
            return frames
        raise ValueError(
            "detector image must have shape (y, x) or (repeats, y, x)"
        )

    def _toggle_view(self):
        self._show_spectrum = not self._show_spectrum
        self.toggle_btn.setText(
            "Show detector image" if self._show_spectrum
            else "Show row-sum spectrum"
        )
        self.pixel_axis_check.setVisible(self._show_spectrum)
        self.fix_y_scale_check.setVisible(self._show_spectrum)
        self._redraw()

    def _on_fix_y_scale_toggled(self, checked):
        if checked:
            self._fixed_y_limits = self.ax.get_ylim()
        else:
            self._fixed_y_limits = None
            self._redraw()

    def _on_start_row_changed(self, start_row):
        if start_row > self.end_row_input.value():
            self.end_row_input.setValue(start_row)
        if self._show_spectrum:
            self._redraw()

    def _on_end_row_changed(self, end_row):
        if end_row < self.start_row_input.value():
            self.start_row_input.setValue(end_row)
        if self._show_spectrum:
            self._redraw()

    def update_frames(self, frames, *, title=None):
        """Replace the live detector data while preserving the chosen view."""
        old_last_row = self.image.shape[0] - 1
        was_full_range = (
            self.start_row_input.value() == 0
            and self.end_row_input.value() == old_last_row
        )
        self.image = self._mean_detector_image(frames)
        last_row = self.image.shape[0] - 1
        self.start_row_input.setMaximum(last_row)
        self.end_row_input.setMaximum(last_row)
        if was_full_range:
            self.start_row_input.setValue(0)
            self.end_row_input.setValue(last_row)
        if title is not None:
            self.setWindowTitle(title)
        self._redraw()

    def _redraw(self, _checked=None):
        self.ax.clear()
        if self._show_spectrum:
            spectrum = sum_detector_rows(
                self.image,
                self.start_row_input.value(),
                self.end_row_input.value(),
            )
            lines = self.ax.plot(spectrum)
            _set_spectral_line_axis(
                self.ax,
                lines,
                self.spectral_calibration,
                self.pixel_axis_check.isChecked(),
            )
            self.ax.set_ylabel("Summed intensity (a.u.)")
            self.ax.set_title(
                f"{self.windowTitle()} | rows "
                f"{self.start_row_input.value()}-{self.end_row_input.value()}"
            )
            if self._fixed_y_limits is not None:
                self.ax.set_ylim(self._fixed_y_limits)
            if self._colorbar is not None:
                self._colorbar.ax.set_visible(False)
        else:
            image_artist = self.ax.imshow(
                self.image,
                cmap="gray",
                aspect="auto",
                origin="lower",
            )
            self.ax.set_xlabel("Detector X pixel")
            self.ax.set_ylabel("Detector Y pixel")
            self.ax.set_title(self.windowTitle())
            if self._colorbar is None:
                self._colorbar = self.fig.colorbar(
                    image_artist, ax=self.ax, label="Intensity (a.u.)"
                )
            else:
                self._colorbar.update_normal(image_artist)
                self._colorbar.ax.set_visible(True)
        self.fig.tight_layout()
        self.canvas.draw_idle()


class SpectrumWindow(QMainWindow):
    """Pop-up plot window with a toggle between mean and all-traces views."""

    def __init__(
        self,
        spec,
        title="Spectrum",
        spectral_calibration=None,
        calibration_changed=None,
        spectral_bias=None,
        remove_spectral_bias=False,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(700, 550)
        self.spec = self._normalize_spectra(spec)
        self.spectral_bias = self._normalize_spectral_bias(spectral_bias)
        self._show_mean = True
        self._fixed_y_limits = None
        self.spectral_calibration = spectral_calibration
        self._calibration_changed = calibration_changed
        self._calibrating = False
        self._pending_pixel = None
        self._calibration_pixels = []
        self._known_shifts = []
        self._show_pixels_before_calibration = spectral_calibration is None
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        central = QWidget()
        layout = QVBoxLayout(central)
        controls = QHBoxLayout()
        self.toggle_btn = QPushButton("Show all traces")
        self.toggle_btn.clicked.connect(self._toggle)
        controls.addWidget(self.toggle_btn)
        self.pixel_axis_check = _make_pixel_axis_checkbox(
            spectral_calibration, self._redraw
        )
        controls.addWidget(self.pixel_axis_check)
        self.fix_y_scale_check = QCheckBox("Fix Y scale")
        self.fix_y_scale_check.setToolTip(
            "Keep the Y limits from the frame shown when checked"
        )
        self.fix_y_scale_check.toggled.connect(
            self._on_fix_y_scale_toggled
        )
        controls.addWidget(self.fix_y_scale_check)
        self.remove_spectral_bias_check = QCheckBox("Remove spectral bias")
        self.remove_spectral_bias_check.setToolTip(
            "Subtract filter_mean(dark noise) in this plot only. The raw "
            "spectra remain unchanged."
        )
        self.remove_spectral_bias_check.setChecked(
            bool(remove_spectral_bias and self.spectral_bias is not None)
        )
        self.remove_spectral_bias_check.toggled.connect(self._redraw)
        self.remove_spectral_bias_check.setVisible(
            self.spectral_bias is not None
        )
        controls.addWidget(self.remove_spectral_bias_check)
        self.calibration_btn = QPushButton("Pixel-to-wavenumber calibration")
        self.calibration_btn.clicked.connect(self._start_calibration)
        controls.addWidget(self.calibration_btn)
        layout.addLayout(controls)

        self.calibration_controls = QWidget()
        calibration_layout = QHBoxLayout(self.calibration_controls)
        calibration_layout.setContentsMargins(0, 0, 0, 0)
        calibration_layout.addWidget(QLabel("Polynomial degree:"))
        self.calibration_degree_input = QSpinBox()
        initial_degree = (
            spectral_calibration.degree
            if spectral_calibration is not None
            else 2
        )
        self.calibration_degree_input.setRange(1, max(9, initial_degree))
        self.calibration_degree_input.setValue(initial_degree)
        self.calibration_degree_input.setToolTip(
            "Degree d requires at least d + 1 calibration peaks. "
            "Degree 2 is recommended unless residual errors justify a "
            "higher degree."
        )
        self.calibration_degree_input.valueChanged.connect(
            self._on_calibration_degree_changed
        )
        calibration_layout.addWidget(self.calibration_degree_input)
        self.calibration_help = QLabel()
        self.calibration_help.setWordWrap(True)
        calibration_layout.addWidget(self.calibration_help, 1)
        self.finish_calibration_btn = QPushButton("Finish and save")
        self.finish_calibration_btn.clicked.connect(self._finish_calibration)
        calibration_layout.addWidget(self.finish_calibration_btn)
        self.cancel_calibration_btn = QPushButton("Cancel")
        self.cancel_calibration_btn.clicked.connect(self._cancel_calibration)
        calibration_layout.addWidget(self.cancel_calibration_btn)
        self.calibration_controls.hide()
        layout.addWidget(self.calibration_controls)
        self.fig = Figure(figsize=(7, 4.5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)
        self.canvas.mpl_connect("button_press_event", self._on_calibration_click)
        self.canvas.mpl_connect("key_press_event", self._on_calibration_key)
        self._redraw()

    @staticmethod
    def _normalize_spectra(spec):
        spec = np.asarray(spec)
        if spec.ndim == 1:
            spec = spec.reshape(1, -1)
        if spec.ndim != 2:
            raise ValueError(
                "spectrum must have shape (pixels,) or (repeats, pixels)"
            )
        return spec

    def _normalize_spectral_bias(self, spectral_bias):
        if spectral_bias is None:
            return None
        spectral_bias = np.asarray(spectral_bias, dtype=float)
        subtract_spectral_bias(self.spec, spectral_bias)
        return spectral_bias

    def _display_spectra(self):
        """Return raw or bias-corrected data for this window's plot only."""
        if (
            self.spectral_bias is not None
            and self.remove_spectral_bias_check.isChecked()
        ):
            return subtract_spectral_bias(self.spec, self.spectral_bias)
        return self.spec

    def update_spectrum(self, spec, *, title=None):
        """Replace plotted data, for example after each live exposure."""
        self.spec = self._normalize_spectra(spec)
        if title is not None:
            self.setWindowTitle(title)
        self._redraw()

    def _on_fix_y_scale_toggled(self, checked):
        if checked:
            self._fixed_y_limits = self.ax.get_ylim()
        else:
            self._fixed_y_limits = None
            self._redraw()

    def _toggle(self):
        self._show_mean = not self._show_mean
        self.toggle_btn.setText(
            "Show all traces" if self._show_mean else "Show mean"
        )
        self._redraw()

    def _redraw(self):
        import matplotlib.cm as cm
        self.ax.clear()
        display_spectra = self._display_spectra()
        lines = []
        if self._show_mean:
            lines.extend(self.ax.plot(filter_mean(display_spectra)))
        else:
            n = display_spectra.shape[0]
            colors = cm.viridis(np.linspace(0, 1, n))
            for i in range(n):
                lines.extend(
                    self.ax.plot(
                        display_spectra[i],
                        color=colors[i],
                        linewidth=0.8,
                    )
                )
        _set_spectral_line_axis(
            self.ax,
            lines,
            self.spectral_calibration,
            self.pixel_axis_check.isChecked(),
        )
        self.ax.set_ylabel("Intensity (a.u.)")
        title = self.windowTitle()
        if self.remove_spectral_bias_check.isChecked():
            title = f"{title} | bias corrected"
        self.ax.set_title(title)
        if self._fixed_y_limits is not None:
            self.ax.set_ylim(self._fixed_y_limits)
        if self._calibrating:
            self._draw_calibration_artists()
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _start_calibration(self):
        self._calibrating = True
        self._show_pixels_before_calibration = (
            self.pixel_axis_check.isChecked()
        )
        self._pending_pixel = None
        self._calibration_pixels = []
        self._known_shifts = []
        self._show_mean = True
        self.toggle_btn.setText("Show all traces")
        self.toggle_btn.setEnabled(False)
        self.calibration_btn.setEnabled(False)
        self.pixel_axis_check.setChecked(True)
        self.pixel_axis_check.setEnabled(False)
        self.calibration_controls.show()
        self.finish_calibration_btn.setEnabled(False)
        self._update_calibration_progress()
        self._redraw()
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

    def _spectrum_for_calibration(self):
        return np.asarray(filter_mean(self._display_spectra()), dtype=float)

    def _nearest_peak_pixel(self, x):
        y = self._spectrum_for_calibration()
        center = int(np.clip(round(x), 0, len(y) - 1))
        start = max(0, center - 8)
        stop = min(len(y), center + 9)
        return start + int(np.nanargmax(y[start:stop]))

    def _on_calibration_click(self, event):
        if (
            not self._calibrating
            or event.inaxes is not self.ax
            or event.xdata is None
            or event.button != 1
        ):
            return
        self._pending_pixel = self._nearest_peak_pixel(event.xdata)
        self._set_calibration_help(
            f"Selected peak at pixel {self._pending_pixel}. "
            "Press Enter to enter its known Raman shift."
        )
        self._redraw()
        self.canvas.setFocus()

    def _on_calibration_key(self, event):
        if not self._calibrating:
            return
        if event.key in ("enter", "return"):
            self._accept_pending_calibration_point()
        elif event.key == "escape":
            self._cancel_calibration()

    def _accept_pending_calibration_point(self):
        if self._pending_pixel is None:
            self._set_calibration_help("Click a peak before pressing Enter.")
            return
        pixel = int(self._pending_pixel)
        if pixel in self._calibration_pixels:
            QMessageBox.warning(
                self,
                "Duplicate calibration pixel",
                f"Pixel {pixel} is already in this calibration.",
            )
            return
        shift, accepted = QInputDialog.getDouble(
            self,
            "Known Raman shift",
            f"Raman shift for pixel {pixel} (cm⁻¹):",
            0.0,
            -10_000_000.0,
            10_000_000.0,
            4,
        )
        if not accepted:
            return
        self._calibration_pixels.append(pixel)
        self._known_shifts.append(float(shift))
        self._pending_pixel = None
        self._update_calibration_progress()
        self._redraw()
        self.canvas.setFocus()

    def _on_calibration_degree_changed(self, _degree):
        if self._calibrating:
            self._update_calibration_progress()

    def _update_calibration_progress(self):
        degree = self.calibration_degree_input.value()
        required = degree + 1
        count = len(self._calibration_pixels)
        self.finish_calibration_btn.setEnabled(count >= required)
        if count == 0:
            self._set_calibration_help(
                f"Degree {degree} needs at least {required} peaks. "
                "Click near a peak; it snaps to the local maximum. "
                "Press Enter, then type its known Raman shift."
            )
            return
        if count >= required:
            suffix = "You may add more peaks, or finish and save."
        else:
            suffix = f"Add {required - count} more calibration peak(s)."
        self._set_calibration_help(f"Accepted {count} point(s). {suffix}")

    def _draw_calibration_artists(self):
        y = self._spectrum_for_calibration()
        if self._calibration_pixels:
            pixels = np.asarray(self._calibration_pixels, dtype=int)
            self.ax.scatter(
                pixels,
                y[pixels],
                marker="o",
                s=55,
                facecolors="none",
                edgecolors="tab:green",
                linewidths=1.5,
                zorder=5,
            )
            for pixel, shift in zip(
                self._calibration_pixels, self._known_shifts
            ):
                self.ax.annotate(
                    f"{shift:g} cm⁻¹",
                    (pixel, y[pixel]),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                    color="tab:green",
                )
        if self._pending_pixel is not None:
            pixel = int(self._pending_pixel)
            self.ax.scatter(
                [pixel], [y[pixel]], marker="x", s=70,
                color="tab:red", linewidths=1.8, zorder=6,
            )

    def _set_calibration_help(self, text):
        self.calibration_help.setText(text)

    def _finish_calibration(self):
        degree = self.calibration_degree_input.value()
        required = degree + 1
        if len(self._calibration_pixels) < required:
            QMessageBox.warning(
                self,
                "More calibration points required",
                f"A degree-{degree} calibration requires at least "
                f"{required} peaks.",
            )
            return
        try:
            calibration = PixelToWavenumberCalibration(
                self._calibration_pixels,
                self._known_shifts,
                degree=degree,
            )
        except ValueError as error:
            QMessageBox.warning(self, "Invalid calibration", str(error))
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save pixel-to-wavenumber calibration",
            "pixel_to_wavenumber_calibration.json",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            saved_path = save_pixel_to_wavenumber_calibration(
                path, calibration
            )
        except (OSError, ValueError) as error:
            QMessageBox.critical(
                self, "Could not save calibration", str(error)
            )
            return
        self.spectral_calibration = calibration
        if self._calibration_changed is not None:
            self._calibration_changed(calibration, saved_path)
        self._stop_calibration(show_wavenumber=True)
        QMessageBox.information(
            self,
            "Calibration saved",
            f"Saved pixel-to-wavenumber calibration to:\n{saved_path}",
        )

    def _cancel_calibration(self):
        self._stop_calibration(show_wavenumber=False)

    def _stop_calibration(self, show_wavenumber):
        self._calibrating = False
        self._pending_pixel = None
        self.calibration_controls.hide()
        self.toggle_btn.setEnabled(True)
        self.calibration_btn.setEnabled(True)
        self.pixel_axis_check.setEnabled(
            self.spectral_calibration is not None
        )
        if show_wavenumber and self.spectral_calibration is not None:
            self.pixel_axis_check.setChecked(False)
        elif self.spectral_calibration is not None:
            self.pixel_axis_check.setChecked(
                self._show_pixels_before_calibration
            )
        self._redraw()


class ReferenceSpectraWindow(QMainWindow):
    """Pop-up showing reference spectra colored by z, with a colorbar."""

    def __init__(
        self, all_raman, zs, title="Reference spectra",
        spectral_calibration=None,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(800, 600)
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.cm as cm
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        from matplotlib.colors import Normalize
        central = QWidget()
        layout = QVBoxLayout(central)
        self.spectral_calibration = spectral_calibration
        self.pixel_axis_check = _make_pixel_axis_checkbox(
            spectral_calibration, self._update_spectral_axis
        )
        layout.addWidget(self.pixel_axis_check)
        self.fig = Figure(figsize=(8, 5.5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        ax = self.fig.add_subplot(111)
        zs = np.asarray(zs)
        n = len(zs)
        norm = Normalize(vmin=float(zs.min()), vmax=float(zs.max()))
        cmap = cm.viridis
        self._spectral_lines = []
        for i in range(n):
            color = cmap(norm(zs[i]))
            self._spectral_lines.extend(
                ax.plot(
                    filter_mean(all_raman[i]), color=color, linewidth=0.9
                )
            )
        self.ax = ax
        self._update_spectral_axis()
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(title)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=ax)
        cbar.set_label("z (um)")
        self.fig.tight_layout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)

    def _update_spectral_axis(self, _checked=None):
        _set_spectral_line_axis(
            self.ax,
            self._spectral_lines,
            self.spectral_calibration,
            self.pixel_axis_check.isChecked(),
        )
        if hasattr(self, "canvas"):
            self.fig.tight_layout()
            self.canvas.draw_idle()


class GridScanPlotWindow(QMainWindow):
    """Pop-up showing grid scan results.

    Single-plane mode: BF, end_BF, extra channels, and a spectrum panel.
    Z-scan mode: z-slider controlling the BF_z image and the spectrum panel.

    In both modes the grid sampling points are overlaid (low alpha) on the
    primary image. Click a point to show its individual spectrum; a toggle
    button switches between the average spectrum and the clicked point's
    spectrum.
    """

    FIXED = ["BF", "end_BF"]

    def __init__(
        self, ds, title="Grid scan result", spectral_calibration=None
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.ds = ds
        self._show_average = True
        self._sel = 0
        self._scat = None
        self._hl = None
        self.spectral_calibration = spectral_calibration
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        # grid sampling points (N, 2) in (row, col) image pixels
        try:
            self._grid = np.asarray(ds["grid_pos"].values, dtype=float)
        except Exception:
            self._grid = np.empty((0, 2))
        self._has_zscan = "BF_z" in ds.data_vars and "z_range" in ds.data_vars
        if self._has_zscan:
            self._init_zscan(ds, title, Figure, FigureCanvasQTAgg,
                             NavigationToolbar2QT)
        else:
            self._init_single(ds, title, Figure, FigureCanvasQTAgg,
                              NavigationToolbar2QT)

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                      #
    # ------------------------------------------------------------------ #
    def _make_mode_button(self):
        btn = QPushButton("Show clicked point")
        btn.clicked.connect(self._toggle_mode)
        self._mode_btn = btn
        return btn

    def _make_spectral_controls(self):
        controls = QHBoxLayout()
        controls.addWidget(self._make_mode_button())
        self.pixel_axis_check = _make_pixel_axis_checkbox(
            self.spectral_calibration, self._on_spectral_axis_changed
        )
        controls.addWidget(self.pixel_axis_check)
        controls.addStretch(1)
        return controls

    def _on_spectral_axis_changed(self, _checked=None):
        self._draw_spec()
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    def _add_grid_scatter(self, ax):
        """Overlay the grid points on ax as a pickable, low-alpha scatter,
        plus a highlight ring on the selected point."""
        if len(self._grid) == 0:
            self._scat = None
            self._hl = None
            return
        xs = self._grid[:, 1]   # column -> image x
        ys = self._grid[:, 0]   # row -> image y
        self._scat = ax.scatter(
            xs, ys, s=18, c="tab:red", alpha=0.35, picker=5,
            edgecolors="none",
        )
        sel = min(self._sel, len(self._grid) - 1)
        self._hl = ax.scatter(
            [xs[sel]], [ys[sel]], s=80, facecolors="none",
            edgecolors="yellow", linewidths=1.5,
        )

    def _current_specs_2d(self):
        """Return (n_pts, spec_dim) spectra for the current state."""
        if self._has_zscan:
            zi = self._z_slider.value()
            return self._specs[zi]
        return self._specs

    def _toggle_mode(self):
        self._show_average = not self._show_average
        self._mode_btn.setText(
            "Show clicked point" if self._show_average else "Show average"
        )
        self._draw_spec()
        self.canvas.draw_idle()

    def _update_highlight(self):
        if self._hl is None or len(self._grid) == 0:
            return
        i = min(self._sel, len(self._grid) - 1)
        self._hl.set_offsets([[self._grid[i, 1], self._grid[i, 0]]])

    def _on_pick(self, event):
        if self._scat is None or event.artist is not self._scat:
            return
        self._sel = int(event.ind[0])
        # a click selects a point -> show it individually
        self._show_average = False
        self._mode_btn.setText("Show average")
        self._update_highlight()
        self._draw_spec()
        self.canvas.draw_idle()

    def _draw_spec(self):
        specs2d = self._current_specs_2d()
        if specs2d is None or len(specs2d) == 0:
            return
        if self._show_average:
            y = specs2d.mean(axis=0)
            title = f"Mean spectrum ({specs2d.shape[0]} points)"
        else:
            i = min(self._sel, specs2d.shape[0] - 1)
            y = specs2d[i]
            title = f"Point {i} spectrum"
        if self._has_zscan:
            z_val = self._z_vals[self._z_slider.value()]
            title += f"  z={z_val:+.2f} um"
        self._spec_line.set_ydata(y)
        _set_spectral_line_axis(
            self._ax_spec,
            [self._spec_line],
            self.spectral_calibration,
            self.pixel_axis_check.isChecked(),
        )
        self._ax_spec.set_title(title)

    # ------------------------------------------------------------------ #
    #  Single-plane mode                                                   #
    # ------------------------------------------------------------------ #
    def _init_single(self, ds, title, Figure, Canvas, Toolbar):
        self.resize(1200, 750)
        self._specs = np.asarray(ds["specs"].values)   # (N, spec_dim)
        skip = {"laser_pos", "grid_pos", "specs"}
        image_vars = [
            name for name in ds.data_vars
            if name not in skip and ds[name].ndim == 2
        ]
        ordered = [c for c in self.FIXED if c in image_vars]
        ordered += [c for c in image_vars if c not in self.FIXED]
        ncols = max(len(ordered), 1)
        self.fig = Figure(figsize=(3 * ncols + 1, 7))
        self.canvas = Canvas(self.fig)
        self.toolbar = Toolbar(self.canvas, self)
        gs = self.fig.add_gridspec(2, ncols, height_ratios=[2, 1], hspace=0.4)
        first_ax = None
        for i, name in enumerate(ordered):
            ax = self.fig.add_subplot(
                gs[0, i], sharex=first_ax, sharey=first_ax,
            )
            if first_ax is None:
                first_ax = ax
            cmap = "gray" if name in self.FIXED else None
            ax.imshow(np.asarray(ds[name].values), cmap=cmap)
            ax.set_title(name)
        # overlay the grid points on the first (primary) image
        if first_ax is not None:
            self._add_grid_scatter(first_ax)
        self._ax_spec = self.fig.add_subplot(gs[1, :])
        (self._spec_line,) = self._ax_spec.plot(self._specs.mean(axis=0))
        self._ax_spec.set_xlabel("Pixels")
        self._ax_spec.set_ylabel("Intensity (a.u.)")
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(self._make_spectral_controls())
        self._draw_spec()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)
        self.fig.tight_layout(h_pad=2.0)
        self.canvas.mpl_connect("pick_event", self._on_pick)

    # ------------------------------------------------------------------ #
    #  Z-scan mode (slider)                                                #
    # ------------------------------------------------------------------ #
    def _init_zscan(self, ds, title, Figure, Canvas, Toolbar):
        self.resize(1000, 820)
        self._bf_z = np.asarray(ds["BF_z"].values)      # (n_z, Y, X)
        self._specs = np.asarray(ds["specs"].values)     # (n_z, n_pts, spec_dim)
        self._z_vals = np.asarray(ds["z_range"].values)  # (n_z,)
        self._n_z = len(self._z_vals)
        # Extra 2D channel images for the top row.
        skip = {"laser_pos", "grid_pos", "specs", "z_range", "BF_z"}
        self._extra_2d = {}
        for name in ds.data_vars:
            if name not in skip and ds[name].ndim == 2:
                self._extra_2d[name] = np.asarray(ds[name].values)
        central = QWidget()
        main_layout = QVBoxLayout(central)
        # --- top controls: mode button + z slider ---
        main_layout.addLayout(self._make_spectral_controls())
        slider_row = QHBoxLayout()
        self._z_label = QLabel(
            f"z = {self._z_vals[0]:+.2f} um  (1/{self._n_z})"
        )
        slider_row.addWidget(self._z_label)
        self._z_slider = QSlider(Qt.Horizontal)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(self._n_z - 1)
        self._z_slider.setValue(0)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        slider_row.addWidget(self._z_slider, 1)
        main_layout.addLayout(slider_row)
        # --- Figure: top row = images, bottom = spectrum ---
        n_extra = len(self._extra_2d)
        ncols = 1 + n_extra  # BF_z + extra channels
        self.fig = Figure(figsize=(4 * ncols + 1, 7))
        self.canvas = Canvas(self.fig)
        self.toolbar = Toolbar(self.canvas, self)
        gs = self.fig.add_gridspec(
            2, ncols, height_ratios=[2, 1], hspace=0.4
        )
        # BF_z image (updates with slider)
        self._ax_bf = self.fig.add_subplot(gs[0, 0])
        self._im_bf = self._ax_bf.imshow(
            self._bf_z[0], cmap="gray", aspect="equal"
        )
        self._ax_bf.set_title(f"BF_z  z={self._z_vals[0]:+.2f}")
        # grid overlay on BF_z
        self._add_grid_scatter(self._ax_bf)
        # Extra 2D channels (static)
        first_ax = self._ax_bf
        for i, (name, img) in enumerate(self._extra_2d.items()):
            ax = self.fig.add_subplot(
                gs[0, i + 1], sharex=first_ax, sharey=first_ax
            )
            cmap = "gray" if name in ("BF", "end_BF") else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(name)
        # Spectrum (updates with slider / selection)
        self._ax_spec = self.fig.add_subplot(gs[1, :])
        (self._spec_line,) = self._ax_spec.plot(self._specs[0].mean(axis=0))
        self._ax_spec.set_xlabel("Pixels")
        self._ax_spec.set_ylabel("Intensity (a.u.)")
        self._draw_spec()
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        self.setCentralWidget(central)
        self.fig.tight_layout(h_pad=2.0)
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def _on_z_changed(self, idx):
        z_val = self._z_vals[idx]
        self._z_label.setText(
            f"z = {z_val:+.2f} um  ({idx + 1}/{self._n_z})"
        )
        # Update BF_z image
        self._im_bf.set_data(self._bf_z[idx])
        self._im_bf.set_clim(self._bf_z[idx].min(), self._bf_z[idx].max())
        self._ax_bf.set_title(f"BF_z  z={z_val:+.2f}")
        # Update spectrum for the new z (respects average / clicked mode)
        self._draw_spec()
        self.canvas.draw_idle()


class DatasetViewerWindow(QMainWindow):
    """Interactive viewer: BF image with laser scatter + spectrum,
    Qt sliders for t/p/z."""

    def __init__(
        self, df, da, title="Dataset viewer", spectral_calibration=None
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 650)
        self.df = df
        self.da = da
        self.spectral_calibration = spectral_calibration
        self.bf = da.sel(c=0).values  # (t, p, z, y, x)
        self._pt_selected = 0
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.cm as mcm
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        self._mcm = mcm
        central = QWidget()
        main_layout = QVBoxLayout(central)
        # --- Sliders ---
        self.t_vals = da.coords["t"].values
        self.p_vals = da.coords["p"].values
        self.z_vals = da.coords["z"].values
        slider_layout = QHBoxLayout()
        self.t_slider, self.t_label = self._make_slider(
            "t", 0, len(self.t_vals) - 1
        )
        self.p_slider, self.p_label = self._make_slider(
            "p", 0, len(self.p_vals) - 1
        )
        self.z_slider, self.z_label = self._make_slider(
            "z", 0, len(self.z_vals) - 1
        )
        for label, slider in [
            (self.t_label, self.t_slider),
            (self.p_label, self.p_slider),
            (self.z_label, self.z_slider),
        ]:
            slider_layout.addWidget(label)
            slider_layout.addWidget(slider, 1)
        self.pixel_axis_check = _make_pixel_axis_checkbox(
            spectral_calibration, self._update_spectral_axis
        )
        slider_layout.addWidget(self.pixel_axis_check)
        main_layout.addLayout(slider_layout)
        # --- Figure ---
        self.fig = Figure(figsize=(12, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax_img = self.fig.add_subplot(121)
        self.ax_spec = self.fig.add_subplot(122)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)
        self.setCentralWidget(central)
        # --- Initial draw ---
        t0 = int(self.t_vals[0])
        p0 = int(self.p_vals[0])
        z0 = int(self.z_vals[0])
        self.im = self.ax_img.imshow(
            self.bf[0, 0, 0], cmap="gray", aspect="equal"
        )
        self.ax_img.set_title(f"t={t0}, p={p0}, z={z0}")
        # Scatter
        try:
            sub = df.loc[t0, p0, z0]
            n = len(sub)
            offsets = np.c_[sub["Y"].to_numpy(), sub["X"].to_numpy()]
        except KeyError:
            n = 0
            offsets = np.empty((0, 2))
        self.scat = self.ax_img.scatter(
            offsets[:, 0] if n else [],
            offsets[:, 1] if n else [],
            c=np.arange(n) if n else [],
            cmap="viridis",
            vmin=0,
            vmax=max(n - 1, 1),
            picker=5,
        )
        # Spectrum line
        try:
            spec0 = df.loc[t0, p0, z0, 0].values[:-3]
        except KeyError:
            spec0 = np.zeros(100)
        colors = self._pt_colors(max(n, 1))
        (self.spec_line,) = self.ax_spec.plot(
            spec0, color=colors[0] if n else "C0"
        )
        self._update_spectral_axis()
        self.ax_spec.set_ylabel("intensity (a.u.)")
        self.ax_spec.set_title(f"pt={self._pt_selected}")
        self.fig.tight_layout()
        # --- Connect signals ---
        self.t_slider.valueChanged.connect(self._on_slider)
        self.p_slider.valueChanged.connect(self._on_slider)
        self.z_slider.valueChanged.connect(self._on_slider)
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def _make_slider(self, name, lo, hi):
        label = QLabel(f"{name}=0")
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(lo)
        slider.setMaximum(hi)
        slider.setValue(0)
        slider.valueChanged.connect(
            lambda v, n=name, lbl=label: lbl.setText(f"{n}={v}")
        )
        return slider, label

    def _pt_colors(self, n):
        if n <= 1:
            return self._mcm.viridis(np.array([0.5]))
        return self._mcm.viridis(np.linspace(0, 1, n))

    def _current_tpz(self):
        ti = self.t_slider.value()
        pi = self.p_slider.value()
        zi = self.z_slider.value()
        t = int(self.t_vals[ti])
        p = int(self.p_vals[pi])
        z = int(self.z_vals[zi])
        return t, p, z, ti, pi, zi

    def _on_slider(self, _=None):
        t, p, z, ti, pi, zi = self._current_tpz()
        # Update image
        self.im.set_data(self.bf[ti, pi, zi])
        self.im.set_clim(
            self.bf[ti, pi, zi].min(), self.bf[ti, pi, zi].max()
        )
        self.ax_img.set_title(f"t={t}, p={p}, z={z}")
        # Update scatter
        try:
            sub = self.df.loc[t, p, z]
            n = len(sub)
            offsets = np.c_[sub["Y"].to_numpy(), sub["X"].to_numpy()]
        except KeyError:
            n = 0
            offsets = np.empty((0, 2))
        self.scat.set_offsets(offsets)
        self.scat.set_array(np.arange(n))
        self.scat.set_clim(0, max(n - 1, 1))
        self._pt_selected = min(self._pt_selected, max(n - 1, 0))
        self._update_spectrum()
        self.canvas.draw_idle()

    def _on_pick(self, event):
        if event.artist is not self.scat:
            return
        self._pt_selected = int(event.ind[0])
        self._update_spectrum()
        self.canvas.draw_idle()

    def _update_spectrum(self):
        t, p, z, _, _, _ = self._current_tpz()
        pt = self._pt_selected
        try:
            y = self.df.loc[t, p, z, pt].values[:-3]
            n = len(self.df.loc[t, p, z])
        except KeyError:
            return
        self.spec_line.set_ydata(y)
        self._update_spectral_axis()
        self.spec_line.set_color(self._pt_colors(n)[min(pt, n - 1)])
        self.ax_spec.set_title(f"pt={pt}")

    def _update_spectral_axis(self, _checked=None):
        if not hasattr(self, "spec_line"):
            return
        _set_spectral_line_axis(
            self.ax_spec,
            [self.spec_line],
            self.spectral_calibration,
            self.pixel_axis_check.isChecked(),
        )
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()
