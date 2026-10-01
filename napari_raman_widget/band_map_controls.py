"""Display-only Raman band/ratio overlays for acquired spatial maps."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from qtpy.QtCore import QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QLabel, QPushButton,
    QToolButton, QVBoxLayout, QWidget,
)

from .band_maps import band_map_values


class BandMapControls(QWidget):
    """Keep map settings separate from spectrum-display processing controls."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.values = None
        self.colorbar = None
        self._spans = []
        self._axis = None
        self._unit = "pixel"
        self._ready = False
        self._applied = False
        self._colorbar_parent_state = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.toggle = QToolButton()
        self.toggle.setText("Raman map")
        self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.RightArrow)
        layout.addWidget(self.toggle)
        self.body = QWidget()
        grid = QGridLayout(self.body)
        grid.setContentsMargins(6, 0, 6, 4)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Points only", "points")
        self.mode_combo.addItem("Band A area", "area")
        self.mode_combo.addItem("Band A / B ratio", "ratio")
        grid.addWidget(QLabel("Map:"), 0, 0)
        grid.addWidget(self.mode_combo, 0, 1, 1, 3)
        self.units_label = QLabel()
        grid.addWidget(self.units_label, 1, 0, 1, 4)
        self.a_low, self.a_high = self._band_row(grid, "Band A:", 2)
        self.b_low, self.b_high = self._band_row(grid, "Band B:", 3)
        self.colormap_combo = QComboBox()
        preferred = ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "coolwarm", "RdBu", "gray"]
        names = [name for name in colormaps if not name.endswith("_r")]
        ordered = [name for name in preferred if name in names]
        ordered.extend(sorted(set(names) - set(ordered), key=str.casefold))
        for name in ordered:
            self.colormap_combo.addItem(self._colormap_icon(colormaps[name]), name)
        self.colormap_combo.setIconSize(QSize(72, 12))
        self.colormap_combo.setCurrentText("viridis")
        self.colormap_combo.setToolTip("Recolor the map immediately without changing band values or color limits")
        self.reverse_check = QCheckBox("Reverse colors")
        grid.addWidget(QLabel("Colormap:"), 4, 0)
        grid.addWidget(self.colormap_combo, 4, 1, 1, 3)
        grid.addWidget(self.reverse_check, 5, 1, 1, 3)
        self.apply_button = QPushButton("Apply map")
        grid.addWidget(self.apply_button, 6, 0, 1, 4)
        hint = QLabel(
            "Integrates raw saved spectra; Processing changes only the spectrum "
            "display. Color scale adjusts to each Z plane. Gray = invalid; "
            "ratios require band B area > 1e-12."
        )
        hint.setWordWrap(True)
        grid.addWidget(hint, 7, 0, 1, 4)
        self.status = QLabel("Points only. Choose bands, then Apply map.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(self.status, 8, 0, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addWidget(self.body)
        self.body.hide()
        self.toggle.toggled.connect(self._toggle)
        self.mode_combo.currentIndexChanged.connect(self._settings_changed)
        self.colormap_combo.currentTextChanged.connect(self._colors_changed)
        self.reverse_check.toggled.connect(self._colors_changed)
        for spin in self.bounds:
            spin.valueChanged.connect(self._settings_changed)
        self.apply_button.clicked.connect(self.apply)
        self.sync_axis()
        self._settings_changed()

    @staticmethod
    def _colormap_icon(cmap):
        """Show the actual palette in the chooser, not just its name."""
        pixmap = QPixmap(72, 12)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        try:
            for x, rgba in enumerate(cmap(np.linspace(0, 1, pixmap.width()))):
                painter.fillRect(x, 0, 1, pixmap.height(), QColor.fromRgbF(*rgba))
        finally:
            painter.end()
        return QIcon(pixmap)

    def _selected_colormap(self):
        cmap = colormaps[self.colormap_combo.currentText()].copy()
        if self.reverse_check.isChecked():
            cmap = cmap.reversed()
        cmap.set_bad("#8c8c8c")
        return cmap

    def _colors_changed(self, *_args):
        # Color-only changes must not integrate spectra again or apply pending
        # band edits. Matplotlib updates the attached colorbar via set_cmap.
        if self._ready and self._applied and self.values is not None:
            self.owner._scat.set_cmap(self._selected_colormap())
            self.owner.canvas.draw_idle()

    @staticmethod
    def _band_row(grid, name, row):
        lower, upper = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (lower, upper):
            spin.setDecimals(6)
            spin.setKeyboardTracking(False)
            spin.setToolTip("Inclusive band boundary; fractional edges are interpolated")
        grid.addWidget(QLabel(name), row, 0)
        grid.addWidget(lower, row, 1)
        grid.addWidget(QLabel("to"), row, 2)
        grid.addWidget(upper, row, 3)
        return lower, upper

    @property
    def bounds(self):
        return self.a_low, self.a_high, self.b_low, self.b_high

    def _toggle(self, expanded):
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)

    def attach(self):
        """Called once the owning plot has created its image/scatter artists."""
        self._ready = True
        self._settings_changed()

    def sync_axis(self):
        """Preserve selected detector-bin locations when axis units change."""
        pixels = np.arange(self.owner._specs.shape[-1], dtype=float)
        calibration = self.owner.spectral_calibration
        calibrated = calibration is not None and self.owner.show_wavenumber_check.isChecked()
        axis = np.asarray(calibration.transform(pixels) if calibrated else pixels, dtype=float)
        unit = "cm⁻¹" if calibrated else "pixel"
        if self._axis is not None and unit == self._unit and np.array_equal(axis, self._axis):
            return
        valid = (len(axis) >= 2 and np.isfinite(axis).all()
                 and (np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0)))
        if not valid:
            self._axis = None
            self._unit = unit
            self.units_label.setText(f"Ranges: {unit} — requires a finite monotonic spectral axis")
            self._settings_changed()
            return
        old_axis = self._axis
        old_bounds = [spin.value() for spin in self.bounds]
        self._axis, self._unit = axis, unit
        lo, hi = float(np.min(axis)), float(np.max(axis))
        if old_axis is None:
            new_bounds = [lo, hi, lo, hi]
        else:
            old_pixels = np.arange(len(old_axis), dtype=float)
            if old_axis[0] > old_axis[-1]:
                old_pixels, old_axis = old_pixels[::-1], old_axis[::-1]
            selected_pixels = np.interp(old_bounds, old_axis, old_pixels)
            new_bounds = np.interp(selected_pixels, pixels, axis)
            new_bounds = [*sorted(new_bounds[:2]), *sorted(new_bounds[2:])]
        for spin, value in zip(self.bounds, new_bounds):
            spin.blockSignals(True)
            # Round inward at the spin-box precision so the displayed range
            # never requests samples outside a calibrated detector axis.
            scale = 10 ** spin.decimals()
            spin.setRange(np.ceil(lo * scale) / scale, np.floor(hi * scale) / scale)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self.units_label.setText(f"Ranges: {unit} (follows Show wavenumber)")
        self._settings_changed()

    def _settings_changed(self, *_args):
        mode = self.mode_combo.currentData()
        available = self._axis is not None
        if self._ready:
            available = available and self.owner._scat is not None
            available = available and len(self.owner._grid) == self.owner._specs.shape[-2]
        for spin in (self.a_low, self.a_high):
            spin.setEnabled(available and mode != "points")
        for spin in (self.b_low, self.b_high):
            spin.setEnabled(available and mode == "ratio")
        self.apply_button.setEnabled(available and mode != "points")
        self._applied = False
        if self._ready:
            self._clear_overlay()
        if not available:
            self.status.setText("Map unavailable: need aligned image-point coordinates and a finite monotonic spectral axis.")
        elif mode == "points":
            self.status.setText("Points only. Choose bands, then Apply map.")
        else:
            self.status.setText("Settings changed — click Apply map. Values use raw saved spectra.")

    def _clear_overlay(self):
        self.values = None
        if self.colorbar is not None:
            parent = self.owner._scat.axes
            self.colorbar.remove()
            self.colorbar = None
            if self._colorbar_parent_state is not None:
                position, anchor, in_layout = self._colorbar_parent_state
                parent.set_position(position)
                parent.set_anchor(anchor)
                parent.set_in_layout(in_layout)
                self._colorbar_parent_state = None
        for span in self._spans:
            span.remove()
        self._spans.clear()
        scatter = self.owner._scat
        if scatter is not None:
            scatter.set_array(None)
            scatter.set_color("tab:red")
            scatter.set_edgecolors("none")
            scatter.set_alpha(0.35)
            scatter.set_sizes([18])
        self.owner.canvas.draw_idle()

    def apply(self):
        if not self._ready or not self.apply_button.isEnabled():
            return
        self._clear_overlay()
        band_a = (self.a_low.value(), self.a_high.value())
        ratio = self.mode_combo.currentData() == "ratio"
        band_b = (self.b_low.value(), self.b_high.value()) if ratio else None
        try:
            values = band_map_values(self.owner._current_specs_2d(), self._axis, band_a, band_b)
        except (ValueError, TypeError) as error:
            self._applied = False
            self.status.setText(f"Cannot apply map: {error}")
            return
        self.values = values
        self._applied = True
        finite = values[np.isfinite(values)]
        from matplotlib.colors import Normalize

        scatter = self.owner._scat
        scatter.set_cmap(self._selected_colormap())
        if finite.size:
            lower, upper = float(finite.min()), float(finite.max())
            if lower == upper:
                padding = max(abs(lower) * 0.05, 1e-12)
                lower, upper = lower - padding, upper + padding
            scatter.set_norm(Normalize(lower, upper))
        else:
            scatter.set_norm(Normalize(0, 1))
        scatter.set_array(np.ma.masked_invalid(values))
        scatter.set_edgecolors("none")
        scatter.set_alpha(0.9)
        scatter.set_sizes([36])
        if finite.size:
            parent = scatter.axes
            self._colorbar_parent_state = (
                parent.get_position(original=True).frozen(),
                parent.get_anchor(), parent.get_in_layout(),
            )
            self.colorbar = self.owner.fig.colorbar(
                scatter, ax=scatter.axes, fraction=0.045, pad=0.04, use_gridspec=False,
            )
            self.colorbar.set_label("Raw band A / B" if ratio else f"Raw band A area (a.u.·{self._unit})")
        self._spans.append(self.owner._ax_spec.axvspan(*band_a, color="orange", alpha=0.18))
        if ratio:
            self._spans.append(self.owner._ax_spec.axvspan(*band_b, color="deepskyblue", alpha=0.18))
        self.selection_changed()
        self.owner.canvas.draw_idle()

    def plane_changed(self):
        if self._applied:
            self.apply()

    def selection_changed(self):
        if self.values is None:
            return
        valid = np.isfinite(self.values)
        index = self.owner._sel
        value = self.values[index]
        unit = "ratio" if self.mode_combo.currentData() == "ratio" else f"a.u.·{self._unit}"
        selected = f"{value:.6g} {unit}" if np.isfinite(value) else "invalid"
        self.status.setText(
            f"{int(valid.sum())}/{len(valid)} valid points; {int((~valid).sum())} gray/invalid. "
            f"Selected point {index}: {selected}. Color range: current Z plane."
        )
