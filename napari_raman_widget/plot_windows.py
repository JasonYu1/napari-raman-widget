"""Dock-ready Matplotlib panels for calibration, spectra, and scans."""
import numpy as np
from napari_raman_widget.band_map_controls import BandMapControls
from napari_raman_widget.figure_panel import (
    _add_matplotlib_background_control,
    _configure_matplotlib_figure,
    _make_white_background_checkbox,
    _style_matplotlib_toolbar,
)
from napari_raman_widget.spectra import (
    asls_baseline,
    filter_mean,
    smooth_spectra,
    subtract_spectral_bias,
    sum_detector_rows,
)
from napari_raman_widget.spectral_calibration import (
    PixelToWavenumberCalibration,
    save_pixel_to_wavenumber_calibration,
)
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


def _panel_layout(owner):
    """Return the compact top-level layout shared by plot panels."""
    layout = QVBoxLayout(owner)
    layout.setContentsMargins(6, 6, 6, 6)
    layout.setSpacing(6)
    return layout


def _add_compact_control_group(layout, title, rows):
    """Add a control group whose rows remain legible in a narrow dock."""
    group = QGroupBox(title)
    grid = QGridLayout(group)
    grid.setContentsMargins(6, 4, 6, 6)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(4)
    column_count = max((len(row) for row in rows), default=1)
    for row_index, row in enumerate(rows):
        if len(row) == 1:
            grid.addWidget(row[0], row_index, 0, 1, column_count)
            continue
        span = max(1, column_count // len(row))
        for item_index, widget in enumerate(row):
            column = item_index * span
            final = item_index == len(row) - 1
            widget_span = column_count - column if final else span
            grid.addWidget(widget, row_index, column, 1, widget_span)
    for column in range(column_count):
        grid.setColumnStretch(column, 1)
    layout.addWidget(group)
    return group


def _make_repeat_controls(owner, changed):
    """Create synchronized aggregate/individual repeat navigation controls."""

    owner.repeat_controls = QWidget()
    repeat_layout = QHBoxLayout(owner.repeat_controls)
    repeat_layout.setContentsMargins(0, 0, 0, 0)
    repeat_layout.setSpacing(6)
    repeat_layout.addWidget(QLabel("Display repeat:"))
    owner.repeat_input = QSpinBox()
    owner.repeat_input.setSpecialValueText("Mean")
    owner.repeat_input.setKeyboardTracking(False)
    owner.repeat_input.setMaximumWidth(90)
    owner.repeat_input.setToolTip(
        "Mean shows the aggregate; 1 through N shows one acquired repeat"
    )
    repeat_layout.addWidget(owner.repeat_input)
    owner.repeat_slider = QSlider(Qt.Horizontal)
    owner.repeat_slider.setSingleStep(1)
    owner.repeat_slider.setPageStep(1)
    owner.repeat_slider.setToolTip(
        "0 shows the aggregate; 1 through N shows one acquired repeat"
    )
    repeat_layout.addWidget(owner.repeat_slider, 1)
    owner.repeat_count_label = QLabel()
    owner.repeat_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    repeat_layout.addWidget(owner.repeat_count_label)
    owner.repeat_input.valueChanged.connect(owner.repeat_slider.setValue)
    owner.repeat_slider.valueChanged.connect(owner.repeat_input.setValue)
    owner.repeat_slider.valueChanged.connect(changed)
    return owner.repeat_controls


def _configure_repeat_controls(owner, repeat_count, *, reset=False):
    """Set repeat bounds and hide navigation for a single measurement."""

    repeat_count = max(1, int(repeat_count))
    selected = int(owner.repeat_slider.value())
    if reset or selected > repeat_count:
        selected = 0
    owner.repeat_slider.blockSignals(True)
    owner.repeat_input.blockSignals(True)
    try:
        owner.repeat_slider.setRange(0, repeat_count)
        owner.repeat_input.setRange(0, repeat_count)
        owner.repeat_slider.setValue(selected)
        owner.repeat_input.setValue(selected)
    finally:
        owner.repeat_slider.blockSignals(False)
        owner.repeat_input.blockSignals(False)
    owner.repeat_count_label.setText(f"of {repeat_count}")
    available = repeat_count > 1 and not getattr(owner, "_calibrating", False)
    owner.repeat_controls.setVisible(repeat_count > 1)
    owner.repeat_controls.setEnabled(available)


def _finish_plot_layout(
    layout, toolbar, canvas, *, background_control_in_layout=True
):
    """Add Matplotlib widgets with policies suited to resizeable docks."""
    owner = toolbar.parentWidget()
    _style_matplotlib_toolbar(toolbar)
    _add_matplotlib_background_control(
        owner,
        layout,
        canvas.figure,
        canvas,
        toolbar,
        add_to_layout=background_control_in_layout,
    )
    toolbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    canvas.setMinimumSize(200, 160)
    layout.addWidget(toolbar)
    layout.addWidget(canvas, 1)


def _set_cosmic_ray_status(owner, text="", *, visible=None):
    """Update the small processing summary when that control is available."""

    label = getattr(owner, "cosmic_ray_status_label", None)
    if label is None:
        return
    label.setText(str(text))
    label.setVisible(bool(text) if visible is None else bool(visible))


def _is_dark_only_fvb(calibration):
    """Return whether a calibration has no illuminated FVB reference."""

    return (
        calibration is not None
        and getattr(calibration, "read_mode", "fvb") == "fvb"
        and hasattr(calibration, "open_reference_path")
        and calibration.open_reference_path is None
    )


def _set_cosmic_ray_unavailable(owner, reason):
    """Disable RACER after an incompatibility or correction failure."""
    reason = str(reason).strip() or "the calibration is incompatible"
    owner._cosmic_ray_error = reason
    checkbox = owner.cosmic_ray_removal_check
    checkbox.blockSignals(True)
    checkbox.setChecked(False)
    checkbox.blockSignals(False)
    checkbox.setEnabled(False)
    checkbox.setToolTip(f"RACER unavailable: {reason}")
    _set_cosmic_ray_status(owner, f"RACER unavailable: {reason}")


def _refresh_cosmic_ray_compatibility(owner):
    """Validate the current raw data without attempting a correction."""
    calibration = owner.cosmic_ray_calibration
    if calibration is None:
        _set_cosmic_ray_unavailable(
            owner, "load a compatible cosmic-ray calibration first"
        )
        return
    raw_values = np.asarray(owner._cosmic_ray_values)
    if (
        owner.cosmic_ray_read_mode == "fvb"
        and _is_dark_only_fvb(calibration)
    ):
        if not owner._cosmic_ray_fvb_repeat_guard:
            _set_cosmic_ray_unavailable(
                owner,
                "dark-only FVB correction is limited to same-point repeats "
                "in the Spectrum view; load an illuminated FVB reference "
                "for this panel",
            )
            return
        if raw_values.ndim != 2 or raw_values.shape[0] < 2:
            _set_cosmic_ray_unavailable(
                owner,
                "dark-only FVB correction needs at least two repeats; "
                "collect more repeats or load an illuminated FVB reference",
            )
            return
    values = raw_values
    if owner.cosmic_ray_read_mode == "image" and values.ndim >= 3:
        values = values.reshape((-1,) + values.shape[-2:])
    elif owner.cosmic_ray_read_mode != "image" and values.ndim >= 2:
        values = values.reshape(-1, values.shape[-1])
    try:
        compatible, reason = calibration.compatibility(
            values,
            exposure_ms=owner.cosmic_ray_exposure_ms,
            read_mode=owner.cosmic_ray_read_mode,
        )
    except Exception as error:
        _set_cosmic_ray_unavailable(
            owner,
            "could not validate the calibration "
            f"({type(error).__name__}: {error})",
        )
        return
    if not compatible:
        _set_cosmic_ray_unavailable(
            owner, reason or "the calibration is incompatible with these data"
        )
        return
    owner._cosmic_ray_error = None
    owner.cosmic_ray_removal_check.setEnabled(True)
    tooltip = (
        "Apply display-only RACER cosmic-ray removal. Raw acquired and saved "
        "data remain unchanged."
    )
    if owner._cosmic_ray_fvb_repeat_guard:
        tooltip += (
            " Repeat-consensus protection preserves stable narrow spectral "
            "features."
        )
    owner.cosmic_ray_removal_check.setToolTip(tooltip)
    if not owner.cosmic_ray_removal_check.isChecked():
        _set_cosmic_ray_status(owner)


def _add_cosmic_ray_controls(
    owner,
    layout,
    redraw,
    values,
    *,
    calibration=None,
    exposure_ms=None,
    read_mode="fvb",
    fvb_repeat_guard=False,
):
    """Add a default-off RACER control and initialize its correction cache."""
    owner.cosmic_ray_calibration = calibration
    owner.cosmic_ray_exposure_ms = exposure_ms
    owner.cosmic_ray_read_mode = read_mode
    owner._cosmic_ray_fvb_repeat_guard = bool(fvb_repeat_guard)
    owner._cosmic_ray_values = np.asarray(values)
    owner._cosmic_ray_data_revision = 0
    owner._cosmic_ray_cache_key = None
    owner._cosmic_ray_corrected_cache = None
    owner._cosmic_ray_correction_stats = None
    owner._cosmic_ray_error = None
    owner._cosmic_ray_redraw = redraw
    owner.cosmic_ray_removal_check = QCheckBox(
        "Remove cosmic rays (RACER)"
    )
    owner.cosmic_ray_removal_check.setChecked(False)
    owner.cosmic_ray_removal_check.toggled.connect(redraw)
    layout.addWidget(owner.cosmic_ray_removal_check)
    owner.cosmic_ray_status_label = QLabel()
    owner.cosmic_ray_status_label.setWordWrap(True)
    owner.cosmic_ray_status_label.setStyleSheet(
        "color: #777; font-size: 11px;"
    )
    owner.cosmic_ray_status_label.hide()
    layout.addWidget(owner.cosmic_ray_status_label)
    _refresh_cosmic_ray_compatibility(owner)


def _replace_cosmic_ray_values(owner, values):
    """Replace live raw data and invalidate any cached RACER output."""
    owner._cosmic_ray_values = np.asarray(values)
    owner._cosmic_ray_data_revision += 1
    owner._cosmic_ray_cache_key = None
    owner._cosmic_ray_corrected_cache = None
    owner._cosmic_ray_correction_stats = None
    _refresh_cosmic_ray_compatibility(owner)


def set_cosmic_ray_calibration(owner, calibration):
    """Replace one open panel's calibration and refresh its display state."""

    if not hasattr(owner, "cosmic_ray_removal_check"):
        return False
    was_checked = owner.cosmic_ray_removal_check.isChecked()
    owner.cosmic_ray_calibration = calibration
    owner._cosmic_ray_cache_key = None
    owner._cosmic_ray_corrected_cache = None
    owner._cosmic_ray_correction_stats = None
    _refresh_cosmic_ray_compatibility(owner)
    compatible = owner.cosmic_ray_removal_check.isEnabled()
    if was_checked:
        owner._cosmic_ray_redraw()
    return compatible


def _protect_repeat_consistent_fvb(owner, raw, corrected):
    """Restore stable spectral features that dark-only RACER misclassified."""

    raw = np.asarray(raw, dtype=float)
    corrected = np.asarray(corrected, dtype=float)
    if raw.ndim != 2 or raw.shape[0] < 2 or corrected.shape != raw.shape:
        return corrected, 0
    changed = corrected != raw
    if not np.any(changed):
        return corrected, 0

    removal = np.abs(raw - corrected)
    calibration = owner.cosmic_ray_calibration
    dark_sigma = getattr(calibration, "fvb_dark_sigma_adu", None)
    if dark_sigma is None:
        dark_sigma = np.full(raw.shape[1], 0.5, dtype=float)
    else:
        dark_sigma = np.asarray(dark_sigma, dtype=float)
        if dark_sigma.shape != (raw.shape[1],):
            dark_sigma = np.full(raw.shape[1], 0.5, dtype=float)
    dark_sigma = np.maximum(
        np.where(np.isfinite(dark_sigma), dark_sigma, 0.5), 0.5
    )
    dark_tolerance = 6.0 * dark_sigma[None, :]

    if raw.shape[0] == 2:
        peer_distance = np.abs(raw - raw[::-1])
        restore = changed & (
            peer_distance <= np.maximum(dark_tolerance, 0.5 * removal)
        )
    else:
        peer_median = np.median(raw, axis=0)
        peer_distance = np.abs(raw - peer_median[None, :])
        repeat_sigma = 1.4826 * np.median(peer_distance, axis=0)
        repeat_sigma = np.where(np.isfinite(repeat_sigma), repeat_sigma, 0.0)
        noise_tolerance = 6.0 * np.maximum(
            repeat_sigma[None, :], dark_sigma[None, :]
        )
        removal_tolerance = np.maximum(dark_tolerance, 0.5 * removal)
        restore = (
            changed
            & (peer_distance <= noise_tolerance)
            & (peer_distance <= removal_tolerance)
        )

    restored = int(np.count_nonzero(restore))
    if restored == 0:
        return corrected, 0
    protected = np.array(corrected, copy=True)
    protected[restore] = raw[restore]
    return protected, restored


def _show_cosmic_ray_summary(owner):
    """Show how much the cached correction actually changed."""

    stats = owner._cosmic_ray_correction_stats
    if stats is None:
        _set_cosmic_ray_status(owner)
        return
    changed, measurements, restored = stats
    noun = "frame" if owner.cosmic_ray_read_mode == "image" else "spectrum"
    noun += "s" if measurements != 1 else ""
    if changed:
        value_noun = "value" if changed == 1 else "values"
        text = (
            f"RACER corrected {changed:,} detector {value_noun} across "
            f"{measurements:,} {noun}."
        )
    else:
        text = f"RACER ran on {measurements:,} {noun}; no values changed."
    if restored:
        text += (
            f" Preserved {restored:,} repeat-consistent spectral values."
        )
    _set_cosmic_ray_status(owner, text)


def _correct_cosmic_ray_measurements(owner, values):
    """Correct each raw spectrum or detector frame independently."""
    raw = np.asarray(values)
    calibration = owner.cosmic_ray_calibration
    if owner.cosmic_ray_read_mode == "image":
        if raw.ndim < 2:
            raise ValueError("detector data must contain two-dimensional frames")
        measurement_shape = raw.shape[-2:]
        leading_shape = raw.shape[:-2]
    else:
        if raw.ndim < 1:
            raise ValueError("spectral data must contain one-dimensional traces")
        measurement_shape = raw.shape[-1:]
        leading_shape = raw.shape[:-1]

    measurements = raw.reshape((-1,) + measurement_shape)
    corrected = []
    for measurement in measurements:
        result = np.asarray(
            calibration.correct(np.array(measurement, copy=True))
        )
        if result.shape != measurement.shape:
            raise ValueError(
                "corrected data shape does not match the raw measurement "
                f"({result.shape} != {measurement.shape})"
            )
        corrected.append(result)
    if not corrected:
        owner._cosmic_ray_correction_stats = (0, 0, 0)
        return np.array(raw, copy=True)
    result = np.stack(corrected).reshape(leading_shape + measurement_shape)
    restored = 0
    if (
        owner.cosmic_ray_read_mode == "fvb"
        and owner._cosmic_ray_fvb_repeat_guard
        and _is_dark_only_fvb(calibration)
        and result.ndim == 2
    ):
        result, restored = _protect_repeat_consistent_fvb(owner, raw, result)
    changed = int(np.count_nonzero(np.asarray(result) != raw))
    owner._cosmic_ray_correction_stats = (
        changed,
        int(measurements.shape[0]),
        restored,
    )
    return result


def _cosmic_ray_values_for_plot(owner):
    """Return raw or cached corrected values, disabling RACER on failure."""
    raw = owner._cosmic_ray_values
    checkbox = owner.cosmic_ray_removal_check
    if not checkbox.isChecked() or not checkbox.isEnabled():
        if checkbox.isEnabled():
            _set_cosmic_ray_status(owner)
        return raw
    key = (id(owner.cosmic_ray_calibration), owner._cosmic_ray_data_revision)
    if owner._cosmic_ray_cache_key == key:
        _show_cosmic_ray_summary(owner)
        return owner._cosmic_ray_corrected_cache
    try:
        corrected = _correct_cosmic_ray_measurements(owner, raw)
    except Exception as error:
        _set_cosmic_ray_unavailable(
            owner, f"correction failed ({type(error).__name__}: {error})"
        )
        return raw
    owner._cosmic_ray_cache_key = key
    owner._cosmic_ray_corrected_cache = corrected
    _show_cosmic_ray_summary(owner)
    return corrected


def _add_processing_section(
    owner,
    layout,
    redraw,
    spectral_length,
    *,
    cosmic_ray_calibration=None,
    cosmic_ray_values=None,
    exposure_ms=None,
    read_mode="fvb",
    fvb_repeat_guard=False,
):
    """Add shared collapsible baseline and smoothing controls."""
    owner._processing_available = True
    owner.processing_toggle = QToolButton()
    owner.processing_toggle.setText("Processing")
    owner.processing_toggle.setCheckable(True)
    owner.processing_toggle.setChecked(False)
    owner.processing_toggle.setArrowType(Qt.RightArrow)
    owner.processing_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    owner.processing_toggle.setSizePolicy(
        QSizePolicy.Expanding, QSizePolicy.Fixed
    )

    owner.processing_panel = QWidget()
    processing_layout = QVBoxLayout(owner.processing_panel)
    processing_layout.setContentsMargins(6, 0, 6, 4)
    processing_layout.setSpacing(4)
    _add_cosmic_ray_controls(
        owner,
        processing_layout,
        redraw,
        np.empty((0, spectral_length))
        if cosmic_ray_values is None
        else cosmic_ray_values,
        calibration=cosmic_ray_calibration,
        exposure_ms=exposure_ms,
        read_mode=read_mode,
        fvb_repeat_guard=fvb_repeat_guard,
    )
    _add_baseline_controls(
        owner, processing_layout, redraw, spectral_length
    )
    _add_smoothing_controls(
        owner, processing_layout, redraw, spectral_length
    )
    owner.processing_panel.hide()

    def on_processing_toggled(expanded):
        owner.processing_toggle.setArrowType(
            Qt.DownArrow if expanded else Qt.RightArrow
        )
        owner.processing_panel.setVisible(
            bool(expanded) and owner._processing_available
        )

    owner.processing_toggle.toggled.connect(on_processing_toggled)
    layout.addWidget(owner.processing_toggle)
    layout.addWidget(owner.processing_panel)


def _set_processing_available(owner, available):
    """Show or hide a panel's processing section without losing its state."""
    available = bool(available)
    owner._processing_available = available
    owner.processing_toggle.setVisible(available)
    owner.processing_panel.setVisible(
        available and owner.processing_toggle.isChecked()
    )


def _spectral_x(length, calibration, show_wavenumber):
    pixels = np.arange(length, dtype=float)
    if calibration is None or not show_wavenumber:
        return pixels
    return calibration.transform(pixels)


def _set_spectral_line_axis(ax, lines, calibration, show_wavenumber):
    """Update spectrum line x data and label for the selected axis units."""
    for line in lines:
        line.set_xdata(
            _spectral_x(
                len(line.get_ydata()), calibration, show_wavenumber
            )
        )
    if calibration is not None and show_wavenumber:
        ax.set_xlabel("Raman shift (cm⁻¹)")
    else:
        ax.set_xlabel("Pixel")
    ax.relim()
    ax.autoscale_view()


def _make_wavenumber_axis_checkbox(calibration, callback):
    """Return an opt-in wavenumber-axis control, defaulting to pixels."""
    checkbox = QCheckBox("Show wavenumber")
    checkbox.setChecked(False)
    checkbox.setEnabled(calibration is not None)
    if calibration is None:
        checkbox.setToolTip("Load a pixel-to-wavenumber calibration first")
    else:
        checkbox.setToolTip("Use calibrated Raman shift instead of pixels")
    checkbox.toggled.connect(callback)
    return checkbox


def _smoothing_window(owner):
    """Map the slider index to an odd Savitzky-Golay window >= 5."""
    return 5 + 2 * owner.smoothing_window_slider.value()


def _update_smoothing_label(owner):
    window = _smoothing_window(owner)
    owner.smoothing_window_label.setText(f"Window: {window}")
    owner.smoothing_window_input.blockSignals(True)
    owner.smoothing_window_input.setValue(window)
    owner.smoothing_window_input.blockSignals(False)


def _configure_smoothing_length(owner, spectral_length):
    """Constrain smoothing choices to valid odd windows for the data."""
    spectral_length = int(spectral_length)
    largest_odd = spectral_length if spectral_length % 2 else spectral_length - 1
    available = largest_odd >= 5

    slider = owner.smoothing_window_slider
    slider.blockSignals(True)
    slider.setMaximum(max(0, (largest_odd - 5) // 2))
    slider.setValue(min(slider.value(), slider.maximum()))
    slider.blockSignals(False)
    owner.smoothing_window_input.blockSignals(True)
    owner.smoothing_window_input.setRange(5, max(5, largest_odd))
    owner.smoothing_window_input.setValue(_smoothing_window(owner))
    owner.smoothing_window_input.blockSignals(False)
    checkbox = owner.smoothing_check
    checkbox.setEnabled(available)
    if not available:
        checkbox.blockSignals(True)
        checkbox.setChecked(False)
        checkbox.blockSignals(False)
        owner.smoothing_window_controls.hide()
    _update_smoothing_label(owner)


def _add_smoothing_controls(owner, layout, redraw, spectral_length):
    """Add a checkbox whose enabled state reveals an odd-window slider."""
    row = QWidget()
    row_layout = QVBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(3)
    owner.smoothing_check = QCheckBox("Smooth")
    owner.smoothing_check.setToolTip(
        "Apply display-only Savitzky-Golay smoothing with polynomial order 3"
    )
    row_layout.addWidget(owner.smoothing_check)

    owner.smoothing_window_controls = QWidget()
    smoothing_layout = QGridLayout(owner.smoothing_window_controls)
    smoothing_layout.setContentsMargins(0, 0, 0, 0)
    smoothing_layout.setHorizontalSpacing(6)
    smoothing_layout.setVerticalSpacing(2)
    owner.smoothing_window_label = QLabel()
    owner.smoothing_window_label.setAlignment(
        Qt.AlignRight | Qt.AlignVCenter
    )
    owner.smoothing_window_input = QSpinBox()
    owner.smoothing_window_input.setSingleStep(2)
    owner.smoothing_window_input.setKeyboardTracking(False)
    owner.smoothing_window_input.setMaximumWidth(84)
    owner.smoothing_window_input.setToolTip(
        "Savitzky-Golay smoothing window (odd samples)"
    )
    owner.smoothing_window_slider = QSlider(Qt.Horizontal)
    owner.smoothing_window_slider.setMinimum(0)
    owner.smoothing_window_slider.setSingleStep(1)
    owner.smoothing_window_slider.setPageStep(5)
    owner.smoothing_window_slider.setToolTip(
        "Odd smoothing window; polynomial order is fixed at 3"
    )
    smoothing_layout.addWidget(owner.smoothing_window_label, 0, 0)
    smoothing_layout.addWidget(owner.smoothing_window_input, 0, 1)
    smoothing_layout.addWidget(owner.smoothing_window_slider, 1, 0, 1, 2)
    smoothing_layout.setColumnStretch(0, 1)
    owner.smoothing_window_controls.hide()
    row_layout.addWidget(owner.smoothing_window_controls)
    layout.addWidget(row)
    _configure_smoothing_length(owner, spectral_length)

    def on_toggled(checked):
        owner.smoothing_window_controls.setVisible(bool(checked))
        redraw()

    def on_window_changed(_index):
        _update_smoothing_label(owner)
        redraw()

    def on_window_input_changed(value):
        maximum = 5 + 2 * owner.smoothing_window_slider.maximum()
        value = min(maximum, max(5, int(value)))
        if value % 2 == 0:
            value = value + 1 if value < maximum else value - 1
        slider_index = (value - 5) // 2
        if owner.smoothing_window_slider.value() != slider_index:
            owner.smoothing_window_slider.setValue(slider_index)
        else:
            _update_smoothing_label(owner)
            redraw()

    owner.smoothing_check.toggled.connect(on_toggled)
    owner.smoothing_window_slider.valueChanged.connect(on_window_changed)
    owner.smoothing_window_input.valueChanged.connect(on_window_input_changed)


def _smooth_for_plot(owner, spectra):
    """Return display data, smoothing only when this window opted in."""
    spectra = np.asarray(spectra)
    if not owner.smoothing_check.isChecked():
        return spectra
    return smooth_spectra(spectra, _smoothing_window(owner))


_BASELINE_LAMBDA_STEPS_PER_DECADE = 4


def _baseline_lambda(owner):
    """Return lambda from a quarter-decade logarithmic slider."""
    exponent = (
        owner.baseline_lambda_slider.value()
        / _BASELINE_LAMBDA_STEPS_PER_DECADE
    )
    return 10.0 ** exponent


def _update_baseline_lambda_label(owner):
    lam = _baseline_lambda(owner)
    exponent = int(np.floor(np.log10(lam)))
    coefficient = lam / (10.0 ** exponent)
    if np.isclose(coefficient, 1.0):
        value = f"1e{exponent}"
    else:
        value = f"{coefficient:.1f}e{exponent}"
    owner.baseline_lambda_label.setText(f"λ: {value}")
    owner.baseline_lambda_input.blockSignals(True)
    owner.baseline_lambda_input.setValue(lam)
    owner.baseline_lambda_input.blockSignals(False)


def _configure_baseline_length(owner, spectral_length):
    """Disable AsLS for spectra too short for a second-difference penalty."""
    available = int(spectral_length) >= 3
    checkbox = owner.baseline_subtraction_check
    checkbox.setEnabled(available)
    if not available:
        checkbox.blockSignals(True)
        checkbox.setChecked(False)
        checkbox.blockSignals(False)
        owner.baseline_lambda_controls.hide()


def _add_baseline_controls(owner, layout, redraw, spectral_length):
    """Add display-only AsLS subtraction with a logarithmic lambda slider."""
    row = QWidget()
    row_layout = QVBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(3)
    owner.baseline_subtraction_check = QCheckBox("Baseline subtraction")
    owner.baseline_subtraction_check.setToolTip(
        "Subtract an AsLS Whittaker baseline in this plot only "
        "(asymmetry p = 0.01)"
    )
    row_layout.addWidget(owner.baseline_subtraction_check)

    owner.baseline_lambda_controls = QWidget()
    baseline_layout = QGridLayout(owner.baseline_lambda_controls)
    baseline_layout.setContentsMargins(0, 0, 0, 0)
    baseline_layout.setHorizontalSpacing(6)
    baseline_layout.setVerticalSpacing(2)
    owner.baseline_lambda_label = QLabel()
    owner.baseline_lambda_label.setAlignment(
        Qt.AlignRight | Qt.AlignVCenter
    )
    owner.baseline_lambda_input = QDoubleSpinBox()
    owner.baseline_lambda_input.setRange(1e2, 1e10)
    owner.baseline_lambda_input.setDecimals(0)
    owner.baseline_lambda_input.setSingleStep(100)
    owner.baseline_lambda_input.setKeyboardTracking(False)
    owner.baseline_lambda_input.setMinimumWidth(
        owner.baseline_lambda_input.fontMetrics().horizontalAdvance(
            "10000000000"
        ) + 40
    )
    owner.baseline_lambda_input.setToolTip(
        "AsLS smoothness lambda (100 to 10,000,000,000). Values snap "
        "to the slider's quarter-decade steps."
    )
    owner.baseline_lambda_slider = QSlider(Qt.Horizontal)
    owner.baseline_lambda_slider.setRange(
        2 * _BASELINE_LAMBDA_STEPS_PER_DECADE,
        10 * _BASELINE_LAMBDA_STEPS_PER_DECADE,
    )
    owner.baseline_lambda_slider.setValue(
        6 * _BASELINE_LAMBDA_STEPS_PER_DECADE
    )
    owner.baseline_lambda_slider.setSingleStep(1)
    owner.baseline_lambda_slider.setPageStep(
        _BASELINE_LAMBDA_STEPS_PER_DECADE
    )
    owner.baseline_lambda_slider.setTracking(False)
    owner.baseline_lambda_slider.setToolTip(
        "AsLS smoothness lambda on a logarithmic scale; larger values "
        "produce a smoother baseline"
    )
    baseline_layout.addWidget(owner.baseline_lambda_label, 0, 0)
    baseline_layout.addWidget(owner.baseline_lambda_input, 0, 1)
    baseline_layout.addWidget(owner.baseline_lambda_slider, 1, 0, 1, 2)
    baseline_layout.setColumnStretch(0, 1)
    owner.baseline_lambda_controls.hide()
    row_layout.addWidget(owner.baseline_lambda_controls)
    layout.addWidget(row)
    _update_baseline_lambda_label(owner)
    _configure_baseline_length(owner, spectral_length)

    def on_toggled(checked):
        owner.baseline_lambda_controls.setVisible(bool(checked))
        redraw()

    def on_lambda_changed(_index):
        _update_baseline_lambda_label(owner)
        redraw()

    def on_lambda_input_changed(value):
        value = min(1e10, max(1e2, float(value)))
        slider_value = int(round(
            np.log10(value) * _BASELINE_LAMBDA_STEPS_PER_DECADE
        ))
        slider_value = min(
            owner.baseline_lambda_slider.maximum(),
            max(owner.baseline_lambda_slider.minimum(), slider_value),
        )
        if owner.baseline_lambda_slider.value() != slider_value:
            owner.baseline_lambda_slider.setValue(slider_value)
        else:
            _update_baseline_lambda_label(owner)
            redraw()

    owner.baseline_subtraction_check.toggled.connect(on_toggled)
    owner.baseline_lambda_slider.valueChanged.connect(on_lambda_changed)
    owner.baseline_lambda_input.valueChanged.connect(on_lambda_input_changed)


def _subtract_baseline_for_plot(owner, spectra):
    """Return display data after optional AsLS baseline subtraction."""
    spectra = np.asarray(spectra, dtype=float)
    if not owner.baseline_subtraction_check.isChecked():
        return spectra
    baseline = asls_baseline(spectra, lam=_baseline_lambda(owner))
    corrected = spectra.copy()
    np.subtract(
        spectra,
        baseline,
        out=corrected,
        where=np.isfinite(spectra),
    )
    return corrected


class CalibrationPlotWindow(QWidget):
    """Calibration image and the spectrum acquired at a selected point."""

    _pick_radius_pixels = 12.0

    def __init__(
        self,
        ds,
        title="Calibration result",
        spectral_calibration=None,
        cosmic_ray_calibration=None,
        exposure_ms=None,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(980, 600)
        self.ds = ds
        self.spectral_calibration = spectral_calibration
        self.selected_index = None
        self.selected_spectrum = None
        self.spectrum_line = None
        self._fixed_y_limits = None
        self.raw_spectra = self._dataset_array("specs")
        self._point_positions = self._read_point_positions()
        self._plotted_indices = np.flatnonzero(
            np.all(np.isfinite(self._point_positions), axis=1)
        )

        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )

        layout = _panel_layout(self)
        self.selection_label = QLabel()
        self.selection_label.setWordWrap(True)
        self.fix_y_scale_check = QCheckBox("Fix Y scale")
        self.fix_y_scale_check.setToolTip(
            "Keep the selected spectrum's current Y limits"
        )
        self.fix_y_scale_check.toggled.connect(
            self._on_fix_y_scale_toggled
        )
        self.show_wavenumber_check = _make_wavenumber_axis_checkbox(
            spectral_calibration, self._redraw_spectrum
        )
        _make_white_background_checkbox(self)
        self.view_controls_group = _add_compact_control_group(
            layout,
            "Selected acquisition",
            [
                [self.selection_label],
                [
                    self.fix_y_scale_check,
                    self.show_wavenumber_check,
                    self.white_background_check,
                ],
            ],
        )
        spectral_length = (
            0
            if self.raw_spectra is None or self.raw_spectra.ndim == 0
            else self.raw_spectra.shape[-1]
        )
        _add_processing_section(
            self,
            layout,
            self._redraw_spectrum,
            spectral_length,
            cosmic_ray_calibration=cosmic_ray_calibration,
            cosmic_ray_values=self.raw_spectra,
            exposure_ms=exposure_ms,
        )

        self.fig = Figure(figsize=(10, 5), constrained_layout=True)
        _configure_matplotlib_figure(self.fig)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax_image, self.ax_spectrum = self.fig.subplots(1, 2)
        # ``ax`` was the only axes attribute implicitly available in the old
        # single-panel implementation.  Keep it as a compatibility alias.
        self.ax = self.ax_image
        self._draw_image(title)
        self._select_initial_point()
        self._click_connection = self.canvas.mpl_connect(
            "button_press_event", self._on_image_click
        )
        _finish_plot_layout(
            layout,
            self.toolbar,
            self.canvas,
            background_control_in_layout=False,
        )
        self._redraw_spectrum()

    def _dataset_array(self, name):
        if name not in self.ds:
            return None
        try:
            return np.asarray(self.ds[name], dtype=float)
        except (TypeError, ValueError):
            return None

    def _read_point_positions(self):
        positions = self._dataset_array("rel_BF_pos")
        if positions is None:
            return np.empty((0, 2), dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if positions.ndim != 2 or positions.shape[1] < 2:
            return np.empty((0, 2), dtype=float)
        return positions[:, :2].copy()

    def _image_projection(self):
        if "imgs" not in self.ds:
            return None
        try:
            # Keep the camera's native dtype and avoid materialising a second
            # full-sized float64 stack merely to calculate the projection.
            images = np.asarray(self.ds["imgs"])
        except (TypeError, ValueError):
            return None
        if images is None or images.ndim < 2 or images.size == 0:
            return None
        if images.ndim == 2:
            return images
        axes = tuple(range(images.ndim - 2))
        try:
            if np.issubdtype(images.dtype, np.floating):
                # fmax is a NaN-tolerant ufunc reduction and allocates only
                # the 2-D result, unlike nanmax's full-size temporary copy.
                return np.fmax.reduce(images, axis=axes)
            return np.max(images, axis=axes)
        except (TypeError, ValueError):
            return None

    def _draw_image(self, title):
        projection = self._image_projection()
        if projection is None:
            self.ax_image.text(
                0.5,
                0.5,
                "Calibration image unavailable",
                ha="center",
                va="center",
                color="#767676",
                transform=self.ax_image.transAxes,
            )
            self.ax_image.set_axis_off()
        else:
            self.image_artist = self.ax_image.imshow(
                projection, cmap="gray"
            )
            self.ax_image.set_xlabel("Image X pixel")
            self.ax_image.set_ylabel("Image Y pixel")
        if len(self._plotted_indices):
            points = self._point_positions[self._plotted_indices]
            self.points_artist = self.ax_image.scatter(
                points[:, 0],
                points[:, 1],
                color="#ff4d4d",
                edgecolor="black",
                linewidth=0.5,
                s=32,
                zorder=3,
            )
        else:
            self.points_artist = self.ax_image.scatter([], [], s=32)
        self.selection_artist = self.ax_image.scatter(
            [],
            [],
            facecolors="none",
            edgecolors="#ffe066",
            linewidth=2.0,
            s=110,
            zorder=4,
        )
        self.ax_image.set_title(f"{title} · click a red point")

    def _spectrum_for_point(self, index):
        if self.raw_spectra is None or self.raw_spectra.ndim == 0:
            return None
        if index < 0 or index >= self.raw_spectra.shape[0]:
            return None
        spectra = _cosmic_ray_values_for_plot(self)
        spectrum = np.asarray(spectra[index], dtype=float)
        if spectrum.ndim == 0 or spectrum.size == 0:
            return None
        if spectrum.ndim == 1:
            result = spectrum.copy()
        else:
            repeats = spectrum.reshape(-1, spectrum.shape[-1])
            if repeats.shape[0] == 0 or not np.isfinite(repeats).any():
                return None
            result = filter_mean(repeats)
        if result.ndim != 1 or not np.isfinite(result).any():
            return None
        result = _subtract_baseline_for_plot(self, result)
        return _smooth_for_plot(self, result)

    def _select_initial_point(self):
        for index in self._plotted_indices:
            if self._spectrum_for_point(int(index)) is not None:
                self.selected_index = int(index)
                return
        if len(self._plotted_indices):
            self.selected_index = int(self._plotted_indices[0])

    def _toolbar_is_active(self):
        return bool(getattr(self.toolbar, "mode", None))

    def _on_image_click(self, event):
        if (
            self._toolbar_is_active()
            or event.button != 1
            or event.inaxes is not self.ax_image
            or event.x is None
            or event.y is None
            or not len(self._plotted_indices)
        ):
            return
        points = self._point_positions[self._plotted_indices]
        display_points = self.ax_image.transData.transform(points)
        distances = np.hypot(
            display_points[:, 0] - event.x,
            display_points[:, 1] - event.y,
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] > self._pick_radius_pixels:
            return
        self.select_point(int(self._plotted_indices[nearest]))

    def select_point(self, index):
        """Select one real, finite calibration acquisition by row index."""
        index = int(index)
        if index not in self._plotted_indices:
            return False
        self.selected_index = index
        self._redraw_spectrum()
        return True

    def _selection_metadata(self, spectrum):
        if self.selected_index is None:
            return "No finite calibration acquisition points are available."
        index = self.selected_index
        x, y = self._point_positions[index]
        parts = [
            f"Point {index + 1} of {len(self._point_positions)}",
            f"image (x={x:.3g}, y={y:.3g})",
        ]
        laser_positions = self._dataset_array("laser_pos")
        if (
            laser_positions is not None
            and laser_positions.ndim >= 2
            and index < laser_positions.shape[0]
            and laser_positions.shape[1] >= 2
        ):
            vx, vy = laser_positions[index, :2]
            if np.isfinite(vx) and np.isfinite(vy):
                parts.append(f"galvo (x={vx:.3g} V, y={vy:.3g} V)")
        if (
            self.raw_spectra is not None
            and self.raw_spectra.ndim >= 1
            and index < self.raw_spectra.shape[0]
        ):
            point_data = np.asarray(self.raw_spectra[index])
            repeat_count = 1 if point_data.ndim <= 1 else int(
                np.prod(point_data.shape[:-1])
            )
            parts.append(
                f"{repeat_count} acquired trace"
                f"{'s' if repeat_count != 1 else ''}"
            )
        if spectrum is None:
            parts.append("spectrum unavailable")
        return " · ".join(parts)

    def _redraw_spectrum(self, _checked=None):
        self.ax_spectrum.clear()
        self.spectrum_line = None
        self.selected_spectrum = None
        if self.selected_index is None:
            self.selection_artist.set_offsets(np.empty((0, 2)))
            self.selection_label.setText(
                "No finite calibration acquisition points are available."
            )
            self.ax_spectrum.text(
                0.5,
                0.5,
                "Select a valid acquisition point",
                ha="center",
                va="center",
                color="#767676",
                transform=self.ax_spectrum.transAxes,
            )
            self.ax_spectrum.set_title("Selected spectrum")
            self.canvas.draw_idle()
            return

        point = self._point_positions[self.selected_index]
        self.selection_artist.set_offsets(point.reshape(1, 2))
        spectrum = self._spectrum_for_point(self.selected_index)
        self.selected_spectrum = spectrum
        self.selection_label.setText(self._selection_metadata(spectrum))
        self.ax_spectrum.set_title(
            f"Point {self.selected_index + 1} spectrum"
        )
        if spectrum is None:
            self.ax_spectrum.text(
                0.5,
                0.5,
                "Spectrum unavailable for this point",
                ha="center",
                va="center",
                color="#767676",
                transform=self.ax_spectrum.transAxes,
            )
            self.ax_spectrum.set_xlabel("Pixel")
            self.ax_spectrum.set_ylabel("Intensity (a.u.)")
        else:
            (self.spectrum_line,) = self.ax_spectrum.plot(
                spectrum, color="#2f80ed"
            )
            _set_spectral_line_axis(
                self.ax_spectrum,
                [self.spectrum_line],
                self.spectral_calibration,
                self.show_wavenumber_check.isChecked(),
            )
            self.ax_spectrum.set_ylabel("Intensity (a.u.)")
            if self._fixed_y_limits is not None:
                self.ax_spectrum.set_ylim(self._fixed_y_limits)
        self.canvas.draw_idle()

    def _on_fix_y_scale_toggled(self, checked):
        if checked and self.spectrum_line is not None:
            self._fixed_y_limits = self.ax_spectrum.get_ylim()
        elif not checked:
            self._fixed_y_limits = None
            self._redraw_spectrum()


class DetectorImageWindow(QWidget):
    """Detector image panel with a switchable row-summed spectrum view."""

    def __init__(
        self,
        frames,
        title="Detector image",
        spectral_calibration=None,
        cosmic_ray_calibration=None,
        exposure_ms=None,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(800, 550)
        self.raw_frames = np.array(frames, copy=True)
        self.image = self._mean_detector_image(self.raw_frames)
        self.spectral_calibration = spectral_calibration
        self._show_spectrum = False
        self._fixed_y_limits = None

        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )

        layout = _panel_layout(self)
        self.view_mode_label = QLabel("Current view: detector image")
        self.view_mode_label.setStyleSheet("font-weight: bold;")
        self.toggle_btn = QPushButton("Show spectrum")
        self.toggle_btn.setToolTip(
            "Switch between the detector image and its selected-row sum"
        )
        self.toggle_btn.clicked.connect(self._toggle_view)
        _make_repeat_controls(self, self._on_repeat_changed)
        _configure_repeat_controls(
            self, self._detector_repeat_count(), reset=True
        )
        rows_label = QLabel("Rows")
        rows_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.start_row_input = QSpinBox()
        self.start_row_input.setRange(0, self.image.shape[0] - 1)
        self.start_row_input.setValue(0)
        self.start_row_input.valueChanged.connect(self._on_start_row_changed)
        to_label = QLabel("to")
        to_label.setAlignment(Qt.AlignCenter)
        self.end_row_input = QSpinBox()
        self.end_row_input.setRange(0, self.image.shape[0] - 1)
        self.end_row_input.setValue(self.image.shape[0] - 1)
        self.end_row_input.valueChanged.connect(self._on_end_row_changed)
        self.show_wavenumber_check = _make_wavenumber_axis_checkbox(
            spectral_calibration, self._redraw
        )
        self.show_wavenumber_check.hide()
        self.fix_y_scale_check = QCheckBox("Fix Y scale")
        self.fix_y_scale_check.setToolTip(
            "Keep the spectrum Y limits from the frame shown when checked"
        )
        self.fix_y_scale_check.toggled.connect(
            self._on_fix_y_scale_toggled
        )
        self.fix_y_scale_check.hide()
        _make_white_background_checkbox(self)
        self.view_controls_group = _add_compact_control_group(
            layout,
            "View",
            [
                [self.view_mode_label, self.toggle_btn],
                [self.repeat_controls],
                [
                    rows_label,
                    self.start_row_input,
                    to_label,
                    self.end_row_input,
                ],
                [
                    self.fix_y_scale_check,
                    self.show_wavenumber_check,
                    self.white_background_check,
                ],
            ],
        )
        _add_processing_section(
            self,
            layout,
            self._redraw,
            self.image.shape[1],
            cosmic_ray_calibration=cosmic_ray_calibration,
            cosmic_ray_values=self.raw_frames,
            exposure_ms=exposure_ms,
            read_mode="image",
        )
        self.baseline_subtraction_check.hide()
        self.smoothing_check.hide()

        self.fig = Figure(figsize=(8, 5))
        _configure_matplotlib_figure(self.fig)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        self._colorbar = None
        _finish_plot_layout(
            layout,
            self.toolbar,
            self.canvas,
            background_control_in_layout=False,
        )
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

    def _detector_repeat_count(self):
        return 1 if self.raw_frames.ndim == 2 else self.raw_frames.shape[0]

    def _display_detector_image(self):
        frames = np.asarray(_cosmic_ray_values_for_plot(self))
        if frames.ndim == 2:
            return frames
        repeat = self.repeat_slider.value()
        if repeat > 0:
            return frames[repeat - 1]
        return np.mean(frames, axis=0)

    def _on_repeat_changed(self, _repeat):
        self._redraw()

    def _toggle_view(self):
        self._show_spectrum = not self._show_spectrum
        self.toggle_btn.setText(
            "Show image" if self._show_spectrum else "Show spectrum"
        )
        self.show_wavenumber_check.setVisible(self._show_spectrum)
        self.fix_y_scale_check.setVisible(self._show_spectrum)
        self.baseline_subtraction_check.setVisible(self._show_spectrum)
        self.baseline_lambda_controls.setVisible(
            self._show_spectrum
            and self.baseline_subtraction_check.isChecked()
        )
        self.smoothing_check.setVisible(self._show_spectrum)
        self.smoothing_window_controls.setVisible(
            self._show_spectrum and self.smoothing_check.isChecked()
        )
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
        self.raw_frames = np.array(frames, copy=True)
        self.image = self._mean_detector_image(self.raw_frames)
        _configure_repeat_controls(
            self, self._detector_repeat_count(), reset=True
        )
        _replace_cosmic_ray_values(self, self.raw_frames)
        _configure_baseline_length(self, self.image.shape[1])
        _configure_smoothing_length(self, self.image.shape[1])
        last_row = self.image.shape[0] - 1
        self.start_row_input.blockSignals(True)
        self.end_row_input.blockSignals(True)
        try:
            self.start_row_input.setMaximum(last_row)
            self.end_row_input.setMaximum(last_row)
            if was_full_range:
                self.start_row_input.setValue(0)
                self.end_row_input.setValue(last_row)
        finally:
            self.start_row_input.blockSignals(False)
            self.end_row_input.blockSignals(False)
        if title is not None:
            self.setWindowTitle(title)
        self._redraw()

    def _redraw(self, _checked=None):
        self.ax.clear()
        display_image = self._display_detector_image()
        repeat_count = self._detector_repeat_count()
        repeat = self.repeat_slider.value()
        if repeat_count > 1:
            repeat_description = (
                f"repeat {repeat} of {repeat_count}" if repeat else "mean"
            )
        else:
            repeat_description = "single acquisition"
        view_description = (
            "row-sum spectrum" if self._show_spectrum else "detector image"
        )
        self.view_mode_label.setText(
            f"Current view: {view_description} · {repeat_description}"
        )
        if self._show_spectrum:
            spectrum = sum_detector_rows(
                display_image,
                self.start_row_input.value(),
                self.end_row_input.value(),
            )
            spectrum = _subtract_baseline_for_plot(self, spectrum)
            spectrum = _smooth_for_plot(self, spectrum)
            lines = self.ax.plot(spectrum)
            _set_spectral_line_axis(
                self.ax,
                lines,
                self.spectral_calibration,
                self.show_wavenumber_check.isChecked(),
            )
            self.ax.set_ylabel("Summed intensity (a.u.)")
            repeat_title = (
                f"Repeat {repeat} of {repeat_count}"
                if repeat
                else "Mean"
            )
            self.ax.set_title(
                f"{repeat_title} row-sum spectrum · rows "
                f"{self.start_row_input.value()}–{self.end_row_input.value()}"
            )
            if self._fixed_y_limits is not None:
                self.ax.set_ylim(self._fixed_y_limits)
            if self._colorbar is not None:
                self._colorbar.ax.set_visible(False)
        else:
            image_artist = self.ax.imshow(
                display_image,
                cmap="gray",
                aspect="auto",
                origin="lower",
            )
            self.ax.set_xlabel("Detector X pixel")
            self.ax.set_ylabel("Detector Y pixel")
            if repeat:
                self.ax.set_title(
                    f"Detector image · repeat {repeat} of {repeat_count}"
                )
            else:
                self.ax.set_title("Mean detector image")
            if self._colorbar is None:
                self._colorbar = self.fig.colorbar(
                    image_artist, ax=self.ax, label="Intensity (a.u.)"
                )
            else:
                self._colorbar.update_normal(image_artist)
                self._colorbar.ax.set_visible(True)
        self.canvas.draw_idle()


class SpectrumWindow(QWidget):
    """Spectrum panel with a toggle between mean and all-traces views."""

    def __init__(
        self,
        spec,
        title="Spectrum",
        spectral_calibration=None,
        calibration_changed=None,
        spectral_bias=None,
        cosmic_ray_calibration=None,
        exposure_ms=None,
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
        self._show_wavenumber_before_calibration = False
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        layout = _panel_layout(self)
        self.view_mode_label = QLabel("Current view: mean spectrum")
        self.view_mode_label.setStyleSheet("font-weight: bold;")
        self.toggle_btn = QPushButton("Show all")
        self.toggle_btn.setToolTip(
            "Switch between the mean spectrum and every acquired trace"
        )
        self.toggle_btn.clicked.connect(self._toggle)
        _make_repeat_controls(self, self._on_repeat_changed)
        _configure_repeat_controls(self, self.spec.shape[0], reset=True)
        self.show_wavenumber_check = _make_wavenumber_axis_checkbox(
            spectral_calibration, self._redraw
        )
        self.fix_y_scale_check = QCheckBox("Fix Y scale")
        self.fix_y_scale_check.setToolTip(
            "Keep the Y limits from the frame shown when checked"
        )
        self.fix_y_scale_check.toggled.connect(
            self._on_fix_y_scale_toggled
        )
        _make_white_background_checkbox(self)
        self.remove_spectral_bias_check = QCheckBox("Remove spectral bias")
        self.remove_spectral_bias_check.setToolTip(
            "Subtract the filter_mean(dark noise) detector pedestal after "
            "RACER in this plot only. This does not identify cosmic rays, "
            "and the raw spectra remain unchanged."
        )
        self.remove_spectral_bias_check.toggled.connect(self._redraw)
        self.remove_spectral_bias_check.setVisible(
            self.spectral_bias is not None
        )
        self.calibration_btn = QPushButton("Calibrate axis")
        self.calibration_btn.setToolTip(
            "Create or replace the pixel-to-wavenumber calibration"
        )
        self.calibration_btn.clicked.connect(self._start_calibration)
        self.view_controls_group = _add_compact_control_group(
            layout,
            "View",
            [
                [self.view_mode_label],
                [self.repeat_controls],
                [
                    self.fix_y_scale_check,
                    self.show_wavenumber_check,
                    self.white_background_check,
                ],
                [
                    self.remove_spectral_bias_check,
                    self.toggle_btn,
                    self.calibration_btn,
                ],
            ],
        )
        _add_processing_section(
            self,
            layout,
            self._redraw,
            self.spec.shape[-1],
            cosmic_ray_calibration=cosmic_ray_calibration,
            cosmic_ray_values=self.spec,
            exposure_ms=exposure_ms,
            fvb_repeat_guard=True,
        )

        self.calibration_controls = QWidget()
        calibration_layout = QGridLayout(self.calibration_controls)
        calibration_layout.setContentsMargins(6, 4, 6, 4)
        calibration_layout.setHorizontalSpacing(6)
        calibration_layout.setVerticalSpacing(4)
        degree_label = QLabel("Polynomial degree")
        degree_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        calibration_layout.addWidget(degree_label, 0, 0)
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
        calibration_layout.addWidget(self.calibration_degree_input, 0, 1)
        self.calibration_help = QLabel()
        self.calibration_help.setWordWrap(True)
        calibration_layout.addWidget(self.calibration_help, 1, 0, 1, 3)
        self.finish_calibration_btn = QPushButton("Finish and save")
        self.finish_calibration_btn.clicked.connect(self._finish_calibration)
        calibration_layout.addWidget(self.finish_calibration_btn, 2, 1)
        self.cancel_calibration_btn = QPushButton("Cancel")
        self.cancel_calibration_btn.clicked.connect(self._cancel_calibration)
        calibration_layout.addWidget(self.cancel_calibration_btn, 2, 2)
        calibration_layout.setColumnStretch(0, 1)
        self.calibration_controls.hide()
        layout.addWidget(self.calibration_controls)
        self.fig = Figure(figsize=(7, 4.5))
        _configure_matplotlib_figure(self.fig)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax = self.fig.add_subplot(111)
        _finish_plot_layout(
            layout,
            self.toolbar,
            self.canvas,
            background_control_in_layout=False,
        )
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
        spectra = _cosmic_ray_values_for_plot(self)
        if (
            self.spectral_bias is not None
            and self.remove_spectral_bias_check.isChecked()
        ):
            return subtract_spectral_bias(spectra, self.spectral_bias)
        return spectra

    def update_spectrum(self, spec, *, title=None):
        """Replace plotted data, for example after each live exposure."""
        self.spec = self._normalize_spectra(spec)
        _configure_repeat_controls(self, self.spec.shape[0], reset=True)
        _replace_cosmic_ray_values(self, self.spec)
        _configure_baseline_length(self, self.spec.shape[-1])
        _configure_smoothing_length(self, self.spec.shape[-1])
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
        self.repeat_slider.blockSignals(True)
        self.repeat_input.blockSignals(True)
        try:
            self.repeat_slider.setValue(0)
            self.repeat_input.setValue(0)
        finally:
            self.repeat_slider.blockSignals(False)
            self.repeat_input.blockSignals(False)
        self._show_mean = not self._show_mean
        self.toggle_btn.setText(
            "Show all" if self._show_mean else "Show mean"
        )
        self._redraw()

    def _on_repeat_changed(self, repeat):
        if repeat > 0 and not self._show_mean:
            self._show_mean = True
            self.toggle_btn.setText("Show all")
        self._redraw()

    def _redraw(self):
        import matplotlib.cm as cm
        self.ax.clear()
        display_spectra = self._display_spectra()
        repeat = self.repeat_slider.value()
        repeat_count = display_spectra.shape[0]
        lines = []
        if repeat > 0:
            selected_spectrum = display_spectra[repeat - 1]
            selected_spectrum = _subtract_baseline_for_plot(
                self, selected_spectrum
            )
            lines.extend(
                self.ax.plot(_smooth_for_plot(self, selected_spectrum))
            )
            title = f"Repeat {repeat} of {repeat_count}"
            self.view_mode_label.setText(
                f"Current view: repeat {repeat} of {repeat_count}"
            )
        elif self._show_mean:
            mean_spectrum = filter_mean(display_spectra)
            mean_spectrum = _subtract_baseline_for_plot(
                self, mean_spectrum
            )
            lines.extend(
                self.ax.plot(_smooth_for_plot(self, mean_spectrum))
            )
            title = "Mean spectrum"
            self.view_mode_label.setText("Current view: mean spectrum")
        else:
            n = display_spectra.shape[0]
            colors = cm.viridis(np.linspace(0, 1, n))
            display_spectra = _subtract_baseline_for_plot(
                self, display_spectra
            )
            display_spectra = _smooth_for_plot(self, display_spectra)
            for i in range(n):
                lines.extend(
                    self.ax.plot(
                        display_spectra[i],
                        color=colors[i],
                        linewidth=0.8,
                    )
                )
            title = "All traces"
            self.view_mode_label.setText("Current view: all traces")
        _set_spectral_line_axis(
            self.ax,
            lines,
            self.spectral_calibration,
            self.show_wavenumber_check.isChecked(),
        )
        self.ax.set_ylabel("Intensity (a.u.)")
        if self.remove_spectral_bias_check.isChecked():
            title = f"{title} · bias corrected"
        if self.baseline_subtraction_check.isChecked():
            title = f"{title} · baseline corrected"
        self.ax.set_title(title)
        if self._fixed_y_limits is not None:
            self.ax.set_ylim(self._fixed_y_limits)
        if self._calibrating:
            self._draw_calibration_artists()
        self.canvas.draw_idle()

    def _start_calibration(self):
        self._calibrating = True
        self._show_wavenumber_before_calibration = (
            self.show_wavenumber_check.isChecked()
        )
        self._pending_pixel = None
        self._calibration_pixels = []
        self._known_shifts = []
        self._show_mean = True
        _configure_repeat_controls(self, self.spec.shape[0], reset=True)
        self.toggle_btn.setText("Show all")
        self.view_mode_label.setText("Current view: mean spectrum")
        self.toggle_btn.setEnabled(False)
        self.calibration_btn.setEnabled(False)
        self.show_wavenumber_check.setChecked(False)
        self.show_wavenumber_check.setEnabled(False)
        self.calibration_controls.show()
        self.finish_calibration_btn.setEnabled(False)
        self._update_calibration_progress()
        self._redraw()
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()

    def _spectrum_for_calibration(self):
        spectrum = filter_mean(self._display_spectra())
        spectrum = _subtract_baseline_for_plot(self, spectrum)
        return np.asarray(_smooth_for_plot(self, spectrum), dtype=float)

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
        self._stop_calibration()
        QMessageBox.information(
            self,
            "Calibration saved",
            f"Saved pixel-to-wavenumber calibration to:\n{saved_path}",
        )

    def _cancel_calibration(self):
        self._stop_calibration()

    def _stop_calibration(self):
        self._calibrating = False
        self._pending_pixel = None
        self.calibration_controls.hide()
        _configure_repeat_controls(self, self.spec.shape[0])
        self.toggle_btn.setEnabled(True)
        self.calibration_btn.setEnabled(True)
        self.show_wavenumber_check.setEnabled(
            self.spectral_calibration is not None
        )
        if self.spectral_calibration is not None:
            self.show_wavenumber_check.setToolTip(
                "Use calibrated Raman shift instead of pixels"
            )
            self.show_wavenumber_check.setChecked(
                self._show_wavenumber_before_calibration
            )
        self._redraw()


class ReferenceSpectraWindow(QWidget):
    """Panel showing reference spectra colored by z, with a colorbar."""

    def __init__(
        self, all_raman, zs, title="Reference spectra",
        spectral_calibration=None,
        cosmic_ray_calibration=None,
        exposure_ms=None,
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
        layout = _panel_layout(self)
        self.spectral_calibration = spectral_calibration
        self._raw_reference_spectra = np.array(all_raman, copy=True)
        self._reference_spectra = np.asarray(
            [filter_mean(repeats) for repeats in self._raw_reference_spectra]
        )
        self.show_wavenumber_check = _make_wavenumber_axis_checkbox(
            spectral_calibration, self._update_spectral_axis
        )
        _make_white_background_checkbox(self)
        self.view_controls_group = _add_compact_control_group(
            layout,
            "View",
            [[self.show_wavenumber_check, self.white_background_check]],
        )
        _add_processing_section(
            self,
            layout,
            self._redraw_spectra,
            self._reference_spectra.shape[-1],
            cosmic_ray_calibration=cosmic_ray_calibration,
            cosmic_ray_values=self._raw_reference_spectra,
            exposure_ms=exposure_ms,
        )
        self.fig = Figure(figsize=(8, 5.5))
        _configure_matplotlib_figure(self.fig)
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
                    self._reference_spectra[i],
                    color=color,
                    linewidth=0.9,
                )
            )
        self.ax = ax
        self._update_spectral_axis()
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Reference spectra")
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=ax)
        cbar.set_label("z (um)")
        _finish_plot_layout(
            layout,
            self.toolbar,
            self.canvas,
            background_control_in_layout=False,
        )

    def _redraw_spectra(self):
        raw_spectra = _cosmic_ray_values_for_plot(self)
        spectra = np.asarray(
            [filter_mean(repeats) for repeats in raw_spectra]
        )
        spectra = _subtract_baseline_for_plot(
            self, spectra
        )
        spectra = _smooth_for_plot(self, spectra)
        for line, spectrum in zip(self._spectral_lines, spectra):
            line.set_ydata(spectrum)
        self._update_spectral_axis()

    def _update_spectral_axis(self, _checked=None):
        _set_spectral_line_axis(
            self.ax,
            self._spectral_lines,
            self.spectral_calibration,
            self.show_wavenumber_check.isChecked(),
        )
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()


class GridScanPlotWindow(QWidget):
    """Panel showing grid scan results.

    Single-plane mode: BF, end_BF, extra channels, and a spectrum panel.
    Z-scan mode: z-slider controlling the BF_z image and the spectrum panel.

    In both modes the grid sampling points are overlaid (low alpha) on the
    primary image. Click a point to show its individual spectrum; a toggle
    button switches between the average spectrum and the clicked point's
    spectrum.
    """

    FIXED = ["BF", "end_BF"]

    def __init__(
        self,
        ds,
        title="Grid scan result",
        spectral_calibration=None,
        cosmic_ray_calibration=None,
        exposure_ms=None,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.ds = ds
        self._show_average = True
        self._sel = 0
        self._scat = None
        self._hl = None
        self.spectral_calibration = spectral_calibration
        self._pending_cosmic_ray_calibration = cosmic_ray_calibration
        self._pending_cosmic_ray_exposure_ms = exposure_ms
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
        self.band_map_controls.attach()

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                      #
    # ------------------------------------------------------------------ #
    def _make_mode_button(self):
        btn = QPushButton("Show point")
        btn.setToolTip(
            "Switch between the mean spectrum and the selected grid point"
        )
        btn.clicked.connect(self._toggle_mode)
        self._mode_btn = btn
        return btn

    def _make_spectral_controls(self):
        rows = QVBoxLayout()
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setSpacing(4)
        self.view_mode_label = QLabel("Current view: mean spectrum")
        self.view_mode_label.setStyleSheet("font-weight: bold;")
        mode_button = self._make_mode_button()
        self.show_wavenumber_check = _make_wavenumber_axis_checkbox(
            self.spectral_calibration, self._on_spectral_axis_changed
        )
        _make_white_background_checkbox(self)
        self.view_controls_group = _add_compact_control_group(
            rows,
            "View",
            [
                [self.view_mode_label, mode_button],
                [self.show_wavenumber_check, self.white_background_check],
            ],
        )
        _add_processing_section(
            self,
            rows,
            self._on_spectral_axis_changed,
            self._specs.shape[-1],
            cosmic_ray_calibration=self._pending_cosmic_ray_calibration,
            cosmic_ray_values=self._specs,
            exposure_ms=self._pending_cosmic_ray_exposure_ms,
        )
        self.band_map_controls = BandMapControls(self)
        rows.insertWidget(1, self.band_map_controls)
        return rows

    def _on_spectral_axis_changed(self, _checked=None):
        self.band_map_controls.sync_axis()
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

    def _current_display_specs_2d(self):
        """Return display spectra while keeping band maps on raw data."""
        spectra = _cosmic_ray_values_for_plot(self)
        if self._has_zscan:
            return spectra[self._z_slider.value()]
        return spectra

    def _toggle_mode(self):
        self._show_average = not self._show_average
        self._mode_btn.setText(
            "Show point" if self._show_average else "Show mean"
        )
        self.view_mode_label.setText(
            "Current view: mean spectrum"
            if self._show_average
            else "Current view: selected point"
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
        self._mode_btn.setText("Show mean")
        self.view_mode_label.setText("Current view: selected point")
        self._update_highlight()
        self._draw_spec()
        self.band_map_controls.selection_changed()
        self.canvas.draw_idle()

    def _draw_spec(self):
        specs2d = self._current_display_specs_2d()
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
        y = _subtract_baseline_for_plot(self, y)
        y = _smooth_for_plot(self, y)
        self._spec_line.set_ydata(y)
        _set_spectral_line_axis(
            self._ax_spec,
            [self._spec_line],
            self.spectral_calibration,
            self.show_wavenumber_check.isChecked(),
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
        _configure_matplotlib_figure(self.fig)
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
        layout = _panel_layout(self)
        layout.addLayout(self._make_spectral_controls())
        self._draw_spec()
        _finish_plot_layout(
            layout,
            self.toolbar,
            self.canvas,
            background_control_in_layout=False,
        )
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
        main_layout = _panel_layout(self)
        # --- top controls: mode button + z slider ---
        main_layout.addLayout(self._make_spectral_controls())
        slider_row = QHBoxLayout()
        self._z_label = QLabel(
            f"z = {self._z_vals[0]:+.2f} um  (1/{self._n_z})"
        )
        slider_row.addWidget(self._z_label)
        self._z_slider = QSlider(Qt.Horizontal)
        # Update large band maps on release rather than throughout a drag.
        self._z_slider.setTracking(False)
        self._z_slider.setMinimum(0)
        self._z_slider.setMaximum(self._n_z - 1)
        self._z_slider.setValue(0)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        slider_row.addWidget(self._z_slider, 1)
        self._z_input = QSpinBox()
        self._z_input.setKeyboardTracking(False)
        self._z_input.setRange(0, self._n_z - 1)
        self._z_input.setValue(0)
        self._z_input.setToolTip("Z-plane index")
        self._z_input.setMaximumWidth(80)
        self._z_slider.valueChanged.connect(self._z_input.setValue)
        self._z_input.valueChanged.connect(self._z_slider.setValue)
        slider_row.addWidget(self._z_input)
        main_layout.addLayout(slider_row)
        # --- Figure: top row = images, bottom = spectrum ---
        n_extra = len(self._extra_2d)
        ncols = 1 + n_extra  # BF_z + extra channels
        self.fig = Figure(figsize=(4 * ncols + 1, 7))
        _configure_matplotlib_figure(self.fig)
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
        _finish_plot_layout(
            main_layout,
            self.toolbar,
            self.canvas,
            background_control_in_layout=False,
        )
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
        self.band_map_controls.plane_changed()
        self.canvas.draw_idle()


class DatasetViewerWindow(QWidget):
    """Interactive viewer: BF image with laser scatter + spectrum,
    Qt sliders for t/p/z."""

    def __init__(
        self,
        df,
        da,
        title="Dataset viewer",
        spectral_calibration=None,
        cosmic_ray_calibration=None,
        exposure_ms=None,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1200, 650)
        self.df = df
        self.da = da
        self._raw_dataset_spectra = np.array(
            df.iloc[:, :-3].to_numpy(), copy=True
        )
        self.spectral_calibration = spectral_calibration
        self.bf = da.sel(c=0).values  # (t, p, z, y, x)
        self._pt_selected = 0
        self._fixed_y_limits = None
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.cm as mcm
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg, NavigationToolbar2QT,
        )
        self._mcm = mcm
        main_layout = _panel_layout(self)
        # --- Dataset navigation ---
        self.t_vals = da.coords["t"].values
        self.p_vals = da.coords["p"].values
        self.z_vals = da.coords["z"].values
        self.navigation_group = QGroupBox("Dataset navigation")
        navigation_layout = QGridLayout(self.navigation_group)
        navigation_layout.setContentsMargins(6, 4, 6, 6)
        navigation_layout.setHorizontalSpacing(6)
        navigation_layout.setVerticalSpacing(4)
        self.t_slider, self.t_label, self.t_input = self._make_slider(
            "Time (t)", 0, len(self.t_vals) - 1
        )
        self.p_slider, self.p_label, self.p_input = self._make_slider(
            "Position (p)", 0, len(self.p_vals) - 1
        )
        self.z_slider, self.z_label, self.z_input = self._make_slider(
            "Z plane", 0, len(self.z_vals) - 1
        )
        for row, (label, slider, value_input) in enumerate([
            (self.t_label, self.t_slider, self.t_input),
            (self.p_label, self.p_slider, self.p_input),
            (self.z_label, self.z_slider, self.z_input),
        ]):
            navigation_layout.addWidget(label, row, 0)
            navigation_layout.addWidget(slider, row, 1)
            navigation_layout.addWidget(value_input, row, 2)
        navigation_layout.setColumnStretch(1, 1)
        main_layout.addWidget(self.navigation_group)

        # --- View and display processing ---
        self.show_wavenumber_check = _make_wavenumber_axis_checkbox(
            spectral_calibration, self._update_spectral_axis
        )
        self.fix_y_scale_check = QCheckBox("Fix Y scale")
        self.fix_y_scale_check.setToolTip(
            "Keep the spectrum Y limits from the frame shown when checked"
        )
        self.fix_y_scale_check.toggled.connect(
            self._on_fix_y_scale_toggled
        )
        _make_white_background_checkbox(self)
        self.view_controls_group = _add_compact_control_group(
            main_layout,
            "View",
            [[
                self.fix_y_scale_check,
                self.show_wavenumber_check,
                self.white_background_check,
            ]],
        )
        _add_processing_section(
            self,
            main_layout,
            self._on_smoothing_changed,
            max(0, int(df.shape[1]) - 3),
            cosmic_ray_calibration=cosmic_ray_calibration,
            cosmic_ray_values=self._raw_dataset_spectra,
            exposure_ms=exposure_ms,
        )
        # --- Figure ---
        self.fig = Figure(figsize=(12, 5))
        _configure_matplotlib_figure(self.fig)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.ax_img = self.fig.add_subplot(121)
        self.ax_spec = self.fig.add_subplot(122)
        _finish_plot_layout(
            main_layout,
            self.toolbar,
            self.canvas,
            background_control_in_layout=False,
        )
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
            spec0 = self._dataset_spectrum((t0, p0, z0, 0))
        except KeyError:
            spec0 = np.zeros(100)
        colors = self._pt_colors(max(n, 1))
        (self.spec_line,) = self.ax_spec.plot(
            spec0, color=colors[0] if n else "C0"
        )
        self._update_spectral_axis()
        self.ax_spec.set_ylabel("intensity (a.u.)")
        self.ax_spec.set_title(f"pt={self._pt_selected}")
        # --- Connect signals ---
        self.t_slider.valueChanged.connect(self._on_slider)
        self.p_slider.valueChanged.connect(self._on_slider)
        self.z_slider.valueChanged.connect(self._on_slider)
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def _make_slider(self, name, lo, hi):
        label = QLabel(name)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(lo)
        slider.setMaximum(hi)
        slider.setValue(0)
        value_input = QSpinBox()
        value_input.setRange(lo, hi)
        value_input.setValue(0)
        value_input.setMaximumWidth(80)
        value_input.setToolTip(f"{name} index")
        slider.valueChanged.connect(value_input.setValue)
        value_input.valueChanged.connect(slider.setValue)
        return slider, label, value_input

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

    def _dataset_spectrum(self, key):
        """Return one raw or corrected spectrum by its DataFrame index."""
        location = self.df.index.get_loc(key)
        if not isinstance(location, (int, np.integer)):
            matches = np.flatnonzero(np.asarray(location))
            if len(matches) != 1:
                raise KeyError(key)
            location = int(matches[0])
        return _cosmic_ray_values_for_plot(self)[int(location)]

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

    def _on_fix_y_scale_toggled(self, checked):
        if checked:
            self._fixed_y_limits = self.ax_spec.get_ylim()
        else:
            self._fixed_y_limits = None
            self.ax_spec.set_autoscaley_on(True)
            self._update_spectral_axis()

    def _update_spectrum(self):
        t, p, z, _, _, _ = self._current_tpz()
        pt = self._pt_selected
        try:
            y = self._dataset_spectrum((t, p, z, pt))
            n = len(self.df.loc[t, p, z])
        except KeyError:
            return
        y = _subtract_baseline_for_plot(self, y)
        y = _smooth_for_plot(self, y)
        self.spec_line.set_ydata(y)
        self._update_spectral_axis()
        self.spec_line.set_color(self._pt_colors(n)[min(pt, n - 1)])
        self.ax_spec.set_title(f"pt={pt}")

    def _on_smoothing_changed(self):
        if not hasattr(self, "spec_line"):
            return
        self._update_spectrum()
        self.canvas.draw_idle()

    def _update_spectral_axis(self, _checked=None):
        if not hasattr(self, "spec_line"):
            return
        _set_spectral_line_axis(
            self.ax_spec,
            [self.spec_line],
            self.spectral_calibration,
            self.show_wavenumber_check.isChecked(),
        )
        if self._fixed_y_limits is not None:
            self.ax_spec.set_ylim(self._fixed_y_limits)
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()
