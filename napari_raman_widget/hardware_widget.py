"""The main HardwareWidget: a dockable napari panel for the Raman rig."""

import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import napari
import numpy as np
import xarray as xr
from qtpy.QtCore import Qt, QThread, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QApplication,
    QVBoxLayout,
    QWidget,
)

from .acquisition import autofocus_w_bkd, unload
from .calibration import (
    Calibrator,
    CoordTransformer,
    ManualImageSelector,
    StagePointPicker,
    apply_vandermonde,
    apply_vandermonde_model,
    fit_vandermonde,
    load_vandermonde_model,
    save_vandermonde_model,
)
from .core_guard import install_core_guard
from .field_help import apply_tooltips
from .hardware_defaults import (
    DEFAULTS_FILENAME,
    load_hardware_defaults,
    resolve_defaults_path,
    update_hardware_defaults,
)
from .hardware_shutdown import shutdown_core_hardware
from .log_window import LogWindow, _StdoutRedirector
from .live_spectra import LiveSpectrumWorker
from .plot_windows import (
    CalibrationPlotWindow,
    DatasetViewerWindow,
    DetectorImageWindow,
    GridScanPlotWindow,
    ReferenceSpectraWindow,
    SpectrumWindow,
)
from .position_specs import resolve_position_specs
from .qt_messages import install_qt_message_filter
from .selection import (
    add_mask_with_hole,
    automated_point_selections,
    center_manual_selections,
    grid_point_selections,
    manual_point_selections,
    refine_cell_source_points,
)
from .slack_notifications import (
    notify_exception,
    run_mda_with_notifications,
)
from .spectra import (
    save_collection_record,
    spectral_bias_from_dark_noise,
)
from .spectral_calibration_ui import (
    add_spectral_calibration_loader,
    browse_spectral_calibration,
    load_spectral_calibration,
    spectral_calibration_created,
)
from .ui_helpers import make_collapsible
from .workflows import set_up_new_seq


class HardwareWidget(QWidget):
    _raman_mda_failed = Signal(str)

    def __init__(
        self,
        viewer: napari.Viewer,
        *,
        show_ai_assistant: bool = True,
    ):
        super().__init__()
        self._raman_mda_failed.connect(self._on_raman_mda_failed)
        install_qt_message_filter()
        self.viewer = viewer
        self.core = None
        self.core_guard = None
        self._raman_mda_guard_paused = False
        self._raman_mda_finish_callback = None
        self._raman_mda_thread = None
        self._active_raman_engine = None
        self._hardware_shutdown_started = False
        self.collector = None
        self.daq = None
        self.transformer = None
        self.vandermonde = None
        self.default_engine = None
        self.calibration_ds = None
        self.calibrator = None
        self.selector = None
        self.scan_ds = None
        self.main_window = None
        self.selection_results = None
        self.mda_channel_rows = []
        self.mda_writer = None
        self.px2stage_picker = None
        self.px2stage_xy = None
        self._live_raman_thread = None
        self._live_raman_worker = None
        self._live_raman_window = None
        self._live_raman_context = None
        self._live_control_states = []
        self._collect_dark_after_live_stop = False
        self._startup_directory = Path.cwd()

        outer = QVBoxLayout()

        self.manual_link = QLabel(
            '<a href="manual" '
            'style="color:white; font-weight:bold; text-decoration:none;">'
            'Help</a>'
        )
        self.manual_link.setAlignment(Qt.AlignRight)
        self.manual_link.setOpenExternalLinks(False)
        self.manual_link.setToolTip("Open the user manual")
        self.manual_link.linkActivated.connect(self.open_user_manual)

        outer.addWidget(self.manual_link)

        # ================= LOADING SECTION =================
        loading_box = make_collapsible("Loading", expanded=True)
        self.loading_box = loading_box
        loading_layout = QVBoxLayout()

        self.hardware_config_controls = QWidget()
        hardware_config_layout = QVBoxLayout(self.hardware_config_controls)
        hardware_config_layout.setContentsMargins(0, 0, 0, 0)

        hardware_config_layout.addWidget(QLabel("Micro-Manager config (.cfg):"))
        cfg_row = QHBoxLayout()
        self.cfg_path = QLineEdit()
        self.cfg_path.setText("test3.cfg")
        self.cfg_path.setPlaceholderText("test3.cfg")
        cfg_browse = QPushButton("...")
        cfg_browse.setFixedWidth(30)
        cfg_browse.clicked.connect(self.browse_cfg)
        cfg_row.addWidget(self.cfg_path)
        cfg_row.addWidget(cfg_browse)
        hardware_config_layout.addLayout(cfg_row)

        hardware_config_layout.addWidget(QLabel("Transformer model (.json):"))
        tf_row = QHBoxLayout()
        self.tf_path = QLineEdit()
        self.tf_path.setPlaceholderText("model_2026-01-08.json")
        tf_browse = QPushButton("...")
        tf_browse.setFixedWidth(30)
        tf_browse.clicked.connect(self.browse_tf)
        tf_row.addWidget(self.tf_path)
        tf_row.addWidget(tf_browse)
        hardware_config_layout.addLayout(tf_row)

        hardware_config_layout.addWidget(QLabel("Vandermonde model (.json):"))
        vdm_row = QHBoxLayout()
        self.sel_vdm_path = QLineEdit()
        self.sel_vdm_path.setPlaceholderText("vandermonde_model.json")
        vdm_browse = QPushButton("...")
        vdm_browse.setFixedWidth(30)
        vdm_browse.clicked.connect(self.browse_vandermonde)
        vdm_row.addWidget(self.sel_vdm_path)
        vdm_row.addWidget(vdm_browse)
        hardware_config_layout.addLayout(vdm_row)
        loading_layout.addWidget(self.hardware_config_controls)

        add_spectral_calibration_loader(self, loading_layout)

        loading_layout.addWidget(QLabel("Dark noise (optional .npy):"))
        dark_noise_layout = QHBoxLayout()
        self.dark_noise_path = QLineEdit()
        self.dark_noise_path.setPlaceholderText(
            "None (raw spectra only)"
        )
        dark_noise_browse = QPushButton("...")
        dark_noise_browse.setFixedWidth(30)
        dark_noise_browse.clicked.connect(self.browse_dark_noise)
        dark_noise_layout.addWidget(self.dark_noise_path)
        dark_noise_layout.addWidget(dark_noise_browse)
        loading_layout.addLayout(dark_noise_layout)

        loading_layout.addWidget(
            QLabel("Output folder (optional, applied on connect):")
        )
        out_row = QHBoxLayout()
        self.out_path = QLineEdit()
        self.out_path.setPlaceholderText("(current directory)")
        out_browse = QPushButton("...")
        out_browse.setFixedWidth(30)
        out_browse.clicked.connect(self.browse_out)
        out_row.addWidget(self.out_path)
        out_row.addWidget(out_browse)
        loading_layout.addLayout(out_row)

        # Wavelength control
        wl_row = QHBoxLayout()
        wl_row.addWidget(QLabel("Center \u03bb (nm):"))
        self.wl_current_label = QLabel("\u2014")
        self.wl_current_label.setStyleSheet("font-weight: bold;")
        wl_row.addWidget(self.wl_current_label)
        loading_layout.addLayout(wl_row)

        wl_set_row = QHBoxLayout()
        self.wl_input = QDoubleSpinBox()
        self.wl_input.setRange(0, 2000)
        self.wl_input.setDecimals(2)
        self.wl_input.setValue(785.0)
        self.wl_input.setSuffix(" nm")
        self.wl_update_btn = QPushButton("Update \u03bb")
        self.wl_update_btn.setEnabled(False)
        self.wl_update_btn.clicked.connect(self.update_wavelength)
        wl_set_row.addWidget(self.wl_input, 2)
        wl_set_row.addWidget(self.wl_update_btn, 1)
        loading_layout.addLayout(wl_set_row)

        # Grating control
        grating_row = QHBoxLayout()
        grating_row.addWidget(QLabel("Grating:"))
        self.grating_combo = QComboBox()
        self.grating_combo.setEnabled(False)
        self.grating_update_btn = QPushButton("Update grating")
        self.grating_update_btn.setEnabled(False)
        self.grating_update_btn.clicked.connect(self.update_grating)
        grating_row.addWidget(self.grating_combo, 2)
        grating_row.addWidget(self.grating_update_btn, 1)
        loading_layout.addLayout(grating_row)

        self.connect_btn = QPushButton("Connect hardware")
        self.disconnect_btn = QPushButton("Disconnect")
        self.reload_tf_btn = QPushButton("Reload transformer")
        self.disconnect_btn.setEnabled(False)
        self.reload_tf_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.connect)
        self.disconnect_btn.clicked.connect(self.disconnect)
        self.reload_tf_btn.clicked.connect(self.reload_transformer)

        loading_layout.addWidget(self.connect_btn)
        loading_layout.addWidget(self.disconnect_btn)
        loading_layout.addWidget(self.reload_tf_btn)

        loading_box.setLayout(loading_layout)
        outer.addWidget(loading_box)

        # ================= COLLECT SPECTRUM SECTION =================
        raman_box = make_collapsible(
            "Collect spectra using points layer", expanded=False
        )
        self.raman_box = raman_box
        raman_layout = QVBoxLayout()

        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exposure (ms):"))
        self.exposure_input = QDoubleSpinBox()
        self.exposure_input.setRange(1, 1_000_000)
        self.exposure_input.setValue(1000)
        self.exposure_input.setDecimals(1)
        exp_row.addWidget(self.exposure_input)
        raman_layout.addLayout(exp_row)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("N (repeats, >=2):"))
        self.n_input = QSpinBox()
        self.n_input.setRange(2, 1000)
        self.n_input.setValue(2)
        n_row.addWidget(self.n_input)
        raman_layout.addLayout(n_row)

        self.live_collect_check = QCheckBox(
            "Live: acquire and refresh until stopped"
        )
        self.live_collect_check.toggled.connect(
            self._update_collect_live_fields
        )
        raman_layout.addWidget(self.live_collect_check)

        read_mode_row = QHBoxLayout()
        read_mode_row.addWidget(QLabel("Read mode:"))
        self.collect_read_mode_combo = QComboBox()
        self.collect_read_mode_combo.addItem(
            "Full vertical binning (FVB)", "fvb"
        )
        self.collect_read_mode_combo.addItem("Single-track", "single_track")
        self.collect_read_mode_combo.addItem("Image", "image")
        self.collect_read_mode_combo.setCurrentIndex(0)
        read_mode_row.addWidget(self.collect_read_mode_combo)
        raman_layout.addLayout(read_mode_row)

        self.collect_single_track_controls = QWidget()
        single_track_layout = QVBoxLayout(self.collect_single_track_controls)
        single_track_layout.setContentsMargins(0, 0, 0, 0)
        track_center_row = QHBoxLayout()
        track_center_row.addWidget(QLabel("Track center (row):"))
        self.collect_track_center_input = QSpinBox()
        self.collect_track_center_input.setRange(0, 65_535)
        self.collect_track_center_input.setValue(185)
        track_center_row.addWidget(self.collect_track_center_input)
        single_track_layout.addLayout(track_center_row)
        track_height_row = QHBoxLayout()
        track_height_row.addWidget(QLabel("Track height (rows):"))
        self.collect_track_height_input = QSpinBox()
        self.collect_track_height_input.setRange(2, 65_535)
        self.collect_track_height_input.setValue(140)
        track_height_row.addWidget(self.collect_track_height_input)
        single_track_layout.addLayout(track_height_row)
        raman_layout.addWidget(self.collect_single_track_controls)
        self.collect_single_track_controls.hide()
        self.collect_read_mode_combo.currentIndexChanged.connect(
            self._update_collect_read_mode_fields
        )
        raman_box.toggled.connect(self._update_collect_read_mode_fields)

        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save as (blank = don't save):"))
        self.collect_save_input = QLineEdit()
        self.collect_save_input.setPlaceholderText(
            "base filename (one .zarr dataset)"
        )
        save_row.addWidget(self.collect_save_input)
        raman_layout.addLayout(save_row)

        self.collect_btn = QPushButton("Collect spectra at selected point")
        self.collect_btn.clicked.connect(self.collect_raman)
        raman_layout.addWidget(self.collect_btn)

        self.collect_dark_noise_btn = QPushButton("Collect dark noise")
        self.collect_dark_noise_btn.clicked.connect(self.collect_dark_noise)
        raman_layout.addWidget(self.collect_dark_noise_btn)

        raman_box.setLayout(raman_layout)
        outer.addWidget(raman_box)

        # ================= LASER AIMING CALIBRATION SECTION =================
        calib_box = make_collapsible("Laser aiming calibration", expanded=False)
        self.calib_box = calib_box
        calib_layout = QVBoxLayout()

        cal_n_row = QHBoxLayout()
        cal_n_row.addWidget(QLabel("N (repeats):"))
        self.cal_n_input = QSpinBox()
        self.cal_n_input.setRange(1, 1000)
        self.cal_n_input.setValue(20)
        cal_n_row.addWidget(self.cal_n_input)
        calib_layout.addLayout(cal_n_row)

        cal_exp_row = QHBoxLayout()
        cal_exp_row.addWidget(QLabel("Exposure (ms):"))
        self.cal_exp_input = QDoubleSpinBox()
        self.cal_exp_input.setRange(1, 1_000_000)
        self.cal_exp_input.setValue(1000)
        self.cal_exp_input.setDecimals(1)
        cal_exp_row.addWidget(self.cal_exp_input)
        calib_layout.addLayout(cal_exp_row)

        cal_volts_row = QHBoxLayout()
        cal_volts_row.addWidget(QLabel("Max volts:"))
        self.cal_volts_input = QDoubleSpinBox()
        self.cal_volts_input.setRange(0.01, 10.0)
        self.cal_volts_input.setValue(1.8)
        self.cal_volts_input.setDecimals(2)
        self.cal_volts_input.setSingleStep(0.1)
        cal_volts_row.addWidget(self.cal_volts_input)
        calib_layout.addLayout(cal_volts_row)

        cal_grid_row = QHBoxLayout()
        cal_grid_row.addWidget(QLabel("Grid size:"))
        self.cal_grid_input = QSpinBox()
        self.cal_grid_input.setRange(2, 100)
        self.cal_grid_input.setValue(10)
        cal_grid_row.addWidget(self.cal_grid_input)
        calib_layout.addLayout(cal_grid_row)

        cal_thres_row = QHBoxLayout()
        cal_thres_row.addWidget(QLabel("Threshold:"))
        self.cal_thres_input = QDoubleSpinBox()
        self.cal_thres_input.setRange(0.0, 100.0)
        self.cal_thres_input.setValue(1.0)
        self.cal_thres_input.setDecimals(2)
        self.cal_thres_input.setSingleStep(0.1)
        cal_thres_row.addWidget(self.cal_thres_input)
        calib_layout.addLayout(cal_thres_row)

        self.calibrate_btn = QPushButton("Run calibration")
        self.calibrate_btn.clicked.connect(self.run_calibration)
        calib_layout.addWidget(self.calibrate_btn)

        # --- recalibration (shown when checked) ---
        self.recal_check = QCheckBox("Recalibration")
        self.recal_check.setChecked(False)
        self.recal_check.toggled.connect(self._toggle_recal_fields)
        calib_layout.addWidget(self.recal_check)

        self._recal_help = QLabel(
            "Opens manual selector on the last calibration dataset.\n"
            "Click points, Enter to advance, Backspace to go back,\n"
            "R to reset, N to mark as NaN. Close window when done,\n"
            "then click Save to write the new model."
        )
        self._recal_help.setWordWrap(True)
        calib_layout.addWidget(self._recal_help)
        model_name_row = QHBoxLayout()
        self._recal_model_name_label = QLabel("Model name:")
        model_name_row.addWidget(self._recal_model_name_label)
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("model_2026-01-08")
        model_name_row.addWidget(self.model_name_input)
        calib_layout.addLayout(model_name_row)
        self.open_selector_btn = QPushButton("Open manual selector")
        self.open_selector_btn.clicked.connect(self.open_selector)
        calib_layout.addWidget(self.open_selector_btn)
        self.save_model_btn = QPushButton("Save recalibrated model")
        self.save_model_btn.clicked.connect(self.save_recalibration)
        calib_layout.addWidget(self.save_model_btn)

        # collect recal widgets so they can be hidden as a group
        self._recal_widgets = [
            self._recal_help,
            self._recal_model_name_label, self.model_name_input,
            self.open_selector_btn, self.save_model_btn,
        ]
        self._toggle_recal_fields(False)   # hidden until checked

        calib_box.setLayout(calib_layout)
        outer.addWidget(calib_box)

        # ================= COLLECT REFERENCE SPECTRA SECTION =================
        ref_box = make_collapsible("Axial background scan", expanded=False)
        ref_layout = QVBoxLayout()

        ref_name_row = QHBoxLayout()
        ref_name_row.addWidget(QLabel("Name:"))
        self.ref_name_input = QLineEdit()
        self.ref_name_input.setPlaceholderText("testing")
        ref_name_row.addWidget(self.ref_name_input)
        ref_layout.addLayout(ref_name_row)

        ref_exp_row = QHBoxLayout()
        ref_exp_row.addWidget(QLabel("Exposure (ms):"))
        self.ref_exp_input = QDoubleSpinBox()
        self.ref_exp_input.setRange(1, 1_000_000)
        self.ref_exp_input.setValue(1000)
        self.ref_exp_input.setDecimals(1)
        ref_exp_row.addWidget(self.ref_exp_input)
        ref_layout.addLayout(ref_exp_row)

        ref_n_row = QHBoxLayout()
        ref_n_row.addWidget(QLabel("N (spectra per z):"))
        self.ref_n_input = QSpinBox()
        self.ref_n_input.setRange(1, 1000)
        self.ref_n_input.setValue(5)
        ref_n_row.addWidget(self.ref_n_input)
        ref_layout.addLayout(ref_n_row)

        ref_range_row = QHBoxLayout()
        ref_range_row.addWidget(QLabel("Search range (um):"))
        self.ref_range_input = QDoubleSpinBox()
        self.ref_range_input.setRange(0.1, 1000.0)
        self.ref_range_input.setValue(10)
        self.ref_range_input.setDecimals(1)
        ref_range_row.addWidget(self.ref_range_input)
        ref_layout.addLayout(ref_range_row)

        ref_pts_row = QHBoxLayout()
        ref_pts_row.addWidget(QLabel("Search pts:"))
        self.ref_pts_input = QSpinBox()
        self.ref_pts_input.setRange(2, 500)
        self.ref_pts_input.setValue(20)
        ref_pts_row.addWidget(self.ref_pts_input)
        ref_layout.addLayout(ref_pts_row)

        self.ref_collect_btn = QPushButton("Collect reference spectra")
        self.ref_collect_btn.clicked.connect(self.collect_reference)
        ref_layout.addWidget(self.ref_collect_btn)

        ref_box.setLayout(ref_layout)
        outer.addWidget(ref_box)

        # ================= SPATIAL MAPPING SECTION =================
        scan_box = make_collapsible("Spatial mapping", expanded=False)
        scan_layout = QVBoxLayout()

        scan_layout.addWidget(QLabel(
            "Draw a rectangle in a Shapes layer first, then run."
        ))

        scan_name_row = QHBoxLayout()
        scan_name_row.addWidget(QLabel("File name:"))
        self.scan_name_input = QLineEdit()
        self.scan_name_input.setPlaceholderText("scan_label")
        scan_name_row.addWidget(self.scan_name_input)
        scan_layout.addLayout(scan_name_row)

        scan_exp_row = QHBoxLayout()
        scan_exp_row.addWidget(QLabel("Raman exposure (ms):"))
        self.scan_exp_input = QDoubleSpinBox()
        self.scan_exp_input.setRange(1, 1_000_000)
        self.scan_exp_input.setValue(1000)
        self.scan_exp_input.setDecimals(1)
        scan_exp_row.addWidget(self.scan_exp_input)
        scan_layout.addLayout(scan_exp_row)

        scan_n_row = QHBoxLayout()
        scan_n_row.addWidget(QLabel("N (grid side):"))
        self.scan_n_input = QSpinBox()
        self.scan_n_input.setRange(2, 500)
        self.scan_n_input.setValue(20)
        scan_n_row.addWidget(self.scan_n_input)
        scan_layout.addLayout(scan_n_row)

        scan_z_row = QHBoxLayout()
        scan_z_row.addWidget(QLabel("Z offset (um):"))
        self.scan_z_input = QDoubleSpinBox()
        self.scan_z_input.setRange(-1000, 1000)
        self.scan_z_input.setValue(4)
        self.scan_z_input.setDecimals(2)
        self.scan_z_input.setSingleStep(0.5)
        scan_z_row.addWidget(self.scan_z_input)
        scan_layout.addLayout(scan_z_row)

        # Z-scan option
        self.scan_zscan_check = QCheckBox("Z-scan (multi-plane Raman)")
        self.scan_zscan_check.setChecked(False)
        self.scan_zscan_check.toggled.connect(self._toggle_zscan_fields)
        scan_layout.addWidget(self.scan_zscan_check)

        self.scan_zrange_row = QHBoxLayout()
        self._zscan_range_label = QLabel("Z range (+/- um):")
        self.scan_zrange_row.addWidget(self._zscan_range_label)
        self.scan_zrange_input = QDoubleSpinBox()
        self.scan_zrange_input.setRange(0.1, 1000.0)
        self.scan_zrange_input.setValue(5.0)
        self.scan_zrange_input.setDecimals(1)
        self.scan_zrange_input.setSingleStep(0.5)
        self.scan_zrange_row.addWidget(self.scan_zrange_input)
        scan_layout.addLayout(self.scan_zrange_row)

        self.scan_zsteps_row = QHBoxLayout()
        self._zscan_steps_label = QLabel("Z steps:")
        self.scan_zsteps_row.addWidget(self._zscan_steps_label)
        self.scan_zsteps_input = QSpinBox()
        self.scan_zsteps_input.setRange(2, 500)
        self.scan_zsteps_input.setValue(10)
        self.scan_zsteps_row.addWidget(self.scan_zsteps_input)
        scan_layout.addLayout(self.scan_zsteps_row)

        # Hide z-scan fields initially
        self._zscan_range_label.setVisible(False)
        self.scan_zrange_input.setVisible(False)
        self._zscan_steps_label.setVisible(False)
        self.scan_zsteps_input.setVisible(False)

        scan_layout.addWidget(QLabel(
            "Extra channels (BF is always captured before & after):"
        ))
        self.channel_rows_layout = QVBoxLayout()
        scan_layout.addLayout(self.channel_rows_layout)
        self.channel_rows = []

        self.add_channel_btn = QPushButton("+ Add channel")
        self.add_channel_btn.clicked.connect(self._add_channel_row)
        scan_layout.addWidget(self.add_channel_btn)

        self.scan_btn = QPushButton("Run grid scan")
        self.scan_btn.clicked.connect(self.run_grid_scan)
        scan_layout.addWidget(self.scan_btn)

        scan_box.setLayout(scan_layout)
        outer.addWidget(scan_box)

        # ================= GENERATE STAGE GRID SECTION =================
        grid_box = make_collapsible("Generate stage grid", expanded=False)
        self.grid_box = grid_box
        grid_layout = QVBoxLayout()

        grid_layout.addWidget(QLabel(
            "Creates a grid of stage positions around the CURRENT stage XY,\n"
            "each carrying the same single fixed point (non-batch)."
        ))

        grid_af_row = QHBoxLayout()
        grid_af_row.addWidget(QLabel("Autofocus object:"))
        self.grid_af_combo = QComboBox()
        self.grid_af_combo.addItems(
            ["None", "laser", "software", "quartz", "glass", "cell"]
        )
        grid_af_row.addWidget(self.grid_af_combo)
        grid_layout.addLayout(grid_af_row)
        # let the grid combo also drive the MDA autofocus-field visibility
        self.grid_af_combo.currentTextChanged.connect(self._toggle_autofocus_fields)

        # Fixed point in image (pixel) coordinates -- same point at every FOV.
        fovx_row = QHBoxLayout()
        fovx_row.addWidget(QLabel("FOV x (px):"))
        self.grid_fovx_input = QSpinBox()
        self.grid_fovx_input.setRange(0, 100000)
        self.grid_fovx_input.setValue(740)
        fovx_row.addWidget(self.grid_fovx_input)
        grid_layout.addLayout(fovx_row)

        fovy_row = QHBoxLayout()
        fovy_row.addWidget(QLabel("FOV y (px):"))
        self.grid_fovy_input = QSpinBox()
        self.grid_fovy_input.setRange(0, 100000)
        self.grid_fovy_input.setValue(540)
        fovy_row.addWidget(self.grid_fovy_input)
        grid_layout.addLayout(fovy_row)

        # Stage grid extent / spacing (real stage units, +/- range about current).
        xr_row = QHBoxLayout()
        xr_row.addWidget(QLabel("X range (+/- um):"))
        self.grid_xrange_input = QDoubleSpinBox()
        self.grid_xrange_input.setRange(0.0, 1_000_000)
        self.grid_xrange_input.setValue(100.0)
        self.grid_xrange_input.setDecimals(2)
        self.grid_xrange_input.setSingleStep(10.0)
        xr_row.addWidget(self.grid_xrange_input)
        grid_layout.addLayout(xr_row)

        yr_row = QHBoxLayout()
        yr_row.addWidget(QLabel("Y range (+/- um):"))
        self.grid_yrange_input = QDoubleSpinBox()
        self.grid_yrange_input.setRange(0.0, 1_000_000)
        self.grid_yrange_input.setValue(100.0)
        self.grid_yrange_input.setDecimals(2)
        self.grid_yrange_input.setSingleStep(10.0)
        yr_row.addWidget(self.grid_yrange_input)
        grid_layout.addLayout(yr_row)

        xs_row = QHBoxLayout()
        xs_row.addWidget(QLabel("X step (um):"))
        self.grid_xstep_input = QDoubleSpinBox()
        self.grid_xstep_input.setRange(0.01, 1_000_000)
        self.grid_xstep_input.setValue(50.0)
        self.grid_xstep_input.setDecimals(2)
        self.grid_xstep_input.setSingleStep(5.0)
        xs_row.addWidget(self.grid_xstep_input)
        grid_layout.addLayout(xs_row)

        ys_row = QHBoxLayout()
        ys_row.addWidget(QLabel("Y step (um):"))
        self.grid_ystep_input = QDoubleSpinBox()
        self.grid_ystep_input.setRange(0.01, 1_000_000)
        self.grid_ystep_input.setValue(50.0)
        self.grid_ystep_input.setDecimals(2)
        self.grid_ystep_input.setSingleStep(5.0)
        ys_row.addWidget(self.grid_ystep_input)
        grid_layout.addLayout(ys_row)

        # Number of identical points placed per position (>=2 for the DAQ,
        # which needs at least 2 samples per channel).
        reps_row = QHBoxLayout()
        reps_row.addWidget(QLabel("Repeats (points per position, >=2):"))
        self.grid_repeats_input = QSpinBox()
        self.grid_repeats_input.setRange(2, 1000)
        self.grid_repeats_input.setValue(2)
        reps_row.addWidget(self.grid_repeats_input)
        grid_layout.addLayout(reps_row)

        # Skip the slow per-position BF pre-scan; use a zero-memory blank
        # placeholder layer to establish dims instead.
        self.grid_blank_check = QCheckBox("Skip BF pre-scan (use blank images)")
        self.grid_blank_check.setChecked(True)
        grid_layout.addWidget(self.grid_blank_check)
        self.run_grid_sel_btn = QPushButton("Generate grid")
        self.run_grid_sel_btn.clicked.connect(self.run_grid_selection)
        grid_layout.addWidget(self.run_grid_sel_btn)

        grid_box.setLayout(grid_layout)
        outer.addWidget(grid_box)

        # ================= AUTOMATED CELL SELECTION SECTION =================
        self.sel_box = make_collapsible("Automated cell selection", expanded=False)
        sel_box = self.sel_box
        sel_layout = QVBoxLayout()

        sel_layout.addWidget(QLabel("Mask region (shared by both buttons):"))

        cy_row = QHBoxLayout()
        cy_row.addWidget(QLabel("Center Y:"))
        self.sel_cy_input = QSpinBox()
        self.sel_cy_input.setRange(0, 100000)
        self.sel_cy_input.setValue(540)
        cy_row.addWidget(self.sel_cy_input)
        sel_layout.addLayout(cy_row)

        cx_row = QHBoxLayout()
        cx_row.addWidget(QLabel("Center X:"))
        self.sel_cx_input = QSpinBox()
        self.sel_cx_input.setRange(0, 100000)
        self.sel_cx_input.setValue(740)
        cx_row.addWidget(self.sel_cx_input)
        sel_layout.addLayout(cx_row)

        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("Radius:"))
        self.sel_r_input = QSpinBox()
        self.sel_r_input.setRange(1, 100000)
        self.sel_r_input.setValue(100)
        r_row.addWidget(self.sel_r_input)
        sel_layout.addLayout(r_row)

        mask_btn_row = QHBoxLayout()
        self.add_mask_btn = QPushButton("Add mask")
        self.add_mask_btn.clicked.connect(self.add_mask)
        self.click_center_btn = QPushButton("Click to center")
        self.click_center_btn.setCheckable(True)
        self.click_center_btn.toggled.connect(self._toggle_click_to_center)
        mask_btn_row.addWidget(self.add_mask_btn)
        mask_btn_row.addWidget(self.click_center_btn)
        sel_layout.addLayout(mask_btn_row)

        sel_layout.addWidget(QLabel("Automated point selection:"))

        af_row = QHBoxLayout()
        af_row.addWidget(QLabel("Autofocus object:"))
        self.sel_af_combo = QComboBox()
        self.sel_af_combo.addItems(
            ["laser", "software", "quartz", "glass", "cell", "None"]
        )
        af_row.addWidget(self.sel_af_combo)
        sel_layout.addLayout(af_row)

        npf_row = QHBoxLayout()
        npf_row.addWidget(QLabel("N per FOV:"))
        self.sel_npf_input = QSpinBox()
        self.sel_npf_input.setRange(1, 1000)
        self.sel_npf_input.setValue(6)
        npf_row.addWidget(self.sel_npf_input)
        sel_layout.addLayout(npf_row)

        # Center-cell mode: split each FOV into one new stage position per
        # detected cell, each shifted so that cell sits exactly at center.
        self.sel_center_cell_check = QCheckBox(
            "Center cell (split FOV into one position per cell, centered)"
        )
        self.sel_center_cell_check.setChecked(False)
        sel_layout.addWidget(self.sel_center_cell_check)

        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Aiming pattern:"))
        self.sel_shape_combo = QComboBox()
        self.sel_shape_combo.addItems(["Square", "Circle"])
        shape_row.addWidget(self.sel_shape_combo)
        sel_layout.addLayout(shape_row)

        sq_size_row = QHBoxLayout()
        sq_size_row.addWidget(QLabel("Pattern size (px):"))
        self.sel_sqsize_input = QDoubleSpinBox()
        self.sel_sqsize_input.setRange(0.0, 10000.0)
        self.sel_sqsize_input.setValue(30)
        self.sel_sqsize_input.setDecimals(1)
        self.sel_sqsize_input.setSingleStep(1.0)
        sq_size_row.addWidget(self.sel_sqsize_input)
        sel_layout.addLayout(sq_size_row)

        sq_n_row = QHBoxLayout()
        sq_n_row.addWidget(QLabel("N_x (subpoints):"))
        self.sel_sqn_input = QSpinBox()
        self.sel_sqn_input.setRange(1, 100)
        self.sel_sqn_input.setValue(1)
        sq_n_row.addWidget(self.sel_sqn_input)
        sel_layout.addLayout(sq_n_row)


        bkd_row = QHBoxLayout()
        bkd_row.addWidget(QLabel("Background distance (px):"))
        self.sel_bkd_input = QDoubleSpinBox()
        self.sel_bkd_input.setRange(0.0, 1_000_000.0)
        self.sel_bkd_input.setValue(80)
        self.sel_bkd_input.setDecimals(1)
        bkd_row.addWidget(self.sel_bkd_input)
        sel_layout.addLayout(bkd_row)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Integrated batch collection:"))
        self.sel_batch_combo = QComboBox()
        self.sel_batch_combo.addItems(["False", "True"])
        batch_row.addWidget(self.sel_batch_combo)
        sel_layout.addLayout(batch_row)

        sel_cp_row = QHBoxLayout()
        sel_cp_row.addWidget(QLabel("Cellpose model:"))
        self.sel_cellpose_combo = QComboBox()
        try:
            from cellpose import models as _cp_models
            _sel_model_names = list(_cp_models.MODEL_NAMES)
        except Exception:
            _sel_model_names = ["cyto2"]
        self.sel_cellpose_combo.addItems(_sel_model_names)
        if "cyto2" in _sel_model_names:
            self.sel_cellpose_combo.setCurrentText("cyto2")
        sel_cp_row.addWidget(self.sel_cellpose_combo)
        sel_layout.addLayout(sel_cp_row)

        self.run_selection_btn = QPushButton("Run automated selection")
        self.run_selection_btn.clicked.connect(self.run_automated_selection)
        sel_layout.addWidget(self.run_selection_btn)

        refine_scale_row = QHBoxLayout()
        refine_scale_row.addWidget(QLabel("Refinement scale:"))
        self.refine_scale_input = QSpinBox()
        self.refine_scale_input.setRange(1, 16)
        self.refine_scale_input.setValue(4)
        refine_scale_row.addWidget(self.refine_scale_input)
        sel_layout.addLayout(refine_scale_row)

        self.refine_cell_points_btn = QPushButton(
            "Refine cell points to centers"
        )
        self.refine_cell_points_btn.clicked.connect(
            self.refine_selected_cell_points
        )
        sel_layout.addWidget(self.refine_cell_points_btn)

        manual_row = QHBoxLayout()
        self.run_manual_btn = QPushButton("Manual selection")
        self.run_manual_btn.clicked.connect(self.run_manual_selection)
        self.center_manual_btn = QPushButton("Center clicked cells")
        self.center_manual_btn.clicked.connect(self.center_manual_cells)
        manual_row.addWidget(self.run_manual_btn)
        manual_row.addWidget(self.center_manual_btn)
        sel_layout.addLayout(manual_row)

        sel_box.setLayout(sel_layout)
        outer.addWidget(sel_box)

        # ================= RUN RAMAN MDA SECTION =================
        mda_box = make_collapsible("Run Raman MDA", expanded=False)
        mda_layout = QVBoxLayout()

        mda_layout.addWidget(QLabel(
            "Run automated cell selection first; this uses its sources "
            "& autofocus_p."
        ))

        mda_dir_row = QHBoxLayout()
        mda_dir_row.addWidget(QLabel("Writer output dir:"))
        self.mda_dir_input = QLineEdit()
        self.mda_dir_input.setText("data/run")
        mda_dir_row.addWidget(self.mda_dir_input)
        mda_layout.addLayout(mda_dir_row)

        afp_row = QHBoxLayout()
        afp_row.addWidget(QLabel("Autofocus positions:"))
        self.mda_afp_input = QLineEdit()
        self.mda_afp_input.setText("")
        self.mda_afp_input.setPlaceholderText(
            "N=every Nth; commas=exact; blank=selection"
        )
        afp_row.addWidget(self.mda_afp_input)
        mda_layout.addLayout(afp_row)

        imgp_row = QHBoxLayout()
        imgp_row.addWidget(QLabel("Imaging positions:"))
        self.mda_imgp_input = QLineEdit()
        self.mda_imgp_input.setText("")
        self.mda_imgp_input.setPlaceholderText(
            "N=every Nth; commas=exact; blank=selection"
        )
        imgp_row.addWidget(self.mda_imgp_input)
        mda_layout.addLayout(imgp_row)

        raman_off_row = QHBoxLayout()
        raman_off_row.addWidget(QLabel("Raman glass offset (um):"))
        self.mda_raman_off_input = QDoubleSpinBox()
        self.mda_raman_off_input.setRange(-1000, 1000)
        self.mda_raman_off_input.setValue(5.0)
        self.mda_raman_off_input.setDecimals(2)
        self.mda_raman_off_input.setSingleStep(0.1)
        raman_off_row.addWidget(self.mda_raman_off_input)
        mda_layout.addLayout(raman_off_row)

        af_range_row = QHBoxLayout()
        self._af_range_label = QLabel("Autofocus search range:")
        af_range_row.addWidget(self._af_range_label)
        self.mda_af_range_input = QDoubleSpinBox()
        self.mda_af_range_input.setRange(0.1, 1000)
        self.mda_af_range_input.setValue(6)
        self.mda_af_range_input.setDecimals(2)
        self.mda_af_range_input.setSingleStep(0.5)
        af_range_row.addWidget(self.mda_af_range_input)
        mda_layout.addLayout(af_range_row)

        # Coarse autofocus search points (all autofocus objects)
        search_pts_row = QHBoxLayout()
        self._search_pts_label = QLabel("Autofocus search pts:")
        search_pts_row.addWidget(self._search_pts_label)
        self.mda_search_pts_input = QSpinBox()
        self.mda_search_pts_input.setRange(2, 500)
        self.mda_search_pts_input.setValue(8)
        search_pts_row.addWidget(self.mda_search_pts_input)
        mda_layout.addLayout(search_pts_row)

        # Laser autofocus FINE scan range
        fine_range_row = QHBoxLayout()
        self._fine_range_label = QLabel("Laser fine search range (+/- um):")
        fine_range_row.addWidget(self._fine_range_label)
        self.mda_fine_range_input = QDoubleSpinBox()
        self.mda_fine_range_input.setRange(0.1, 1000)
        self.mda_fine_range_input.setValue(1.5)
        self.mda_fine_range_input.setDecimals(2)
        self.mda_fine_range_input.setSingleStep(0.5)
        fine_range_row.addWidget(self.mda_fine_range_input)
        mda_layout.addLayout(fine_range_row)

        # Laser autofocus FINE scan points
        fine_pts_row = QHBoxLayout()
        self._fine_pts_label = QLabel("Laser fine search pts:")
        fine_pts_row.addWidget(self._fine_pts_label)
        self.mda_fine_pts_input = QSpinBox()
        self.mda_fine_pts_input.setRange(2, 500)
        self.mda_fine_pts_input.setValue(8)
        fine_pts_row.addWidget(self.mda_fine_pts_input)
        mda_layout.addLayout(fine_pts_row)

        # Show/hide autofocus fields based on the selected autofocus object.
        self.sel_af_combo.currentTextChanged.connect(self._toggle_autofocus_fields)
        self._toggle_autofocus_fields(self.sel_af_combo.currentText())

        # Segment-and-track toggle (independent of autofocus)
        self.mda_seg_track_check = QCheckBox(
            "Segment and track (update aiming)"
        )
        self.mda_seg_track_check.setChecked(False)
        self.mda_seg_track_check.toggled.connect(self._toggle_seg_track_fields)
        mda_layout.addWidget(self.mda_seg_track_check)
        # Seg-track options (shown only when the box is checked)
        seg_ch_row = QHBoxLayout()
        self._seg_ch_label = QLabel("Segment channel:")
        seg_ch_row.addWidget(self._seg_ch_label)
        self.mda_seg_ch_combo = QComboBox()
        self.mda_seg_ch_combo.addItem("BF")
        seg_ch_row.addWidget(self.mda_seg_ch_combo)
        mda_layout.addLayout(seg_ch_row)
        seg_scale_row = QHBoxLayout()
        self._seg_scale_label = QLabel("Image rescale factor:")
        seg_scale_row.addWidget(self._seg_scale_label)
        self.mda_seg_scale_input = QDoubleSpinBox()
        self.mda_seg_scale_input.setRange(1.0, 100.0)
        self.mda_seg_scale_input.setValue(2.0)
        self.mda_seg_scale_input.setDecimals(1)
        self.mda_seg_scale_input.setSingleStep(0.5)
        seg_scale_row.addWidget(self.mda_seg_scale_input)
        mda_layout.addLayout(seg_scale_row)
        
        # Cellpose model dropdown
        seg_model_row = QHBoxLayout()
        self._seg_model_label = QLabel("Cellpose model:")
        seg_model_row.addWidget(self._seg_model_label)
        self.mda_seg_model_combo = QComboBox()
        try:
            from cellpose import models as _cp_models
            model_names = list(_cp_models.MODEL_NAMES)
        except Exception:
            model_names = ["cyto2"]
        self.mda_seg_model_combo.addItems(model_names)
        if "cyto2" in model_names:
            self.mda_seg_model_combo.setCurrentText("cyto2")
        seg_model_row.addWidget(self.mda_seg_model_combo)
        mda_layout.addLayout(seg_model_row)
        # Crop-around-mask dropdown
        seg_crop_row = QHBoxLayout()
        self._seg_crop_label = QLabel("Crop image around mask:")
        seg_crop_row.addWidget(self._seg_crop_label)
        self.mda_seg_crop_combo = QComboBox()
        self.mda_seg_crop_combo.addItems(["True", "False"])
        seg_crop_row.addWidget(self.mda_seg_crop_combo)
        mda_layout.addLayout(seg_crop_row)
        # Tracking config file
        seg_track_cfg_row = QHBoxLayout()
        self._seg_track_cfg_label = QLabel("Tracking config (.json):")
        seg_track_cfg_row.addWidget(self._seg_track_cfg_label)
        self.mda_track_cfg_input = QLineEdit()
        self.mda_track_cfg_input.setText("particle_config.json")
        self.mda_track_cfg_input.setPlaceholderText("particle_config.json")
        self._seg_track_cfg_browse = QPushButton("...")
        self._seg_track_cfg_browse.setFixedWidth(30)
        self._seg_track_cfg_browse.clicked.connect(self.browse_tracking_cfg)
        seg_track_cfg_row.addWidget(self.mda_track_cfg_input)
        seg_track_cfg_row.addWidget(self._seg_track_cfg_browse)
        mda_layout.addLayout(seg_track_cfg_row)
        # hidden until seg-track is checked
        self._toggle_seg_track_fields(False)

        mda_exp_row = QHBoxLayout()
        mda_exp_row.addWidget(QLabel("Exposure per cell (ms):"))
        self.mda_exp_input = QDoubleSpinBox()
        self.mda_exp_input.setRange(1, 1_000_000)
        self.mda_exp_input.setValue(1000)
        self.mda_exp_input.setDecimals(1)
        mda_exp_row.addWidget(self.mda_exp_input)
        mda_layout.addLayout(mda_exp_row)

        loops_row = QHBoxLayout()
        loops_row.addWidget(QLabel("Loops (time points):"))
        self.mda_loops_input = QSpinBox()
        self.mda_loops_input.setRange(1, 1_000_000)
        self.mda_loops_input.setValue(100)
        loops_row.addWidget(self.mda_loops_input)
        mda_layout.addLayout(loops_row)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Interval (s):"))
        self.mda_interval_input = QDoubleSpinBox()
        self.mda_interval_input.setRange(0.0, 1_000_000)
        self.mda_interval_input.setValue(600)
        self.mda_interval_input.setDecimals(1)
        interval_row.addWidget(self.mda_interval_input)
        mda_layout.addLayout(interval_row)

        # Refocus / re-segment cadence: run autofocus AND segment-and-track
        # only every Nth timepoint (1 = every timepoint).
        refocus_row = QHBoxLayout()
        refocus_row.addWidget(QLabel("Refocus & segment every (timepoints):"))
        self.mda_refocus_input = QSpinBox()
        self.mda_refocus_input.setRange(1, 1_000_000)
        self.mda_refocus_input.setValue(1)
        refocus_row.addWidget(self.mda_refocus_input)
        mda_layout.addLayout(refocus_row)

        zrel_row = QHBoxLayout()
        zrel_row.addWidget(QLabel("Z relative (comma-sep um):"))
        self.mda_zrel_input = QLineEdit()
        self.mda_zrel_input.setText("0, 4")
        self.mda_zrel_input.setPlaceholderText("e.g. 0, 3.33")
        zrel_row.addWidget(self.mda_zrel_input)
        mda_layout.addLayout(zrel_row)

        rz_row = QHBoxLayout()
        rz_row.addWidget(QLabel("Raman z indices:"))
        self.mda_rz_input = QLineEdit()
        self.mda_rz_input.setText("0")
        self.mda_rz_input.setPlaceholderText("e.g. 0, 1")
        rz_row.addWidget(self.mda_rz_input)
        mda_layout.addLayout(rz_row)

        mda_layout.addWidget(QLabel(
            "Extra fluorescence channels (added to the sequence):"
        ))
        self.mda_channel_rows_layout = QVBoxLayout()
        mda_layout.addLayout(self.mda_channel_rows_layout)

        self.mda_add_channel_btn = QPushButton("+ Add channel")
        self.mda_add_channel_btn.clicked.connect(
            lambda: self._add_mda_channel_row()
        )
        mda_layout.addWidget(self.mda_add_channel_btn)

        mda_btns_row = QHBoxLayout()
        self.run_mda_btn = QPushButton("Run Raman MDA")
        self.run_mda_btn.clicked.connect(self.run_raman_mda)
        self.stop_mda_btn = QPushButton("Stop")
        self.stop_mda_btn.clicked.connect(self.stop_raman_mda)
        mda_btns_row.addWidget(self.run_mda_btn, 3)
        mda_btns_row.addWidget(self.stop_mda_btn, 1)

        mda_layout.addLayout(mda_btns_row)
 
        # --- separator ---
        sep = QLabel("-" * 45)
        sep.setAlignment(Qt.AlignCenter)
        mda_layout.addWidget(sep)
 
        self.gen_dataset_btn = QPushButton("Generate dataset")
        self.gen_dataset_btn.clicked.connect(self.generate_dataset)
        mda_layout.addWidget(self.gen_dataset_btn)
 
        # --- pixel-to-stage calibration ---
        self.px2stage_check = QCheckBox("Pixel-to-stage calibration")
        self.px2stage_check.setChecked(False)
        self.px2stage_check.toggled.connect(self._toggle_px2stage_fields)
        mda_layout.addWidget(self.px2stage_check)

        self._px2s_help = QLabel(
            "Pick the same feature in each grid position, then fit a\n"
            "pixel->stage Vandermonde model. Uses stage XY from the\n"
            "dataset's useq_sequence attribute."
        )
        self._px2s_help.setWordWrap(True)
        mda_layout.addWidget(self._px2s_help)

        px2s_ds_row = QHBoxLayout()
        self._px2s_ds_label = QLabel("Dataset (.zarr):")
        px2s_ds_row.addWidget(self._px2s_ds_label)
        self.px2stage_ds_path = QLineEdit()
        self.px2stage_ds_path.setPlaceholderText("data/dataset/ds_run_7.zarr")
        self._px2s_ds_browse = QPushButton("...")
        self._px2s_ds_browse.setFixedWidth(30)
        self._px2s_ds_browse.clicked.connect(self.browse_px2stage_ds)
        px2s_ds_row.addWidget(self.px2stage_ds_path)
        px2s_ds_row.addWidget(self._px2s_ds_browse)
        mda_layout.addLayout(px2s_ds_row)

        px2s_deg_row = QHBoxLayout()
        self._px2s_deg_label = QLabel("Vandermonde degree:")
        px2s_deg_row.addWidget(self._px2s_deg_label)
        self.px2stage_degree_input = QSpinBox()
        self.px2stage_degree_input.setRange(1, 5)
        self.px2stage_degree_input.setValue(1)
        px2s_deg_row.addWidget(self.px2stage_degree_input)
        mda_layout.addLayout(px2s_deg_row)

        self.px2stage_pick_btn = QPushButton("Pick points...")
        self.px2stage_pick_btn.clicked.connect(self.open_pixel_stage_picker)
        mda_layout.addWidget(self.px2stage_pick_btn)

        px2s_name_row = QHBoxLayout()
        self._px2s_name_label = QLabel("Model file:")
        px2s_name_row.addWidget(self._px2s_name_label)
        self.px2stage_name_input = QLineEdit()
        self.px2stage_name_input.setText("vandermonde_model.json")
        px2s_name_row.addWidget(self.px2stage_name_input)
        mda_layout.addLayout(px2s_name_row)

        self.px2stage_save_btn = QPushButton("Fit && save model")
        self.px2stage_save_btn.clicked.connect(self.fit_and_save_pixel_stage)
        mda_layout.addWidget(self.px2stage_save_btn)

        # collect the px2stage widgets so they can be hidden as a group
        self._px2stage_widgets = [
            self._px2s_help,
            self._px2s_ds_label, self.px2stage_ds_path, self._px2s_ds_browse,
            self._px2s_deg_label, self.px2stage_degree_input,
            self.px2stage_pick_btn,
            self._px2s_name_label, self.px2stage_name_input,
            self.px2stage_save_btn,
        ]
        # hidden until the box is checked
        self._toggle_px2stage_fields(False)
 
        mda_box.setLayout(mda_layout)
        outer.addWidget(mda_box)

        # make_collapsible re-shows ALL descendants on expand, which clobbers
        # our conditional field hiding -- re-apply it after any box expands.
        for _box in (calib_box, scan_box, self.sel_box, mda_box):
            _box.toggled.connect(lambda checked: self._reapply_toggles())

        self.chat_panel = None
        if show_ai_assistant:
            from .chat_panel import ChatPanel

            self.chat_panel = ChatPanel(self)
            outer.addWidget(self.chat_panel)
        
        outer.addStretch()

        # # ================= LIVE STAGE POSITION =================
        # self.pos_label = QLabel("Stage:  X --  Y --")
        # self.pos_label.setStyleSheet(
        #     "QLabel { border-top: 1px solid palette(mid); padding: 4px; "
        #     "font-family: monospace; }"
        # )
        # outer.addWidget(self.pos_label)

        # ================= STATUS BAR (bottom) =================
        self.status = QLabel("Status: disconnected")
        self.status.setStyleSheet(
            "QLabel { border-top: 1px solid palette(mid); padding: 4px; }"
        )
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        # Wrap everything in a scroll area so the panel doesn't get cut off
        # when many sections are expanded.
        inner = QWidget()
        inner.setLayout(outer)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        wrapper = QVBoxLayout(self)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(scroll)

        # Keep references to pop-up windows so they don't get garbage collected.
        self._plot_windows = []
        self.defaults_path = None
        self._load_user_defaults()
        # attach hover help text to every field
        apply_tooltips(self)
        # # Poll the stage position periodically for the live X/Y/Z readout.
        # self._pos_timer = QTimer(self)
        # self._pos_timer.setInterval(500)  # ms
        # self._pos_timer.timeout.connect(self._update_position_label)
        # self._pos_timer.start()

    def open_user_manual(self, _link=None):
        pdf_path = (
            Path(__file__).resolve().parent
            / "resources"
            / "napari-raman-widget-manual.pdf"
        )

        if not pdf_path.exists():
            QMessageBox.warning(
                self,
                "Manual not found",
                f"The user manual could not be found:\n{pdf_path}",
            )
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(pdf_path)))

    def _load_user_defaults(self):
        """Prefill hardware controls from a machine-local JSON file."""
        try:
            defaults_path, values = load_hardware_defaults()
        except (OSError, ValueError) as error:
            print(f"[hardware defaults] {error}")
            return

        if defaults_path is None:
            return

        self.defaults_path = defaults_path
        path_fields = {
            "micro_manager_config": self.cfg_path,
            "transformer_model": self.tf_path,
            "vandermonde_model": self.sel_vdm_path,
            "pixel_to_wavenumber_calibration": (
                self.spectral_calibration_path
            ),
            "tracking_config": self.mda_track_cfg_input,
            "dark_noise_file": self.dark_noise_path,
        }
        text_fields = {
            "mda_output_directory": self.mda_dir_input,
        }
        number_fields = {
            "center_wavelength_nm": self.wl_input,
            "raman_exposure_ms": self.exposure_input,
            "raman_repeats": self.n_input,
            "selection_center_x": self.sel_cx_input,
            "selection_center_y": self.sel_cy_input,
            "selection_radius": self.sel_r_input,
        }
        combo_fields = {
            "selection_cellpose_model": self.sel_cellpose_combo,
            "mda_cellpose_model": self.mda_seg_model_combo,
        }
        for key, widget in path_fields.items():
            if key in values and values[key] is not None:
                widget.setText(resolve_defaults_path(values[key], defaults_path))
        for key, widget in text_fields.items():
            if key in values and values[key] is not None:
                widget.setText(str(values[key]))
        for key, widget in number_fields.items():
            if key in values and values[key] is not None:
                try:
                    number = float(values[key])
                    if isinstance(widget, QSpinBox):
                        widget.setValue(int(round(number)))
                    else:
                        widget.setValue(number)
                except (TypeError, ValueError):
                    print(
                        f"[hardware defaults] ignored invalid {key}: "
                        f"{values[key]!r}"
                    )
        for key, widget in combo_fields.items():
            if key not in values or values[key] is None:
                continue
            choice = str(values[key])
            index = widget.findText(choice)
            if index >= 0:
                widget.setCurrentIndex(index)
            else:
                print(
                    f"[hardware defaults] ignored unavailable {key}: "
                    f"{choice!r}"
                )
        supported = set(path_fields) | set(text_fields) | set(number_fields)
        supported |= set(combo_fields)
        unknown = sorted(set(values) - supported)
        if unknown:
            print(f"[hardware defaults] ignored unknown keys: {unknown}")
        print(f"[hardware defaults] loaded {defaults_path}")
        if self.spectral_calibration_path.text().strip():
            self.load_spectral_calibration(show_success=False)

    # -------- file pickers --------
    def browse_spectral_calibration(self):
        browse_spectral_calibration(self)

    def load_spectral_calibration(
        self, _checked=False, *, show_success=True
    ):
        return load_spectral_calibration(self, show_success=show_success)

    def _spectral_calibration_created(self, calibration, path):
        spectral_calibration_created(self, calibration, path)

    def browse_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Micro-Manager config", "",
            "Config files (*.cfg);;All files (*)",
        )
        if path:
            self.cfg_path.setText(path)

    def browse_tf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select transformer model", "",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.tf_path.setText(path)

    def browse_out(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select output folder", ""
        )
        if path:
            self.out_path.setText(path)

    def browse_vandermonde(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Vandermonde model", "",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.sel_vdm_path.setText(path)

    def browse_dark_noise(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select dark-noise spectra",
            "",
            "NumPy arrays (*.npy);;All files (*)",
        )
        if path:
            self.dark_noise_path.setText(path)

    def browse_tracking_cfg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select tracking config", "",
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.mda_track_cfg_input.setText(path)

    # -------- helpers --------
    def _get_image_xy(self):
        """Return (X_size, Y_size) from the camera via the core if connected,
        else from the first image layer in the viewer. Falls back to
        (1344, 1024) if neither is available."""
        if self.core is not None:
            try:
                return int(self.core.getImageWidth()), int(self.core.getImageHeight())
            except Exception:
                pass
        from napari.layers import Image
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                shape = layer.data.shape
                Y, X = shape[-2], shape[-1]
                return int(X), int(Y)
        return 1344, 1024
    
    def _load_vandermonde(self):
        """Load the Vandermonde model from the path in the selection section.
        Stores (C, degree) on self.vandermonde. Returns a status fragment."""
        path = self.sel_vdm_path.text().strip()
        if not path:
            self.vandermonde = None
            return " (no vandermonde)"
        try:
            C, degree = load_vandermonde_model(path)
            self.vandermonde = (C, degree)
            return f" (vandermonde deg={degree} OK)"
        except Exception as e:
            self.vandermonde = None
            return f" (vandermonde load failed: {e})"

    def _make_point_transformer(self, size_px, n):
        """Build the selected aiming transformer from a pixel size.

        Converts px -> normalized using image width. Square uses it as edge
        length, Circle as radius.
        """
        from raman_mda_engine.aiming.transformers import Square, Circle
        img_x, _ = self._get_image_xy()
        length = float(size_px) / float(img_x)
        n = max(1, int(n))
        if self.sel_shape_combo.currentText() == "Circle":
            return Circle(length, n)
        return Square(length, n)

    def _pt_to_volts(self, pt):
        X, Y = self._get_image_xy()
        return self.transformer.BF_to_volts(
            (pt.reshape(1, -1)) / np.array([Y, X]),
            max_volts=1.8,
        )

    def _parse_float_list(self, text, label="list"):
        """Parse a comma-separated string of floats."""
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{label} is empty")
        try:
            return [float(p) for p in parts]
        except ValueError:
            raise ValueError(
                f"{label} contains non-numeric entries: {text!r}"
            )

    def _parse_int_list(self, text, label="list"):
        """Parse a comma-separated string of ints."""
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError(f"{label} is empty")
        try:
            return [int(p) for p in parts]
        except ValueError:
            raise ValueError(
                f"{label} contains non-integer entries: {text!r}"
            )

    def _update_position_label(self):
        """Poll the stage for X/Y and update the live readout label.

        Z is deliberately NOT read here: reading the focus device on a timer can
        contend with the engine while it drives Z during acquisition. X/Y live
        only. Never raises; transient failures keep the last good reading."""
        if self.core is None:
            self.pos_label.setText("Stage:  X --  Y --")
            return
        try:
            x = self.core.getXPosition()
            y = self.core.getYPosition()
            self.pos_label.setText(
                f"Stage:  X {x:9.2f}  Y {y:9.2f}"
            )
        except Exception:
            # transient failure (device busy, reload in progress) -- leave the
            # last good reading up rather than flickering an error.
            pass

    def _toggle_zscan_fields(self, checked):
        """Show/hide the z-scan range and steps fields."""
        self._zscan_range_label.setVisible(checked)
        self.scan_zrange_input.setVisible(checked)
        self._zscan_steps_label.setVisible(checked)
        self.scan_zsteps_input.setVisible(checked)

    def _toggle_seg_track_fields(self, checked):
        """Show/hide the segment-channel and rescale fields."""
        self._seg_ch_label.setVisible(checked)
        self.mda_seg_ch_combo.setVisible(checked)
        self._seg_scale_label.setVisible(checked)
        self.mda_seg_scale_input.setVisible(checked)
        self._seg_model_label.setVisible(checked)
        self.mda_seg_model_combo.setVisible(checked)
        self._seg_crop_label.setVisible(checked)
        self.mda_seg_crop_combo.setVisible(checked)
        self._seg_track_cfg_label.setVisible(checked)
        self.mda_track_cfg_input.setVisible(checked)
        self._seg_track_cfg_browse.setVisible(checked)

    def _toggle_px2stage_fields(self, checked):
        """Show/hide the pixel-to-stage calibration fields."""
        for w in self._px2stage_widgets:
            w.setVisible(checked)

    def _toggle_recal_fields(self, checked):
        """Show/hide the recalibration fields."""
        for w in self._recal_widgets:
            w.setVisible(checked)

    def _toggle_autofocus_fields(self, method):
        """Show/hide the MDA autofocus fields based on the chosen object.

        - None  : hide everything (no autofocus).
        - laser : show coarse range/pts AND the laser fine range/pts.
        - other : show coarse range/pts only (fine is laser-specific).
        """
        method = (method or "").lower()
        has_autofocus = method not in ("none", "")
        is_laser = method == "laser"

        # coarse fields: shown for any real autofocus object
        self._af_range_label.setVisible(has_autofocus)
        self.mda_af_range_input.setVisible(has_autofocus)
        self._search_pts_label.setVisible(has_autofocus)
        self.mda_search_pts_input.setVisible(has_autofocus)

        # fine fields: laser only
        self._fine_range_label.setVisible(is_laser)
        self.mda_fine_range_input.setVisible(is_laser)
        self._fine_pts_label.setVisible(is_laser)
        self.mda_fine_pts_input.setVisible(is_laser)

    def _reapply_toggles(self):
        self._toggle_seg_track_fields(self.mda_seg_track_check.isChecked())
        self._toggle_zscan_fields(self.scan_zscan_check.isChecked())
        self._toggle_autofocus_fields(self.sel_af_combo.currentText())
        self._toggle_px2stage_fields(self.px2stage_check.isChecked())
        self._toggle_recal_fields(self.recal_check.isChecked())

    # -------- channel row helpers --------
    def _available_channels(self):
        """Query MM for available channels in the 'Channel' group, excluding BF."""
        if self.core is None:
            return []
        try:
            channels = list(self.core.getAvailableConfigs("Channel"))
            return [c for c in channels if c != "BF"]
        except Exception:
            return []

    def _add_channel_row(self, *, channel=None, exposure=500.0):
        """Append a new channel row to the spatial-mapping section."""
        row = QHBoxLayout()
        combo = QComboBox()
        combo.setEditable(False)
        available = self._available_channels()
        if available:
            combo.addItems(available)
            if channel and channel in available:
                combo.setCurrentText(channel)
        else:
            combo.addItem("(connect first)")
            combo.setEnabled(False)

        exp_spin = QDoubleSpinBox()
        exp_spin.setRange(1, 1_000_000)
        exp_spin.setValue(exposure)
        exp_spin.setDecimals(1)
        exp_spin.setSuffix(" ms")

        remove_btn = QPushButton("x")
        remove_btn.setFixedWidth(30)

        row.addWidget(combo, 2)
        row.addWidget(exp_spin, 1)
        row.addWidget(remove_btn)
        self.channel_rows_layout.addLayout(row)

        entry = {
            "row": row, "combo": combo, "exp": exp_spin, "remove": remove_btn,
        }
        self.channel_rows.append(entry)
        remove_btn.clicked.connect(lambda: self._remove_channel_row(entry))

    def _remove_channel_row(self, entry):
        """Remove a channel row from the layout and the bookkeeping list."""
        while entry["row"].count():
            item = entry["row"].takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.channel_rows_layout.removeItem(entry["row"])
        if entry in self.channel_rows:
            self.channel_rows.remove(entry)

    def _add_mda_channel_row(self, *, channel=None, exposure=10.0):
        """Append a channel row to the MDA section."""
        row = QHBoxLayout()
        combo = QComboBox()
        combo.setEditable(False)
        try:
            available_all = (
                list(self.core.getAvailableConfigs("Channel"))
                if self.core else []
            )
        except Exception:
            available_all = []
        if available_all:
            combo.addItems(available_all)
            if channel and channel in available_all:
                combo.setCurrentText(channel)
        else:
            combo.addItem("(connect first)")
            combo.setEnabled(False)

        exp_spin = QDoubleSpinBox()
        exp_spin.setRange(1, 1_000_000)
        exp_spin.setValue(exposure)
        exp_spin.setDecimals(1)
        exp_spin.setSuffix(" ms")

        remove_btn = QPushButton("x")
        remove_btn.setFixedWidth(30)

        row.addWidget(combo, 2)
        row.addWidget(exp_spin, 1)
        row.addWidget(remove_btn)
        self.mda_channel_rows_layout.addLayout(row)

        entry = {
            "row": row, "combo": combo, "exp": exp_spin, "remove": remove_btn,
        }
        self.mda_channel_rows.append(entry)
        remove_btn.clicked.connect(lambda: self._remove_mda_channel_row(entry))

    def _remove_mda_channel_row(self, entry):
        while entry["row"].count():
            item = entry["row"].takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.mda_channel_rows_layout.removeItem(entry["row"])
        if entry in self.mda_channel_rows:
            self.mda_channel_rows.remove(entry)

    def _refresh_channel_combos(self):
        """Repopulate every channel combo with the current MM channel list."""
        available_no_bf = self._available_channels()
        try:
            available_all = (
                list(self.core.getAvailableConfigs("Channel"))
                if self.core else []
            )
        except Exception:
            available_all = []

        for entry in self.channel_rows:
            combo = entry["combo"]
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if available_no_bf:
                combo.addItems(available_no_bf)
                combo.setEnabled(True)
                if current in available_no_bf:
                    combo.setCurrentText(current)
            else:
                combo.addItem("(connect first)")
                combo.setEnabled(False)
            combo.blockSignals(False)

        for entry in self.mda_channel_rows:
            combo = entry["combo"]
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if available_all:
                combo.addItems(available_all)
                combo.setEnabled(True)
                if current in available_all:
                    combo.setCurrentText(current)
            else:
                combo.addItem("(connect first)")
                combo.setEnabled(False)
            combo.blockSignals(False)
        
        combo = self.mda_seg_ch_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        if available_all:
            combo.addItems(available_all)
            combo.setCurrentText(current if current in available_all else "BF")
        else:
            combo.addItem("BF")
        combo.blockSignals(False)

    def _prepare_for_selection(self):
        """Read, normalize, and write back the napari MM MDA sequence."""
        try:
            self.core.stopSequenceAcquisition()
        except Exception as e:
            print(f"[live mode stop] {e}")

        if self.main_window is None:
            raise RuntimeError("napari-micromanager MDA widget is unavailable")

        try:
            mda_dock = self.main_window._dock_widgets["MDA"]
            mda_settings = mda_dock.children()[4]
        except Exception as e:
            raise RuntimeError(f"couldn't locate napari MDA controls: {e}") from e

        try:
            from raman_mda_engine.utils import get_seq_from_napari
            from useq import Channel, TIntervalLoops, ZRangeAround
            import datetime as _dt

            seq = get_seq_from_napari(self.main_window)
            replacements = {
                "axis_order": ("t", "p", "c", "z"),
                "z_plan": ZRangeAround(range=0.0, step=1.0),
                "time_plan": TIntervalLoops(
                    interval=_dt.timedelta(seconds=0), loops=1
                ),
            }
            new_seq = seq.replace(**replacements)
            if hasattr(mda_settings, "setValue"):
                mda_settings.setValue(new_seq)
                print(
                    "[mda setup] get_seq_from_napari -> "
                    f"{len(new_seq.stage_positions)} position(s), BF, "
                    "z=single, t=1; pushed to MDA widget"
                )
            else:
                raise RuntimeError("napari MDA sequence control has no setValue")
            return new_seq
        except Exception as e:
            raise RuntimeError(f"failed to prepare napari MDA sequence: {e}") from e

    # -------- dataset generation --------
    def generate_dataset(self):
        # Let the user pick which run folder to load.
        default_dir = self.mda_dir_input.text().strip() or "data/run"
        run_dir = QFileDialog.getExistingDirectory(
            self, "Select MDA run folder", default_dir
        )
        if not run_dir:
            return  # user cancelled

        batch = self.sel_batch_combo.currentText() == "True"

        log = LogWindow(title="Dataset generation log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: generating dataset...")
        self.repaint()

        try:
            from .dataset import load_experiment
            from pathlib import Path

            run_path = Path(run_dir)
            run_name = run_path.name              # e.g. "run_9"
            parent = run_path.parent              # e.g. "data/"
            dataset_dir = parent / "dataset"      # e.g. "data/dataset/"

            os.makedirs(dataset_dir, exist_ok=True)
            zarr_path = str(dataset_dir / f"ds_{run_name}.zarr")
            pkl_path = str(dataset_dir / f"df_{run_name}.pkl")

            with _StdoutRedirector(log):
                df, df_locs, da = load_experiment(
                    run_dir, zarr_output=zarr_path, batch=batch,
                )
                df.to_pickle(pkl_path)
                print(f"Saved DataFrame to {pkl_path}")

            log.append(f"\n--- dataset ready: {len(df)} spectra ---\n")

            win = DatasetViewerWindow(
                df,
                da,
                title=f"Dataset: {run_name}",
                spectral_calibration=self.spectral_calibration,
            )
            win.show()
            self._plot_windows.append(win)

            self.status.setText(
                f"Status: dataset generated ({len(df)} spectra) "
                f"-> {zarr_path}, {pkl_path}"
            )
        except Exception as e:
            log.append(f"\n--- generation failed: {e} ---\n")
            self.status.setText(f"Status: dataset generation failed - {e}")

    
    def browse_px2stage_ds(self):
        # zarr stores are directories
        path = QFileDialog.getExistingDirectory(
            self, "Select dataset (.zarr)", "data/dataset"
        )
        if path:
            self.px2stage_ds_path.setText(path)
 
    def open_pixel_stage_picker(self):
        """Open the frame-by-frame point picker on the selected dataset."""
        import json as _json
        path = self.px2stage_ds_path.text().strip()
        if not path:
            self.status.setText("Status: select a dataset (.zarr) first")
            return
        try:
            ds = xr.open_zarr(path)
            if "useq_sequence" not in ds.attrs:
                self.status.setText(
                    "Status: dataset has no useq_sequence attr -- "
                    "regenerate it with the current loader"
                )
                return
            seq = _json.loads(ds.attrs["useq_sequence"])
            self.px2stage_xy = np.array(
                [[s["x"], s["y"]] for s in seq["stage_positions"]]
            )
            # one frame per position: first t / c / z
            imgs = ds["image"].isel(t=0, c=0, z=0).values
            if len(imgs) != len(self.px2stage_xy):
                print(
                    f"[px2stage] warning: {len(imgs)} frames vs "
                    f"{len(self.px2stage_xy)} stage positions"
                )
            import matplotlib
            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt
            plt.ion()
            self.px2stage_picker = StagePointPicker(imgs)
            plt.show()
            self.status.setText(
                f"Status: picker open ({len(imgs)} frames) -- click through, "
                "then Fit & save"
            )
        except Exception as e:
            self.status.setText(f"Status: picker failed -- {e}")
 
    def fit_and_save_pixel_stage(self):
        """Fit the centered Vandermonde model on picked points and save it."""
        if self.px2stage_picker is None or self.px2stage_xy is None:
            self.status.setText("Status: no picked points -- pick points first")
            return
        log = LogWindow(title="Pixel-to-stage fit log")
        log.show()
        self._plot_windows.append(log)
        try:
            points = np.asarray(self.px2stage_picker.points, dtype=float)
            xy = self.px2stage_xy
            n = min(len(points), len(xy))
            points, xy = points[:n], xy[:n]
            valid = ~np.isnan(points).any(axis=1)
            degree = int(self.px2stage_degree_input.value())
            n_terms = (degree + 1) * (degree + 2) // 2
            if valid.sum() < n_terms:
                self.status.setText(
                    f"Status: need >= {n_terms} points for degree {degree}, "
                    f"got {valid.sum()}"
                )
                return
            with _StdoutRedirector(log):
                img_center = points[valid].mean(axis=0)
                xy_center = xy[valid].mean(axis=0)
                points_c = points[valid] - img_center
                xy_c = xy[valid] - xy_center
                print(f"Fitting on {valid.sum()}/{n} points")
                print(f"img_center={img_center}, xy_center={xy_center}")
                # RMSE comparison across degrees (offset domain)
                for deg in (1, 2, 3):
                    if valid.sum() < (deg + 1) * (deg + 2) // 2:
                        print(f"degree={deg}  (not enough points)")
                        continue
                    C_deg = fit_vandermonde(points_c, xy_c, deg)
                    res = xy_c - apply_vandermonde(points_c, C_deg, deg)
                    rmse = np.sqrt(np.mean(res**2, axis=0))
                    print(
                        f"degree={deg}  RMSE: x={rmse[0]:.4f}, y={rmse[1]:.4f}"
                    )
                C = fit_vandermonde(points_c, xy_c, degree)
                # sanity check: stage step for a 5 px x-offset
                test = apply_vandermonde(np.array([[5.0, 0.0]]), C, degree)[0]
                print(f"5px-x step -> stage (degree={degree}): {test}")
            # save dialog so the user can rename / choose location
            default_name = (
                self.px2stage_name_input.text().strip()
                or "vandermonde_model.json"
            )
            if not default_name.lower().endswith(".json"):
                default_name += ".json"
            save_path, _ = QFileDialog.getSaveFileName(
                self, "Save Vandermonde model", default_name,
                "JSON files (*.json);;All files (*)",
            )
            if not save_path:
                self.status.setText("Status: save cancelled")
                return
            save_vandermonde_model(
                save_path, C, degree,
                img_center=img_center, xy_center=xy_center,
            )
            self.px2stage_name_input.setText(save_path)
            # make it immediately usable by center-cell mode
            self.sel_vdm_path.setText(save_path)
            log.append(f"\n--- saved {save_path} ---\n")
            self.status.setText(
                f"Status: Vandermonde model (degree={degree}) saved -> "
                f"{save_path}"
            )
        except Exception as e:
            log.append(f"\n--- fit failed: {e} ---\n")
            self.status.setText(f"Status: pixel-to-stage fit failed -- {e}")

    # -------- loading actions --------
    def connect(self):
        self.status.setText("Status: connecting...")
        self.repaint()

        out = self.out_path.text().strip()
        if out:
            try:
                os.makedirs(out, exist_ok=True)
                os.chdir(out)
                print(f"[cwd] changed to {os.getcwd()}")
            except Exception as e:
                self.status.setText(
                    f"Status: couldn't cd to output folder -- {e}"
                )
                return

        try:
            from pymmcore_plus import CMMCorePlus
            from raman_control.andor import AndorSpectraCollector

            self.core = CMMCorePlus.instance()
            cfg = self.cfg_path.text().strip()
            # Patch the shared singleton before napari-micromanager is loaded,
            # so its controller, stage, live, and snap widgets are guarded too.
            self.core_guard = install_core_guard(
                self.core,
                config_file=(cfg or None),
            )

            try:
                self.core.unloadAllDevices()
                time.sleep(0.5)
            except Exception:
                pass

            if cfg:
                self.core.loadSystemConfiguration(cfg)
                try:
                    self.core.setConfig("Channel", "GFP")
                    time.sleep(1)
                    self.core.setConfig("Channel", "BF")
                except Exception as e:
                    print(f"[channel warm-up] {e}")

            try:
                result = self.viewer.window.add_plugin_dock_widget(
                    "napari-micromanager"
                )
                if isinstance(result, tuple) and len(result) >= 2:
                    self.main_window = result[1]
            except Exception as e:
                print(f"[napari-micromanager load] {e}")

            self.collector = AndorSpectraCollector()
            self.daq = self.collector.daq
            self._configure_collect_detector_rows()
            self.default_engine = self.core.mda.engine

            tf = self.tf_path.text().strip()
            if tf:
                self.transformer = CoordTransformer.from_json(tf)

            vdm_msg = self._load_vandermonde()

            self._refresh_channel_combos()
            self.wl_update_btn.setEnabled(True)
            self.refresh_wavelength()
            self.refresh_wavelength()
            self.grating_update_btn.setEnabled(True)   
            self.refresh_gratings()

            msg = "Status: connected OK (core retries + config recovery enabled)"
            if not cfg:
                msg += " (reload disabled: no cfg loaded)"
            if not tf:
                msg += " (no transformer)"
            msg += vdm_msg
            self.status.setText(msg)
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.reload_tf_btn.setEnabled(True)
        except Exception as e:
            self.status.setText(f"Status: failed -- {e}")

    def reload_transformer(self):
        tf = self.tf_path.text().strip()
        if not tf:
            self.status.setText("Status: no transformer path set")
            return
        try:
            self.transformer = CoordTransformer.from_json(tf)
            vdm_msg = self._load_vandermonde()
            self.status.setText(f"Status: transformer reloaded OK{vdm_msg}")
        except Exception as e:
            self.status.setText(f"Status: transformer reload failed -- {e}")

    def refresh_wavelength(self):
        """Read and display the current center wavelength."""
        if self.collector is None:
            return
        try:
            wl = self.collector.get_wavelength()
            self.wl_current_label.setText(f"{wl:.2f} nm")
            self.wl_input.setValue(wl)
        except Exception as e:
            self.wl_current_label.setText(f"error: {e}")

    def update_wavelength(self):
        """Set a new center wavelength on the spectrometer."""
        if self.collector is None:
            self.status.setText("Status: not connected")
            return
        wl = float(self.wl_input.value())
        try:
            self.collector.set_wavelength(wl)
            self.refresh_wavelength()
            self.status.setText(f"Status: wavelength set to {wl:.2f} nm")
        except Exception as e:
            self.status.setText(f"Status: wavelength update failed -- {e}")

    def refresh_gratings(self):
        if self.collector is None:
            return
        try:
            n = int(self.collector.get_number_gratings())
            current = int(self.collector.get_grating())
            # ensure the list is at least long enough to hold `current`
            n = max(n, current)
            self.grating_combo.blockSignals(True)
            self.grating_combo.clear()
            self.grating_combo.addItems([str(i) for i in range(1, n + 1)])
            self.grating_combo.blockSignals(False)
            idx = current - 1
            if 0 <= idx < self.grating_combo.count():
                self.grating_combo.setCurrentIndex(idx)
            self.grating_combo.setEnabled(True)
            print(f"[gratings] n={n} current={current} "
                  f"count={self.grating_combo.count()} "
                  f"shown={self.grating_combo.currentText()}")
        except Exception as e:
            self.status.setText(f"Status: couldn't read gratings -- {e}")

    def update_grating(self):
        """Move the turret to the selected grating, then report groove
        density and center wavelength in the status bar."""
        if self.collector is None:
            self.status.setText("Status: not connected")
            return
        text = self.grating_combo.currentText()
        if not text:
            self.status.setText("Status: no grating selected")
            return
        grating = int(text)
        self.status.setText(f"Status: moving to grating {grating}...")
        self.repaint()
        try:
            self.collector.set_grating(grating)
            lines, blaze, home, offset = self.collector.get_grating_info(grating)
            wl = self.collector.get_wavelength()
            self.refresh_wavelength()               # keep lambda display in sync
            self.status.setText(
                f"Status: grating {grating} set -- {lines:.0f} grooves/mm, "
                f"center {wl:.2f} nm"
            )
        except Exception as e:
            self.status.setText(f"Status: grating update failed -- {e}")

    def disconnect(self):
        if self._live_raman_worker is not None:
            self._stop_live_raman()
            return
        try:
            if self.core is not None:
                unload(self.core)
        except Exception as e:
            print(f"unload error: {e}")
        self.core = None
        self.collector = None
        self.daq = None
        self.transformer = None
        self.default_engine = None
        self.calibration_ds = None
        self.calibrator = None
        self.selector = None
        self.scan_ds = None
        self.main_window = None
        self.selection_results = None
        self.mda_writer = None
        self.px2stage_picker = None
        self.px2stage_xy = None

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown_hardware)
        self.vandermonde = None
        self.status.setText("Status: disconnected")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.reload_tf_btn.setEnabled(False)
        self.wl_update_btn.setEnabled(False)
        self.grating_update_btn.setEnabled(False)
        self.grating_combo.setEnabled(False)
        self.grating_combo.clear()
        self.wl_current_label.setText("\u2014")
        self.calib_box.setEnabled(True)
        self.calib_box.setToolTip("")

    # -------- raman collection --------
    def _raman_image_center_yx(self):
        """Return the center of the active/newest image in napari coordinates."""
        active = self.viewer.layers.selection.active
        if isinstance(active, napari.layers.Image):
            height, width = np.asarray(active.data).shape[-2:]
            return np.array([height / 2.0, width / 2.0], dtype=float)

        for layer in reversed(list(self.viewer.layers)):
            if isinstance(layer, napari.layers.Image):
                height, width = np.asarray(layer.data).shape[-2:]
                return np.array([height / 2.0, width / 2.0], dtype=float)

        width, height = self._get_image_xy()
        return np.array([height / 2.0, width / 2.0], dtype=float)

    def _ensure_raman_points_layer(self):
        """Select a Points layer, creating a centered point when needed."""
        active = self.viewer.layers.selection.active
        if isinstance(active, napari.layers.Points):
            layer = active
        else:
            point_layers = [
                layer
                for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Points)
            ]
            layer = point_layers[-1] if point_layers else None

        if layer is None:
            center = self._raman_image_center_yx()
            layer = self.viewer.add_points(
                center.reshape(1, 2),
                name="Raman points",
            )
            self.viewer.layers.selection.active = layer
            layer.selected_data = {0}
            self.status.setText(
                "Status: created Raman points layer at the image center"
            )
            return layer

        data = np.asarray(layer.data)
        if data.ndim != 2 or len(data) == 0:
            center = self._raman_image_center_yx()
            layer.add(center.reshape(1, 2))
            data = np.asarray(layer.data)
            layer.selected_data = {len(data) - 1}

        self.viewer.layers.selection.active = layer
        return layer

    def _raman_point_from_active_layer(self):
        """Return the selected (or newest) point from the active Points layer."""
        layer = self._ensure_raman_points_layer()

        data = np.asarray(layer.data)
        if data.ndim != 2 or len(data) == 0 or data.shape[1] < 2:
            raise ValueError("the active Points layer does not contain a point")

        selected = sorted(int(i) for i in layer.selected_data)
        point_index = selected[-1] if selected else len(data) - 1
        return np.asarray(data[point_index, -2:], dtype=float), point_index

    def _set_scan_imaging_channel(self):
        """Select a camera channel that exists on the connected microscope."""
        self.core.setConfig("Channel", "BF")

    def _set_scan_raman_mode(self, enabled):
        """Control hardware-only Raman channel/shutter state."""
        if enabled:
            self.core.setConfig("Channel", "RM")
            self.core.setShutterOpen("Fluoshutter", True)
        else:
            self.core.setShutterOpen("Fluoshutter", False)

    def _update_collect_read_mode_fields(self, _index=None):
        """Show detector-track options only for single-track readout."""
        mode = self.collect_read_mode_combo.currentData()
        self.collect_single_track_controls.setVisible(
            self.raman_box.isChecked() and mode == "single_track"
        )

    def _collect_read_mode_options(self, read_mode):
        options = {}
        if read_mode != "fvb":
            options["read_mode"] = read_mode
        if read_mode == "single_track":
            options.update(
                track_center=int(self.collect_track_center_input.value()),
                track_height=int(self.collect_track_height_input.value()),
            )
        return options

    def _load_spectral_bias(self):
        path_text = self.dark_noise_path.text().strip()
        if not path_text:
            return None
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"dark-noise file not found: {path}")
        dark_noise = np.load(path, allow_pickle=False)
        return spectral_bias_from_dark_noise(dark_noise)

    def _save_dark_noise_as_default(self, path):
        defaults_path = self.defaults_path
        if defaults_path is None:
            defaults_path = self._startup_directory / DEFAULTS_FILENAME
        self.defaults_path = update_hardware_defaults(
            defaults_path,
            {
                "dark_noise_file": str(Path(path).resolve()),
            },
        )

    def collect_dark_noise(self):
        """Stop live acquisition, collect repeated dark spectra, and save."""
        if self.live_collect_check.isChecked():
            self.live_collect_check.setChecked(False)
        if self._live_raman_worker is not None:
            self._collect_dark_after_live_stop = True
            self.collect_dark_noise_btn.setEnabled(False)
            self.collect_dark_noise_btn.setText(
                "Waiting for live exposure to stop..."
            )
            self._stop_live_raman()
            return
        self._collect_dark_after_live_stop = False

        if self.collector is None or self.daq is None:
            self.status.setText("Status: not connected")
            return

        read_mode = self.collect_read_mode_combo.currentData()
        if read_mode == "image":
            self.status.setText(
                "Status: dark-noise bias removal supports FVB or "
                "single-track readout"
            )
            return

        exposure = float(self.exposure_input.value())
        repeats = int(self.n_input.value())
        try:
            if self.core is not None:
                try:
                    self.core.stopSequenceAcquisition()
                except Exception as error:
                    print(f"[dark noise live stop] {error}")
                self._set_scan_raman_mode(False)

            self.status.setText(
                f"Status: collecting {repeats} dark spectra..."
            )
            self.repaint()
            self.daq.galvo.stop()
            self.daq.galvo.start()
            dark_noise = self.collector.collect_spectra_pts(
                np.zeros((repeats, 2), dtype=float),
                exposure,
                **self._collect_read_mode_options(read_mode),
            )
            dark_noise = np.asarray(dark_noise)
            spectral_bias = spectral_bias_from_dark_noise(dark_noise)

            path = (
                Path.cwd()
                / f"dark_noise_{exposure:g}ms_{uuid.uuid4()}.npy"
            )
            np.save(path, dark_noise, allow_pickle=False)
            self.dark_noise_path.setText(str(path.resolve()))

            defaults_warning = ""
            try:
                self._save_dark_noise_as_default(path)
            except Exception as error:
                defaults_warning = f"; couldn't update defaults: {error}"
                print(f"[dark noise defaults] {error}")

            title = (
                f"Dark noise | {read_mode} | {repeats}x{exposure:.0f} ms"
            )
            window = SpectrumWindow(
                dark_noise,
                title=title,
                spectral_calibration=self.spectral_calibration,
                calibration_changed=self._spectral_calibration_created,
                spectral_bias=spectral_bias,
            )
            window.show()
            self._plot_windows.append(window)
            self.status.setText(
                f"Status: dark noise saved and selected as default -> "
                f"{path}{defaults_warning}"
            )
        except Exception as error:
            self.status.setText(
                f"Status: dark-noise collection failed -- {error}"
            )
        finally:
            self.collect_dark_noise_btn.setEnabled(True)
            self.collect_dark_noise_btn.setText("Collect dark noise")

    def _update_collect_live_fields(self, checked=None):
        """Switch between finite repeats and one-at-a-time live collection."""
        if checked is None:
            checked = self.live_collect_check.isChecked()
        if self._live_raman_worker is None:
            self.n_input.setEnabled(not checked)
            self.collect_save_input.setEnabled(not checked)
            self.collect_btn.setText(
                "Start live spectra at selected point"
                if checked
                else "Collect spectra at selected point"
            )
        self.collect_save_input.setToolTip(
            "Live display is not saved. Turn off Live to save a finite "
            "repeated collection."
            if checked
            else "Optional base filename for one xarray .zarr dataset."
        )

    def _configure_collect_detector_rows(self):
        """Constrain single-track fields to the connected Andor detector."""
        sdk = getattr(self.collector, "_sdk", None)
        if sdk is None:
            return
        detector_result = sdk.GetDetector()
        if len(detector_result) < 3:
            return
        detector_rows = int(detector_result[2])
        if detector_rows < 2:
            return
        self.collect_track_center_input.setRange(0, detector_rows - 1)
        self.collect_track_center_input.setValue(min(185, detector_rows - 1))
        self.collect_track_height_input.setRange(2, detector_rows)
        self.collect_track_height_input.setValue(min(140, detector_rows))

    def collect_raman(self):
        if self._live_raman_worker is not None:
            self._stop_live_raman()
            return
        if self.collector is None or self.daq is None:
            self.status.setText("Status: not connected")
            return
        if self.transformer is None:
            self.status.setText("Status: no transformer loaded")
            return
        exposure = float(self.exposure_input.value())
        N = int(self.n_input.value())
        read_mode = self.collect_read_mode_combo.currentData()

        if self.live_collect_check.isChecked():
            self._start_live_raman(exposure, read_mode)
            return

        try:
            collection_started_at = datetime.now().astimezone()
            collection_started = time.perf_counter()
            pt, point_index = self._raman_point_from_active_layer()
            spectral_bias = (
                self._load_spectral_bias()
                if read_mode != "image"
                else None
            )
            bias_available = spectral_bias is not None

            self.daq.galvo.stop()
            self.daq.galvo.start()

            volts = self._pt_to_volts(pt)
            read_mode_options = self._collect_read_mode_options(read_mode)
            spec = self.collector.collect_spectra_pts(
                np.tile(volts[0], (N, 1)),
                exposure,
                **read_mode_options,
            )
            collection_elapsed = time.perf_counter() - collection_started
            collection_finished_at = datetime.now().astimezone()

            wavelength = float(self.collector.get_wavelength())
            grating = int(self.collector.get_grating())
            peak = float(np.max(spec))

            save_name = self.collect_save_input.text().strip()
            saved_msg = ""
            if save_name:
                track_center = (
                    int(self.collect_track_center_input.value())
                    if read_mode == "single_track"
                    else None
                )
                track_height = (
                    int(self.collect_track_height_input.value())
                    if read_mode == "single_track"
                    else None
                )
                store_path = save_collection_record(
                    save_name,
                    spec,
                    {
                        "record_type": "point_layer_raman_collection",
                        "format_version": 1,
                        "collection_started_at": collection_started_at.isoformat(),
                        "collection_finished_at": collection_finished_at.isoformat(),
                        "collection_elapsed_seconds": collection_elapsed,
                        "point_index": point_index,
                        "point_yx": pt.tolist(),
                        "galvo_volts": np.asarray(volts[0]).tolist(),
                        "read_mode": read_mode,
                        "track_center": track_center,
                        "track_height": track_height,
                        "exposure_ms": exposure,
                        "repeats": N,
                        "center_wavelength_nm": wavelength,
                        "grating": grating,
                        "peak_intensity": peak,
                        "spectral_bias_available": bias_available,
                        "dark_noise_file": (
                            self.dark_noise_path.text().strip()
                            if bias_available
                            else None
                        ),
                    },
                )
                saved_msg = f" -> {store_path}"
                print(f"Saved collection dataset to {store_path}")

            title = (
                f"Raman point {point_index} | RM | {read_mode} | "
                f"{wavelength:.1f} nm | grating {grating} | "
                f"{exposure:.0f} ms"
            )
            if read_mode == "image":
                win = DetectorImageWindow(
                    spec,
                    title=title,
                    spectral_calibration=self.spectral_calibration,
                )
            else:
                win = SpectrumWindow(
                    spec,
                    title=title,
                    spectral_calibration=self.spectral_calibration,
                    calibration_changed=self._spectral_calibration_created,
                    spectral_bias=spectral_bias,
                )
            win.show()
            self._plot_windows.append(win)

            self.status.setText(
                f"Status: collected point {point_index} on RM, "
                f"{N}x{exposure:.0f} ms, {wavelength:.1f} nm, "
                f"grating {grating}, {read_mode}, {collection_elapsed:.3f} s, "
                f"peak {peak:.1f}{saved_msg}"
            )
        except Exception as e:
            self.status.setText(f"Status: collection failed -- {e}")

    def _start_live_raman(self, exposure, read_mode):
        """Start one-detector-frame acquisition cycles off the UI thread."""
        try:
            pt, point_index = self._raman_point_from_active_layer()
            spectral_bias = (
                self._load_spectral_bias()
                if read_mode != "image"
                else None
            )
            self.daq.galvo.stop()
            self.daq.galvo.start()
            volts = np.asarray(self._pt_to_volts(pt)[0], dtype=float)

            read_mode_options = self._collect_read_mode_options(read_mode)

            wavelength = float(self.collector.get_wavelength())
            grating = int(self.collector.get_grating())
            title = (
                f"Raman point {point_index} | RM | {read_mode} | "
                f"{wavelength:.1f} nm | grating {grating} | "
                f"{exposure:.0f} ms"
            )
            collector = self.collector
            batch_collector = getattr(
                collector, "collect_spectra_pts_batch", None
            )

            def acquire_one():
                repeated_volts = np.tile(volts, (2, 1))
                if callable(batch_collector):
                    spectrum = batch_collector(
                        repeated_volts, exposure, **read_mode_options
                    )
                else:
                    spectra = collector.collect_spectra_pts(
                        repeated_volts, exposure, **read_mode_options
                    )
                    spectrum = np.asarray(spectra)[-1:]
                return spectrum

            thread = QThread(self)
            worker = LiveSpectrumWorker(acquire_one)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.spectrum_ready.connect(self._on_live_raman_spectrum)
            worker.failed.connect(self._on_live_raman_failed)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(self._on_live_raman_finished)

            self._live_raman_thread = thread
            self._live_raman_worker = worker
            self._live_raman_window = None
            self._live_raman_context = {
                "point_index": point_index,
                "read_mode": read_mode,
                "exposure": exposure,
                "wavelength": wavelength,
                "grating": grating,
                "title": title,
                "count": 0,
                "failed": None,
                "spectral_bias": spectral_bias,
            }
            self._set_live_raman_controls(True)
            self.collect_btn.setText("Stop live spectra")
            self.status.setText(
                f"Status: live collection started at point {point_index}; "
                "waiting for spectrum 1..."
            )
            thread.start()
        except Exception as error:
            self.status.setText(f"Status: live collection failed -- {error}")

    def _on_live_raman_spectrum(self, spec, count, elapsed):
        context = self._live_raman_context
        if context is None:
            return
        context["count"] = count
        title = f"{context['title']} | live #{count}"
        if self._live_raman_window is None:
            if context["read_mode"] == "image":
                window = DetectorImageWindow(
                    spec,
                    title=title,
                    spectral_calibration=self.spectral_calibration,
                )
            else:
                window = SpectrumWindow(
                    spec,
                    title=title,
                    spectral_calibration=self.spectral_calibration,
                    calibration_changed=self._spectral_calibration_created,
                    spectral_bias=context.get("spectral_bias"),
                )
            window.show()
            self._plot_windows.append(window)
            self._live_raman_window = window
        elif context["read_mode"] == "image":
            self._live_raman_window.update_frames(spec, title=title)
        else:
            self._live_raman_window.update_spectrum(spec, title=title)

        peak = float(np.max(spec))
        self.status.setText(
            f"Status: live point {context['point_index']}, spectrum {count}, "
            f"{context['exposure']:.0f} ms, {elapsed:.3f} s elapsed, "
            f"peak {peak:.1f}"
        )

    def _stop_live_raman(self):
        worker = self._live_raman_worker
        if worker is None:
            return
        worker.request_stop()
        self.collect_btn.setEnabled(False)
        self.collect_btn.setText("Stopping after current exposure...")
        self.status.setText(
            "Status: live stop requested; waiting for the current exposure"
        )

    def _on_live_raman_failed(self, message):
        if self._live_raman_context is not None:
            self._live_raman_context["failed"] = message
        self.status.setText(f"Status: live collection failed -- {message}")

    def _on_live_raman_finished(self):
        context = self._live_raman_context or {}
        failed = context.get("failed")
        count = context.get("count", 0)
        thread = self._live_raman_thread
        self._live_raman_worker = None
        self._live_raman_thread = None
        self._live_raman_context = None
        self._live_raman_window = None
        self._set_live_raman_controls(False)
        self.collect_btn.setEnabled(True)
        self._update_collect_live_fields()
        collect_dark = self._collect_dark_after_live_stop
        self._collect_dark_after_live_stop = False
        if not failed and not collect_dark:
            self.status.setText(
                f"Status: live collection stopped after {count} spectrum"
                f"{'s' if count != 1 else ''}"
            )
        if thread is not None:
            thread.deleteLater()
        if collect_dark:
            self.collect_dark_noise_btn.setEnabled(True)
            self.collect_dark_noise_btn.setText("Collect dark noise")
            QTimer.singleShot(0, self.collect_dark_noise)

    def _set_live_raman_controls(self, running):
        controls = (
            self.loading_box,
            self.calib_box,
            self.grid_box,
            self.sel_box,
            self.exposure_input,
            self.n_input,
            self.live_collect_check,
            self.collect_read_mode_combo,
            self.collect_single_track_controls,
            self.collect_save_input,
        )
        if running:
            self._live_control_states = [
                (control, control.isEnabled()) for control in controls
            ]
            for control, _enabled in self._live_control_states:
                control.setEnabled(False)
        else:
            for control, enabled in self._live_control_states:
                control.setEnabled(enabled)
            self._live_control_states = []

    # -------- laser aiming calibration --------
    def run_calibration(self):
        if self.core is None or self.daq is None or self.collector is None:
            self.status.setText("Status: not connected")
            return
        if self.transformer is None:
            self.status.setText("Status: no transformer loaded")
            return

        N = int(self.cal_n_input.value())
        exp = float(self.cal_exp_input.value())
        max_volts = float(self.cal_volts_input.value())
        grid = int(self.cal_grid_input.value())
        thres = float(self.cal_thres_input.value())

        log = LogWindow(title="Calibration log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: calibrating...")
        self.repaint()

        try:
            self.calibrator = Calibrator(
                self.core, self.daq, self.transformer, self.collector,
                repeats=N, exposure=exp, max_volts=max_volts,
            )
            with _StdoutRedirector(log):
                self.calibration_ds = self.calibrator.calibrate(
                    grid, threshold=thres, plot=False
                )

            log.append("\n--- calibration complete ---\n")

            plot_win = CalibrationPlotWindow(
                self.calibration_ds, title="Calibration result"
            )
            plot_win.show()
            self._plot_windows.append(plot_win)

            self.status.setText("Status: calibration done OK")
        except Exception as e:
            log.append(f"\n--- calibration failed: {e} ---\n")
            self.status.setText(f"Status: calibration failed -- {e}")

    # -------- recalibration --------
    def open_selector(self):
        if self.calibration_ds is None:
            self.status.setText(
                "Status: no calibration dataset -- run calibration first"
            )
            return
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt

            plt.ion()
            self.selector = ManualImageSelector(self.calibration_ds)
            plt.show()
            self.status.setText(
                "Status: selector open -- click through, then save"
            )
        except Exception as e:
            self.status.setText(f"Status: selector failed -- {e}")

    def save_recalibration(self):
        if self.selector is None:
            self.status.setText("Status: no selector -- open it first")
            return
        if self.calibrator is None:
            self.status.setText(
                "Status: no calibrator -- run calibration first"
            )
            return
        if self.calibration_ds is None:
            self.status.setText("Status: no calibration dataset")
            return

        model_name = self.model_name_input.text().strip()
        if not model_name:
            self.status.setText("Status: enter a model name")
            return

        try:
            selected_points = self.selector.selected_points
            self.calibrator.save_new_model(
                self.calibration_ds, selected_points, model_name
            )
            self.transformer = CoordTransformer.from_json(f"{model_name}.json")
            self.tf_path.setText(f"{model_name}.json")
            self.status.setText(f"Status: saved & loaded {model_name}.json OK")
        except Exception as e:
            self.status.setText(f"Status: save failed -- {e}")

    # -------- collect reference spectra --------
    def collect_reference(self):
        if self.core is None or self.daq is None or self.collector is None:
            self.status.setText("Status: not connected")
            return
        if self.transformer is None:
            self.status.setText("Status: no transformer loaded")
            return
        if len(self.viewer.layers) == 0:
            self.status.setText("Status: no layer to read point from")
            return

        name = self.ref_name_input.text().strip()
        if not name:
            self.status.setText("Status: enter a name for the reference")
            return

        exp = float(self.ref_exp_input.value())
        N = int(self.ref_n_input.value())
        search_range = float(self.ref_range_input.value())
        search_pts = int(self.ref_pts_input.value())

        self.status.setText("Status: collecting reference spectra...")
        self.repaint()

        log = LogWindow(title="Reference collection log")
        log.show()
        self._plot_windows.append(log)

        try:
            pt, point_index = self._raman_point_from_active_layer()

            volts = self._pt_to_volts(pt)
            volts_tiled = np.array([volts[0] for _ in range(N)])

            with _StdoutRedirector(log):
                focusZ, coarse_raman, all_raman = autofocus_w_bkd(
                    self.core, self.daq, self.collector, volts_tiled,
                    search_range=search_range,
                    search_pts=search_pts,
                    exposure=exp,
                )
            self.core.setZPosition(focusZ)

            zs = np.linspace(-search_range, search_range, search_pts)

            win = ReferenceSpectraWindow(
                all_raman,
                zs,
                title=f"Reference spectra: {name}",
                spectral_calibration=self.spectral_calibration,
            )
            win.show()
            self._plot_windows.append(win)

            os.makedirs("reference", exist_ok=True)
            uid = str(uuid.uuid1())[:8]

            ds = xr.Dataset(
                {
                    "spec": (["z", "n", "pixel"], all_raman),
                },
                coords={
                    "z":        ("z",  zs),
                    "x":        pt[1],
                    "y":        pt[0],
                    "exposure": exp,
                },
            )

            zarr_path = f"reference/{name}_{uid}.zarr"
            ds.to_zarr(zarr_path)

            log.append(f"\n--- saved to {zarr_path} ---\n")
            self.status.setText(f"Status: reference saved ({zarr_path}) OK")

        except Exception as e:
            log.append(f"\n--- reference collection failed: {e} ---\n")
            self.status.setText(
                f"Status: reference collection failed -- {e}"
            )

    # -------- spatial mapping --------
    def run_grid_scan(self):
        if self.core is None or self.daq is None or self.collector is None:
            self.status.setText("Status: not connected")
            return
        if self.transformer is None:
            self.status.setText("Status: no transformer loaded")
            return
        if len(self.viewer.layers) == 0:
            self.status.setText("Status: no layer to read shape from")
            return

        file_name = self.scan_name_input.text().strip()
        if not file_name:
            self.status.setText("Status: enter a file name")
            return

        exp = float(self.scan_exp_input.value())
        N = int(self.scan_n_input.value())
        z_offset = float(self.scan_z_input.value())
        do_zscan = self.scan_zscan_check.isChecked()

        if do_zscan:
            z_half = float(self.scan_zrange_input.value())
            z_steps = int(self.scan_zsteps_input.value())
            z_range = np.linspace(-z_half, z_half, z_steps)
        else:
            z_range = np.array([0.0])

        extra_channels = []
        seen = set()
        for entry in self.channel_rows:
            if not entry["combo"].isEnabled():
                continue
            ch = entry["combo"].currentText()
            if not ch or ch in seen:
                continue
            seen.add(ch)
            extra_channels.append((ch, float(entry["exp"].value())))

        log = LogWindow(title="Grid scan log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: grid scanning...")
        self.repaint()

        try:
            import xarray as xr
            from datetime import datetime

            with _StdoutRedirector(log):
                shapes = self.viewer.layers.selection.active
                if shapes is None or not isinstance(shapes, napari.layers.Shapes):
                    raise RuntimeError(
                        "Select a Shapes layer and draw a rectangle first."
                    )
                if len(shapes.data) == 0:
                    raise RuntimeError("The active Shapes layer is empty.")
                selected_shapes = sorted(int(i) for i in shapes.selected_data)
                shape_index = selected_shapes[-1] if selected_shapes else len(shapes.data) - 1
                shape0 = np.asarray(shapes.data[shape_index], dtype=float)

                x_min = float(np.min(shape0[:, 0]))
                x_max = float(np.max(shape0[:, 0]))
                y_min = float(np.min(shape0[:, 1]))
                y_max = float(np.max(shape0[:, 1]))
                x = np.linspace(x_min, x_max, N)
                y = np.linspace(y_min, y_max, N)
                Xg, Yg = np.meshgrid(x, y)
                grid = np.column_stack([Xg.ravel(), Yg.ravel()])

                print(f"Grid: {N}x{N} = {grid.shape[0]} points")
                if do_zscan:
                    print(
                        f"Z-scan: {len(z_range)} planes, "
                        f"range [{z_range[0]:.1f}, {z_range[-1]:.1f}] um"
                    )

                self._set_scan_imaging_channel()
                self.core.setExposure(10)
                BF = self.core.snap()

                extra_imgs = {}
                for ch, ch_exp in extra_channels:
                    print(f"Snapping {ch} at {ch_exp:.0f} ms")
                    self.core.setConfig("Channel", ch)
                    self.core.setExposure(ch_exp)
                    extra_imgs[ch] = self.core.snap()

                self.daq.galvo.stop()
                self.daq.galvo.start()
                currentz = self.core.getPosition()
                base_z = currentz - z_offset
                self._set_scan_raman_mode(True)

                X_img, Y_img = self._get_image_xy()
                volts = self.transformer.BF_to_volts(
                    grid / np.array([Y_img, X_img]), max_volts=1.8
                )
                self.core.stopSequenceAcquisition()
                self.core.setExposure(1)

                all_specs = []
                all_BF_z = []
                for i, dz in enumerate(z_range):
                    self.core.setPosition(base_z + dz)
                    print(
                        f"  z-plane {i+1}/{len(z_range)}: "
                        f"dz={dz:+.2f} um, collecting "
                        f"{grid.shape[0]} spectra..."
                    )
                    specs = self.collector.collect_spectra_pts(volts, exp)
                    all_specs.append(specs)

                    if do_zscan:
                        self._set_scan_raman_mode(False)
                        self._set_scan_imaging_channel()
                        self.core.setExposure(10)
                        BF_z = self.core.snap()
                        all_BF_z.append(BF_z)
                        self._set_scan_raman_mode(True)
                        self.core.setExposure(1)

                self._set_scan_raman_mode(False)
                self.core.setPosition(currentz)
                self._set_scan_imaging_channel()
                self.core.setExposure(10)
                end_BF = self.core.snap()

                if do_zscan:
                    specs_stack = np.stack(all_specs, axis=0)
                    BF_stack = np.stack(all_BF_z, axis=0)
                    data_vars = {
                        "laser_pos": xr.DataArray(
                            volts, dims=("idx", "volt")
                        ),
                        "grid_pos": xr.DataArray(
                            grid, dims=("idx", "xy")
                        ),
                        "specs": xr.DataArray(
                            specs_stack, dims=("z", "idx", "spec_dim")
                        ),
                        "z_range": xr.DataArray(
                            z_range, dims=("z",)
                        ),
                        "BF": xr.DataArray(BF, dims=("Y", "X")),
                        "BF_z": xr.DataArray(
                            BF_stack, dims=("z", "Y", "X")
                        ),
                        "end_BF": xr.DataArray(end_BF, dims=("Y", "X")),
                    }
                    for ch, img in extra_imgs.items():
                        data_vars[ch] = xr.DataArray(img, dims=("Y", "X"))

                    ds = xr.Dataset(data_vars)
                    ds.attrs["time"] = str(datetime.now())
                    ds.attrs["raman_exposure_ms"] = exp
                    ds.attrs["z_offset"] = z_offset
                    ds.attrs["z_range_min"] = float(z_range[0])
                    ds.attrs["z_range_max"] = float(z_range[-1])
                    ds.attrs["z_steps"] = len(z_range)
                    ds.attrs["channel_exposures_ms"] = {
                        ch: ch_exp for ch, ch_exp in extra_channels
                    }

                    uid = uuid.uuid4().hex[:8]
                    zarr_name = f"grid_scan_z_{file_name}_{uid}.zarr"
                else:
                    data_vars = {
                        "laser_pos": xr.DataArray(
                            volts, dims=("idx", "volt")
                        ),
                        "grid_pos": xr.DataArray(
                            grid, dims=("idx", "volt")
                        ),
                        "specs": xr.DataArray(
                            all_specs[0], dims=("N", "spec_dim")
                        ),
                        "BF": xr.DataArray(BF, dims=("Y", "X")),
                        "end_BF": xr.DataArray(end_BF, dims=("Y", "X")),
                    }
                    for ch, img in extra_imgs.items():
                        data_vars[ch] = xr.DataArray(img, dims=("Y", "X"))

                    ds = xr.Dataset(data_vars)
                    ds.attrs["time"] = str(datetime.now())
                    ds.attrs["raman_exposure_ms"] = exp
                    ds.attrs["channel_exposures_ms"] = {
                        ch: ch_exp for ch, ch_exp in extra_channels
                    }

                    uid = uuid.uuid4().hex[:8]
                    zarr_name = f"grid_scan_data_{file_name}_{uid}.zarr"

                ds.to_zarr(zarr_name)
                print(f"Saved grid scan to {zarr_name}")

            self.scan_ds = ds

            win = GridScanPlotWindow(
                ds,
                title=f"Grid scan: {file_name}",
                spectral_calibration=self.spectral_calibration,
            )
            win.show()
            self._plot_windows.append(win)

            self.status.setText(f"Status: grid scan saved -> {zarr_name}")
        except Exception as e:
            log.append(f"\n--- grid scan failed: {e} ---\n")
            self.status.setText(f"Status: grid scan failed -- {e}")

    # -------- automated cell selection --------
    def add_mask(self):
        """Add the masked overlay to the viewer using current center/radius."""

        X, Y = self._get_image_xy()
        cy = int(self.sel_cy_input.value())
        cx = int(self.sel_cx_input.value())
        r = int(self.sel_r_input.value())

        try:
            add_mask_with_hole(
                self.viewer,
                image_size=(Y, X),
                circle_center=(cy, cx),
                circle_radius=r,
                small_circle_radius=10,
                color=(255, 0, 0),
                alpha=60,
                small_circle_color=(0, 255, 0),
                small_circle_alpha=255,
            )
            self.status.setText(
                f"Status: mask added at ({cy},{cx}) r={r} OK"
            )
        except Exception as e:
            self.status.setText(f"Status: add_mask failed -- {e}")

    def run_automated_selection(self):
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.default_engine is None:
            self.status.setText("Status: no default engine -- reconnect")
            return

        cy = int(self.sel_cy_input.value())
        cx = int(self.sel_cx_input.value())
        r = int(self.sel_r_input.value())
        autofocus_object = self.sel_af_combo.currentText()
        N_per_fov = int(self.sel_npf_input.value())
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        bkd_thres = float(self.sel_bkd_input.value())
        batch = self.sel_batch_combo.currentText() == "True"

        center_cell = self.sel_center_cell_check.isChecked()
        vandermonde_model_path = self.sel_vdm_path.text().strip()
        cellpose_model = self.sel_cellpose_combo.currentText() or "cyto2"

        if center_cell and not vandermonde_model_path:
            self.status.setText(
                "Status: Center cell mode requires a Vandermonde model (.json)"
            )
            return

        log = LogWindow(title="Automated selection log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText(
            "Status: running Cellpose from the napari MDA sequence..."
        )
        self.repaint()

        try:
            with _StdoutRedirector(log):
                self._prepare_for_selection()
                self.core.register_mda_engine(self.default_engine)
                point_transformer = self._make_point_transformer(sq_size, sq_n)
                no_autofocus = (
                    autofocus_object is None
                    or str(autofocus_object).strip().lower()
                    in {"", "none"}
                )

                selection_limit = (
                    N_per_fov
                    if center_cell or no_autofocus
                    else N_per_fov + 1
                )

                sources, autofocus_p, new_seq = automated_point_selections(
                    self.core,
                    self.viewer,
                    self.main_window,
                    point_transformer,
                    N=selection_limit,
                    center=(cy, cx),
                    radius=r,
                    autofocus_object=autofocus_object,
                    bkd_thres=bkd_thres,
                    batch=batch,
                    center_cell=center_cell,
                    vandermonde_model_path=(
                        vandermonde_model_path
                        if center_cell
                        else None
                    ),
                    cellpose_model=cellpose_model,
                    direct_pixel_to_stage=False,
                    stage_settle_time=5.0,
                    image_settle_time=1.0,
                    block_mda=False,
                    show_masks=False,
                    cellpose_upsample=1,
                    cell_point_refiner=None,
                )
            cell_point_count = sum(
                len(source._points.data)
                for source in sources
                if "cell" in str(getattr(source, "name", "")).lower()
            )
            if cell_point_count == 0:
                self.selection_results = None
                log.append(
                    "\n--- Cellpose returned no selectable cells; "
                    "inspect the Cellpose mask layer ---\n"
                )
                self.status.setText(
                    "Status: Cellpose found no selectable cells -- "
                    "inspect 'Cellpose mask p0' and adjust Center/Radius"
                )
                return
            self.selection_results = {
                "sources": sources,
                "autofocus_p": autofocus_p,
                "new_seq": new_seq,
            }
            n_new = len(new_seq.stage_positions)
            log.append("\n--- selection complete ---\n")
            extra = (
                f" ({n_new} centered positions)" if center_cell else ""
            )
            self.status.setText(
                f"Status: automated selection done OK{extra}"
            )
        except Exception as e:
            log.append(f"\n--- selection failed: {e} ---\n")
            self.status.setText(f"Status: selection failed -- {e}")


    def _toggle_click_to_center(self, checked):
        """Arm/disarm one-shot click-to-center mode."""
        if checked:
            if self._click_center_cb not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.append(self._click_center_cb)
            self.status.setText(
                "Status: click-to-center ARMED -- click a spot in the image"
            )
        else:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._click_center_cb)
            except ValueError:
                pass

    def refine_selected_cell_points(self):
        """Re-segment selected FOVs and snap only cell points to centroids."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.selection_results is None:
            self.status.setText(
                "Status: run automated, manual, or grid selection first"
            )
            return

        sources = self.selection_results.get("sources", ())
        cell_source = next(
            (
                source
                for source in sources
                if "cell" in str(getattr(source, "name", "")).lower()
            ),
            None,
        )
        if cell_source is None:
            self.status.setText("Status: selection has no cells layer")
            return

        log = LogWindow(title="Refine cell points log")
        log.show()
        self._plot_windows.append(log)
        self.status.setText(
            "Status: acquiring BF images and refining cell points..."
        )
        self.repaint()

        try:
            with _StdoutRedirector(log):
                result = refine_cell_source_points(
                    self.core,
                    cell_source,
                    self.selection_results["new_seq"],
                    center_yx=(
                        float(self.sel_cy_input.value()),
                        float(self.sel_cx_input.value()),
                    ),
                    radius=float(self.sel_r_input.value()),
                    max_distance=80.0,
                    cellpose_model=(
                        self.sel_cellpose_combo.currentText() or "cyto2"
                    ),
                    channel=(self.mda_seg_ch_combo.currentText() or "BF"),
                    stage_settle_time=0.5,
                    segmentation_scale=int(self.refine_scale_input.value()),
                )
            log.append(
                "\n--- refinement complete: "
                f"{result['moved']} moved, "
                f"{result['unmatched']} unmatched ---\n"
            )
            self.status.setText(
                f"Status: refined {result['matched']}/{result['total']} "
                f"cell point(s); {result['moved']} moved"
            )
        except Exception as e:
            log.append(f"\n--- refinement failed: {e} ---\n")
            self.status.setText(f"Status: point refinement failed -- {e}")

    def _click_center_cb(self, viewer, event):
        """napari mouse callback: fires on press while armed."""
        if event.button != 1:          # left click only
            return
        yx = np.array(event.position[-2:], dtype=float)
        # disarm BEFORE moving so a slow move can't eat a second click
        self.click_center_btn.setChecked(False)
        self._move_clicked_to_center(yx)

    def _move_clicked_to_center(self, yx):
        """Move the stage so the clicked pixel lands at (cy, cx)."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.vandermonde is None:
            self._load_vandermonde()
        if self.vandermonde is None:
            self.status.setText(
                "Status: no Vandermonde model loaded (set it in Loading)"
            )
            return
        try:
            C, degree = self.vandermonde
            cy = int(self.sel_cy_input.value())
            cx = int(self.sel_cx_input.value())
            offset_yx = yx - np.array([cy, cx], dtype=float)
            offset_xy = np.array([offset_yx[1], offset_yx[0]])
            stage_dx, stage_dy = apply_vandermonde_model(offset_xy, C, degree)
            x, y = self.core.getXYPosition()
            self.core.setXYPosition(float(x - stage_dx), float(y - stage_dy))
            self.core.waitForSystem()
            self.status.setText(
                f"Status: moved ({yx[0]:.0f},{yx[1]:.0f}) -> center "
                f"({cy},{cx})  [dX={-stage_dx:.2f}, dY={-stage_dy:.2f} um]"
            )
        except Exception as e:
            self.status.setText(f"Status: click-to-center failed -- {e}")

    def run_manual_selection(self):
        """Create empty point-source layers for hand-clicking, mirroring the
        automated selection's (sources, autofocus_p, new_seq) contract.

        N cells-per-FOV is fixed up front by the 'N per FOV' spinbox. In batch
        mode positions are repeated N times and you click N cells per FOV; in
        non-batch mode you click freely."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.default_engine is None:
            self.status.setText("Status: no default engine -- reconnect")
            return

        autofocus_object = self.sel_af_combo.currentText()
        N_per_fov = int(self.sel_npf_input.value())
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        batch = self.sel_batch_combo.currentText() == "True"

        log = LogWindow(title="Manual selection log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: setting up manual selection...")
        self.repaint()

        try:
            with _StdoutRedirector(log):
                self._prepare_for_selection()
                self.core.register_mda_engine(self.default_engine)
                point_transformer = self._make_point_transformer(sq_size, sq_n)
                sources, autofocus_p, new_seq = manual_point_selections(
                    self.core, self.viewer, self.main_window,
                    point_transformer,
                    N=N_per_fov,
                    autofocus_object=autofocus_object,
                    batch=batch,
                )

            self.selection_results = {
                "sources": sources,
                "autofocus_p": autofocus_p,
                "new_seq": new_seq,
            }
            hint = (
                f"click {N_per_fov} cell(s) per FOV"
                if batch else "click points freely"
            )
            log.append("\n--- manual layers ready ---\n")
            self.status.setText(
                f"Status: manual selection ready ({len(sources)} layer(s)) -- "
                f"{hint}, then Run Raman MDA"
            )
        except Exception as e:
            log.append(f"\n--- manual selection failed: {e} ---\n")
            self.status.setText(f"Status: manual selection failed -- {e}")

    def center_manual_cells(self):
        """Convert hand-clicked cells (from a non-batch manual selection)
        into one centered stage position per cell via the Vandermonde model."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.selection_results is None:
            self.status.setText(
                "Status: run Manual selection and click cells first"
            )
            return
        if self.sel_batch_combo.currentText() == "True":
            self.status.setText(
                "Status: centering requires a NON-batch manual selection"
            )
            return
        vandermonde_model_path = self.sel_vdm_path.text().strip()
        if not vandermonde_model_path:
            self.status.setText(
                "Status: set a Vandermonde model (.json) in Loading first"
            )
            return
        autofocus_object = self.sel_af_combo.currentText()
        cy = int(self.sel_cy_input.value())
        cx = int(self.sel_cx_input.value())
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        log = LogWindow(title="Center clicked cells log")
        log.show()
        self._plot_windows.append(log)
        self.status.setText("Status: centering clicked cells...")
        self.repaint()
        try:
            with _StdoutRedirector(log):
                point_transformer = self._make_point_transformer(sq_size, sq_n)
                sources, autofocus_p, new_seq = center_manual_selections(
                    self.core, self.viewer, self.main_window,
                    point_transformer,
                    sources=self.selection_results["sources"],
                    vandermonde_model_path=vandermonde_model_path,
                    autofocus_object=autofocus_object,
                    center=(cy, cx),
                    direct_pixel_to_stage=False,
                )
            self.selection_results = {
                "sources": sources,
                "autofocus_p": autofocus_p,
                "new_seq": new_seq,
                "autofocus_object": autofocus_object,
                "batch": False,
            }
            n_new = len(new_seq.stage_positions)
            log.append("\n--- centering complete ---\n")
            self.status.setText(
                f"Status: centered {n_new} cell position(s) -- "
                "then Run Raman MDA"
            )
        except Exception as e:
            log.append(f"\n--- centering failed: {e} ---\n")
            self.status.setText(f"Status: centering failed -- {e}")

    def run_grid_selection(self):
        """Build a grid of stage positions around the current stage XY, each
        carrying the same single fixed point (grid_point_selections). No
        autofocus, non-batch. Stores the (sources, autofocus_p, new_seq) that
        Run Raman MDA consumes."""
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.default_engine is None:
            self.status.setText("Status: no default engine -- reconnect")
            return

        fov_x = int(self.grid_fovx_input.value())
        fov_y = int(self.grid_fovy_input.value())
        x_range = float(self.grid_xrange_input.value())
        y_range = float(self.grid_yrange_input.value())
        x_step = float(self.grid_xstep_input.value())
        y_step = float(self.grid_ystep_input.value())
        repeats = int(self.grid_repeats_input.value())
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        autofocus_object = self.grid_af_combo.currentText()

        log = LogWindow(title="Stage grid log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: generating stage grid...")
        self.repaint()

        try:
            with _StdoutRedirector(log):
                self.core.register_mda_engine(self.default_engine)
                self._prepare_for_selection()
                point_transformer = self._make_point_transformer(sq_size, sq_n)
                sources, autofocus_p, new_seq = grid_point_selections(
                    self.core, self.viewer, self.main_window,
                    point_transformer,
                    fov_x=fov_x, fov_y=fov_y,
                    x_range=x_range, y_range=y_range,
                    x_step=x_step, y_step=y_step,
                    repeats=repeats,
                    use_blank_images=self.grid_blank_check.isChecked(),
                    autofocus_object=autofocus_object,
                    sequence=None,
                    block_mda=False,
                )

            self.selection_results = {
                "sources": sources,
                "autofocus_p": autofocus_p,
                "new_seq": new_seq,
                "autofocus_object": autofocus_object,
                "batch": False,
            }
            n_pos = len(autofocus_p)
            log.append("\n--- stage grid ready ---\n")
            self.status.setText(
                f"Status: stage grid ready ({n_pos} positions, "
                f"{repeats} pts each at ({fov_x},{fov_y})) -- then Run Raman MDA"
            )
        except Exception as e:
            log.append(f"\n--- stage grid failed: {e} ---\n")
            self.status.setText(f"Status: stage grid failed -- {e}")

    # -------- run raman MDA --------
    def run_raman_mda(self):
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.collector is None or self.transformer is None:
            self.status.setText(
                "Status: collector/transformer missing -- reconnect"
            )
            return
        if self.selection_results is None:
            self.status.setText(
                "Status: no selection results -- "
                "run automated selection first"
            )
            return

        sources = self.selection_results["sources"]
        autofocus_p = self.selection_results["autofocus_p"]
        new_seq = self.selection_results["new_seq"]
        n_pos = len(new_seq.stage_positions)
        afp_text = self.mda_afp_input.text().strip()
        imgp_text = self.mda_imgp_input.text().strip()
        try:
            autofocus_values, imaging_values = resolve_position_specs(
                afp_text,
                imgp_text,
                n_pos,
                autofocus_p,
            )
        except ValueError as e:
            self.status.setText(f"Status: {e}")
            return
        autofocus_p = np.asarray(autofocus_values, dtype=int)
        image_p = np.asarray(imaging_values, dtype=int)
        if afp_text and afp_text.lower() != "none":
            print(f"[autofocus_p] manual override: {autofocus_p.tolist()}")
        if imgp_text and imgp_text.lower() != "none":
            print(f"[image_p] manual override: {image_p.tolist()}")

        af_choice = self.selection_results.get(
            "autofocus_object", self.sel_af_combo.currentText()
        )
        autofocus_enabled = af_choice not in ("None", "none", "", None)
        autofocus_object = af_choice if autofocus_enabled else "laser"
        segment_and_track = self.mda_seg_track_check.isChecked()
        batch = self.selection_results.get(
            "batch", self.sel_batch_combo.currentText() == "True"
        )
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        if batch and self._make_point_transformer(sq_size, sq_n).multiplier < 2:
            self.status.setText(
                "Status: batch mode needs a pattern with >= 2 points "
                "(increase N)"
            )
            return

        out_dir = self.mda_dir_input.text().strip() or "data/run"
        os.makedirs(out_dir, exist_ok=True)
        raman_offset = float(self.mda_raman_off_input.value())
        af_range = float(self.mda_af_range_input.value())
        search_pts = int(self.mda_search_pts_input.value())
        fine_search_range = float(self.mda_fine_range_input.value())
        fine_search_pts = int(self.mda_fine_pts_input.value())
        total_exp = float(self.mda_exp_input.value())
        loops = int(self.mda_loops_input.value())
        interval = float(self.mda_interval_input.value())
        refocus_every = int(self.mda_refocus_input.value())
        segment_channel = self.mda_seg_ch_combo.currentText() or "BF"
        seg_scale = float(self.mda_seg_scale_input.value())
        cellpose_model = self.mda_seg_model_combo.currentText() or "cyto2"
        segment_crop = self.mda_seg_crop_combo.currentText() == "True"
        tracking_config = self.mda_track_cfg_input.text().strip() or "particle_config.json"
        cy = int(self.sel_cy_input.value())
        cx = int(self.sel_cx_input.value())
        circle_center=(cy, cx)
        circle_radius = int(self.sel_r_input.value())

        try:
            z_relative = self._parse_float_list(
                self.mda_zrel_input.text(), "Z relative"
            )
            raman_z_indices = self._parse_int_list(
                self.mda_rz_input.text(), "Raman z indices"
            )
        except ValueError as e:
            self.status.setText(f"Status: {e}")
            return

        extra_channels = []
        seen = set()
        for entry in self.mda_channel_rows:
            if not entry["combo"].isEnabled():
                continue
            ch = entry["combo"].currentText()
            if not ch or ch in seen:
                continue
            seen.add(ch)
            extra_channels.append((ch, float(entry["exp"].value())))

        log = LogWindow(title="Raman MDA log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: starting Raman MDA...")
        self.repaint()

        try:
            import datetime as _dt
            from useq import ZRelativePositions
            from raman_mda_engine import (
                RamanEngine, RamanTiffAndNumpyWriter,
            )
            try:
                img_x = int(self.core.getImageWidth())
                img_y = int(self.core.getImageHeight())
            except Exception:
                img_x, img_y = self._get_image_xy()

            with _StdoutRedirector(log):
                engine = RamanEngine(
                    spectra_collector=self.collector,
                    transformer=self.transformer,
                    batch=batch,
                    autofocus=autofocus_enabled,
                    autofocus_p=autofocus_p,
                    image_p=image_p,
                    autofocus_object=autofocus_object,
                    segment_and_track=segment_and_track,
                    scale=seg_scale,
                    segment_channel=segment_channel,
                    cellpose_model=cellpose_model,
                    segment_crop=segment_crop,
                    tracking_config=tracking_config,
                    raman_glass_offset=raman_offset,
                    autofocus_search_range=af_range,
                    search_pts=search_pts,
                    fine_search_range=fine_search_range,
                    fine_search_pts=fine_search_pts,
                    refocus_every=refocus_every,
                    image_x=img_x,
                    image_y=img_y,
                    skip_imaging_for_same_pos=True,
                    config_file=self.cfg_path.text().strip() or "test3.cfg",
                    circle_center=circle_center,
                    circle_radius=circle_radius,
                    shutter_device="Fluoshutter",
                )
                self._active_raman_engine = engine

                self.core.register_mda_engine(engine)

                self.mda_writer = RamanTiffAndNumpyWriter(out_dir)
                engine.aiming_sources = sources

                point_transformer = self._make_point_transformer(sq_size, sq_n)

                if batch:
                    final_seq = set_up_new_seq(
                        self.main_window, point_transformer, engine,
                        seq=new_seq, total_exposure=total_exp,
                        batch=batch, z_plan="middle",
                    )
                else:
                    final_seq = set_up_new_seq(
                        self.main_window, point_transformer, engine,
                        seq=new_seq,
                        total_exposure=(
                            total_exp * point_transformer.multiplier
                        ),
                        batch=batch, z_plan="middle",
                    )

                new_time_plan = final_seq.time_plan.replace(
                    loops=loops,
                    interval=_dt.timedelta(seconds=interval),
                )
                new_z_plan = ZRelativePositions(relative=z_relative)

                if final_seq.channels:
                    template = final_seq.channels[0]
                    extra_channel_objs = tuple(
                        template.replace(config=ch, exposure=ch_exp)
                        for ch, ch_exp in extra_channels
                    )
                    sequence_channels = final_seq.channels + extra_channel_objs
                else:
                    sequence_channels = final_seq.channels
                    if extra_channels:
                        print(
                            "[warn] no template channel in sequence; "
                            "can't add extra channels"
                        )

                final_seq = final_seq.replace(
                    axis_order=("t", "p", "c", "z"),
                    time_plan=new_time_plan,
                    z_plan=new_z_plan,
                    channels=sequence_channels,
                )

                if "raman" in final_seq.metadata:
                    final_seq.metadata["raman"]["z"] = raman_z_indices
                else:
                    print(
                        "[warn] final_seq.metadata has no 'raman' key; "
                        "skipping raman['z']"
                    )

                print(
                    f"Starting MDA: {loops} loops, interval={interval}s, "
                    f"z_rel={z_relative}, raman_z={raman_z_indices}, "
                    f"autofocus={autofocus_enabled} ({af_choice}), "
                    f"segment_and_track={segment_and_track}, "
                    f"search_pts={search_pts}, fine_range={fine_search_range}, "
                    f"fine_pts={fine_search_pts}, refocus_every={refocus_every}, "
                    f"image=({img_x}x{img_y}), "
                    f"extra channels={[ch for ch, _ in extra_channels]}, "
                    f"shutter={engine._shutter_device!r}"
                )
                print(
                    f"[debug] engine._autofocus={engine._autofocus}, "
                    f"engine._segment_and_track={engine._segment_and_track}"
                )
                self._pause_core_guard_for_raman_mda()
                self._raman_mda_thread = run_mda_with_notifications(
                    self.core,
                    final_seq,
                    on_error=lambda error: self._raman_mda_failed.emit(
                        f"{type(error).__name__}: {error}"
                    ),
                )

            log.append("\n--- MDA started ---\n")
            self.status.setText("Status: Raman MDA started OK")
        except Warning as warning:
            self._resume_core_guard_after_raman_mda()
            log.append(f"\n--- MDA warning: {warning} ---\n")
            self.status.setText(f"Status: MDA warning -- {warning}")
        except Exception as e:
            self._resume_core_guard_after_raman_mda()
            notify_exception(e, context="Raman MDA startup")
            log.append(f"\n--- MDA failed: {e} ---\n")
            self.status.setText(f"Status: MDA failed -- {e}")

    def _on_raman_mda_failed(self, message):
        """Display an acquisition-thread failure on the Qt main thread."""
        self.status.setText(f"Status: Raman MDA failed -- {message}")

    def _pause_core_guard_for_raman_mda(self):
        """Give RamanEngine sole ownership of retries for one MDA run."""
        guard = self.core_guard
        if guard is None or self._raman_mda_guard_paused:
            return

        def _resume_after_finish(*_args):
            self._resume_core_guard_after_raman_mda()

        events = self.core.mda.events
        events.sequenceFinished.connect(_resume_after_finish)
        self._raman_mda_finish_callback = _resume_after_finish
        guard.pause()
        self._raman_mda_guard_paused = True

    def _resume_core_guard_after_raman_mda(self):
        """Restore widget-level retries after Raman MDA finishes or fails."""
        callback = self._raman_mda_finish_callback
        self._raman_mda_finish_callback = None
        if callback is not None and self.core is not None:
            try:
                self.core.mda.events.sequenceFinished.disconnect(callback)
            except Exception:
                pass

        if self._raman_mda_guard_paused:
            self._raman_mda_guard_paused = False
            if self.core_guard is not None:
                self.core_guard.resume()

    def stop_raman_mda(self):
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        try:
            # cancel() requests a clean stop; the MDA finishes its current
            # event and then exits. For an immediate halt, also stop sequence
            # acq.
            self.core.mda.cancel()
            try:
                self.core.stopSequenceAcquisition()
            except Exception:
                pass
            self.status.setText("Status: stop requested OK")
        except Exception as e:
            self.status.setText(f"Status: stop failed -- {e}")

    def shutdown_hardware(self):
        """Release MMCore devices before the Napari process can linger."""
        if self._hardware_shutdown_started:
            return
        self._hardware_shutdown_started = True

        live_worker = self._live_raman_worker
        live_thread = self._live_raman_thread
        if live_worker is not None:
            live_worker.request_stop()
            sdk = getattr(self.collector, "_sdk", None)
            abort = getattr(sdk, "AbortAcquisition", None)
            if callable(abort):
                try:
                    abort()
                except Exception:
                    pass
            if live_thread is not None:
                live_thread.quit()
                live_thread.wait(5000)

        if self.core is None:
            return

        engine = self._active_raman_engine
        request_shutdown = getattr(engine, "request_shutdown", None)
        if callable(request_shutdown):
            try:
                request_shutdown()
            except Exception:
                pass

        print("[shutdown] stopping acquisition and releasing hardware...")
        result = shutdown_core_hardware(
            self.core,
            guard=self.core_guard,
            mda_thread=self._raman_mda_thread,
            join_timeout=5.0,
            release_delay=1.0,
        )
        self._resume_core_guard_after_raman_mda()

        if result.errors:
            for error in result.errors:
                print(f"[shutdown] {error}")
        if result.devices_unloaded:
            print("[shutdown] MMCore devices unloaded; COM ports released")


__all__ = ["HardwareWidget"]
