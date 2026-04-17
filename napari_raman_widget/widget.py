"""The main HardwareWidget: a dockable napari panel for the CNS Raman rig."""
import os
import time
import uuid

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from .log_window import LogWindow, _StdoutRedirector
from .plot_windows import (
    CalibrationPlotWindow, GridScanPlotWindow, ReferenceSpectraWindow,
    SpectrumWindow, DatasetViewerWindow,
)
from .ui_helpers import make_collapsible


class HardwareWidget(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
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
        self.mda_channel_rows = []
        self.mda_writer = None

        outer = QVBoxLayout()

        # ================= LOADING SECTION =================
        loading_box = make_collapsible("Loading", expanded=True)
        loading_layout = QVBoxLayout()

        loading_layout.addWidget(QLabel("Micro-Manager config (.cfg):"))
        cfg_row = QHBoxLayout()
        self.cfg_path = QLineEdit()
        self.cfg_path.setText("test3.cfg")
        self.cfg_path.setPlaceholderText("test3.cfg")
        cfg_browse = QPushButton("…")
        cfg_browse.setFixedWidth(30)
        cfg_browse.clicked.connect(self.browse_cfg)
        cfg_row.addWidget(self.cfg_path)
        cfg_row.addWidget(cfg_browse)
        loading_layout.addLayout(cfg_row)

        loading_layout.addWidget(QLabel("Transformer model (.json):"))
        tf_row = QHBoxLayout()
        self.tf_path = QLineEdit()
        self.tf_path.setPlaceholderText("model_2026-01-08.json")
        tf_browse = QPushButton("…")
        tf_browse.setFixedWidth(30)
        tf_browse.clicked.connect(self.browse_tf)
        tf_row.addWidget(self.tf_path)
        tf_row.addWidget(tf_browse)
        loading_layout.addLayout(tf_row)

        loading_layout.addWidget(
            QLabel("Output folder (optional, applied on connect):")
        )
        out_row = QHBoxLayout()
        self.out_path = QLineEdit()
        self.out_path.setPlaceholderText("(current directory)")
        out_browse = QPushButton("…")
        out_browse.setFixedWidth(30)
        out_browse.clicked.connect(self.browse_out)
        out_row.addWidget(self.out_path)
        out_row.addWidget(out_browse)
        loading_layout.addLayout(out_row)

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
        n_row.addWidget(QLabel("N (repeats, ≥2):"))
        self.n_input = QSpinBox()
        self.n_input.setRange(2, 1000)
        self.n_input.setValue(2)
        n_row.addWidget(self.n_input)
        raman_layout.addLayout(n_row)

        self.collect_btn = QPushButton("Collect spectra at last point")
        self.collect_btn.clicked.connect(self.collect_raman)
        raman_layout.addWidget(self.collect_btn)

        raman_box.setLayout(raman_layout)
        outer.addWidget(raman_box)

        # ================= LASER AIMING CALIBRATION SECTION =================
        calib_box = make_collapsible("Laser aiming calibration", expanded=False)
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

        calib_box.setLayout(calib_layout)
        outer.addWidget(calib_box)

        # ================= RECALIBRATION SECTION =================
        recal_box = make_collapsible("Recalibration", expanded=False)
        recal_layout = QVBoxLayout()

        recal_help = QLabel(
            "Opens manual selector on the last calibration dataset.\n"
            "Click points, Enter to advance, Backspace to go back,\n"
            "R to reset, N to mark as NaN. Close window when done,\n"
            "then click Save to write the new model."
        )
        recal_help.setWordWrap(True)
        recal_layout.addWidget(recal_help)

        model_name_row = QHBoxLayout()
        model_name_row.addWidget(QLabel("Model name:"))
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("model_2026-01-08")
        model_name_row.addWidget(self.model_name_input)
        recal_layout.addLayout(model_name_row)

        self.open_selector_btn = QPushButton("Open manual selector")
        self.open_selector_btn.clicked.connect(self.open_selector)
        recal_layout.addWidget(self.open_selector_btn)

        self.save_model_btn = QPushButton("Save recalibrated model")
        self.save_model_btn.clicked.connect(self.save_recalibration)
        recal_layout.addWidget(self.save_model_btn)

        recal_box.setLayout(recal_layout)
        outer.addWidget(recal_box)

        # ================= COLLECT REFERENCE SPECTRA SECTION =================
        ref_box = make_collapsible("Collect reference spectra", expanded=False)
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
        ref_range_row.addWidget(QLabel("Search range (±µm):"))
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
        scan_z_row.addWidget(QLabel("Z offset (µm):"))
        self.scan_z_input = QDoubleSpinBox()
        self.scan_z_input.setRange(-1000, 1000)
        self.scan_z_input.setValue(4)
        self.scan_z_input.setDecimals(2)
        self.scan_z_input.setSingleStep(0.5)
        scan_z_row.addWidget(self.scan_z_input)
        scan_layout.addLayout(scan_z_row)

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

        # ================= AUTOMATED CELL SELECTION SECTION =================
        sel_box = make_collapsible("Automated cell selection", expanded=False)
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

        self.add_mask_btn = QPushButton("Add mask")
        self.add_mask_btn.clicked.connect(self.add_mask)
        sel_layout.addWidget(self.add_mask_btn)

        sel_layout.addWidget(QLabel("Automated point selection:"))

        af_row = QHBoxLayout()
        af_row.addWidget(QLabel("Autofocus object:"))
        self.sel_af_combo = QComboBox()
        self.sel_af_combo.addItems(
            ["laser", "software", "quartz", "glass", "cell"]
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

        sq_size_row = QHBoxLayout()
        sq_size_row.addWidget(QLabel("Square size:"))
        self.sel_sqsize_input = QDoubleSpinBox()
        self.sel_sqsize_input.setRange(0.001, 10.0)
        self.sel_sqsize_input.setValue(0.002)
        self.sel_sqsize_input.setDecimals(4)
        self.sel_sqsize_input.setSingleStep(0.0005)
        sq_size_row.addWidget(self.sel_sqsize_input)
        sel_layout.addLayout(sq_size_row)

        sq_n_row = QHBoxLayout()
        sq_n_row.addWidget(QLabel("Square N (subpoints):"))
        self.sel_sqn_input = QSpinBox()
        self.sel_sqn_input.setRange(1, 100)
        self.sel_sqn_input.setValue(1)
        sq_n_row.addWidget(self.sel_sqn_input)
        sel_layout.addLayout(sq_n_row)

        bkd_row = QHBoxLayout()
        bkd_row.addWidget(QLabel("Background threshold:"))
        self.sel_bkd_input = QDoubleSpinBox()
        self.sel_bkd_input.setRange(0.0, 1_000_000.0)
        self.sel_bkd_input.setValue(80)
        self.sel_bkd_input.setDecimals(1)
        bkd_row.addWidget(self.sel_bkd_input)
        sel_layout.addLayout(bkd_row)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch:"))
        self.sel_batch_combo = QComboBox()
        self.sel_batch_combo.addItems(["False", "True"])
        batch_row.addWidget(self.sel_batch_combo)
        sel_layout.addLayout(batch_row)

        self.run_selection_btn = QPushButton("Run automated selection")
        self.run_selection_btn.clicked.connect(self.run_automated_selection)
        sel_layout.addWidget(self.run_selection_btn)

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

        raman_off_row = QHBoxLayout()
        raman_off_row.addWidget(QLabel("Raman glass offset (µm):"))
        self.mda_raman_off_input = QDoubleSpinBox()
        self.mda_raman_off_input.setRange(-1000, 1000)
        self.mda_raman_off_input.setValue(5.0)
        self.mda_raman_off_input.setDecimals(2)
        self.mda_raman_off_input.setSingleStep(0.1)
        raman_off_row.addWidget(self.mda_raman_off_input)
        mda_layout.addLayout(raman_off_row)

        af_range_row = QHBoxLayout()
        af_range_row.addWidget(QLabel("Autofocus search range:"))
        self.mda_af_range_input = QDoubleSpinBox()
        self.mda_af_range_input.setRange(0.1, 1000)
        self.mda_af_range_input.setValue(4.5)
        self.mda_af_range_input.setDecimals(2)
        self.mda_af_range_input.setSingleStep(0.5)
        af_range_row.addWidget(self.mda_af_range_input)
        mda_layout.addLayout(af_range_row)

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

        zrel_row = QHBoxLayout()
        zrel_row.addWidget(QLabel("Z relative (comma-sep µm):"))
        self.mda_zrel_input = QLineEdit()
        self.mda_zrel_input.setText("0, 3")
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

        self.gen_dataset_btn = QPushButton("Generate dataset")
        self.gen_dataset_btn.clicked.connect(self.generate_dataset)
        mda_layout.addWidget(self.gen_dataset_btn)

        mda_box.setLayout(mda_layout)
        outer.addWidget(mda_box)

        outer.addStretch()

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

    # -------- file pickers --------
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

    # -------- helpers --------
    def _get_image_xy(self):
        """Return (X_size, Y_size) from the first image layer in the viewer.
        Falls back to (1344, 1024) if no image layer is present."""
        from napari.layers import Image
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                shape = layer.data.shape
                Y, X = shape[-2], shape[-1]
                return int(X), int(Y)
        return 1344, 1024

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

        remove_btn = QPushButton("✕")
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

        remove_btn = QPushButton("✕")
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

    def _prepare_for_selection(self):
        """Stop live mode, set axis order to tpcz, set z plan to RangeAround."""
        try:
            self.core.stopSequenceAcquisition()
        except Exception as e:
            print(f"[live mode stop] {e}")

        if self.main_window is None:
            print("[mda setup] no main_window — can't configure MDA widget")
            return

        try:
            mda_dock = self.main_window._dock_widgets["MDA"]
            mda_settings = mda_dock.children()[4]
        except Exception as e:
            print(f"[mda setup] couldn't locate MDA widget: {e}")
            return

        try:
            from useq import ZRangeAround
            seq = mda_settings.value()
            new_seq = seq.replace(
                axis_order=("t", "p", "c", "z"),
                z_plan=ZRangeAround(range=0.0, step=1.0),
            )
            if hasattr(mda_settings, "setValue"):
                mda_settings.setValue(new_seq)
                print("[mda setup] axis_order=tpcz, z_plan=ZRangeAround ✓")
            else:
                print("[mda setup] no setValue method — can't push sequence back")
                print(
                    "[mda setup] available: "
                    f"{[m for m in dir(mda_settings) if 'value' in m.lower() or 'set' in m.lower()][:10]}"
                )
        except Exception as e:
            print(f"[mda setup] failed to update sequence: {e}")

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
                df, da, title=f"Dataset: {run_name}"
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

    # -------- loading actions --------
    def connect(self):
        self.status.setText("Status: connecting…")
        self.repaint()

        # cd to output folder before doing anything else — all subsequent
        # relative paths (model json, reference/, grid_scan_*.zarr, data/run,
        # etc.) will then land inside it.
        out = self.out_path.text().strip()
        if out:
            try:
                os.makedirs(out, exist_ok=True)
                os.chdir(out)
                print(f"[cwd] changed to {os.getcwd()}")
            except Exception as e:
                self.status.setText(
                    f"Status: couldn't cd to output folder — {e}"
                )
                return

        try:
            from pymmcore_plus import CMMCorePlus
            from raman_control.andor import AndorSpectraCollector
            from cns_control.coordtransformer import CoordTransformer

            self.core = CMMCorePlus.instance()

            try:
                self.core.unloadAllDevices()
                time.sleep(0.5)
            except Exception:
                pass

            cfg = self.cfg_path.text().strip()
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
            self.default_engine = self.core.mda.engine

            tf = self.tf_path.text().strip()
            if tf:
                self.transformer = CoordTransformer.from_json(tf)

            self._refresh_channel_combos()

            msg = "Status: connected ✓"
            if not cfg:
                msg += " (no cfg loaded)"
            if not tf:
                msg += " (no transformer)"
            self.status.setText(msg)
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.reload_tf_btn.setEnabled(True)
        except Exception as e:
            self.status.setText(f"Status: failed — {e}")

    def reload_transformer(self):
        tf = self.tf_path.text().strip()
        if not tf:
            self.status.setText("Status: no transformer path set")
            return
        try:
            from cns_control.coordtransformer import CoordTransformer
            self.transformer = CoordTransformer.from_json(tf)
            self.status.setText("Status: transformer reloaded ✓")
        except Exception as e:
            self.status.setText(f"Status: transformer reload failed — {e}")

    def disconnect(self):
        try:
            if self.core is not None:
                from cns_control.utils import unload
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
        self.status.setText("Status: disconnected")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.reload_tf_btn.setEnabled(False)

    # -------- raman collection --------
    def collect_raman(self):
        if self.collector is None or self.daq is None:
            self.status.setText("Status: not connected")
            return
        if self.transformer is None:
            self.status.setText("Status: no transformer loaded")
            return
        if len(self.viewer.layers) == 0:
            self.status.setText("Status: no layer to read point from")
            return

        exposure = float(self.exposure_input.value())
        N = int(self.n_input.value())

        try:
            self.daq.galvo.stop()
            self.daq.galvo.start()

            pt = self.viewer.layers[-1].data[0, -2:]
            volts = self._pt_to_volts(pt)
            spec = self.collector.collect_spectra_pts(
                np.tile(volts[0], (N, 1)), exposure
            )

            win = SpectrumWindow(spec, title="Raman spectra")
            win.show()
            self._plot_windows.append(win)

            self.status.setText(f"Status: collected {N}x{exposure:.0f}ms ✓")
        except Exception as e:
            self.status.setText(f"Status: collection failed — {e}")

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

        self.status.setText("Status: calibrating…")
        self.repaint()

        try:
            from cns_control.calibration import Calibrator

            self.calibrator = Calibrator(
                self.core, self.daq, self.transformer, self.collector,
                N=N, exp=exp, max_volts=max_volts,
            )
            with _StdoutRedirector(log):
                self.calibration_ds = self.calibrator.calibrate(
                    grid, thres=thres, plot=False
                )

            log.append("\n--- calibration complete ---\n")

            plot_win = CalibrationPlotWindow(
                self.calibration_ds, title="Calibration result"
            )
            plot_win.show()
            self._plot_windows.append(plot_win)

            self.status.setText("Status: calibration done ✓")
        except Exception as e:
            log.append(f"\n--- calibration failed: {e} ---\n")
            self.status.setText(f"Status: calibration failed — {e}")

    # -------- recalibration --------
    def open_selector(self):
        if self.calibration_ds is None:
            self.status.setText(
                "Status: no calibration dataset — run calibration first"
            )
            return
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            import matplotlib.pyplot as plt
            from cns_control.calibration import ManualImageSelector

            plt.ion()
            self.selector = ManualImageSelector(self.calibration_ds)
            plt.show()
            self.status.setText(
                "Status: selector open — click through, then save"
            )
        except Exception as e:
            self.status.setText(f"Status: selector failed — {e}")

    def save_recalibration(self):
        if self.selector is None:
            self.status.setText("Status: no selector — open it first")
            return
        if self.calibrator is None:
            self.status.setText(
                "Status: no calibrator — run calibration first"
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
            from cns_control.coordtransformer import CoordTransformer

            selected_points = self.selector.selected_points
            self.calibrator.save_new_model(
                self.calibration_ds, selected_points, model_name
            )
            self.transformer = CoordTransformer.from_json(f"{model_name}.json")
            self.tf_path.setText(f"{model_name}.json")
            self.status.setText(f"Status: saved & loaded {model_name}.json ✓")
        except Exception as e:
            self.status.setText(f"Status: save failed — {e}")

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

        self.status.setText("Status: collecting reference spectra…")
        self.repaint()

        log = LogWindow(title="Reference collection log")
        log.show()
        self._plot_windows.append(log)

        try:
            from cns_control.autofocus import autofocus_w_bkd

            pt = self.viewer.layers[-1].data[0, -2:]
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
                all_raman, zs, title=f"Reference spectra: {name}"
            )
            win.show()
            self._plot_windows.append(win)

            os.makedirs("reference", exist_ok=True)
            uid = str(uuid.uuid1())[:8]
            np.save(
                f"reference/reference_spec_zs_{name}_{uid}.npy", zs
            )
            np.save(
                f"reference/reference_spec_{int(exp)}s_{name}_{uid}.npy",
                all_raman,
            )
            np.save(
                f"reference/reference_spec_xy_{name}_{uid}.npy",
                np.asarray(pt),
            )

            log.append(f"\n--- saved to reference/*_{name}_{uid}.npy ---\n")
            self.status.setText(
                f"Status: reference saved "
                f"(reference/*_{name}_{uid}.npy) ✓"
            )
        except Exception as e:
            log.append(f"\n--- reference collection failed: {e} ---\n")
            self.status.setText(
                f"Status: reference collection failed — {e}"
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

        self.status.setText("Status: grid scanning…")
        self.repaint()

        try:
            import xarray as xr
            from datetime import datetime

            with _StdoutRedirector(log):
                shapes = self.viewer.layers[-1]
                try:
                    shape0 = shapes.data[0]
                except Exception:
                    raise RuntimeError(
                        "Last layer has no shape data — "
                        "draw a rectangle first."
                    )

                x_min = float(np.min(shape0[:, 0]))
                x_max = float(np.max(shape0[:, 0]))
                y_min = float(np.min(shape0[:, 1]))
                y_max = float(np.max(shape0[:, 1]))
                x = np.linspace(x_min, x_max, N)
                y = np.linspace(y_min, y_max, N)
                Xg, Yg = np.meshgrid(x, y)
                grid = np.column_stack([Xg.ravel(), Yg.ravel()])

                print(f"Grid: {N}x{N} = {grid.shape[0]} points")

                self.core.setConfig("Channel", "BF")
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
                self.core.setPosition(currentz - z_offset)
                self.core.setConfig("Channel", "RM")
                self.core.setShutterOpen("Fluoshutter", True)

                X_img, Y_img = self._get_image_xy()
                volts = self.transformer.BF_to_volts(
                    grid / np.array([X_img, Y_img]), max_volts=1.8
                )
                self.core.stopSequenceAcquisition()
                self.core.setExposure(1)
                print(
                    f"Collecting {grid.shape[0]} spectra at "
                    f"{exp:.0f} ms each…"
                )
                specs = self.collector.collect_spectra_pts(volts, exp)

                self.core.setShutterOpen("Fluoshutter", False)
                self.core.setPosition(currentz)
                self.core.setConfig("Channel", "BF")
                self.core.setExposure(10)
                end_BF = self.core.snap()

                data_vars = {
                    "laser_pos": xr.DataArray(volts, dims=("idx", "volt")),
                    "grid_pos": xr.DataArray(grid, dims=("idx", "volt")),
                    "specs": xr.DataArray(specs, dims=("N", "spec_dim")),
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

            win = GridScanPlotWindow(ds, title=f"Grid scan: {file_name}")
            win.show()
            self._plot_windows.append(win)

            self.status.setText(f"Status: grid scan saved → {zarr_name} ✓")
        except Exception as e:
            log.append(f"\n--- grid scan failed: {e} ---\n")
            self.status.setText(f"Status: grid scan failed — {e}")

    # -------- automated cell selection --------
    def add_mask(self):
        """Add the masked overlay to the viewer using current center/radius."""
        try:
            from cns_control.utils import add_mask_with_hole
        except Exception as e:
            self.status.setText(f"Status: import failed — {e}")
            return

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
                f"Status: mask added at ({cy},{cx}) r={r} ✓"
            )
        except Exception as e:
            self.status.setText(f"Status: add_mask failed — {e}")

    def run_automated_selection(self):
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.default_engine is None:
            self.status.setText("Status: no default engine — reconnect")
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

        log = LogWindow(title="Automated selection log")
        log.show()
        self._plot_windows.append(log)

        self.status.setText("Status: running automated selection…")
        self.repaint()

        try:
            from cns_control.utils import automated_point_selections
            from raman_mda_engine.aiming.transformers import Square

            with _StdoutRedirector(log):
                self._prepare_for_selection()
                self.core.register_mda_engine(self.default_engine)
                point_transformer = Square(sq_size, sq_n)
                sources, autofocus_p, new_seq = automated_point_selections(
                    self.core, self.viewer, self.main_window,
                    point_transformer,
                    N=N_per_fov + 1,
                    center=(cy, cx),
                    radius=r,
                    autofocus_object=autofocus_object,
                    bkd_thres=bkd_thres,
                    batch=batch,
                )

            self.selection_results = {
                "sources": sources,
                "autofocus_p": autofocus_p,
                "new_seq": new_seq,
            }
            log.append("\n--- selection complete ---\n")
            self.status.setText("Status: automated selection done ✓")
        except Exception as e:
            log.append(f"\n--- selection failed: {e} ---\n")
            self.status.setText(f"Status: selection failed — {e}")

    # -------- run raman MDA --------
    def run_raman_mda(self):
        if self.core is None:
            self.status.setText("Status: not connected")
            return
        if self.collector is None or self.transformer is None:
            self.status.setText(
                "Status: collector/transformer missing — reconnect"
            )
            return
        if self.selection_results is None:
            self.status.setText(
                "Status: no selection results — "
                "run automated selection first"
            )
            return

        sources = self.selection_results["sources"]
        autofocus_p = self.selection_results["autofocus_p"]
        new_seq = self.selection_results["new_seq"]

        autofocus_object = self.sel_af_combo.currentText()
        batch = self.sel_batch_combo.currentText() == "True"
        sq_size = float(self.sel_sqsize_input.value())
        sq_n = int(self.sel_sqn_input.value())
        if batch and sq_n < 2:
            self.status.setText(
                "Status: batch mode requires Square N >= 2 "
                "(DAQ needs at least 2 samples per channel)"
            )
            return

        out_dir = self.mda_dir_input.text().strip() or "data/run"
        os.makedirs(out_dir, exist_ok=True)
        raman_offset = float(self.mda_raman_off_input.value())
        af_range = float(self.mda_af_range_input.value())
        total_exp = float(self.mda_exp_input.value())
        loops = int(self.mda_loops_input.value())
        interval = float(self.mda_interval_input.value())

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

        self.status.setText("Status: starting Raman MDA…")
        self.repaint()

        try:
            import datetime as _dt
            from useq import ZRelativePositions
            from raman_mda_engine import (
                RamanEngine, RamanTiffAndNumpyWriter,
            )
            from raman_mda_engine.aiming.transformers import Square
            from cns_control.utils import set_up_new_seq

            with _StdoutRedirector(log):
                engine = RamanEngine(
                    spectra_collector=self.collector,
                    scale=2,
                    transformer=self.transformer,
                    batch=batch,
                    autofocus_p=autofocus_p,
                    autofocus_object=autofocus_object,
                    raman_glass_offset=raman_offset,
                    autofocus_search_range=40,
                    skip_imaging_for_same_pos=True,
                )
                engine._autofocus_search_range = af_range

                self.core.register_mda_engine(engine)

                self.mda_writer = RamanTiffAndNumpyWriter(out_dir)
                engine.aiming_sources = sources

                point_transformer = Square(sq_size, sq_n)

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
                else:
                    extra_channel_objs = ()
                    if extra_channels:
                        print(
                            "[warn] no template channel in sequence; "
                            "can't add extra channels"
                        )

                final_seq = final_seq.replace(
                    axis_order=("t", "p", "c", "z"),
                    time_plan=new_time_plan,
                    z_plan=new_z_plan,
                    channels=final_seq.channels + extra_channel_objs,
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
                    f"extra channels={[ch for ch, _ in extra_channels]}"
                )
                self.core.run_mda(final_seq)

            log.append("\n--- MDA started ---\n")
            self.status.setText("Status: Raman MDA started ✓")
        except Exception as e:
            log.append(f"\n--- MDA failed: {e} ---\n")
            self.status.setText(f"Status: MDA failed — {e}")

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
            self.status.setText("Status: stop requested ✓")
        except Exception as e:
            self.status.setText(f"Status: stop failed — {e}")