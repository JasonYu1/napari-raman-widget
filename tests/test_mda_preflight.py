"""MDA review counts, validation, and the no-start-on-cancel boundary."""

import ast
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QDialog, QLabel, QWidget

from napari_raman_widget.mda_preflight import (
    MdaReviewDialog, build_mda_review, confirm_raman_mda,
)
from napari_raman_widget.position_specs import resolve_position_specs


class Control:
    def __init__(self, value):
        self.data = value

    def text(self):
        return str(self.data)

    currentText = text

    def value(self):
        return self.data

    def isChecked(self):
        return bool(self.data)

    def isEnabled(self):
        return True


class MdaPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # GitHub's Windows runners can return 8.3 paths (e.g. RUNNER~1).
        # Match the canonical path displayed by the pre-run review.
        self.root = Path(temporary.name).resolve()
        self.owner = QWidget()
        self.addCleanup(self.owner.deleteLater)
        owner = self.owner
        owner.status = QLabel(owner)
        owner.core = SimpleNamespace(mda=SimpleNamespace(is_running=lambda: False),
                                     isSequenceRunning=lambda: False)
        owner.collector = object()
        owner.transformer = object()
        self.source = SimpleNamespace(
            name="cells", _pos_idx=1,
            _points=SimpleNamespace(data=np.array([
                [0, 0, 0, 0, 10, 20], [0, 0, 0, 0, 30, 40],
                [0, 1, 0, 0, 50, 60],
            ], dtype=float)),
            transformer=SimpleNamespace(multiplier=4),
        )
        self.seq = SimpleNamespace(stage_positions=[object(), object()],
                                   time_plan=object(), channels=[SimpleNamespace(config="BF")])
        owner.selection_results = dict(sources=[self.source], new_seq=self.seq,
                                       autofocus_p=[0, 1], autofocus_object="laser", batch=False)
        values = dict(
            mda_dir_input=str(self.root / "new-run"), sel_batch_combo="False",
            sel_af_combo="None", mda_afp_input="", mda_imgp_input="",
            mda_seg_track_check=False, mda_auto_add_cells_check=False,
            mda_refocus_input=1, mda_exp_input=1000, mda_loops_input=3,
            mda_interval_input=60, mda_raman_off_input=5,
            mda_zrel_input="0, 4, 8", mda_rz_input="0, 2",
        )
        for name, value in values.items():
            setattr(owner, name, Control(value))
        owner.mda_channel_rows = []

    def review(self):
        return build_mda_review(self.owner)

    def test_summary_counts_patterns_z_planes_and_scheduled_intervals(self):
        review = self.review()
        self.assertEqual(review.errors, ())
        rows = dict(review.rows)
        self.assertEqual(rows["Stage positions"], "2")
        self.assertIn("3 (current selection)", rows["Selected cells / target anchors"])
        self.assertIn("72", rows["Planned cell Raman collections"])
        # 3 anchors * 4 points * 2 Raman planes * 1 s = 24 s per loop.
        self.assertEqual(review.minimum_seconds, 144)
        self.assertIn("laser, 2 positions", rows["Autofocus"])
        self.assertEqual(rows["Output folder"], str(self.root / "new-run"))
        self.assertIn("not a completion-time prediction", review.text)

    def test_batch_counts_one_integrated_collection_per_anchor(self):
        self.owner.selection_results["batch"] = True
        review = self.review()
        self.assertFalse(review.errors)
        self.assertIn("18", dict(review.rows)["Planned cell Raman collections"])
        self.assertEqual(review.minimum_seconds, 126)

    def test_single_timepoint_does_not_add_interval(self):
        self.owner.mda_loops_input.data = 1
        self.assertEqual(self.review().minimum_seconds, 24)

    def test_short_and_zero_intervals_use_exposure_floor(self):
        self.owner.mda_interval_input.data = 10
        review = self.review()
        self.assertEqual(review.minimum_seconds, 72)
        self.assertIn("behind schedule", review.text)
        self.owner.mda_interval_input.data = 0
        self.assertEqual(self.review().minimum_seconds, 72)

    def test_missing_connection_selection_and_channels_are_blocking(self):
        self.owner.core = self.owner.collector = self.owner.transformer = None
        self.owner.selection_results = None
        review = self.review()
        self.assertIn("Not connected", review.text)
        self.assertIn("Collector or coordinate transformer", review.text)
        self.assertIn("No complete prepared", review.text)
        self.assertIsNone(review.minimum_seconds)
        self.owner.selection_results = dict(sources=[self.source], new_seq=self.seq,
                                            autofocus_p=[0, 1])
        self.seq.channels = []
        self.seq.time_plan = None
        self.assertIn("needs a BF channel", self.review().text)
        self.assertIn("no time plan", self.review().text)

    def test_invalid_z_values_and_indices_are_blocking(self):
        for name, value in (("mda_zrel_input", "nan"), ("mda_zrel_input", ""),
                            ("mda_rz_input", "3"), ("mda_rz_input", "-1"),
                            ("mda_rz_input", "0.5")):
            with self.subTest(name=name, value=value):
                control = getattr(self.owner, name)
                old = control.data
                control.data = value
                self.assertTrue(self.review().errors)
                control.data = old

    def test_minimum_exposure_and_invalid_position_overrides(self):
        self.owner.mda_exp_input.data = 20
        self.owner.mda_afp_input.data = "0, 9"
        self.assertIn("73.8", self.review().text)
        self.assertIn("out of range", self.review().text)

    def test_empty_invalid_or_stale_cell_sources_are_blocking(self):
        for data in (np.empty((0, 6)), np.array([[0, 8, 0, 0, 1, 2]]),
                     np.array([[0, 0, 0, 0, float("nan"), 2]])):
            with self.subTest(data=data):
                self.source._points.data = data
                self.assertTrue(self.review().errors)

    def test_batch_tracking_incompatibility_and_conditional_warning(self):
        self.owner.selection_results["batch"] = True
        self.owner.mda_seg_track_check.data = True
        self.owner.mda_auto_add_cells_check.data = True
        self.assertIn("not supported", self.review().text)
        self.assertIn("Tracking can change", self.review().text)
        self.owner.mda_seg_track_check.data = False
        self.assertFalse(self.review().errors)

    def test_live_and_other_acquisitions_block_start(self):
        self.owner._acquisition_jobs = SimpleNamespace(is_running=True)
        self.owner._live_raman_worker = object()
        self.owner.core.mda.is_running = lambda: True
        self.owner.core.isSequenceRunning = lambda: True
        text = self.review().text
        for expected in ("Another acquisition", "live Raman", "already running", "live camera"):
            self.assertIn(expected, text)

    def test_output_file_and_file_parent_are_blocking(self):
        file = self.root / "existing.txt"
        file.touch()
        for output in (file, file / "run"):
            self.owner.mda_dir_input.data = str(output)
            self.assertTrue(self.review().errors)

    def test_nonempty_output_warning_and_no_preflight_writes(self):
        self.owner.mda_dir_input.data = str(self.root)
        (self.root / "data.txt").touch()
        self.assertIn("not empty", self.review().text)
        self.owner.mda_dir_input.data = str(self.root / "new-run")
        with patch.object(Path, "mkdir") as mkdir, patch("os.makedirs") as makedirs:
            self.review()
        mkdir.assert_not_called()
        makedirs.assert_not_called()
        self.assertFalse((self.root / "new-run").exists())

    @unittest.skipUnless(os.name == "nt", "Windows short-path regression")
    def test_short_windows_output_path_is_resolved_without_creating_folder(self):
        import ctypes

        get_short_path = ctypes.windll.kernel32.GetShortPathNameW
        get_short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        get_short_path.restype = ctypes.c_uint
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_short_path(str(self.root), buffer, len(buffer))
        if not length or length >= len(buffer):
            self.skipTest("No short path available for the temporary directory")
        if os.path.normcase(buffer.value) == os.path.normcase(str(self.root)):
            self.skipTest("8.3 names are not enabled on this volume")
        self.owner.mda_dir_input.data = str(Path(buffer.value) / "new-run")

        review = self.review()

        self.assertFalse(review.errors)
        self.assertEqual(dict(review.rows)["Output folder"], str(self.root / "new-run"))
        self.assertFalse((self.root / "new-run").exists())

    def test_demo_explains_ignored_extra_channels(self):
        self.owner.mda_channel_rows = [{"combo": Control("GFP"), "exp": Control(50)}]
        review = build_mda_review(self.owner, demo=True)
        self.assertIn("simulated acquisition", review.text)
        self.assertIn("extra fluorescence channels are not acquired", review.text)

    def test_dialog_defaults_to_back_and_blocks_start_for_errors(self):
        dialog = MdaReviewDialog(self.review())
        self.addCleanup(dialog.deleteLater)
        dialog.show()
        self.app.processEvents()
        self.assertTrue(dialog.start_button.isEnabled())
        self.assertTrue(dialog.back_button.isDefault())
        QTest.keyClick(dialog.back_button, Qt.Key_Return)
        self.assertEqual(dialog.result(), QDialog.Rejected)
        self.owner.collector = None
        invalid = MdaReviewDialog(self.review())
        self.addCleanup(invalid.deleteLater)
        self.assertFalse(invalid.start_button.isEnabled())

    def test_only_explicit_acceptance_passes_confirmation(self):
        with patch.object(MdaReviewDialog, "exec_", return_value=QDialog.Rejected):
            self.assertFalse(confirm_raman_mda(self.owner))
        self.assertIn("cancelled", self.owner.status.text())
        with patch.object(MdaReviewDialog, "exec_", return_value=QDialog.Accepted):
            self.assertTrue(confirm_raman_mda(self.owner))
        self.assertIn("Stage positions: 2", self.owner._confirmed_mda_summary)
        self.assertFalse(self.owner._mda_review_open)
        self.owner.collector = None
        with patch.object(MdaReviewDialog, "exec_", return_value=QDialog.Accepted):
            self.assertFalse(confirm_raman_mda(self.owner))

    def test_background_change_during_review_requires_another_review(self):
        def change():
            self.owner.core.mda.is_running = lambda: True
            return QDialog.Accepted
        with patch.object(MdaReviewDialog, "exec_", side_effect=change):
            self.assertFalse(confirm_raman_mda(self.owner))
        self.assertIn("changed during review", self.owner.status.text())

    def test_point_move_during_review_requires_another_review(self):
        def change():
            self.source._points.data[0, -1] += 10
            return QDialog.Accepted
        with patch.object(MdaReviewDialog, "exec_", side_effect=change):
            self.assertFalse(confirm_raman_mda(self.owner))
        self.assertIn("changed during review", self.owner.status.text())

    def test_demo_live_timer_blocks_start_without_camera_sequence(self):
        self.owner.demo_live_timer = SimpleNamespace(isActive=lambda: True)
        self.assertIn("Turn off demo live", self.review().text)

    def test_hardware_and_demo_run_methods_return_before_any_start_on_cancel(self):
        package = Path(__file__).resolve().parents[1] / "napari_raman_widget"
        for filename, demo in (("hardware_widget.py", False), ("demo_widget.py", True)):
            with self.subTest(filename=filename):
                tree = ast.parse((package / filename).read_text(encoding="utf-8"))
                method = next(node for node in ast.walk(tree)
                              if isinstance(node, ast.FunctionDef) and node.name == "run_raman_mda")
                confirm = Mock(return_value=False)
                namespace = {"confirm_raman_mda": confirm}
                exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
                # No core or other attributes: touching them would fail.
                sentinel = object()
                namespace["run_raman_mda"](sentinel)
                confirm.assert_called_once_with(sentinel, **({"demo": True} if demo else {}))

    def test_accepted_review_reaches_existing_run_path_and_logs_summary(self):
        package = Path(__file__).resolve().parents[1] / "napari_raman_widget"
        owner = self.owner
        values = dict(sel_sqsize_input=3, sel_sqn_input=2,
                      mda_suppress_pillars_check=False, mda_af_range_input=6,
                      mda_search_pts_input=8, mda_fine_range_input=1.5,
                      mda_fine_pts_input=8, mda_seg_ch_combo="BF",
                      mda_seg_scale_input=2, mda_seg_model_combo="cyto2",
                      mda_seg_crop_combo="True", mda_track_cfg_input="particle_config.json",
                      sel_cy_input=50, sel_cx_input=50, sel_r_input=40,
                      cfg_path="test.cfg")
        for name, value in values.items():
            setattr(owner, name, Control(value))
        owner.main_window = None
        owner._show_plot = Mock()
        owner._make_point_transformer = Mock(return_value=SimpleNamespace(multiplier=4))
        owner._parse_float_list = lambda text, label: [float(v) for v in text.split(",")]
        owner._parse_int_list = lambda text, label: [int(v) for v in text.split(",")]
        owner._pause_core_guard_for_raman_mda = Mock()
        owner._resume_core_guard_after_raman_mda = Mock()
        owner.core.getImageWidth = lambda: 100
        owner.core.getImageHeight = lambda: 100
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                owner._raman_mda_thread = None
                owner.core.register_mda_engine = Mock()
                owner.core.run_mda = Mock()
                engine = SimpleNamespace(_shutter_device=None, _autofocus=True,
                                         _segment_and_track=False)
                engine_type = Mock(return_value=engine)
                final_seq = SimpleNamespace(
                    time_plan=SimpleNamespace(replace=Mock()),
                    channels=(SimpleNamespace(config="BF"),), metadata={"raman": {}},
                )
                final_seq.replace = Mock(return_value=final_seq)
                log = Mock()
                run = Mock()
                setup = Mock(return_value=final_seq)
                namespace = dict(
                    confirm_raman_mda=confirm_raman_mda, np=np, os=os,
                    resolve_position_specs=resolve_position_specs,
                    LogWindow=Mock(return_value=log),
                    make_live_cell_engine_type=lambda cls: cls,
                    _StdoutRedirector=lambda log: nullcontext(),
                    pillar_suppression_kwargs=lambda *args: {},
                    set_up_new_seq=setup, run_mda_with_notifications=run,
                    notify_exception=Mock(),
                )
                tree = ast.parse((package / filename).read_text(encoding="utf-8"))
                method = next(node for node in ast.walk(tree)
                              if isinstance(node, ast.FunctionDef) and node.name == "run_raman_mda")
                exec(compile(ast.Module(body=[method], type_ignores=[]), filename, "exec"), namespace)
                modules = {
                    "raman_mda_engine": SimpleNamespace(RamanEngine=engine_type,
                                                       RamanTiffAndNumpyWriter=Mock()),
                    "useq": SimpleNamespace(ZRelativePositions=lambda **kw: kw,
                                            Channel=lambda **kw: SimpleNamespace(**kw)),
                }
                with patch.dict("sys.modules", modules), patch("os.makedirs") as mkdir, \
                        patch.object(MdaReviewDialog, "exec_", return_value=QDialog.Accepted):
                    namespace["run_raman_mda"](owner)
                self.assertIn("started OK", owner.status.text())
                owner.core.register_mda_engine.assert_called_once_with(engine)
                mkdir.assert_called_once()
                self.assertIn("Stage positions: 2", log.append.call_args_list[0].args[0])
                self.assertEqual(setup.call_args.kwargs["total_exposure"], 4000)
                if filename == "hardware_widget.py":
                    run.assert_called_once()
                else:
                    owner.core.run_mda.assert_called_once_with(final_seq)


if __name__ == "__main__":
    unittest.main()
