"""Exercise the real MDA layout without initializing hardware or models."""

import ast
import os
from pathlib import Path
import textwrap
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtWidgets
from qtpy.QtCore import QCoreApplication, QEvent

from napari_raman_widget.ui_helpers import make_collapsible


PACKAGE = Path(__file__).resolve().parents[1] / "napari_raman_widget"
WIDGETS = ("hardware_widget.py", "demo_widget.py")
STANDARD = (
    "mda_dir_input", "mda_raman_off_input", "mda_exp_input",
    "mda_loops_input", "mda_interval_input", "mda_zrel_input",
    "mda_rz_input", "mda_add_channel_btn", "run_mda_btn", "stop_mda_btn",
)
ADVANCED = (
    "mda_afp_input", "mda_imgp_input", "mda_af_range_input",
    "mda_search_pts_input", "mda_fine_range_input", "mda_fine_pts_input",
    "mda_seg_track_check", "mda_refocus_input",
)
TRACKING = (
    "mda_auto_add_cells_check", "mda_suppress_pillars_check",
    "mda_seg_ch_combo", "mda_seg_scale_input", "mda_seg_model_combo",
    "mda_seg_crop_combo", "mda_track_cfg_input", "_seg_track_cfg_browse",
)


def make_mda_owner(filename):
    source = (PACKAGE / filename).read_text(encoding="utf-8")
    owner = QtWidgets.QWidget()
    owner.sel_af_combo = QtWidgets.QComboBox(owner)
    owner.sel_af_combo.addItems(["laser", "None", "image"])
    namespace = {name: getattr(QtWidgets, name) for name in (
        "QVBoxLayout", "QHBoxLayout", "QLabel", "QLineEdit",
        "QDoubleSpinBox", "QSpinBox", "QCheckBox", "QComboBox", "QPushButton",
    )}
    namespace.update(self=owner, make_collapsible=make_collapsible)
    methods = [node for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.FunctionDef) and node.name in {
                   "_toggle_autofocus_fields", "_toggle_seg_track_fields",
               }]
    exec(compile(ast.Module(body=methods, type_ignores=[]), filename, "exec"),
         namespace)
    for method in methods:
        setattr(owner, method.name, MethodType(namespace[method.name], owner))
    for name in ("browse_tracking_cfg", "_add_mda_channel_row",
                 "run_raman_mda", "stop_raman_mda"):
        setattr(owner, name, Mock())
    start = source.index('        mda_box = make_collapsible("Run Raman MDA"')
    end = source.index("        # ================= DATASET TOOLS", start)
    # Avoid importing Cellpose (and loading its ML dependencies) for a UI test.
    with patch.dict("sys.modules", {
        "cellpose": SimpleNamespace(models=SimpleNamespace(MODEL_NAMES=["cyto2"]))
    }):
        exec(compile(textwrap.dedent(source[start:end]), filename, "exec"),
             namespace)
    layout = QtWidgets.QVBoxLayout(owner)
    layout.addWidget(owner.mda_box)
    owner.mda_box.setChecked(True)
    owner.show()
    QtWidgets.QApplication.processEvents()
    return owner


class MdaAdvancedUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def owner(self, filename):
        owner = make_mda_owner(filename)
        self.addCleanup(owner.deleteLater)
        self.addCleanup(owner.close)
        return owner

    def tearDown(self):
        self.doCleanups()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_default_collapsed_keeps_standard_controls_visible(self):
        for filename in WIDGETS:
            with self.subTest(filename=filename):
                owner = self.owner(filename)
                advanced = owner.mda_advanced_box
                self.assertEqual(advanced.title(), "Advanced")
                self.assertFalse(advanced.isChecked())
                for name in STANDARD:
                    control = getattr(owner, name)
                    self.assertTrue(control.isVisible(), name)
                    self.assertFalse(advanced.isAncestorOf(control), name)
                for name in ADVANCED + TRACKING:
                    control = getattr(owner, name)
                    self.assertFalse(control.isVisible(), name)
                    self.assertTrue(advanced.isAncestorOf(control), name)
                owner.run_mda_btn.click()
                owner.stop_mda_btn.click()
                owner.run_raman_mda.assert_called_once_with(False)
                owner.stop_raman_mda.assert_called_once_with(False)

    def test_expand_and_collapse_preserve_values_and_tracking_visibility(self):
        for filename in WIDGETS:
            with self.subTest(filename=filename):
                owner = self.owner(filename)
                advanced = owner.mda_advanced_box
                advanced.setChecked(True)
                for name in ADVANCED:
                    self.assertTrue(getattr(owner, name).isVisible(), name)
                for name in TRACKING:
                    self.assertFalse(getattr(owner, name).isVisible(), name)
                owner.mda_af_range_input.setValue(12.5)
                owner.mda_search_pts_input.setValue(15)
                owner.mda_afp_input.setText("0, 3")
                owner.mda_imgp_input.setText("2")
                owner.mda_refocus_input.setValue(4)
                owner.mda_seg_track_check.setChecked(True)
                for name in TRACKING:
                    self.assertTrue(getattr(owner, name).isVisible(), name)
                advanced.setChecked(False)
                for name in ADVANCED + TRACKING:
                    self.assertFalse(getattr(owner, name).isVisible(), name)
                advanced.setChecked(True)
                self.assertEqual(owner.mda_af_range_input.value(), 12.5)
                self.assertEqual(owner.mda_search_pts_input.value(), 15)
                self.assertEqual(owner.mda_afp_input.text(), "0, 3")
                self.assertEqual(owner.mda_imgp_input.text(), "2")
                self.assertEqual(owner.mda_refocus_input.value(), 4)
                self.assertTrue(owner.mda_seg_track_check.isChecked())
                for name in TRACKING:
                    self.assertTrue(getattr(owner, name).isVisible(), name)

    def test_autofocus_changes_while_collapsed_keep_conditional_fields(self):
        for filename in WIDGETS:
            with self.subTest(filename=filename):
                owner = self.owner(filename)
                for method in ("None", "image", "laser"):
                    owner.mda_advanced_box.setChecked(False)
                    owner.sel_af_combo.setCurrentText(method)
                    self.assertFalse(owner.mda_af_range_input.isVisible())
                    owner.mda_advanced_box.setChecked(True)
                    self.assertEqual(owner.mda_af_range_input.isVisible(),
                                     method != "None")
                    self.assertEqual(owner.mda_fine_range_input.isVisible(),
                                     method == "laser")


if __name__ == "__main__":
    unittest.main()
