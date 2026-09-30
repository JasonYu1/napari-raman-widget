"""Exercise each widget's real, hardware-free sampling controls with Qt."""

import ast
import os
import unittest
from pathlib import Path
from types import MethodType

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_raman_widget.chat_panel import WIDGET_PARAMS, _set_widget_parameter
from napari_raman_widget.field_help import HELP
from napari_raman_widget.ui_helpers import make_collapsible


PACKAGE = Path(__file__).resolve().parents[1] / "napari_raman_widget"


class _SamplingOwner:
    pass


def _sampling_owner(filename):
    """Load the exact GUI-only methods, without initializing microscope code."""
    tree = ast.parse((PACKAGE / filename).read_text(encoding="utf-8"))
    method_names = {"_make_scan_sampling_controls", "_update_scan_sampling_fields"}
    methods = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    if {method.name for method in methods} != method_names:
        raise AssertionError(f"Sampling controls missing from {filename}")
    namespace = {
        "QComboBox": QComboBox, "QDoubleSpinBox": QDoubleSpinBox,
        "QHBoxLayout": QHBoxLayout, "QLabel": QLabel, "QSpinBox": QSpinBox,
        "QVBoxLayout": QVBoxLayout, "QWidget": QWidget,
    }
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(module, str(PACKAGE / filename), "exec"), namespace)
    owner = _SamplingOwner()
    for name in method_names:
        setattr(owner, name, MethodType(namespace[name], owner))
    return owner


class ScanSamplingControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_modes_defaults_bounds_and_conditional_rows_match_in_both_widgets(self):
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                owner = _sampling_owner(filename)
                controls = owner._make_scan_sampling_controls()
                try:
                    combo = owner.scan_sampling_mode_combo
                    self.assertEqual(combo.currentText(), "Total points")
                    self.assertEqual(combo.currentData(), "count")
                    self.assertEqual(owner.scan_total_points_input.value(), 400)
                    self.assertEqual(owner.scan_total_points_input.minimum(), 2)
                    self.assertEqual(owner.scan_total_points_input.maximum(), 250000)
                    self.assertEqual(owner.scan_spacing_input.value(), 10)
                    self.assertEqual(owner.scan_spacing_input.minimum(), 0.01)
                    self.assertEqual(owner.scan_spacing_input.maximum(), 100000)
                    self.assertEqual(owner.scan_spacing_input.suffix(), " px")
                    self.assertFalse(owner.scan_count_controls.isHidden())
                    self.assertFalse(owner.scan_count_hint.isHidden())
                    self.assertTrue(owner.scan_count_hint.wordWrap())
                    self.assertIn("approximate", owner.scan_count_hint.text())
                    self.assertIn("Uniform X/Y spacing", owner.scan_count_hint.text())
                    self.assertTrue(owner.scan_spacing_controls.isHidden())
                    self.assertFalse(hasattr(owner, "scan_n_input"))
                    labels = [label.text() for label in controls.findChildren(QLabel)]
                    self.assertIn("Target points per Z plane:", labels)

                    combo.setCurrentText("Pixel spacing")
                    self.assertEqual(combo.currentData(), "spacing")
                    self.assertTrue(owner.scan_count_controls.isHidden())
                    self.assertTrue(owner.scan_count_hint.isHidden())
                    self.assertFalse(owner.scan_spacing_controls.isHidden())
                    combo.setCurrentText("Total points")
                    self.assertFalse(owner.scan_count_controls.isHidden())
                    self.assertFalse(owner.scan_count_hint.isHidden())
                    self.assertTrue(owner.scan_spacing_controls.isHidden())
                finally:
                    controls.close()

    def test_collapse_and_assistant_setting_changes_preserve_chosen_mode(self):
        params = {param["name"]: param for param in WIDGET_PARAMS}
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                owner = _sampling_owner(filename)
                controls = owner._make_scan_sampling_controls()
                group = make_collapsible("Spatial mapping", expanded=True)
                layout = QVBoxLayout()
                layout.addWidget(controls)
                group.setLayout(layout)
                try:
                    group.show()
                    self.app.processEvents()
                    _set_widget_parameter(owner, params["scan_total_points"], 120)
                    _set_widget_parameter(owner, params["scan_spacing_px"], 3.5)
                    _set_widget_parameter(owner, params["scan_sampling_mode"], "Pixel spacing")
                    group.setChecked(False)
                    group.setChecked(True)
                    self.app.processEvents()
                    self.assertEqual(owner.scan_total_points_input.value(), 120)
                    self.assertEqual(owner.scan_spacing_input.value(), 3.5)
                    self.assertTrue(owner.scan_spacing_controls.isVisible())
                    self.assertFalse(owner.scan_count_controls.isVisible())
                    self.assertFalse(owner.scan_count_hint.isVisible())
                    _set_widget_parameter(owner, params["scan_sampling_mode"], "Total points")
                    self.assertTrue(owner.scan_count_controls.isVisible())
                    self.assertTrue(owner.scan_count_hint.isVisible())
                    self.assertFalse(owner.scan_spacing_controls.isVisible())
                finally:
                    group.close()

    def test_help_explains_approximate_target_and_uniform_spacing_in_both_modes(self):
        self.assertIn("Target number", HELP["scan_total_points_input"])
        self.assertIn("not a grid side", HELP["scan_total_points_input"])
        self.assertIn("actual count is approximate", HELP["scan_total_points_input"])
        self.assertIn("Uniform X/Y spacing", HELP["scan_total_points_input"])
        self.assertIn("Both modes", HELP["scan_sampling_mode_combo"])
        self.assertNotIn("exactly", HELP["scan_sampling_mode_combo"])
        self.assertIn("square sampling lattice", HELP["scan_spacing_input"])
        self.assertIn("at least two", HELP["scan_spacing_input"])
        self.assertIn("ellipse", HELP["scan_btn"])
        self.assertNotIn("scan_n_input", HELP)


if __name__ == "__main__":
    unittest.main()
