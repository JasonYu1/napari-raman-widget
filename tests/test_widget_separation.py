"""Architecture checks for independent hardware and demo widgets."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "napari_raman_widget"


class WidgetSeparationTests(unittest.TestCase):
    def test_refine_cell_points_button_exists_in_both_widgets(self) -> None:
        expected = 'QPushButton(\n            "Refine cell points to centers"'
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = (PACKAGE / filename).read_text(encoding="utf-8")
                self.assertIn(expected, source)
                self.assertIn("self.refine_scale_input.setValue(4)", source)

    def test_image_rescale_minimum_is_one_in_both_widgets(self) -> None:
        expected = "self.mda_seg_scale_input.setRange(1.0, 100.0)"
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = (PACKAGE / filename).read_text(encoding="utf-8")
                self.assertIn(expected, source)

    def test_grid_fov_defaults_are_separate(self) -> None:
        hardware = (PACKAGE / "hardware_widget.py").read_text(encoding="utf-8")
        demo = (PACKAGE / "demo_widget.py").read_text(encoding="utf-8")
        self.assertIn("self.grid_fovx_input.setValue(740)", hardware)
        self.assertIn("self.grid_fovy_input.setValue(540)", hardware)
        self.assertIn("self.grid_fovx_input.setValue(256)", demo)
        self.assertIn("self.grid_fovy_input.setValue(256)", demo)

    def test_hardware_module_has_no_demo_mode_branches(self) -> None:
        source = (PACKAGE / "hardware_widget.py").read_text(encoding="utf-8")
        self.assertNotIn("demo_backend", source)
        self.assertNotIn("create_demo_backend", source)
        self.assertNotIn("DemoWorld", source)

    def test_hardware_mda_temporarily_pauses_the_core_guard(self) -> None:
        hardware = (PACKAGE / "hardware_widget.py").read_text(encoding="utf-8")
        self.assertIn("self._pause_core_guard_for_raman_mda()", hardware)
        self.assertIn("events.sequenceFinished.connect", hardware)
        self.assertIn("self._resume_core_guard_after_raman_mda()", hardware)

    def test_example_config_assigns_the_default_xy_stage(self) -> None:
        config = (
            PACKAGE.parent
            / "example_configs"
            / "microscope"
            / "example_micromanager.cfg"
        ).read_text(encoding="utf-8")
        self.assertIn("Property,Core,XYStage,XYStage", config)

    def test_demo_widget_is_a_standalone_qwidget(self) -> None:
        source = (PACKAGE / "demo_widget.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        demo_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DemoWidget"
        )
        self.assertEqual([ast.unparse(base) for base in demo_class.bases], ["QWidget"])
        self.assertNotIn("HardwareWidget", source)
        self.assertNotIn("AndorSpectraCollector", source)

    def test_old_widget_module_is_only_a_hardware_compatibility_import(self) -> None:
        source = (PACKAGE / "widget.py").read_text(encoding="utf-8")
        self.assertIn("from .hardware_widget import HardwareWidget", source)
        self.assertNotIn("DemoWidget", source)


if __name__ == "__main__":
    unittest.main()
