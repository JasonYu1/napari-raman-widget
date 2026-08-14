"""Architecture checks for independent hardware and demo widgets."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "napari_raman_widget"


class WidgetSeparationTests(unittest.TestCase):
    def test_spectral_calibration_has_no_redundant_load_button(self) -> None:
        source = (PACKAGE / "spectral_calibration_ui.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('QPushButton("Load")', source)
        self.assertIn("owner.load_spectral_calibration()", source)

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

    def test_detector_read_mode_controls_exist_in_both_widgets(self) -> None:
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = (PACKAGE / filename).read_text(encoding="utf-8")
                self.assertIn("self.collect_read_mode_combo", source)
                self.assertIn("self.collect_track_center_input", source)
                self.assertIn("self.collect_track_height_input", source)
                self.assertIn(
                    "self.collect_read_mode_combo.setCurrentIndex(0)", source
                )
                self.assertIn(
                    "self.collect_single_track_controls.hide()", source
                )
                self.assertIn(
                    "self.collect_track_center_input.setValue(185)", source
                )
                self.assertIn(
                    "self.collect_track_height_input.setValue(140)", source
                )
                self.assertIn(
                    "self.raman_box.isChecked() and mode == \"single_track\"",
                    source,
                )
                self.assertIn('if read_mode == "image":', source)
                self.assertIn("DetectorImageWindow(", source)
                self.assertIn("spectral_calibration=self.spectral_calibration", source)

    def test_live_spectrum_controls_exist_in_both_widgets(self) -> None:
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = (PACKAGE / filename).read_text(encoding="utf-8")
                self.assertIn("self.live_collect_check", source)
                self.assertIn("def _start_live_raman", source)
                self.assertIn("def _stop_live_raman", source)
                self.assertIn("update_spectrum(spec, title=title)", source)
                self.assertIn("update_frames(spec, title=title)", source)

    def test_center_point_and_spectral_bias_controls_exist_in_both_widgets(
        self,
    ) -> None:
        for filename in ("hardware_widget.py", "demo_widget.py"):
            with self.subTest(filename=filename):
                source = (PACKAGE / filename).read_text(encoding="utf-8")
                self.assertIn("def _ensure_raman_points_layer", source)
                self.assertIn('name="Raman points"', source)
                self.assertIn("height / 2.0, width / 2.0", source)
                self.assertIn("self.remove_spectral_bias_check", source)
                self.assertIn("self.dark_noise_controls", source)
                self.assertIn("self.collect_dark_noise_btn", source)
                self.assertIn("def collect_dark_noise", source)
                self.assertIn("spectral_bias_from_dark_noise", source)
                self.assertIn("spectral_bias=spectral_bias", source)
                self.assertNotIn("subtract_spectral_bias", source)
                self.assertIn(
                    'f"dark_noise_{exposure:g}ms_{uuid.uuid4()}.npy"',
                    source,
                )
                collect_button = source.index(
                    'self.collect_btn = QPushButton('
                )
                dark_button = source.index(
                    'self.collect_dark_noise_btn = QPushButton('
                )
                supply_checkbox = source.index(
                    'self.remove_spectral_bias_check = QCheckBox('
                )
                dark_loader = source.index(
                    'self.dark_noise_controls = QWidget()'
                )
                self.assertLess(collect_button, dark_button)
                self.assertLess(dark_button, supply_checkbox)
                self.assertLess(supply_checkbox, dark_loader)

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

    def test_hardware_launcher_guarantees_shutdown_cleanup(self) -> None:
        hardware = (PACKAGE / "hardware_widget.py").read_text(encoding="utf-8")
        launcher = (PACKAGE.parent / "run_napari.py").read_text(encoding="utf-8")
        self.assertIn("app.aboutToQuit.connect(self.shutdown_hardware)", hardware)
        self.assertIn("finally:", launcher)
        self.assertIn("widget.shutdown_hardware()", launcher)

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
