"""Architecture checks for independent hardware and demo widgets."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1] / "napari_raman_widget"


class WidgetSeparationTests(unittest.TestCase):
    def test_hardware_module_has_no_demo_mode_branches(self) -> None:
        source = (PACKAGE / "hardware_widget.py").read_text(encoding="utf-8")
        self.assertNotIn("demo_backend", source)
        self.assertNotIn("create_demo_backend", source)
        self.assertNotIn("DemoWorld", source)

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
