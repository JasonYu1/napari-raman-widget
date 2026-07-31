import ast
import unittest
from pathlib import Path

from napari_raman_widget.chat_panel import (
    ACTIONS_BY_NAME,
    CONFIGURABLE_WIDGET_PARAMS,
    WIDGET_PARAMS,
    _read_widget_settings,
    _set_widget_parameter,
    build_tools,
)


ROOT = Path(__file__).parents[1]
CONTROL_TYPES = {
    "QCheckBox",
    "QComboBox",
    "QDoubleSpinBox",
    "QLineEdit",
    "QSpinBox",
}


def _declared_controls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    controls = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
            continue
        if value.func.id not in CONTROL_TYPES:
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                controls.add(target.attr)
    return controls


class _ValueControl:
    def __init__(self, value=0):
        self._value = value

    def setValue(self, value):
        self._value = value

    def value(self):
        return self._value


class _CheckControl:
    def __init__(self, checked=False):
        self._checked = checked

    def setChecked(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _FakeWidget:
    def __init__(self):
        self.sel_sqn_input = _ValueControl(1)
        self.sel_center_cell_check = _CheckControl(False)


class ChatPanelTests(unittest.TestCase):
    def test_widget_registry_covers_every_editable_control(self):
        registered = {param["attr"] for param in WIDGET_PARAMS}
        declared = set()
        for filename in ("hardware_widget.py", "demo_widget.py"):
            declared.update(
                _declared_controls(ROOT / "napari_raman_widget" / filename)
            )

        self.assertLessEqual(declared, registered)
        self.assertEqual(
            registered - declared,
            {"channel_rows", "mda_channel_rows"},
        )

    def test_automated_selection_exposes_every_setting_including_n_x(self):
        params = {
            param["name"]: param
            for param in ACTIONS_BY_NAME["run_automated_selection"]["params"]
        }

        self.assertEqual(
            set(params),
            {
                "aiming_pattern",
                "autofocus_object",
                "background_distance_px",
                "batch",
                "cellpose_model",
                "center_cell",
                "center_x",
                "center_y",
                "n_per_fov",
                "n_x",
                "pattern_size_px",
                "radius",
                "vandermonde_model",
            },
        )
        self.assertEqual(params["n_x"]["attr"], "sel_sqn_input")
        self.assertEqual(params["n_x"]["kind"], "int")

    def test_settings_only_tool_configures_n_x_without_running_selection(self):
        params = {
            param["name"]: param
            for param in ACTIONS_BY_NAME["configure_widget"]["params"]
        }
        self.assertEqual(params["n_x"]["attr"], "sel_sqn_input")
        self.assertIs(
            CONFIGURABLE_WIDGET_PARAMS,
            ACTIONS_BY_NAME["configure_widget"]["params"],
        )

    def test_tool_schemas_include_dynamic_channel_rows(self):
        tools = {tool["name"]: tool for tool in build_tools()}
        scan_channels = tools["run_grid_scan"]["input_schema"]["properties"][
            "extra_channels"
        ]
        mda_channels = tools["run_raman_mda"]["input_schema"]["properties"][
            "extra_channels"
        ]

        self.assertEqual(scan_channels["type"], "array")
        self.assertEqual(
            mda_channels["items"]["required"],
            ["channel", "exposure_ms"],
        )

    def test_widget_parameter_setter_and_state_reader_use_registry_names(self):
        widget = _FakeWidget()
        by_name = {param["name"]: param for param in WIDGET_PARAMS}

        _set_widget_parameter(widget, by_name["n_x"], 4)
        _set_widget_parameter(widget, by_name["center_cell"], "false")

        self.assertEqual(widget.sel_sqn_input.value(), 4)
        self.assertFalse(widget.sel_center_cell_check.isChecked())
        settings = _read_widget_settings(widget)
        self.assertIn("n_x=4", settings)
        self.assertIn("center_cell=False", settings)
