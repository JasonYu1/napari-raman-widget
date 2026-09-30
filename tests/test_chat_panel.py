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
from napari_raman_widget.assistant_calibration_tools import (
    CALIBRATION_CONTROL_WIDGET_ATTRIBUTES,
)
from napari_raman_widget.assistant_plot_tools import PLOT_CONTROL_WIDGET_ATTRIBUTES


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
                and target.value.id in {"self", "owner"}
            ):
                controls.add(target.attr)
    return controls


def _plot_controls(path):
    """Include helper factories and local-variable controls assigned to owners."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    factories = CONTROL_TYPES | {"QSlider", "_make_wavenumber_axis_checkbox"}
    result = set()
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # A name such as `checkbox` belongs to this function only, not every
        # method in the file that accepts an existing control as an argument.
        assignments = [node for node in ast.walk(scope) if isinstance(node, ast.Assign)]
        local_controls = {
            target.id
            for node in assignments
            if isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in factories
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        for node in assignments:
            value = node.value
            direct = (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in factories
            )
            alias = isinstance(value, ast.Name) and value.id in local_controls
            if not (direct or alias):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {"self", "owner"}
                ):
                    result.add(target.attr)
    return result


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
        for filename in (
            "hardware_widget.py", "demo_widget.py", "spectral_calibration_ui.py"
        ):
            declared.update(
                _declared_controls(ROOT / "napari_raman_widget" / filename)
            )

        self.assertLessEqual(declared, registered)
        self.assertEqual(
            registered - declared,
            {"channel_rows", "mda_channel_rows"},
        )

    def test_new_capabilities_are_exposed_in_the_api_schema(self):
        expected = {
            "get_assistant_capabilities", "list_plots", "configure_plot",
            "show_plot_workspace", "hide_plot_workspace",
            "load_wavenumber_calibration", "clear_wavenumber_calibration",
            "clear_dark_noise", "collect_dark_noise", "stop_live_spectra",
            "inspect_calibration_result", "query_calibration_progress",
            "control_spectral_axis_calibration",
        }
        tools = build_tools()
        names = [tool["name"] for tool in tools]
        self.assertEqual(len(names), len(set(names)))
        self.assertLessEqual(expected, set(names))
        for tool in tools:
            schema = tool["input_schema"]
            self.assertFalse(schema["additionalProperties"])
            self.assertLessEqual(set(schema["required"]), set(schema["properties"]))

    def test_dark_collection_is_gated_and_live_stop_is_not(self):
        collect = ACTIONS_BY_NAME["collect_dark_noise"]
        self.assertFalse(collect.get("readonly", False))
        self.assertFalse(collect.get("always_run", False))
        self.assertTrue(ACTIONS_BY_NAME["stop_live_spectra"]["always_run"])
        self.assertTrue(ACTIONS_BY_NAME["clear_dark_noise"]["readonly"])

    def test_plot_and_shared_controls_have_adapters_or_synchronized_aliases(self):
        declared = set()
        for filename in ("plot_windows.py", "figure_panel.py"):
            declared |= _plot_controls(ROOT / "napari_raman_widget" / filename)
        registered = set(PLOT_CONTROL_WIDGET_ATTRIBUTES.values())
        registered |= set(CALIBRATION_CONTROL_WIDGET_ATTRIBUTES.values())
        # These sliders mirror their corresponding numeric inputs, which are
        # configured via the same signal handlers rather than a second tool.
        synchronized = {
            "smoothing_window_slider": "smoothing_window_input",
            "baseline_lambda_slider": "baseline_lambda_input",
            "_z_slider": "_z_input",
        }
        self.assertLessEqual(set(synchronized.values()), registered)
        self.assertLessEqual(declared, registered | set(synchronized))
        # Dataset controls returned by _make_slider are tuple assignments,
        # not direct constructors, so cover their adapter mapping explicitly.
        self.assertLessEqual({"t_input", "p_input", "z_input"}, registered)
        self.assertLessEqual(
            {"white_background_check", "show_wavenumber_check"}, declared
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
                "suppress_pillars",
                "vandermonde_model",
            },
        )
        self.assertEqual(params["n_x"]["attr"], "sel_sqn_input")
        self.assertEqual(params["n_x"]["kind"], "int")
        self.assertEqual(
            params["suppress_pillars"]["attr"],
            "sel_suppress_pillars_check",
        )
        self.assertEqual(params["suppress_pillars"]["kind"], "check")

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

    def test_pillar_suppression_is_a_boolean_mda_setting(self):
        widget_params = {param["name"]: param for param in WIDGET_PARAMS}
        pillar = widget_params["suppress_pillars"]
        self.assertEqual(pillar["attr"], "mda_suppress_pillars_check")
        self.assertEqual(pillar["kind"], "check")

        mda_params = {
            param["name"]: param
            for param in ACTIONS_BY_NAME["run_raman_mda"]["params"]
        }
        self.assertEqual(
            mda_params["suppress_pillars"]["attr"],
            "mda_suppress_pillars_check",
        )
        mda_schema = {
            tool["name"]: tool for tool in build_tools()
        }["run_raman_mda"]["input_schema"]
        self.assertEqual(
            mda_schema["properties"]["suppress_pillars"]["type"],
            "boolean",
        )

    def test_automatic_new_cells_is_a_boolean_mda_setting(self):
        widget_params = {param["name"]: param for param in WIDGET_PARAMS}
        automatic = widget_params["auto_add_new_cells"]
        self.assertEqual(
            automatic["attr"],
            "mda_auto_add_cells_check",
        )
        self.assertEqual(automatic["kind"], "check")

        mda_params = {
            param["name"]: param
            for param in ACTIONS_BY_NAME["run_raman_mda"]["params"]
        }
        self.assertEqual(
            mda_params["auto_add_new_cells"]["attr"],
            "mda_auto_add_cells_check",
        )
        mda_schema = {
            tool["name"]: tool for tool in build_tools()
        }["run_raman_mda"]["input_schema"]
        self.assertEqual(
            mda_schema["properties"]["auto_add_new_cells"]["type"],
            "boolean",
        )

    def test_pillar_suppression_is_a_boolean_selection_setting(self):
        widget_params = {param["name"]: param for param in WIDGET_PARAMS}
        pillar = widget_params["selection_suppress_pillars"]
        self.assertEqual(pillar["attr"], "sel_suppress_pillars_check")
        self.assertEqual(pillar["kind"], "check")

        selection_schema = {
            tool["name"]: tool for tool in build_tools()
        }["run_automated_selection"]["input_schema"]
        self.assertEqual(
            selection_schema["properties"]["suppress_pillars"]["type"],
            "boolean",
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

    def test_spatial_scan_modes_replace_grid_side_without_reinterpreting_it(self):
        actions = ACTIONS_BY_NAME["run_grid_scan"]["params"]
        scan_params = {param["name"]: param for param in actions}
        widget_params = {param["name"]: param for param in WIDGET_PARAMS}
        for action_name, widget_name, attr in (
            ("sampling_mode", "scan_sampling_mode", "scan_sampling_mode_combo"),
            ("total_points", "scan_total_points", "scan_total_points_input"),
            ("spacing_px", "scan_spacing_px", "scan_spacing_input"),
        ):
            with self.subTest(action_name=action_name):
                self.assertEqual(scan_params[action_name]["attr"], attr)
                self.assertEqual(widget_params[widget_name]["attr"], attr)
                self.assertEqual(
                    scan_params[action_name]["description"],
                    widget_params[widget_name]["description"],
                )
        self.assertNotIn("grid_side", scan_params)
        self.assertNotIn("scan_grid_side", widget_params)
        self.assertNotIn("scan_n_input", {param["attr"] for param in WIDGET_PARAMS})
        tools = {tool["name"]: tool for tool in build_tools()}
        for tool_name, mode_name, old_name in (
            ("run_grid_scan", "sampling_mode", "grid_side"),
            ("configure_widget", "scan_sampling_mode", "scan_grid_side"),
        ):
            properties = tools[tool_name]["input_schema"]["properties"]
            self.assertEqual(properties[mode_name]["enum"], ["Total points", "Pixel spacing"])
            self.assertNotIn(old_name, properties)
            self.assertFalse(tools[tool_name]["input_schema"]["additionalProperties"])
        self.assertIn("inside one selected", ACTIONS_BY_NAME["run_grid_scan"]["description"])
        description = ACTIONS_BY_NAME["run_grid_scan"]["description"]
        self.assertIn("approximate target count", description)
        self.assertIn("uniform X/Y spacing", description)
        self.assertIn("target versus actual counts", description)
        self.assertNotIn("exact count", description)
        self.assertIn("approximate", scan_params["total_points"]["description"])

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
