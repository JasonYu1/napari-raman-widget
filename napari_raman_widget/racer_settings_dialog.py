"""Optional editor for remembered RACER calibration parameters."""

from __future__ import annotations

from copy import deepcopy

from qtpy.QtCore import QLocale
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .racer_settings import (
    GROUP_LABELS,
    calibration_parameter_specs,
    normalize_calibration_options,
)


class RacerCalibrationSettingsDialog(QDialog):
    """Edit upstream fit options without changing an active calibration."""

    def __init__(
        self, options=None, parent=None, *, read_mode="fvb", settings_warning=None
    ):
        super().__init__(parent)
        self.setWindowTitle("RACER calibration settings")
        self.resize(740, 620)
        self.specs = calibration_parameter_specs()
        try:
            self._selected_options = normalize_calibration_options(options)
        except (TypeError, ValueError) as error:
            self._selected_options = {}
            settings_warning = (
                "Saved settings are incompatible with this RACER version; "
                "showing defaults. Save settings to replace them. "
                f"{error}"
            )
        self.editors = {}
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Save settings remembers these values next time you open the widget. "
            "Reload a calibration to apply changes."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        metadata_note = QLabel(
            "Exposure, detector gain, and saturation use the calibration stores."
        )
        metadata_note.setWordWrap(True)
        layout.addWidget(metadata_note)

        self.tabs = QTabWidget()
        self._group_tabs = {}
        for group in ("spectrum_dark", "spectrum_reference", "image"):
            specs = self.specs.get(group, ())
            if not specs:
                continue
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            form.setVerticalSpacing(10)
            group_note = QLabel(
                {
                    "spectrum_dark": "Fits the dark noise and cosmic-ray charge distribution for FVB spectra.",
                    "spectrum_reference": "Fits FVB reference variance. With no illuminated reference, this uses the dark stack.",
                    "image": "Fits the detector-image profile and noise. Trace half-width controls the region around the optical trace.",
                }[group]
            )
            group_note.setWordWrap(True)
            form.addRow(group_note)
            for spec in specs:
                value = self._selected_options.get(group, {}).get(
                    spec.name, spec.default
                )
                if spec.is_integer:
                    editor = QSpinBox()
                    editor.setRange(
                        int(spec.minimum) if spec.minimum is not None else -2_147_483_648,
                        min(int(spec.maximum), 2_147_483_647)
                        if spec.maximum is not None else 2_147_483_647,
                    )
                    editor.setValue(value)
                    editor.setKeyboardTracking(False)
                else:
                    editor = QLineEdit(str(value))
                    validator = QDoubleValidator(editor)
                    validator.setLocale(QLocale.c())
                    validator.setNotation(QDoubleValidator.ScientificNotation)
                    editor.setValidator(validator)
                editor.setObjectName(f"{group}.{spec.name}")
                editor.setMinimumWidth(130)
                tooltip = (
                    f"{spec.tooltip}\nParameter: {spec.name}\n"
                    f"RACER default: {spec.default}"
                )
                editor.setToolTip(tooltip)
                label = QLabel(spec.label)
                label.setToolTip(tooltip)
                label.setBuddy(editor)
                field = QHBoxLayout()
                field.addWidget(editor, 1)
                default_label = QLabel(f"Default: {spec.default}")
                default_label.setMinimumWidth(120)
                field.addWidget(default_label)
                form.addRow(label, field)
                self.editors[(group, spec.name)] = editor
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(page)
            self._group_tabs[group] = self.tabs.addTab(scroll, GROUP_LABELS[group])
        initial_group = "image" if read_mode == "image" else "spectrum_dark"
        if initial_group in self._group_tabs:
            self.tabs.setCurrentIndex(self._group_tabs[initial_group])
        layout.addWidget(self.tabs, 1)

        self.error_label = QLabel(settings_warning or "")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(bool(settings_warning))
        layout.addWidget(self.error_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        self.buttons.button(QDialogButtonBox.Ok).setText("Save settings")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self.restore_defaults
        )
        layout.addWidget(self.buttons)

    @property
    def calibration_options(self):
        """Return a copy so edits never mutate the owner's saved options."""
        return deepcopy(self._selected_options)

    def restore_defaults(self):
        for group, specs in self.specs.items():
            for spec in specs:
                editor = self.editors[(group, spec.name)]
                if spec.is_integer:
                    editor.setValue(spec.default)
                else:
                    editor.setText(str(spec.default))
        self.error_label.clear()
        self.error_label.hide()

    def accept(self):
        draft = {}
        try:
            for group, specs in self.specs.items():
                draft[group] = {}
                for spec in specs:
                    editor = self.editors[(group, spec.name)]
                    if spec.is_integer:
                        editor.interpretText()
                        value = editor.value()
                    else:
                        if not editor.hasAcceptableInput():
                            self.tabs.setCurrentIndex(self._group_tabs[group])
                            editor.setFocus()
                            editor.selectAll()
                            raise ValueError(f"{spec.label}: enter a finite number.")
                        value = float(editor.text())
                    draft[group][spec.name] = value
            selected = normalize_calibration_options(draft)
        except (TypeError, ValueError) as error:
            self.error_label.setText(f"Please check the settings: {error}")
            self.error_label.show()
            return
        self._selected_options = selected
        super().accept()
