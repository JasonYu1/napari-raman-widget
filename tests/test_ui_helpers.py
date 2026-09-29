"""Focused tests for the shared dock-panel layout helpers."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QCoreApplication, QEvent  # noqa: E402
from qtpy.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_raman_widget.ui_helpers import (  # noqa: E402
    align_form_rows,
    make_collapsible,
    make_panel_header,
    make_workflow_tabs,
)


class TestUiHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_initial_visibility_timer_is_cancelled_when_section_is_deleted(self):
        box = make_collapsible("Temporary section")
        fired = []
        self.assertIs(box._initial_visibility_timer.parent(), box)
        box._initial_visibility_timer.timeout.connect(lambda: fired.append(True))
        box.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()
        self.assertEqual(fired, [])

    def test_collapsible_preserves_conditional_child_visibility(self):
        box = make_collapsible("Options", expanded=True)
        layout = QVBoxLayout()
        ordinary = QLabel("Always available")
        conditional = QLabel("Conditional")
        layout.addWidget(ordinary)
        layout.addWidget(conditional)
        box.setLayout(layout)
        conditional.hide()

        box.setChecked(False)
        self.assertTrue(box.content_widget.isHidden())
        box.setChecked(True)

        self.assertFalse(box.content_widget.isHidden())
        self.assertFalse(ordinary.isHidden())
        self.assertTrue(conditional.isHidden())

    def test_workflow_tabs_give_each_page_its_own_scroll_area(self):
        setup_section = make_collapsible("Loading")
        acquire_section = make_collapsible("Acquire", expanded=False)
        tabs = make_workflow_tabs(
            [
                ("Setup", [setup_section]),
                ("Acquire", [acquire_section]),
            ]
        )

        self.assertEqual(tabs.count(), 2)
        self.assertEqual(tabs.tabText(0), "Setup")
        self.assertIsInstance(tabs.widget(0), QScrollArea)
        self.assertIs(setup_section.parent(), tabs.widget(0).widget())

    def test_header_plots_button_uses_callback(self):
        calls = []
        header, plots_button = make_panel_header(
            QLabel("Help"), lambda: calls.append("plots")
        )

        self.assertIsNotNone(header.layout())
        plots_button.click()
        self.assertEqual(calls, ["plots"])

    def test_form_rows_share_label_width_and_expand_fields(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        labels = []
        fields = []
        for text in ("Short:", "A longer setting:"):
            row = QHBoxLayout()
            label = QLabel(text)
            field = QLineEdit()
            labels.append(label)
            fields.append(field)
            row.addWidget(label)
            row.addWidget(field)
            root_layout.addLayout(row)

        align_form_rows([root])

        self.assertEqual(labels[0].minimumWidth(), labels[1].minimumWidth())
        self.assertEqual(
            fields[0].sizePolicy().horizontalPolicy(),
            QSizePolicy.Ignored,
        )

    def test_long_form_controls_fit_narrow_workflow_viewport(self):
        section = make_collapsible("Loading", expanded=True)
        section_layout = QVBoxLayout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Detector read mode with long label:"))
        mode_combo = QComboBox()
        mode_combo.addItem("Full vertical binning (FVB) with long option")
        mode_row.addWidget(mode_combo)
        section_layout.addLayout(mode_row)

        section_layout.addWidget(
            QLabel("Output folder (optional, applied on connect):")
        )
        file_row = QHBoxLayout()
        file_path = QLineEdit()
        file_path.setPlaceholderText("A long folder placeholder")
        browse = QPushButton("...")
        browse.setFixedWidth(30)
        file_row.addWidget(file_path)
        file_row.addWidget(browse)
        section_layout.addLayout(file_row)
        section.setLayout(section_layout)

        align_form_rows([section])
        tabs = make_workflow_tabs([("Setup", [section])])
        tabs.resize(340, 400)
        tabs.show()
        self.app.processEvents()

        scroll = tabs.widget(0)
        page = scroll.widget()
        try:
            self.assertLessEqual(page.width(), scroll.viewport().width())
            self.assertLessEqual(
                page.minimumSizeHint().width(), scroll.viewport().width()
            )
        finally:
            tabs.close()
            tabs.deleteLater()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
