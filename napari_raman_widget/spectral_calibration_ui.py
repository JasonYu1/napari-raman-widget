"""Shared Qt controls for loading a spectral-axis calibration."""

from __future__ import annotations

from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
)

from .spectral_calibration import load_pixel_to_wavenumber_calibration


def add_spectral_calibration_loader(owner, loading_layout) -> None:
    """Add shared calibration controls to *owner*'s Loading section."""
    owner.spectral_calibration = None
    loading_layout.addWidget(
        QLabel("Pixel-to-wavenumber calibration (optional .json):")
    )
    row = QHBoxLayout()
    owner.spectral_calibration_path = QLineEdit()
    owner.spectral_calibration_path.setPlaceholderText(
        "None (pixel axis by default)"
    )
    browse = QPushButton("...")
    browse.setFixedWidth(30)
    browse.clicked.connect(owner.browse_spectral_calibration)
    owner.clear_spectral_calibration_btn = QPushButton("Clear")
    owner.clear_spectral_calibration_btn.setToolTip(
        "Remove the optional calibration for new plots; existing plots "
        "keep their calibration."
    )
    owner.clear_spectral_calibration_btn.clicked.connect(
        lambda: clear_spectral_calibration(owner)
    )
    row.addWidget(owner.spectral_calibration_path)
    row.addWidget(browse)
    row.addWidget(owner.clear_spectral_calibration_btn)
    loading_layout.addLayout(row)


def clear_spectral_calibration(owner) -> None:
    """Return new plots to an uncalibrated pixel axis."""
    owner.spectral_calibration = None
    owner.spectral_calibration_path.clear()
    if hasattr(owner, "status"):
        owner.status.setText(
            "Status: optional wavenumber calibration cleared for new plots"
        )


def browse_spectral_calibration(owner) -> None:
    """Select and immediately load a calibration JSON file."""
    path, _ = QFileDialog.getOpenFileName(
        owner,
        "Select pixel-to-wavenumber calibration",
        "",
        "JSON files (*.json);;All files (*)",
    )
    if path:
        owner.spectral_calibration_path.setText(path)
        owner.load_spectral_calibration()


def load_spectral_calibration(owner, *, show_success=True):
    """Load the calibration named by *owner*'s path field."""
    path = owner.spectral_calibration_path.text().strip()
    if not path:
        QMessageBox.warning(
            owner,
            "No calibration selected",
            "Choose a pixel-to-wavenumber calibration JSON file first.",
        )
        return None
    try:
        calibration = load_spectral_calibration_file(owner, path)
    except (OSError, ValueError) as error:
        owner.spectral_calibration = None
        QMessageBox.warning(
            owner, "Could not load spectral calibration", str(error)
        )
        if hasattr(owner, "status"):
            owner.status.setText(
                f"Status: spectral calibration load failed -- {error}"
            )
        return None
    if show_success:
        QMessageBox.information(
            owner,
            "Calibration loaded",
            "New spectrum tabs still start in pixels. "
            "Check 'Show wavenumber' in a plot to use this calibration.",
        )
    return calibration


def load_spectral_calibration_file(owner, path):
    """Validate then apply a calibration, without dialogs or partial changes."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Select a calibration JSON file first.")
    path = path.strip()
    calibration = load_pixel_to_wavenumber_calibration(path)
    owner.spectral_calibration = calibration
    owner.spectral_calibration_path.setText(path)
    if hasattr(owner, "status"):
        owner.status.setText(
            "Status: pixel-to-wavenumber calibration loaded "
            f"(degree {calibration.degree}, "
            f"{len(calibration.pixel_positions)} points)"
        )
    return calibration


def spectral_calibration_created(owner, calibration, path) -> None:
    """Store a calibration created inside one of *owner*'s plot panels."""
    owner.spectral_calibration = calibration
    owner.spectral_calibration_path.setText(str(path))
    if hasattr(owner, "status"):
        owner.status.setText(
            f"Status: pixel-to-wavenumber calibration saved -> {path}"
        )
