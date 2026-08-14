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
        QLabel("Pixel-to-wavenumber calibration (.json):")
    )
    row = QHBoxLayout()
    owner.spectral_calibration_path = QLineEdit()
    owner.spectral_calibration_path.setPlaceholderText(
        "pixel_to_wavenumber_calibration.json"
    )
    browse = QPushButton("...")
    browse.setFixedWidth(30)
    browse.clicked.connect(owner.browse_spectral_calibration)
    row.addWidget(owner.spectral_calibration_path)
    row.addWidget(browse)
    loading_layout.addLayout(row)


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
        calibration = load_pixel_to_wavenumber_calibration(path)
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
    owner.spectral_calibration = calibration
    if hasattr(owner, "status"):
        owner.status.setText(
            "Status: pixel-to-wavenumber calibration loaded "
            f"(degree {calibration.degree}, "
            f"{len(calibration.pixel_positions)} points)"
        )
    if show_success:
        QMessageBox.information(
            owner,
            "Calibration loaded",
            "Spectrum popups will now use Raman shift by default. "
            "Use 'Show pixels' in a popup to switch back.",
        )
    return calibration


def spectral_calibration_created(owner, calibration, path) -> None:
    """Store a calibration created inside one of *owner*'s popups."""
    owner.spectral_calibration = calibration
    owner.spectral_calibration_path.setText(str(path))
    if hasattr(owner, "status"):
        owner.status.setText(
            f"Status: pixel-to-wavenumber calibration saved -> {path}"
        )
