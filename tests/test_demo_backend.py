"""Behavioral tests for the demonstration backend."""

from __future__ import annotations

import unittest

import numpy as np

from napari_raman_widget.demo import create_demo_backend


class DemoBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = create_demo_backend(image_height=128, image_width=160)

    def test_devices_share_world_state(self) -> None:
        self.backend.core.setXYPosition(12.5, -8.0)
        self.backend.core.setZPosition(2.0)
        self.assertEqual(self.backend.core.getXYPosition(), (12.5, -8.0))
        self.assertEqual(self.backend.world.stage_z, 2.0)

    def test_snap_returns_camera_sized_image(self) -> None:
        image = self.backend.core.snap()
        self.assertEqual(image.shape, (128, 160))
        self.assertEqual(image.dtype, np.uint16)

    def test_collector_returns_point_by_pixel_spectra(self) -> None:
        volts = np.array([[0.0, 0.0], [0.2, -0.1], [0.2, -0.1]])
        spectra = self.backend.collector.collect_spectra_pts(volts, exposure=1000)
        repeated_run = self.backend.collector.collect_spectra_pts(
            volts,
            exposure=1000,
        )
        self.assertEqual(spectra.shape, (3, self.backend.world.spectrum_pixels))
        np.testing.assert_allclose(spectra, repeated_run)

    def test_raman_signal_is_stronger_near_focus(self) -> None:
        volts = np.zeros((1, 2))
        self.backend.core.setZPosition(self.backend.world.focus_z)
        focused = self.backend.collector.collect_spectra_pts(volts, 1000)
        self.backend.core.setZPosition(self.backend.world.focus_z + 20)
        defocused = self.backend.collector.collect_spectra_pts(volts, 1000)
        focused_peak = focused[:, 820:890].mean()
        defocused_peak = defocused[:, 820:890].mean()
        self.assertGreater(focused_peak, defocused_peak)

    def test_rm_shutter_adds_calibration_spot(self) -> None:
        self.backend.core.setConfig("Channel", "RM")
        self.backend.core.setShutterOpen("Fluoshutter", False)
        without_laser = self.backend.core.snap()
        self.backend.daq.galvo.write(np.array([0.4, -0.2]))
        self.backend.core.setShutterOpen("Fluoshutter", True)
        with_laser = self.backend.core.snap()
        self.assertGreater(int(with_laser.max()), int(without_laser.max()))

    def test_spectrograph_controls_match_widget_contract(self) -> None:
        self.backend.collector.set_wavelength(785.0)
        self.backend.collector.set_grating(2)
        self.assertEqual(self.backend.collector.get_wavelength(), 785.0)
        self.assertEqual(self.backend.collector.get_grating(), 2)
        lines, blaze, home, offset = self.backend.collector.get_grating_info(2)
        self.assertEqual(lines, 600.0)
        self.assertTrue(all(np.isfinite((lines, blaze, home, offset))))


if __name__ == "__main__":
    unittest.main()
