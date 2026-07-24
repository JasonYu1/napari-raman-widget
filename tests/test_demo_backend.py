"""Behavioral tests for the demonstration backend."""

from __future__ import annotations

import unittest

import numpy as np

from napari_raman_widget.demo import configure_demo_channels, create_demo_backend
from napari_raman_widget.demo.cellpose import segment_upsampled_demo_region
from napari_raman_widget.demo.transformer import make_demo_point_transformer


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

    def test_demo_cells_have_varied_sizes(self) -> None:
        radii = self.backend.world.cell_radii_px
        self.assertEqual(len(radii), len(self.backend.world.cell_positions_um))
        self.assertGreater(float(np.ptp(radii)), 5.0)
        self.assertLess(float(np.mean(radii)), 7.0)

    def test_demo_sample_is_dense_and_has_varied_colors(self) -> None:
        world = self.backend.world
        visible = world._cell_pixels_yx()
        visible = visible[
            (visible[:, 0] >= 0)
            & (visible[:, 0] < world.image_height)
            & (visible[:, 1] >= 0)
            & (visible[:, 1] < world.image_width)
        ]
        self.assertGreater(self.backend.world.number_of_cells, 4_000)
        self.assertGreater(len(visible), 1)
        self.assertGreater(len(np.unique(world.cell_colors_rgb, axis=0)), 3)

    def test_demo_cellpose_preprocessing_restores_original_coordinates(self) -> None:
        image = np.zeros((32, 40), dtype=np.uint16)
        image[15:18, 19:22] = 1000
        received = {}

        def fake_cellpose(frame, **kwargs):
            received["shape"] = frame.shape
            received.update(kwargs)
            return (frame > 0).astype(np.uint16)

        mask = segment_upsampled_demo_region(
            image,
            center_yx=(16, 20),
            radius=10,
            upsample=3,
            cellpose_model="nuclei",
            segmenter=fake_cellpose,
        )
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(received["cellpose_model"], "nuclei")
        self.assertEqual(received["scale"], 1)
        self.assertGreater(received["shape"][0], 32)
        np.testing.assert_array_equal(mask > 0, image > 0)

    def test_fake_mda_channels_are_bf_and_rm(self) -> None:
        class ConfigCore:
            def __init__(self):
                self.groups = {"Channel": ["DAPI", "FITC", "Cy5"]}
                self.channel_group = ""
                self.current = ""
                self.config_values = {}

            def getAvailableConfigGroups(self):
                return tuple(self.groups)

            def deleteConfigGroup(self, group):
                del self.groups[group]

            def getCameraDevice(self):
                return "Camera"

            def getProperty(self, device, prop):
                return "0"

            def defineConfig(self, group, preset, device, prop, value):
                self.groups.setdefault(group, []).append(preset)
                self.config_values[(group, preset)] = (device, prop, value)

            def setChannelGroup(self, group):
                self.channel_group = group

            def setConfig(self, group, preset):
                self.current = preset

        core = ConfigCore()
        configure_demo_channels(core)
        self.assertEqual(core.groups["Channel"], ["BF", "RM"])
        self.assertEqual(core.channel_group, "Channel")
        self.assertEqual(core.current, "BF")
        self.assertNotEqual(
            core.config_values[("Channel", "BF")],
            core.config_values[("Channel", "RM")],
        )

    def test_demo_cells_are_solid_spheres_not_rings(self) -> None:
        world = create_demo_backend(image_height=64, image_width=64).world
        world.cell_positions_um = np.array([[0.0, 0.0]])
        world.cell_radii_px = np.array([8.0])
        world.cell_colors_rgb = np.array([[76.0, 201.0, 240.0]])
        image = world.render_image()
        center_y, center_x = 32, 32
        self.assertGreater(image[center_y, center_x], image[center_y, center_x + 6])
        color = world.render_color_image()
        self.assertEqual(color.shape, (64, 64, 3))
        self.assertGreater(color[center_y, center_x].mean(), color[0, 0].mean())

    def test_demo_sample_extends_beyond_initial_fov(self) -> None:
        world = create_demo_backend().world
        fov_width_um = world.image_width * world.microns_per_pixel
        fov_height_um = world.image_height * world.microns_per_pixel
        self.assertGreater(world.sample_width_um, fov_width_um * 5)
        self.assertGreater(world.sample_height_um, fov_height_um * 5)

        initial_visible = world._cell_pixels_yx()
        initial_visible = initial_visible[
            (initial_visible[:, 0] >= 0)
            & (initial_visible[:, 0] < world.image_height)
            & (initial_visible[:, 1] >= 0)
            & (initial_visible[:, 1] < world.image_width)
        ]
        self.assertGreater(len(initial_visible), 10)

    def test_demo_pixel_to_stage_mapping_is_one_to_one(self) -> None:
        before = self.backend.world._cell_pixels_yx()[0]
        self.backend.core.setXYPosition(12.0, -7.0)
        after = self.backend.world._cell_pixels_yx()[0]
        np.testing.assert_allclose(after - before, [7.0, -12.0])

    def test_single_demo_pattern_preserves_the_selected_pixel(self) -> None:
        selected = np.array([[0.5, 0.5]])
        for shape in ("Square", "Circle"):
            transformer = make_demo_point_transformer(shape, 30.0, 1, 512)
            np.testing.assert_allclose(transformer.transform(selected), selected)
            self.assertEqual(transformer.multiplier, 1)

    def test_detected_demo_points_are_refined_to_sphere_centers(self) -> None:
        world = self.backend.world
        centers = world._cell_pixels_yx()
        visible = centers[
            (centers[:, 0] >= 0)
            & (centers[:, 0] < world.image_height)
            & (centers[:, 1] >= 0)
            & (centers[:, 1] < world.image_width)
        ]
        self.assertGreater(len(visible), 0)
        detected = visible[:2] + np.array([[2.0, -3.0], [-1.5, 2.5]])[: len(visible[:2])]
        refined = world.snap_points_to_cell_centers(detected)
        for point in refined:
            self.assertTrue(np.any(np.all(np.isclose(visible, point), axis=1)))

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
        self.backend.world.cell_positions_um[0] = (0.0, 0.0)
        volts = np.zeros((1, 2))
        self.backend.core.setZPosition(self.backend.world.focus_z)
        focused = self.backend.collector.collect_spectra_pts(volts, 1000)
        self.backend.core.setZPosition(self.backend.world.focus_z + 20)
        defocused = self.backend.collector.collect_spectra_pts(volts, 1000)
        focused_peak = focused[:, 820:890].mean()
        defocused_peak = defocused[:, 820:890].mean()
        self.assertGreater(focused_peak, defocused_peak)

    def test_background_has_no_raman_peaks(self) -> None:
        self.backend.world.cell_positions_um[0] = (0.0, 0.0)
        cell = self.backend.collector.collect_spectra_pts([[0.0, 0.0]], 1000)
        background = self.backend.collector.collect_spectra_pts(
            [[1.5, 1.5]], 1000
        )
        self.assertGreater(cell.max(), background.max() * 5)

    def test_visible_cell_pixel_produces_raman_signal(self) -> None:
        """The demo collector samples the clicked image pixel directly."""
        world = self.backend.world
        cell_yx = world._cell_pixels_yx()[0]
        cell = self.backend.collector.collect_spectra_image_points(
            [cell_yx], 1000
        )
        background = self.backend.collector.collect_spectra_image_points(
            [[2.0, 2.0]], 1000
        )
        self.assertGreater(cell.max(), background.max() * 5)

    def test_spatial_map_samples_each_image_pixel(self) -> None:
        world = self.backend.world
        cell_yx = world._cell_pixels_yx()[0]
        grid_yx = np.vstack((cell_yx, [2.0, 2.0]))
        spectra = self.backend.collector.collect_spectra_image_points(
            grid_yx, 1000
        )
        self.assertEqual(spectra.shape, (2, world.spectrum_pixels))
        self.assertGreater(spectra[0].max(), spectra[1].max() * 5)

    def test_mda_voltage_path_hits_the_selected_image_cell(self) -> None:
        backend = create_demo_backend(image_height=512, image_width=512)
        world = backend.world
        # Use an off-center, asymmetric target so swapped axes or a reduced
        # galvo range cannot accidentally pass this test.
        world.cell_positions_um[0] = (80.0, -40.0)
        cell_yx = world._cell_pixels_yx()[0]
        normalized_yx = cell_yx / np.array(
            [world.image_height, world.image_width], dtype=float
        )
        cell_volts = backend.transformer.BF_to_volts(
            normalized_yx, max_volts=1.8
        )
        background_yx = np.array([10.0, 10.0])
        background_volts = backend.transformer.BF_to_volts(
            background_yx / np.array([512.0, 512.0]), max_volts=1.8
        )
        cell = backend.collector.collect_spectra_pts(cell_volts, 1000)
        background = backend.collector.collect_spectra_pts(
            background_volts, 1000
        )
        np.testing.assert_allclose(world.galvo_pixel_yx(), background_yx)
        self.assertGreater(np.ptp(cell), np.ptp(background) * 5)

    def test_axial_cell_profile_peaks_near_focus(self) -> None:
        world = self.backend.world
        cell_yx = world._cell_pixels_yx()[0]
        number_of_positions = 9
        z_positions = np.linspace(-10.0, 10.0, number_of_positions)
        peak_intensities = []
        for z in z_positions:
            self.backend.core.setZPosition(float(z))
            spectra = self.backend.collector.collect_spectra_image_points(
                np.tile(cell_yx, (5, 1)), 1000
            )
            peak_intensities.append(float(np.mean(spectra, axis=0).max()))
        self.assertEqual(len(peak_intensities), number_of_positions)
        self.assertLess(abs(z_positions[int(np.argmax(peak_intensities))]), 3.0)
        self.assertGreater(max(peak_intensities), min(peak_intensities) * 5)

    def test_exposure_changes_spectrum_intensity(self) -> None:
        self.backend.world.cell_positions_um[0] = (0.0, 0.0)
        short = self.backend.collector.collect_spectra_pts([[0.0, 0.0]], 100)
        long = self.backend.collector.collect_spectra_pts([[0.0, 0.0]], 1000)
        self.assertGreater(long.max(), short.max() * 5)

    def test_wavelength_and_grating_change_the_spectrum(self) -> None:
        self.backend.world.cell_positions_um[0] = (0.0, 0.0)
        collector = self.backend.collector
        initial = collector.collect_spectra_pts([[0.0, 0.0]], 1000)
        collector.set_wavelength(730.0)
        wavelength_changed = collector.collect_spectra_pts(
            [[0.0, 0.0]], 1000
        )
        collector.set_grating(2)
        grating_changed = collector.collect_spectra_pts([[0.0, 0.0]], 1000)
        self.assertFalse(np.allclose(initial, wavelength_changed))
        self.assertFalse(np.allclose(wavelength_changed, grating_changed))

    def test_rm_channel_shows_last_laser_target_without_a_shutter(self) -> None:
        self.backend.core.setConfig("Channel", "RM")
        without_laser = self.backend.core.snap()
        self.backend.daq.galvo.write(np.array([0.4, -0.2]))
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
