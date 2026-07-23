"""Image acquisition and point selection for Raman calibration."""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nidaqmx
import numpy as np
import scipy.ndimage as ndi
import xarray as xr
from raman_mda_engine.aiming import SimpleGridSource
from scipy.interpolate import griddata
from scipy.ndimage import binary_dilation, center_of_mass
from skimage.measure import label
from tqdm.auto import tqdm

from .coordinate_transform import CoordTransformer

__all__ = [
    "Calibrator",
    "ManualImageSelector",
]


class Calibrator:
    """Acquire data used to calibrate Raman targeting.

    Parameters
    ----------
    core
        Microscope control object.
    daq
        DAQ object controlling the galvo.
    transformer
        Current brightfield-to-galvo coordinate transformer.
    collector
        Raman spectral collector.
    repeats
        Number of spectral acquisitions at each calibration position.
    exposure
        Raman exposure time at each position.
    max_volts
        Maximum galvo voltage used during calibration.
    """

    def __init__(
        self,
        core: Any,
        daq: Any,
        transformer: CoordTransformer,
        collector: Any,
        repeats: int,
        exposure: float,
        max_volts: float = 1.5,
    ) -> None:
        self.core = core
        self.daq = daq
        self.transformer = transformer
        self.collector = collector
        self.max_volts = float(max_volts)

        # These names are retained because the widget currently uses them.
        self.N = int(repeats)
        self.exp = float(exposure)

    def collect_calibration_images(
        self,
        volts: np.ndarray,
        threshold: float,
        relative_positions: np.ndarray | None = None,
        save_directory: str | Path = ".",
    ) -> xr.Dataset:
        """Collect images and spectra at specified galvo voltages."""
        volts = np.asarray(volts, dtype=float)

        if volts.ndim != 2 or volts.shape[1] != 2:
            raise ValueError("volts must have shape (N, 2).")

        if relative_positions is not None:
            relative_positions = np.asarray(
                relative_positions,
                dtype=float,
            )

            if relative_positions.shape != volts.shape:
                raise ValueError(
                    "relative_positions must have the same shape as volts."
                )

        self.daq._galvo.out_stream.output_buf_size = 1000
        self.daq._galvo.timing.cfg_samp_clk_timing(
            1e4,
            sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
        )

        outside_range = np.any(
            np.abs(volts) > threshold,
            axis=1,
        )
        accepted_volts = volts[~outside_range]

        images = []
        spectra = []

        self.core.setAutoShutter(False)

        try:
            for voltage_xy in tqdm(
                accepted_volts,
                desc="Collecting calibration data",
            ):
                repeated_voltages = np.tile(
                    voltage_xy,
                    (self.N, 1),
                )

                spectrum = self.collector.collect_spectra_pts(
                    repeated_voltages,
                    self.exp,
                )

                spectra.append(spectrum)
                images.append(self.core.snap())
                time.sleep(0.1)
        finally:
            self.core.setAutoShutter(True)

        images = np.asarray(images)
        spectra = np.asarray(spectra)

        dataset = xr.Dataset(
            {
                "laser_pos": xr.DataArray(
                    accepted_volts,
                    dims=("idx", "volt"),
                ),
                "imgs": xr.DataArray(
                    images,
                    dims=("idx", "Y", "X"),
                ),
                "specs": xr.DataArray(
                    spectra,
                    dims=("idx", "N", "spec_dim"),
                ),
                "BF_bkd": xr.DataArray(
                    self.core.snap(),
                    dims=("Y", "X"),
                ),
            }
        )

        if relative_positions is not None:
            dataset["rel_BF_pos"] = xr.DataArray(
                relative_positions[~outside_range],
                dims=("idx", "rel_BF"),
            )

        dataset.attrs["time"] = datetime.now().isoformat()

        save_directory = Path(save_directory)
        save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_path = (
            save_directory
            / f"calibration_{uuid.uuid4()}.zarr"
        )

        dataset.to_zarr(save_path)
        print(f"Saved calibration dataset to {save_path}")

        return dataset

    def calibrate(
        self,
        N: int = 5,
        threshold: float = 1.5,
        plot: bool = True,
        save_directory: str | Path = ".",
    ) -> xr.Dataset:
        """Acquire a grid of Raman calibration measurements."""
        self.daq.galvo.stop()
        self.core.setConfig("Channel", "RM")
        self.core.setShutterOpen("Fluoshutter", True)

        try:
            width = self.core.getImageWidth()
            height = self.core.getImageHeight()

            grid = SimpleGridSource(N, N)
            relative_brightfield = np.asarray(
                grid.get_current_points()
            )

            pixel_positions = (
                relative_brightfield
                * np.array([width, height])
            )

            transformer_input = (
                pixel_positions[:, ::-1]
                / np.array([height, width])
            )

            volts = self.transformer.BF_to_volts(
                transformer_input,
                max_volts=self.max_volts,
            )

            self.core.stopSequenceAcquisition()
            self.core.setExposure(1)

            dataset = self.collect_calibration_images(
                volts=volts,
                threshold=threshold,
                relative_positions=pixel_positions,
                save_directory=save_directory,
            )
        finally:
            self.core.setShutterOpen(
                "Fluoshutter",
                False,
            )

        if plot:
            plt.figure()
            plt.imshow(
                dataset["imgs"].max(axis=0),
                cmap="gray",
            )
            plt.scatter(
                pixel_positions[:, 0],
                pixel_positions[:, 1],
                color="red",
            )
            plt.title("Calibration positions")

        return dataset

    def save_new_model(
        self,
        dataset: xr.Dataset,
        selected_points: np.ndarray,
        model_name: str | Path,
    ) -> CoordTransformer:
        """Fit and save a new coordinate transformation model."""
        selected_points = np.asarray(
            selected_points,
            dtype=float,
        )

        valid = ~np.isnan(selected_points).any(axis=1)

        image_shape = np.asarray(
            dataset["imgs"].shape[-2:],
            dtype=float,
        )

        relative_brightfield = (
            selected_points / image_shape
        )[valid]

        relative_raman = (
            (
                dataset["laser_pos"].values
                + self.max_volts
            )
            / (2 * self.max_volts)
        )[valid]

        degrees = (3, 3)

        model = CoordTransformer.fit_model(
            relative_brightfield,
            relative_raman,
            degrees,
            alpha=0.001,
        )

        model_path = Path(model_name)

        if model_path.suffix.lower() != ".json":
            model_path = model_path.with_suffix(".json")

        CoordTransformer.save_model(
            model_path,
            model,
            degrees,
        )

        return CoordTransformer.from_json(model_path)

    def interpolate2d(
        self,
        dataset: xr.Dataset,
        plot: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate calibration intensity over the camera image."""
        intensity = np.median(
            dataset["specs"].values,
            axis=1,
        )
        coordinates = dataset["rel_BF_pos"].values

        height = dataset["imgs"].shape[-2]
        width = dataset["imgs"].shape[-1]

        grid_x, grid_y = np.meshgrid(
            np.linspace(0, width, width),
            np.linspace(0, height, height),
        )

        grid_z = griddata(
            coordinates,
            intensity,
            (grid_x, grid_y),
            method="cubic",
        )

        if plot:
            plt.figure()
            plt.imshow(
                grid_z,
                extent=(0, width, 0, height),
                origin="lower",
                aspect="auto",
            )
            plt.imshow(
                dataset["imgs"].max(axis=0),
                alpha=0.1,
                cmap="gray",
            )
            plt.scatter(
                coordinates[:, 0],
                coordinates[:, 1],
                c=intensity,
                edgecolor="black",
            )
            plt.title("Interpolated calibration intensity")

        return grid_x, grid_y, grid_z


class ManualImageSelector:
    """Review or manually select the laser position in each image."""

    def __init__(
        self,
        dataset: xr.Dataset,
    ) -> None:
        self.images = np.asarray(
            dataset["imgs"].values
        )

        if len(self.images) == 0:
            raise ValueError(
                "The calibration dataset contains no images."
            )

        self.coms = self.find_coms(self.images)
        self.num_images = self.images.shape[0]
        self.current_idx = 0

        self.selected_points = [
            (None, None)
            for _ in range(self.num_images)
        ]
        self.manual_selections = [
            False
            for _ in range(self.num_images)
        ]

        self.fig = plt.figure(figsize=(15, 7))
        self.ax_full = self.fig.add_subplot(121)
        self.ax_zoom = self.fig.add_subplot(122)

        self.zoom_window_size = 100
        self.zoom_scale = 4
        self.zoom_coords = (0, 0, 0, 0)

        self.cid = None

        self.show_image()

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.on_key_press,
        )

    @staticmethod
    def find_coms(
        images: np.ndarray,
    ) -> np.ndarray:
        """Estimate a center of mass for each image."""
        centers = []

        for image in images:
            mask = (
                image
                > image.mean() + 4 * image.std()
            )

            if mask.any():
                center = ndi.center_of_mass(mask)
            else:
                center = (
                    image.shape[0] / 2,
                    image.shape[1] / 2,
                )

            centers.append(center)

        return np.asarray(centers)

    @staticmethod
    def make_image_mask(
        image: np.ndarray,
        point: tuple[float, float],
    ) -> np.ndarray:
        """Create an intensity mask near a selected point."""
        center_y, center_x = point
        rows, columns = np.indices(image.shape)

        within_radius = (
            np.sqrt(
                (rows - center_y) ** 2
                + (columns - center_x) ** 2
            )
            <= 300
        )

        mask = (
            image > image.mean() + 3 * image.std()
        ) & within_radius

        mask = binary_dilation(
            mask,
            structure=np.ones((1, 1)),
        )

        if not mask.any():
            return np.zeros_like(image)

        labeled_mask = label(mask)
        sizes = np.bincount(labeled_mask.ravel())
        sizes[0] = 0

        largest_label = sizes.argmax()
        return (labeled_mask == largest_label) * image

    def update_zoom_window(
        self,
        center: tuple[float, float] | None = None,
    ) -> None:
        """Update the zoomed view around a selected point."""
        image = self.images[self.current_idx]

        if center is None:
            center = self.selected_points[
                self.current_idx
            ]

        center_y, center_x = center

        if center_y is None or center_x is None:
            center_y, center_x = (
                image.shape[0] / 2,
                image.shape[1] / 2,
            )

        half_size = self.zoom_window_size // 2

        y_min = max(
            0,
            int(center_y - half_size),
        )
        y_max = min(
            image.shape[0],
            int(center_y + half_size),
        )
        x_min = max(
            0,
            int(center_x - half_size),
        )
        x_max = min(
            image.shape[1],
            int(center_x + half_size),
        )

        self.ax_zoom.clear()
        self.ax_zoom.imshow(
            image[y_min:y_max, x_min:x_max],
            cmap="gray",
        )

        if (
            y_min <= center_y <= y_max
            and x_min <= center_x <= x_max
        ):
            self.ax_zoom.scatter(
                center_x - x_min,
                center_y - y_min,
                color="red",
                marker="x",
                s=100,
            )

        self.ax_zoom.set_title("Zoomed view")
        self.zoom_coords = (
            x_min,
            x_max,
            y_min,
            y_max,
        )

        self.fig.canvas.draw_idle()

    def show_image(self) -> None:
        """Show the current image and its selected center."""
        self.ax_full.clear()
        image = self.images[self.current_idx]

        if self.manual_selections[self.current_idx]:
            center_y, center_x = (
                self.selected_points[
                    self.current_idx
                ]
            )
        else:
            selected = self.selected_points[
                self.current_idx
            ]

            if selected == (None, None):
                center_y, center_x = self.coms[
                    self.current_idx
                ]
                self.selected_points[
                    self.current_idx
                ] = (center_y, center_x)
            else:
                center_y, center_x = selected

        if (
            center_x is not None
            and center_y is not None
            and not np.isnan([center_y, center_x]).any()
        ):
            masked_image = self.make_image_mask(
                image,
                (center_y, center_x),
            )

            if (
                not self.manual_selections[
                    self.current_idx
                ]
                and np.any(masked_image)
            ):
                center_y, center_x = center_of_mass(
                    masked_image
                )

                self.selected_points[
                    self.current_idx
                ] = (
                    int(center_y),
                    int(center_x),
                )

        self.ax_full.imshow(
            image,
            cmap="gray",
        )

        valid_point = (
            center_x is not None
            and center_y is not None
            and not np.isnan(
                [center_y, center_x]
            ).any()
        )

        if valid_point:
            label_text = (
                "Manual selection"
                if self.manual_selections[
                    self.current_idx
                ]
                else "Automatic center"
            )

            self.ax_full.scatter(
                center_x,
                center_y,
                color="red",
                marker="x",
                s=100,
                label=label_text,
            )
            self.ax_full.legend()

        self.ax_full.set_title(
            f"Image {self.current_idx + 1}/"
            f"{self.num_images}\n"
            "Click to set center | Enter next | "
            "Backspace previous | R reset | N unavailable"
        )

        self.update_zoom_window(
            (center_y, center_x)
        )

        if self.cid is not None:
            self.fig.canvas.mpl_disconnect(
                self.cid
            )

        self.cid = self.fig.canvas.mpl_connect(
            "button_press_event",
            self.on_click,
        )

    def on_click(self, event) -> None:
        """Store a point selected in either image view."""
        if event.xdata is None or event.ydata is None:
            return

        if event.inaxes is self.ax_full:
            center_y = int(event.ydata)
            center_x = int(event.xdata)

        elif event.inaxes is self.ax_zoom:
            x_min, _, y_min, _ = self.zoom_coords
            center_x = int(event.xdata + x_min)
            center_y = int(event.ydata + y_min)

        else:
            return

        self.selected_points[self.current_idx] = (
            center_y,
            center_x,
        )
        self.manual_selections[self.current_idx] = True
        self.show_image()

    def on_key_press(self, event) -> None:
        """Handle image-selection keyboard controls."""
        key = event.key

        if key == "enter":
            if self.current_idx < self.num_images - 1:
                self.current_idx += 1
                self.show_image()
            else:
                print("Finished calibration image selection.")
                plt.close(self.fig)

        elif key == "backspace":
            if self.current_idx > 0:
                self.current_idx -= 1
                self.show_image()

        elif key and key.lower() == "r":
            self.manual_selections[self.current_idx] = False
            self.selected_points[self.current_idx] = (
                None,
                None,
            )
            self.show_image()

        elif key and key.lower() == "n":
            self.manual_selections[self.current_idx] = True
            self.selected_points[self.current_idx] = (
                np.nan,
                np.nan,
            )

            if self.current_idx < self.num_images - 1:
                self.current_idx += 1
                self.show_image()
            else:
                print("Finished calibration image selection.")
                plt.close(self.fig)

    def start(
        self,
    ) -> list[tuple[float | None, float | None]]:
        """Display the selector and return its selected points."""
        plt.show()
        return self.selected_points