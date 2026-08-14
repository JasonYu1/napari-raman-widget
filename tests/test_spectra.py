import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from napari_raman_widget.spectra import (
    save_collection_record,
    sum_detector_rows,
)


class CollectionRecordTests(unittest.TestCase):
    def test_saves_data_and_metadata_in_one_zarr_dataset(self) -> None:
        data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        metadata = {
            "read_mode": "image",
            "collection_elapsed_seconds": 1.25,
        }
        with tempfile.TemporaryDirectory() as directory:
            store_path = save_collection_record(
                Path(directory) / "point-collection.npy",
                data,
                metadata,
            )

            self.assertEqual(store_path.suffix, ".zarr")
            self.assertEqual(
                [path.name for path in Path(directory).iterdir()],
                ["point-collection.zarr"],
            )
            saved = xr.open_zarr(store_path, consolidated=True)
            try:
                np.testing.assert_array_equal(saved["signal"].values, data)
                self.assertEqual(
                    saved["signal"].dims,
                    ("repeat", "detector_y", "detector_x"),
                )
                self.assertEqual(saved.attrs["data_shape"], [2, 3, 4])
                self.assertEqual(saved.attrs["data_dtype"], "uint16")
                self.assertEqual(
                    saved.attrs["collection_elapsed_seconds"], 1.25
                )
            finally:
                saved.close()


class DetectorRowSumTests(unittest.TestCase):
    def test_sums_an_inclusive_row_range(self) -> None:
        image = np.arange(20).reshape(4, 5)

        result = sum_detector_rows(image, 1, 2)

        np.testing.assert_array_equal(result, image[1] + image[2])
        self.assertEqual(result.dtype, np.dtype(float))

    def test_rejects_reversed_or_out_of_bounds_ranges(self) -> None:
        image = np.zeros((4, 5))

        with self.assertRaisesRegex(ValueError, "end_row"):
            sum_detector_rows(image, 3, 2)
        with self.assertRaisesRegex(ValueError, "start_row"):
            sum_detector_rows(image, -1, 2)

    def test_requires_a_detector_image(self) -> None:
        with self.assertRaisesRegex(ValueError, "detector image"):
            sum_detector_rows(np.zeros((2, 3, 4)), 0, 1)


if __name__ == "__main__":
    unittest.main()
