import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from napari_raman_widget.spectra import save_collection_record


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


if __name__ == "__main__":
    unittest.main()
