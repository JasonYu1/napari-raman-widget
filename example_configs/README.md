# Example configurations

These files are examples showing the formats expected by
napari-raman-widget. They are not universal calibration files.

## Microscope configuration

`microscope/example_micromanager.cfg` demonstrates where a Micro-Manager
configuration may be stored. Replace device names and properties with those
for the local microscope.

## Coordinate-transform model

`models/example_coordinate_transform.json` maps normalized brightfield
coordinates to Raman targeting coordinates.

## Vandermonde model

`models/example_vandermonde_model.json` maps image-pixel offsets to stage
offsets for centered-cell acquisition.

Calibration models are specific to the microscope alignment, camera,
objective, image orientation, and stage. Generate new models for each
instrument rather than using these examples for acquisition.