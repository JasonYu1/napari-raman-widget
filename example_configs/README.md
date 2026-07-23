# Example configurations

These files are examples showing the formats expected by
napari-raman-widget. They are not universal calibration or instrument files.

## Microscope configuration

`microscope/example_micromanager.cfg` demonstrates where a Micro-Manager
configuration may be stored. Replace its device names, adapters, ports, and
properties with those used by the local microscope.

## Coordinate-transform model

`models/example_coordinate_transform_model.json` maps normalized brightfield
coordinates to Raman targeting coordinates.

This model depends on the optical alignment, camera orientation, image size,
galvanometer response, and objective used during calibration.

## Vandermonde model

`models/example_vandermonde_model.json` maps image-pixel offsets to stage
offsets for centered-cell acquisition.

This model allows a cell selected away from the image center to be repositioned
at the center of a new field of view.

## Particle-tracking configuration

`tracking/particle_config.json` contains the configuration used by btrack for
particle linking and tracking.

The file defines tracking behavior such as the motion model, observation model,
search behavior, and track-generation settings. Adjust these values for the
expected cell motion, image interval, segmentation quality, and experiment
duration.

The configuration should be selected in the tracking controls before running an
acquisition that uses segmentation and tracking. Confirm its performance on
representative data before using it for a full experiment.

## Instrument-specific use

The files in this directory are templates and examples. Calibration and
tracking parameters may depend on:

- Microscope hardware and device names
- Camera dimensions and orientation
- Objective and optical alignment
- Stage direction and coordinate convention
- Galvanometer response
- Cell type and expected motion
- Image interval and acquisition duration
- Segmentation and detection quality

Generate new calibration models for each instrument. Validate the tracking
configuration with representative recordings rather than assuming the example
settings are suitable for every experiment.