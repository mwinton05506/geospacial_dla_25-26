# Geospacial DLA

Tools for turning DMSP SSUSI auroral EDR files into polar dial-plot images, then using those images for PBI and streamer model experiments.

## What the Code Does

The repo has two main workflows:

1. Build image datasets from geospace data.
   `src/scripts/create_images.py` pulls AE index values from CDAWeb OMNI data, finds high-AE dates, matches SSJ midnight passes to overlapping SSUSI EDR files, and saves SSUSI polar images plus metadata.

2. Train and evaluate image models.
   The notebooks under `src/models/` train or evaluate TensorFlow models for automatically detecting PBIs and streamers. The shared `DenseCL` implementation is in `src/models/utils/DenseCL.py`.

## Repository Map

- `src/utils/ssusi_edr.py` reads one SSUSI EDR file, extracts the selected FUV radiance band, normalizes negative radiance values to zero, converts magnetic local time to degrees, and exposes pass data for plotting or model input.
- `src/utils/plotting_ssusi_edr.py` coordinates the image-generation workflow: finds SSJ midnight passes, matches them with SSUSI EDR files, plots each pass, and writes `ssusi_metadata.csv`.
- `src/utils/omni_data.py` wraps CDAWeb access for OMNI hourly data, mainly AE index retrieval.
- `src/utils/helper_funcs.py` contains coordinate, polar-plot, and nearest-neighbor interpolation helpers used by the SSUSI utilities.
- `src/utils/time_dict.py` provides nearest-timestamp lookup for matching pass end times to AE values.
- `src/scripts/create_images.py` is the main batch script for generating EDR image datasets.
- `src/scripts/collect_images.py` samples generated images into smaller labeling batches.
- `src/scripts/collect_annotated_images.py` groups annotated images by category from a COCO-style annotation JSON.
- `src/scripts/collect_non_annotaed_images.py` creates negative-category folders for images without PBI or streamer annotations.
- `src/scripts/add_metadata_to_annotations.py` is currently only a stub.
- `src/models/utils/` contains reusable TensorFlow callbacks and the DenseCL model class.

## Setup

Install dependencies with:

```bash
python setup.py
```

or manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some packages depend on geospace/scientific libraries that may need system-level support, especially Fortran tooling for `apexpy`.

## Typical Image-Generation Flow

1. Fetch AE index data for the configured date range.
2. Keep dates where AE exceeds `AE_CUTOFF`.
3. Search SSJ CDF files for passes near midnight magnetic local time.
4. Match those pass times to SSUSI EDR NetCDF files.
5. Plot SSUSI radiance as polar dial images.
6. Save generated images by date and append metadata to `ssusi_metadata.csv`.

## Modeling Notes

`DenseCL.py` implements a student/teacher contrastive model with:

- a shared image backbone,
- global and dense projection heads,
- a momentum teacher updated after each training step,
- a queue of negative samples,
- a warmup/ramp schedule that introduces dense contrastive loss after the global loss has started training.