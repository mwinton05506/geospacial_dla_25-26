## Model Creation

This directory contains the model experiments for automatically detecting PBIs and streamers from generated SSUSI dial-plot images.

## Main Workflow

The modeling code is organized around three steps:

1. Pretrain a visual backbone with DenseCL on unlabeled SSUSI images.
2. Freeze that backbone and train a small classification head for PBI or streamer labels.
3. Evaluate learned representations with linear probes, kNN, confusion matrices, and t-SNE plots.

## Notebooks

- `densecl.ipynb` trains or resumes DenseCL pretraining. It builds a ResNet50 backbone, creates two augmented views of each image, trains with global and dense contrastive losses, and saves checkpoints under `logs/<run_name>/checkpoints/`.
- `classification_head.ipynb` freezes a backbone and trains a binary classifier for either `pbi` or `streamer`. It supports contrastive, ImageNet, or random ResNet50 backbones and logs training metrics under `logs/classification_head_<timestamp>_<mode>/`.
- `evaluate_representations.ipynb` compares frozen backbones without training a full classifier. It extracts features from labeled image folders, then runs logistic-regression probes, kNN, classification reports, confusion matrices, t-SNE, and per-class plots.
- `plot_maker.ipynb` makes summary figures from saved training and evaluation CSV files.
- `tensorflow_test.ipynb` is a small environment check notebook for TensorFlow.

## Utility Code

- `utils/DenseCL.py` defines the custom TensorFlow `DenseCL` model. It uses a student backbone, momentum teacher, global and dense projection heads, a negative-sample queue, and a warmup/ramp schedule for the dense loss.
- `utils/EpochLogger.py` updates `model.current_epoch` so `DenseCL.train_step()` can adjust the dense-loss weight over time.
- `utils/HistoryCSV.py` writes epoch metrics and learning rate values to CSV during training.

## Expected Data Layout

Most notebooks expect image folders where subdirectory names are class labels:

```text
data/
  data_sample_2_pbi/
    No_PBI/
    PBI/
  data_sample_2_streamer/
    No_Streamer/
    Streamer/
```

`densecl.ipynb` currently points to unlabeled images at `/glade/work/winton/data/images/edr_images`. Update `images_dir` before running it somewhere else.

## Outputs

- DenseCL checkpoints: `logs/dense_<timestamp>/checkpoints/cp-####.keras`
- DenseCL training metrics: `logs/dense_<timestamp>/training.csv`
- Classification-head metrics: `logs/classification_head_<timestamp>_<mode>/training.csv`
- Evaluation plots and CSVs: usually written under `<MODELS_DIR>/results/`

