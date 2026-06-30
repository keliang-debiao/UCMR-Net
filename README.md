# UCMR-Net

This repository contains the code-only package for **UCMR-Net**, a text-anchored residual fusion framework with adaptive residual weighting for multimodal sentiment intensity prediction.

The package is intended to provide runnable code for model training, unified baseline execution, controlled ablation, missing-modality evaluation, calibration analysis, residual reliability analysis, modality corruption analysis, CMU-MOSEI validation, and significance testing.

This repository does **not** include experimental result tables, pretrained checkpoints, prediction files, generated figures, or processed datasets.

## Repository contents

```text
UCMR-Net_full_code_only/
├── configs/
│   ├── mosi_ucmr.yaml
│   └── mosei_ucmr.yaml
├── scripts/
│   ├── train_ucmr_net.py
│   ├── run_unified_baselines.py
│   ├── run_controlled_ablation.py
│   ├── run_missing_modality.py
│   ├── run_random_missing.py
│   ├── run_calibration.py
│   ├── run_reliability_error.py
│   ├── run_modality_corruption.py
│   ├── run_mosei_validation.py
│   ├── run_significance_tests.py
│   ├── run_all_experiments.py
│   └── _bootstrap.py
└── src/
    └── ucmr_net/
        ├── data/
        ├── models/
        ├── training/
        └── utils/
```

## Main modules

### `configs/`

This directory stores YAML configuration files for the supported experimental settings.

- `mosi_ucmr.yaml`: configuration for CMU-MOSI-style experiments.
- `mosei_ucmr.yaml`: configuration for CMU-MOSEI-style validation.

The configuration files define runtime options, modality dimensions, model hyperparameters, training phases, batch size, random seeds, loss weights, and optimization settings.

### `src/ucmr_net/data/`

This module provides the data-loading utilities for multimodal numeric feature files.

Main files:

- `multimodal_dataset.py`: implements a generic NPZ dataset loader for text, audio, visual, and label arrays.
- `factory.py`: builds train, validation, and test data loaders.

The default data loader expects precomputed numeric features. Text input is expected to be precomputed BERT-derived contextual features by default. Audio and visual inputs may be either fixed-size vectors or temporal feature sequences.

### `src/ucmr_net/models/`

This module contains the model definitions.

Main files:

- `ucmr_net.py`: implements the UCMR-Net architecture.
- `components.py`: contains reusable neural network components.
- `baselines.py`: contains baseline model implementations used by the unified baseline runner.

The UCMR-Net implementation includes text-anchored residual fusion, acoustic and visual encoders, adaptive residual weighting, residual reliability-risk scoring, and multi-task prediction heads.

### `src/ucmr_net/training/`

This module contains the training and loss functions.

Main files:

- `trainer.py`: implements the two-phase training loop, validation-based checkpoint selection, evaluation, and masked-modality distillation workflow.
- `losses.py`: implements regression, correlation-oriented, binary polarity, ordinal, and distillation-related loss components.

### `src/ucmr_net/utils/`

This module contains utility functions.

Main files:

- `config.py`: loads YAML configuration files and saves JSON outputs.
- `metrics.py`: implements regression, binary polarity, calibration, correlation, confidence-bin, and summary metric utilities.

## Running scripts

All runnable scripts are stored in the `scripts/` directory.

### Train UCMR-Net

```bash
python scripts/train_ucmr_net.py \
  --config configs/mosi_ucmr.yaml \
  --data path/to/mosi_features.npz \
  --output-dir outputs/ucmr
```

### Run unified baselines

```bash
python scripts/run_unified_baselines.py \
  --config configs/mosi_ucmr.yaml \
  --data path/to/mosi_features.npz \
  --output-dir outputs/baselines
```

### Run controlled ablation

```bash
python scripts/run_controlled_ablation.py \
  --config configs/mosi_ucmr.yaml \
  --data path/to/mosi_features.npz \
  --output-dir outputs/ablation
```

### Run fixed missing-modality evaluation

```bash
python scripts/run_missing_modality.py \
  --config configs/mosi_ucmr.yaml \
  --data path/to/mosi_features.npz \
  --checkpoint path/to/checkpoint.pt \
  --output-dir outputs/missing_modality
```

### Run random missing-rate evaluation

```bash
python scripts/run_random_missing.py \
  --config configs/mosi_ucmr.yaml \
  --data path/to/mosi_features.npz \
  --checkpoint path/to/checkpoint.pt \
  --output-dir outputs/random_missing
```

### Run calibration analysis

```bash
python scripts/run_calibration.py \
  --config configs/mosi_ucmr.yaml \
  --data path/to/mosi_features.npz \
  --checkpoint path/to/checkpoint.pt \
  --output-dir outputs/calibration
```

### Run residual reliability-error analysis

```bash
python scripts/run_reliability_error.py \
  --config configs/mosi_ucmr.yaml \
  --data path/to/mosi_features.npz \
  --checkpoint path/to/checkpoint.pt \
  --output-dir outputs/reliability_error
```

### Run modality corruption analysis

```bash
python scripts/run_modality_corruption.py \
  --config configs/mosi_ucmr.yaml \
  --data path/to/mosi_features.npz \
  --checkpoint path/to/checkpoint.pt \
  --output-dir outputs/corruption
```

### Run CMU-MOSEI validation

```bash
python scripts/run_mosei_validation.py \
  --config configs/mosei_ucmr.yaml \
  --mosei-data path/to/mosei_features.npz \
  --output-dir outputs/mosei
```

### Run significance testing

```bash
python scripts/run_significance_tests.py \
  --metrics-json path/to/unified_baseline_metrics.json \
  --output-dir outputs/significance
```

### Run the full experiment pipeline

```bash
python scripts/run_all_experiments.py \
  --config configs/mosi_ucmr.yaml \
  --mosi-data path/to/mosi_features.npz \
  --mosei-data path/to/mosei_features.npz \
  --output-dir outputs/full_pipeline
```

## Expected data format

The data loader expects one `.npz` file containing train, validation, and test arrays.

Supported key styles include split-first or modality-first naming, for example:

```text
train_text, train_audio, train_visual, train_y
val_text, val_audio, val_visual, val_y
test_text, test_audio, test_visual, test_y
```

or:

```text
text_train, audio_train, visual_train, y_train
text_val, audio_val, visual_val, y_val
text_test, audio_test, visual_test, y_test
```

Text arrays should contain numeric contextual features. Audio and visual arrays may be either fixed-size vectors or temporal sequences.

## Dependencies

The code uses Python, PyTorch, NumPy, scikit-learn-compatible utilities, and YAML configuration loading.

A typical environment should include:

```bash
pip install torch numpy pyyaml scikit-learn
```

Additional packages may be required depending on the local feature extraction or data preparation pipeline used before running this repository.

## Notes

- This repository is code-only.
- No experimental result tables are included.
- No pretrained model checkpoints are included.
- No prediction outputs are included.
- No processed CMU-MOSI or CMU-MOSEI data files are included.
- Users should prepare their own feature files and run the scripts from the command line.
