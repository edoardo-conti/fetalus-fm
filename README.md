# Fetal Ultrasound Foundation Model (FUS Foundation Model)

A PyTorch-based global foundation model for fetal ultrasound imaging using self-supervised DINO models for plane segmentation and related tasks.

## Features

- Self-supervised learning with DINOv2 and DINOv3 backbones
- Segmentation of fetal ultrasound structures and planes
- Support for multiple public fetal ultrasound datasets
- Configurable experiments for multi-dataset training
- PyTorch implementation with CUDA support

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/edoardo-conti/fetalus-fm
cd fetalus-fm
pip install -r requirements.txt
```

## Usage

### Training

Run training with specific experiment configuration:

```bash
python train.py --root /path/to/project --datasets /path/to/datasets --exp-id 0 --fine-tune
```

**Arguments:**
- `--root`: Path to project root (default: current directory)
- `--datasets`: Path to datasets directory
- `--exp-id`: Experiment ID from `configs/experiments.json` (default: 0)
- `--fine-tune`: Fine-tune the DINO backbone

### Dataset Loading

To load and inspect datasets:

```bash
python main.py --dataset /path/to/dataset
```

Options:
- `--reset`: Reset project's dataset files and splits

### Model Testing

Evaluate trained models on test sets:

```bash
python model_testing.py ...
```

## Supported Datasets

The project supports the following fetal ultrasound datasets:

- **HC18**: Fetal head circumference measurement
- **FABD**: Fetal abdominal structures segmentation
- **FPDB**: Fetal Planes Database (common maternal-fetal ultrasound planes)
- **IPSFH**: Pubic Symphysis and Fetal Head Segmentation
- **ACSLC**: ACOUSLIC Fetal Echocardiography

Datasets are automatically processed and split into train/validation/test sets with patient-level holdout.

## DINO Weights

Pre-trained DINOv2 weights are included in `dinov2_weights/`. DINOv3 requires external download or training.

## Project Structure

- `datasets/`: Dataset loading and processing classes
- `foundation/`: Core model architecture and DINO segmentator
- `dinov2/` & `dinov3/`: DINO model implementations
- `utils/`: Utility functions for data handling and analysis
- `configs/`: Experiment configurations
- `outputs/`: Training outputs, logs, and metrics
- `testing_preds/`: Model predictions on test sets

## License

Licensed under the terms in the LICENSE file.
