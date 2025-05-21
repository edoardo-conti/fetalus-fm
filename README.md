# fetal-foundation-model

A PyTorch-based project for working with various fetal ultrasound datasets.

## Supported Datasets

The project currently supports the following fetal ultrasound datasets:

- **Fetal_HC18_Z1327317**: Dataset for fetal head circumference measurement
- **Fetal_Planes_DB_Z3904280**: Common maternal-fetal ultrasound plane images
- **Fetal_Planes_Africa_Z7540448**: Maternal fetal US planes from low-resource imaging settings in five African countries
- **Fetal_Abdominal_MD4GCPM9DSC3**: Dataset for fetal abdominal structures segmentation
- **Fetal_PSFH_Z7851339**: Dataset for Pubic Symphysis and Fetal Head Segmentation (PSFHS)

## Requirements

The following Python packages are required:
- scikit-learn
- pillow
- torch
- torchvision
- matplotlib
- seaborn
- SimpleITK

You can install the requirements using:
```bash
pip install -r requirements.txt
```

## Usage

To use a dataset, run the main script with the path to your dataset:

```bash
python main.py --dataset <path_to_dataset>
```

Optional arguments:
- `--reset`: Reset the project's dataset files

## Dataset Classes

Each dataset is implemented as a PyTorch VisionDataset with standardized interfaces:

- `FetalHC18`: Head circumference measurement dataset
- `FetalPlanesDB`: Common maternal-fetal ultrasound planes
- `FetalPlanesAfrica`: Fetal planes from African countries
- `FetalAbdominal`: Abdominal structures segmentation
- `FetalPSFH`: Pubic symphysis and fetal head segmentation

Each dataset class provides:
- Train/test splits
- Image loading with transforms
- Access to targets and class labels
- Patient ID tracking
- Metadata handling specific to each dataset