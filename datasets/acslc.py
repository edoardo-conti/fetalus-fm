from pathlib import Path
from typing import Literal, Callable, Optional, Tuple, List, Set
from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk


# ================================================================================== 
# =================================== ACOUSLIC =====================================
# ================================================================================== 
class VD_ACOUSLIC(VisionDataset):
    def __init__(
        self,
        root: Path,
        split: Literal['train', 'test', 'val'],
        data_dir: Path = Path("./data_csv"),
        val_size: Optional[float] = 0.20,
        test_size: Optional[float] = 0.10,
        annotated: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Dataset 'VD_ACOUSLIC' for Prenatal Ultrasound Frames acquired using a pre-specified blind-sweep protocol.
        https://zenodo.org/records/12697994
        
        Args:
            root (str): Root directory of the dataset.
            split (Literal['train', 'val', 'test']): Dataset split, either 'train', 'val' or 'test'.
            data_dir (str): Directory where the processed dataset csv files will be stored.
            val_size (Optional[float]): Proportion of the training set to use for validation.
            test_size (Optional[float]): Proportion of the dataset to use for testing.
            target (Optional[Callable]): A function/transform that takes in the image and transforms it.
            target_transform (Optional[Callable]): A function/transform that takes in the target and transforms it.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of ['train', 'val', 'test']")
        
        self.root = Path(root)
        self.data_dir = Path(data_dir) / self.root.name
        self.split = split
        self.val_size = val_size
        self.test_size = test_size
        self.annotated = annotated
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        
        # Ensure the data directory exists or create it
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # CSV file paths
        self.data_csv = self.data_dir / "acslc.csv"
        self.train_csv = self.data_dir / "acslc_train.csv"
        self.val_csv = self.data_dir / "acslc_val.csv"
        self.test_csv = self.data_dir / "acslc_test.csv"
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)

        # Creating csv file of the dataset
        self.process_csv()

        # Split train/val/test
        self.split_std()

        # Loading dataframes based on the split
        self.data = pd.read_csv(getattr(self, f'{self.split}_csv'))
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}__{self.split}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        Loads the image and the corresponding target.

        Args:
            index (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, str]: Transformed image and target.
        """
        # TODO: Implement the logic to load the image and segmentation mask

    @property
    def targets(self) -> List[str]:
        """Returns the list of labels in the current dataset."""
        targets = self.data["class"]

        if self.target_transform:
            return self.target_transform(targets)
        else:
            return targets.tolist()

    @property
    def classes(self) -> List[str]:
        """Returns the unique classes in the dataset."""
        return sorted(set(self.targets))
    
    @property
    def patients(self) -> Set[str]:
        patients = self.data["patient_id"]

        if self.target_transform:
            return self.target_transform(patients)
        else:
            return patients.tolist()

    @property
    def structures_distr(self) -> pd.Series:
        """Returns a dict with structure counts and proportions in the split."""
        if "mask_structures" not in self.data.columns:
            return {}
        s = self.data["mask_structures"].dropna().apply(eval).explode()
        t = pd.read_csv(self.data_csv)["mask_structures"].dropna().apply(eval).explode()
        split_counts = s.value_counts()
        total_counts = t.value_counts()
        return {k: {"count": int(v), "perc": round(v / total_counts.get(k, 1), 2)} for k, v in split_counts.items()}
    
    
    def _load_mha(self, filepath: str):
        """Carica un file MHA e lo converte in un array NumPy."""
        return sitk.GetArrayFromImage(sitk.ReadImage(filepath))

    def process_csv(self):
        """Crea un CSV unico con tutte le informazioni del dataset a partire dal csv di riferimento."""
        if self.data_csv.exists():
            return

        csv_source = self.root / "circumferences/fetal_abdominal_circumferences_per_sweep.csv"
        df_source = pd.read_csv(csv_source, usecols=["uuid", "subject_id"])

        images_dir = self.root / "images/stacked_fetal_ultrasound"
        masks_dir = self.root / "masks/stacked_fetal_abdomen"

        data_list = []
        for _, row in tqdm(df_source.iterrows(), total=len(df_source), desc=f"Processing {csv_source}"):
            uuid = row["uuid"]
            patient_id = row["subject_id"]

            image_file = images_dir / f"{uuid}.mha"
            mask_file = masks_dir / f"{uuid}.mha"

            if not image_file.exists() or not mask_file.exists():
                print(f"⚠️ file '{uuid}' not found in '{images_dir}' or '{masks_dir}' - skipped")
                continue
            
            # Carica il file MHA e converte in array NumPy
            mask_array = self._load_mha(mask_file)

            num_frames = mask_array.shape[0]
            resolution = mask_array.shape[1:]

            # Ottimizzazione: cerca i frame non vuoti in modo vettoriale
            non_empty = mask_array.sum(axis=(1, 2)) > 0
            non_empty_masks = non_empty.sum()

            # Identifica dove, nei frame non vuoti, compare il label 1
            optimal = np.any(mask_array == 1, axis=(1, 2))
            optimal_masks_indices = np.where(non_empty & optimal)[0].tolist()
            suboptimal_masks_indices = np.where(non_empty & (~optimal))[0].tolist()

            data_list.append({
                "images_path": f"images/stacked_fetal_ultrasound/{uuid}.mha",
                "class": "Abdomen",
                "patient_id": int(patient_id),
                "num_frames": int(num_frames),
                "resolution": resolution,
                "masks_path": f"masks/stacked_fetal_abdomen/{uuid}.mha",
                "masks_structures": ['Abdomen'],
                "non_empty_masks": int(non_empty_masks),
                "optplane_masks": len(optimal_masks_indices),
                "optplane_masks_idxs": optimal_masks_indices,
                "suboptplane_masks": len(suboptimal_masks_indices),
                "suboptplane_masks_idxs": suboptimal_masks_indices
            })

        df_final = pd.DataFrame(data_list).sort_values(by='patient_id', ignore_index=True)
        df_final.to_csv(self.data_csv, index=False)
        print(f"✅ Full dataset saved in {self.data_csv}")


    def split_std(self):
        """Manages train and test partitioning."""
        if self.train_csv.exists() and self.test_csv.exists():
            return
        
        # Load dataset
        df = pd.read_csv(self.data_csv)

        # Get unique patient IDs
        patient_ids = sorted(df['patient_id'].unique())

        # Split patient IDs into train+val and test
        trainval_ids, test_ids = train_test_split(
            patient_ids,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
        )

        # Further split trainval into train and val
        if self.val_size and self.val_size > 0:
            val_relative_size = self.val_size / (1 - self.test_size)
            train_ids, val_ids = train_test_split(
                trainval_ids,
                test_size=val_relative_size,
                random_state=self.seed,
                shuffle=True,
            )
        else:
            train_ids = trainval_ids
            val_ids = []

        # Create DataFrames for each split
        train_df = df[df['patient_id'].isin(train_ids)]
        val_df = df[df['patient_id'].isin(val_ids)]
        test_df = df[df['patient_id'].isin(test_ids)]

        # Save splits
        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)
        test_df.to_csv(self.test_csv, index=False)

        print(f"✅ Split train/val/test completed and saved in {self.train_csv.parent}")