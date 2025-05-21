from pathlib import Path
from typing import Literal, Callable, Optional, Tuple, List, Set
from PIL import Image
from torchvision.datasets import VisionDataset
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk

# ================================================================================== 
# =========================== Fetal_ACOUSLIC_Z12697994 =============================
# ================================================================================== 
class FetalACOUSLIC(VisionDataset):
    def __init__(
        self,
        root: Path,
        data_dir: Path = Path("./data_csv"),
        split: Literal['train', 'test'] = 'train',
        #test_size: float = 0.2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Dataset 'Fetal_ACOUSLIC_Z12697994' for Prenatal Ultrasound Frames acquired using a pre-specified blind-sweep protocol.
        https://zenodo.org/records/12697994
        
        Args:
            root (str): Root directory of the dataset.
            data_dir (str): Directory where the processed dataset csv files will be stored.
            split (Literal['train', 'test']): Dataset split, either 'train' or 'test'.
            target (Optional[Callable]): A function/transform that takes in the image and transforms it.
            target_transform (Optional[Callable]): A function/transform that takes in the target and transforms it.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        if split not in ['train', 'test']:
            raise ValueError("split must be one of ['train', 'test']")
        
        self.root = Path(root)
        self.data_dir = data_dir / self.root.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        #self.test_size = test_size
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        
        # CSV file paths
        self.data_csv = self.data_dir / "facouslic.csv"
        self.train_csv = self.data_dir / "facouslic_train.csv"
        self.test_csv = self.data_dir / "facouslic_test.csv"
        
        # Creating csv file of the dataset
        self.process_csv()

        # Split train/test
        self.split_std()

        # Loading dataframes based on the split
        self.data = pd.read_csv(getattr(self, f'{self.split}_csv'))

    def __str__(self) -> str:
        return f"FetalACOUSLIC_{self.split}"

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
        img_path = self.root / self.data.iloc[index]["path"]
        image = Image.open(img_path)
        image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # TODO: target
        target = str(self.data.iloc[index]["class"])
        if self.target_transform:
            target = self.target_transform(target)
        
        return image, target

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

    
    def _load_mha(self, filepath: str):
        """Carica un file MHA e lo converte in un array NumPy."""
        image = sitk.ReadImage(filepath)
        array = sitk.GetArrayFromImage(image)  # Convert to NumPy array
        return array


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

        train_df = df
        train_df.to_csv(self.train_csv, index=False)

        print(f"✅ Split train/test completed and saved in {self.train_csv.parent}")