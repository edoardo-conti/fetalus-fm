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
# ==================================== VD_IPSFH ====================================
# ================================================================================== 
class VD_IPSFH(VisionDataset):
    def __init__(
        self,
        root: Path,
        split: Literal['train', 'val', 'test'],
        data_dir: Path = Path("./data_csv"),
        val_size: Optional[float] = 0.15,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Dataset 'VD_IPSFH' for Pubic Symphysis and Fetal Head Segmentation (PSFHS).
        https://zenodo.org/records/7851339 , https://zenodo.org/records/10969427
        
        Args:
            root (str): Root directory of the dataset.
            split (Literal['train', 'val', 'test']): Dataset split, either 'train', 'val' or 'test'.
            data_dir (str): Directory where the processed dataset csv files will be stored.
            val_size (Optional[float]): Proportion of the training set to use for validation.
            target (Optional[Callable]): A function/transform that takes in the image and transforms it.
            target_transform (Optional[Callable]): A function/transform that takes in the target and transforms it.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        if split not in ['train', 'test', 'val']:
            raise ValueError("split must be one of ['train', 'test', 'val']")
        
        self.root = Path(root)
        self.data_dir = Path(data_dir) / self.root.name
        self.split = split
        self.val_size = val_size
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        # Ensure the data directory exists or create it
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # CSV file paths
        self.data_csv = self.data_dir / "ipsfh.csv"
        self.train_csv = self.data_dir / "ipsfh_train.csv"
        self.val_csv = self.data_dir / "ipsfh_val.csv"
        self.test_csv = self.data_dir / "ipsfh_test.csv"
        
        # Creating csv file of the dataset
        self.process_csv()
        
        # Split train/test and train/val if requested
        self.split_std()
        self.split_train_val()

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
        image = sitk.ReadImage(filepath)
        array = sitk.GetArrayFromImage(image)  # Convert to NumPy array
        return array


    def process_csv(self):
        """Create a unique CSV file with all the dataset information."""
        if self.data_csv.exists():
            return
        
        # data
        images_dir = self.root / "image_mha"
        masks_dir = self.root / "label_mha"
        image_files = [f.name for f in images_dir.iterdir() if f.is_file() and f.name.endswith(".mha")]
        
        # Process data
        data_list = []
        for img_file in tqdm(image_files, total=len(image_files), desc=f"Processing {images_dir}"):
            image_path = f"image_mha/{img_file}"
            
            # annotations
            masks_full_path = masks_dir / img_file
            masks_path = f"label_mha/{img_file}" if masks_full_path.exists() else None
            
            # Analyze mask
            structures = []
            if masks_full_path.exists():
                mask = self._load_mha(masks_full_path)
                if 1 in mask:
                    structures.append("Pubic Symphysis")
                if 2 in mask:
                    structures.append("Fetal Head")
            
            # add data to the list with split info based on filename
            file_num = int(img_file.split('.')[0])  # extract number from filename (e.g. "03998.mha" -> 3998)
            split = "train" if file_num <= 4000 else "test"

            data_list.append({
                "image_path": image_path,  
                "class": "PSFH",
                "mask_path": masks_path,
                "mask_structures": structures,
                "split": split
            })
    
        df_final = pd.DataFrame(data_list).sort_values(by='image_path', ignore_index=True)
        df_final.to_csv(self.data_csv, index=False)
        print(f"✅ Full dataset saved in {self.data_csv}")
    
    def split_std(self):
        """Manages train and test partitioning with patient holdout."""
        if self.train_csv.exists() and self.test_csv.exists():
            return
        
        # Load dataset
        df = pd.read_csv(self.data_csv)
        
        # Split based on the split column
        train_df = df[df['split'] == 'train'].drop(columns=['split'])
        test_df = df[df['split'] == 'test'].drop(columns=['split'])

        train_df.to_csv(self.train_csv, index=False)
        test_df.to_csv(self.test_csv, index=False)

        print(f"✅ Split train/test completed and saved in {self.train_csv.parent}")
    
    def split_train_val(self):
        """
        Splits the training set into training and validation sets, ensuring no 
        data leakage and stratifying by the structures present in the masks.

        Args:
            val_size (float): Percentage of the training set to use for validation.
        """
        if self.val_csv.exists():
            return
        
        if not (0 < self.val_size < 1):
            raise ValueError("val_size must be between 0 and 1")

        # Read the training data
        train_df = pd.read_csv(self.train_csv)
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_size,
            random_state=self.seed,
            stratify=train_df['mask_structures']
        )

        # Save the new train and validation sets
        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)

        print(f"✅ Split train/val completed and saved in {self.val_csv.parent}")