from pathlib import Path
from typing import Literal, Callable, Optional, Tuple, List, Set
from PIL import Image
from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk


# ================================================================================== 
# =========================== Fetal_PSFH_Z7851339 =============================
# ================================================================================== 
class FetalPSFH(VisionDataset):
    def __init__(
        self,
        root: Path,
        data_dir: Path = Path("./data_csv"),
        split: Literal['train', 'val', 'test'] = 'train',
        val_percentage: Optional[float] = None,
        test_size: float = 0.2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Dataset 'Fetal_PSFH_Z7851339' for Pubic Symphysis and Fetal Head Segmentation (PSFHS).
        https://zenodo.org/records/7851339 , https://zenodo.org/records/10969427
        
        Args:
            root (str): Root directory of the dataset.
            data_dir (str): Directory where the processed dataset csv files will be stored.
            split (Literal['train', 'test']): Dataset split, either 'train' or 'test'.
            target (Optional[Callable]): A function/transform that takes in the image and transforms it.
            target_transform (Optional[Callable]): A function/transform that takes in the target and transforms it.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        if split not in ['train', 'test', 'val']:
            raise ValueError("split must be one of ['train', 'test', 'val']")
        
        self.root = Path(root)
        self.data_dir = data_dir / self.root.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.val_percentage = val_percentage
        self.test_size = test_size
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        # CSV file paths
        self.data_csv = self.data_dir / "fpsfh.csv"
        self.train_csv = self.data_dir / "fpsfh_train.csv"
        self.val_csv = self.data_dir / "fpsfh_val.csv"
        self.test_csv = self.data_dir / "fpsfh_test.csv"
        
        # Creating csv file of the dataset
        self.process_csv()
        
        # Split train/test
        self.split_std()

        # Split train/val if requested
        if self.val_percentage is not None and not self.val_csv.exists():
            self.split_train_val(self.val_percentage)

        # Loading dataframes based on the split
        self.data = pd.read_csv(getattr(self, f'{self.split}_csv'))

    def __str__(self) -> str:
        return f"FetalPSFH_{self.split}"

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
    
    def split_train_val(self, val_percentage: float = 0.2):
        """
        Splits the training set into training and validation sets, ensuring no 
        data leakage and stratifying by the structures present in the masks.

        Args:
            val_percentage (float): Percentage of the training set to use for validation.
        """
        if not (0 < val_percentage < 1):
            raise ValueError("val_percentage must be between 0 and 1")

        # Read the training data
        train_df = pd.read_csv(self.train_csv)
        
        # Split into train and validation sets
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_percentage,
            random_state=self.seed,
            stratify=train_df['mask_structures']
        )

        # Save the new train and validation sets
        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)

        print(f"✅ Split train/val completed and saved in {self.val_csv.parent}")