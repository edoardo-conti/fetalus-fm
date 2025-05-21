from pathlib import Path
from typing import Literal, Callable, Optional, Tuple, List, Set
from PIL import Image
from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

# ================================================================================== 
# =========================== Fetal_Abdominal_MD4GCPM9DSC3 =============================
# ================================================================================== 
class FetalAbdominal(VisionDataset):
    def __init__(
        self,
        root: Path,
        data_dir: Path = Path("./data_csv"),
        split: Literal['train', 'test', 'val'] = 'train',
        val_percentage: Optional[float] = None,
        test_size: float = 0.3,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Dataset 'Fetal_Abdominal_MD4GCPM9DSC3' for fetal abdominal structures segmentation.
        https://data.mendeley.com/datasets/4gcpm9dsc3/1

        Args:
            root (str): Root directory of the dataset.
            data_dir (str): Directory where the processed dataset csv files will be stored.
            split (Literal['train', 'test']): Dataset split, either 'train' or 'test'.
            target (Optional[Callable]): A function/transform that takes in the image and transforms it.
            target_transform (Optional[Callable]): A function/transform that takes in the target and transforms it.
            seed (int): Random seed for reproducibility.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of ['train', 'val', 'test']")
        
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
        self.data_csv = self.data_dir / "fabdominal.csv"
        self.train_csv = self.data_dir / "fabdominal_train.csv"
        self.val_csv = self.data_dir / "fabdominal_val.csv"
        self.test_csv = self.data_dir / "fabdominal_test.csv"
        
        # Creating csv file of the dataset
        self.process_csv()
        
        # Split train/test
        self.split_pholdout()

        # Split train/val if requested
        if self.val_percentage is not None and not self.val_csv.exists():
            self.split_train_val(self.val_percentage)

        # Loading dataframes based on the split
        self.data = pd.read_csv(getattr(self, f'{self.split}_csv'))

    def __str__(self) -> str:
        return f"FetalAbdominal_{self.split}"

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

    def process_csv(self):
        """Create a unique CSV file with all the dataset information."""
        if self.data_csv.exists():
            return
        
        # data
        images_dir = self.root / "IMAGES"
        npy_dir = self.root / "ARRAY_FORMAT"
        image_files = [f.name for f in images_dir.iterdir() if f.is_file() and f.name.endswith(".png")]
        
        # Process data
        data_list = []
        for img_file in tqdm(image_files, total=len(image_files), desc=f"Processing {images_dir}"):
            image_path = f"IMAGES/{img_file}"
            patient_id = int(img_file.split("_")[0][1:])
            
            # annotations
            mask_file = img_file.replace(".png", ".npy")
            mask_full_path = npy_dir / mask_file
            mask_path = f"ARRAY_FORMAT/{mask_file}" if mask_full_path.exists() else None

            # metadata
            structures = list(np.load(mask_full_path, allow_pickle=True).item()['structures'])
            vendors = ["Siemens Acuson", "Voluson 730", "Philips-EPIQ Elite"]
            transducer = "Curvilinear"
            freq_range = "2-9 MHz"

            # add data to the list
            data_list.append({
                "image_path": image_path,  
                "class": "Abdomen",
                "patient_id": patient_id,
                "vendors": vendors,
                "transducer": transducer,
                "freq_range": freq_range,
                "mask_path": mask_path,
                "mask_structures": structures
            })
        
        df_final = pd.DataFrame(data_list).sort_values(by='patient_id', ignore_index=True)
        df_final.to_csv(self.data_csv, index=False)
        print(f"✅ Full dataset saved in {self.data_csv}")
    
    def split_pholdout(self):
        """Manages train and test partitioning with patient holdout."""
        if self.train_csv.exists() and self.test_csv.exists():
            return
        
        # Load dataset
        df = pd.read_csv(self.data_csv)
        
        # Get unique patient IDs
        patient_ids = sorted(df['patient_id'].unique())
        
        # Split patient IDs into train and test
        train_ids, test_ids = train_test_split(patient_ids, test_size=self.test_size, random_state=self.seed)
        
        # Assign images based on patient split
        train_df = df[df['patient_id'].isin(train_ids)]
        test_df = df[df['patient_id'].isin(test_ids)]
        
        # Save splits
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
        
        # Stratify by mask structures
        train_df['mask_structures_str'] = train_df['mask_structures'].apply(lambda x: ','.join(sorted(eval(x))) if pd.notna(x) else '')
        
        # Shuffle groups
        grouped = train_df.groupby(['patient_id', 'mask_structures_str'])
        grouped_indices = list(grouped.groups.keys())
        np.random.seed(self.seed)
        np.random.shuffle(grouped_indices)

        # Split into train and validation groups
        val_size = int(len(grouped_indices) * val_percentage)
        val_groups = grouped_indices[:val_size]
        train_groups = grouped_indices[val_size:]
        
        # Create new train and validation DataFrames
        val_df = pd.concat([grouped.get_group(g) for g in val_groups])
        train_df = pd.concat([grouped.get_group(g) for g in train_groups])

        # Drop the helper columns
        train_df = train_df.drop(columns=['mask_structures_str'])
        val_df = val_df.drop(columns=['mask_structures_str'])

        # Save the new train and validation sets
        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)

        print(f"✅ Split train/val completed and saved in {self.val_csv.parent}")