from pathlib import Path
from typing import Literal, Callable, Optional, Tuple, List, Set
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import VisionDataset

import numpy as np
import pandas as pd
import re
import torch

# ================================================================================== 
# =========================== Fetal_Planes_DB_Z3904280 =============================
# ================================================================================== 
class FetalPlanesDB(VisionDataset):
    def __init__(
        self,
        root: Path,
        data_dir: Path = Path("./data_csv"),
        split: Literal['train', 'val', 'test'] = 'train',
        val_percentage: Optional[float] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Dataset 'Fetal_Planes_DB_Z3904280' for common maternal-fetal ultrasound images.
        https://zenodo.org/records/3904280
        
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
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        
        # CSV file paths
        self.data_csv = self.data_dir / "fpdb.csv"
        self.train_csv = self.data_dir / "fpdb_train.csv"
        self.val_csv = self.data_dir / "fpdb_val.csv"
        self.test_csv = self.data_dir / "fpdb_test.csv"
        
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
        return f"FetalPlanesDB_{self.split}"

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
        img_path = self.root / self.data.iloc[index]["image_path"]
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

    def _process_mask(self, mask_path: Path) -> List[str]:
        """Process the mask image and return the structures present."""
        mask_image = Image.open(self.root / mask_path)
        mask_image = mask_image.convert("RGB")
        mask_array = np.array(mask_image)
        threshold = 200
        
        structures_present = []
        # red channel -> Brain
        if mask_array[..., 0].max() >= threshold:
            structures_present.append('Brain')
        # green channel -> CSP
        if mask_array[..., 1].max() >= threshold:
            structures_present.append('CSP')
        # blue channel -> LV
        if mask_array[..., 2].max() >= threshold:
            structures_present.append('LV')

        return structures_present

    def process_csv(self):
        """Create a unique CSV file with all the dataset information."""
        if self.data_csv.exists():
            return

        original_csv = self.root / "FETAL_PLANES_DB_data.csv"
        df = pd.read_csv(original_csv, sep=';')
        
        # Mapping for class names
        plane_mapping = {
            "Fetal abdomen": "Abdomen",
            "Fetal brain": "Brain",
            "Fetal femur": "Femur",
            "Fetal thorax": "Thorax",
            "Maternal cervix": "Maternal-Cervix",
            "Other": "Other"
        }
        
        # Process data
        data_list = []
        images_dir = self.root / "Images"
        masks_dir = self.root / "Masks"

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {original_csv}"):
            image_name = row["Image_name"] + ".png"
            image_path = images_dir / image_name
            
            if not image_path.exists():
                print(f"Warning: Image not found {image_path}")
                continue
            
            # get the class (fetal plane) and brain_plane
            fetal_plane = plane_mapping.get(row["Plane"], row["Plane"])
            brain_plane = "NaB" if row["Brain_plane"] == "Not A Brain" else row["Brain_plane"]
            
            # link the mask if the plane is Brain
            mask_path = None
            structures_present = []
            if fetal_plane == "Brain":
                brain_plane_dir = masks_dir / fetal_plane / brain_plane
                mask_file = brain_plane_dir / image_name
                if mask_file.exists():
                    mask_path = f"Masks/{fetal_plane}/{brain_plane}/{image_name}"
                    structures_present = self._process_mask(mask_path)                     
                    
            # extract the plane number and sequence from the image_name
            plane_match = re.search(r"Plane(\d+)", image_name)
            seq_match = re.search(r"_(\d+)_of_(\d+)", image_name)
            img_plane = int(plane_match.group(1)) if plane_match else -1
            img_plane_seq = f"{seq_match.group(1)}/{seq_match.group(2)}" if seq_match else "-"
            
            data_list.append({
                "image_path": f"Images/{image_name}",
                "class": fetal_plane,
                "brain_plane": brain_plane,
                "image_plane": img_plane,
                "image_plane_seq": img_plane_seq,
                "patient_id": row["Patient_num"],
                "operator": row["Operator"],
                "us_machine": row["US_Machine"],
                "mask_path": mask_path,
                "mask_structures": structures_present,
                "split": "train" if row["Train "] == 1 else "test"
            })

        df_final = pd.DataFrame(data_list)
        df_final.to_csv(self.data_csv, index=False)
        print(f"✅ Full dataset saved in {self.data_csv}")

    def split_std(self):
        """Manages train and test partitioning."""
        if self.train_csv.exists() and self.test_csv.exists():
            return
        
        df = pd.read_csv(self.data_csv)
        
        train_df = df[df["split"] == "train"].drop(columns=["split"])
        test_df = df[df["split"] == "test"].drop(columns=["split"])
        
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
        
        # Shuffle groups
        grouped = train_df.groupby(['patient_id', 'brain_plane'])
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

        # Save the new train and validation sets
        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)

        print(f"✅ Split train/val completed and saved in {self.val_csv.parent}")