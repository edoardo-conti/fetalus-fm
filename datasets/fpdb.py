from pathlib import Path
from typing import Literal, Callable, Optional, Tuple, List, Set
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import VisionDataset

import numpy as np
import pandas as pd
import re
import torch
from sklearn.model_selection import StratifiedShuffleSplit

# ================================================================================== 
# ==================================== VD_FPDB =====================================
# ================================================================================== 
class VD_FPDB(VisionDataset):
    def __init__(
        self,
        root: Path,
        split: Literal['train', 'val', 'test'],
        data_dir: Path = Path("./data_csv"),
        val_size: Optional[float] = 0.2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Dataset 'VD_FPDB' for common maternal-fetal ultrasound images.
        https://zenodo.org/records/3904280
        
        Args:
            root (str): Root directory of the dataset.
            split (Literal['train', 'test', 'val']): Dataset split, either 'train', 'test' or 'val'.
            data_dir (str): Directory where the processed dataset csv files will be stored.
            val_size (Optional[float]): Proportion of the training set to use for validation.
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
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        
        # Ensure the data directory exists or create it
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # CSV file paths
        self.data_csv = self.data_dir / "fpdb.csv"
        self.train_csv = self.data_dir / "fpdb_train.csv"
        self.val_csv = self.data_dir / "fpdb_val.csv"
        self.test_csv = self.data_dir / "fpdb_test.csv"
        
        # Set random seed for reproducibility
        np.random.seed(self.seed)

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

    def split_train_val(self):
        """
        Splits the training set into training and validation sets, ensuring no 
        data leakage at patient level and stratifying by brain_plane.

        Args:
            val_size (float): Percentage of the training set to use for validation.
        """
        if self.val_csv.exists():
            return

        if not (0 < self.val_size < 1):
            raise ValueError("val_size must be between 0 and 1")
        
        # Read the training data
        train_df = pd.read_csv(self.train_csv)

        # Get unique patients and their most frequent brain_plane
        patient_brain = (
            train_df.groupby("patient_id")["brain_plane"]
            .agg(lambda x: x.value_counts().idxmax())
            .reset_index()
        )

        # Stratified split by brain_plane at patient level
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_size, random_state=self.seed
        )
        train_idx, val_idx = next(
            splitter.split(patient_brain["patient_id"], patient_brain["brain_plane"])
        )
        train_patients = set(patient_brain.loc[train_idx, "patient_id"])
        val_patients = set(patient_brain.loc[val_idx, "patient_id"])

        # Assign samples to train/val based on patient_id
        train_split_df = train_df[train_df["patient_id"].isin(train_patients)]
        val_split_df = train_df[train_df["patient_id"].isin(val_patients)]

        # Save the new train and validation sets
        train_split_df.to_csv(self.train_csv, index=False)
        val_split_df.to_csv(self.val_csv, index=False)

        print(f"✅ Split train/val completed and saved in {self.val_csv.parent}")