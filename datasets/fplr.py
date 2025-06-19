from pathlib import Path
from typing import Literal, Callable, Optional, Tuple, List, Set

import pandas as pd
import torch
from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split

# ================================================================================== 
# ==================================== VD_FPLR =====================================
# ================================================================================== 
class VD_FPLR(VisionDataset):
    def __init__(
        self,
        root: Path,
        split: Literal['train', 'test', 'val'],
        data_dir: Path = Path("./data_csv"),
        val_size: Optional[float] = 0.2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Dataset 'VD_FPLR' for maternal fetal US planes from low-resource imaging settings in five African countries.
        https://zenodo.org/records/7540448
        
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
        self.data_csv = self.data_dir / "fplr.csv"
        self.train_csv = self.data_dir / "fplr_train.csv"
        self.val_csv = self.data_dir / "fplr_val.csv"
        self.test_csv = self.data_dir / "fplr_test.csv"
        
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


    def process_csv(self):
        """Create a unique CSV file with all the dataset information."""
        if self.data_csv.exists():
            return

        original_csv = self.root / "African_planes_database.csv"
        df = pd.read_csv(original_csv)

        # Mapping for class names
        plane_mapping = {
            "Fetal abdomen": "Abdomen",
            "Fetal brain": "Brain",
            "Fetal femur": "Femur",
            "Fetal thorax": "Thorax"
        }
        
        # Process data
        data_list = []
        for _, row in df.iterrows():
            image_name = row["Filename"] + ".png"
            country = row["Center"]
            image_path = self.root / country / image_name

            if not image_path.exists():
                print(f"Warning: Image not found {image_path}")
                continue
            
            data_list.append({
                "image_path": f"{country}/{image_name}",  
                "class": plane_mapping.get(row["Plane"], row["Plane"]), 
                "patient_id": row["Patient_num"],
                "country": country,
                "split": "train" if row["Train"] == 0 else "test"
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
        patient data leakage by grouping by patient_id.

        Args:
            val_size (float): Percentage of the training set to use for validation.
        """
        if self.val_csv.exists() or not (0 < self.val_size < 1):
            return
        
        # Read the training data
        df = pd.read_csv(self.train_csv)
        
        # Get unique patients and split them
        patients = df["patient_id"].unique()
        train_patients, val_patients = train_test_split(
            patients,
            test_size=self.val_size,
            random_state=self.seed,
        )

        # Split data based on patient groups
        train_df = df[df["patient_id"].isin(train_patients)]
        val_df = df[df["patient_id"].isin(val_patients)]
        
        # Save the new train and validation sets
        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)

        print(f"✅ Split train/val completed and saved in {self.val_csv.parent}")