from pathlib import Path
from typing import Literal, Callable, Optional, Tuple, List
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import VisionDataset
from utils import set_class_values, get_label_mask

import numpy as np
import pandas as pd
import re
import torch
import cv2

# ================================================================================== 
# ============================== Fetal_HC18_Z1327317 ===============================
# ================================================================================== 
class FetalHC18(VisionDataset):
    def __init__(
        self,
        root: Path,
        data_dir: Path = Path("./data_csv"),
        split: Literal['train', 'test', 'val'] = 'train',
        val_percentage: Optional[float] = None,
        label_colors_list: List[Tuple[int, int, int]] = None,
        all_classes: List[str] = None,
        classes_to_train: List[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        seed: int = 42
    ):
        """
        Dataset 'Fetal_HC18_Z1327317' for head circumference measurement.
        https://zenodo.org/records/1327317
        
        Args:
            root (str): Root directory of the dataset.
            data_dir (str): Directory where the processed dataset csv files will be stored.
            split (Literal['train', 'test', 'val']): Dataset split, either 'train', 'test', or 'val'.
            transform (Optional[Callable]): A function/transform that takes in the image and transforms it.
            target_transform (Optional[Callable]): A function/transform that takes in the target and transforms it.
            seed (int): Random seed for reproducibility.
            val_percentage (Optional[float]): Percentage of the training set to use for validation. If provided, splits the training set.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        if split not in ['train', 'test', 'val']:
            raise ValueError("split must be one of ['train', 'test', 'val']")
        
        self.root = Path(root)
        self.data_dir = data_dir / self.root.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.val_percentage = val_percentage
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(self.all_classes, self.classes_to_train)
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation
        self.seed = seed
        
        # CSV file paths
        self.data_csv = self.data_dir / "fhc18.csv"
        self.train_csv = self.data_dir / "fhc18_train.csv"
        self.test_csv = self.data_dir / "fhc18_test.csv"
        self.val_csv = self.data_dir / "fhc18_val.csv"
        
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
        return f"FetalHC18_{self.split}"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the image and the corresponding target.

        Args:
            index (int): Index of the sample.
        
        Returns:
            Tuple[torch.Tensor, torch.LongTensor]: Transformed image and target.
        """
        img_path = self.root / self.data.iloc[index]["image_path"]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        
        mask_path = self.root / self.data.iloc[index]["mask_path"]
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        augmented = self.augmentation(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        mask = get_label_mask(mask, self.class_values, self.label_colors_list).astype('uint8')

        # To C, H, W.
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), torch.LongTensor(mask)
    
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
    def structures(self) -> List[str]:
        """Returns all unique mask structures present in the dataset."""
        mask_structures = set()
        for struct_list in self.data["mask_structures"].dropna():
            mask_structures.update(eval(struct_list))
        return sorted(mask_structures)
    
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
        """Create a unique CSV file with train and test information if it does not already exist."""
        if self.data_csv.exists():
            return

        csv_paths = ["training_set_pixel_size_and_HC.csv", "test_set_pixel_size.csv"]
        data_list = []
        
        for csv_file in csv_paths:
            set_name = re.search(r"(\w+)_set", csv_file).group(1)
            df = pd.read_csv(self.root / csv_file)
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {csv_file}"):
                filename = row["filename"]
                pixel_size = row["pixel size(mm)"]
                head_circumference = row.get("head circumference (mm)", None)
                
                # image path
                img_path = f"{set_name}_set/{filename}"
                
                # annotation paths (only train)
                hc_annotation_path = None
                hc_annotation_filename = filename.replace(".png", "_Annotation.png")
                hc_annotation_full_path = self.root / f"{set_name}_set" / hc_annotation_filename
                if set_name == "training" and hc_annotation_full_path.exists():
                    hc_annotation_path = f"{set_name}_set/{hc_annotation_filename}"
                
                # segmentation masks (only train)
                mask_path = None 
                mask_full_path = self.root / "training_masks" / filename
                if set_name == "training" and mask_full_path.exists():
                    mask_path = f"training_masks/{filename}"
                    structures_present = self._process_mask(mask_path)
                else:
                    structures_present = None 
                
                data_list.append({
                    "image_path": img_path,
                    "class": "Head",
                    "pixel_size": pixel_size,
                    "head_circumference": head_circumference,
                    "hc_annotation_path": hc_annotation_path,
                    "mask_path": mask_path,
                    "mask_structures": structures_present,
                    "split": "train" if set_name == "training" else "test"
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
        test_df = df[df["split"] == "test"].drop(columns=["head_circumference", "hc_annotation_path", "mask_path", "mask_structures", "split"])
        
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

        # Group by base filename (e.g., '010' from '010_HC.png')
        train_df['base_filename'] = train_df['image_path'].apply(lambda x: x.split('/')[-1].split('_')[0])
        
        # Stratify by mask structures
        train_df['mask_structures_str'] = train_df['mask_structures'].apply(lambda x: ','.join(sorted(eval(x))) if pd.notna(x) else '')
        
        # Shuffle groups
        grouped = train_df.groupby(['base_filename', 'mask_structures_str'])
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
        train_df = train_df.drop(columns=['base_filename', 'mask_structures_str'])
        val_df = val_df.drop(columns=['base_filename', 'mask_structures_str'])

        # Save the new train and validation sets
        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)

        print(f"✅ Split train/val completed and saved in {self.val_csv.parent}")
