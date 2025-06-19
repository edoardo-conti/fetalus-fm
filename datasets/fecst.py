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
# =================================== VD_FECST =====================================
# ================================================================================== 
class VD_FECST(VisionDataset):
    def __init__(
        self,
        root: Path,
        data_dir: Path = Path("./data_csv"),
        split: Literal['train', 'test', 'val'] = 'train',
        val_size: Optional[int] = 2,
        test_size: int = 2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Dataset 'VD_FECST' for fetal abdominal structures segmentation.
        https://figshare.com/articles/figure/Second_Trimester_Fetal_Echocardiography_Data_Set_for_Image_Segmentation/21215597?file=37624532

        Args:
            root (str): Root directory of the dataset.
            data_dir (str): Directory where the processed dataset csv files will be stored.
            split (Literal['train', 'val', 'test']): Dataset split, either 'train', 'val' or 'test'.
            val_size (Optional[int]): Number of samples to use for validation.
            test_size (int): Number of samples to use for testing.
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
        self.val_size = val_size
        self.test_size = test_size
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        
        # CSV file paths
        self.data_csv = self.data_dir / "fecst.csv"
        self.train_csv = self.data_dir / "fecst_train.csv"
        self.val_csv = self.data_dir / "fecst_val.csv"
        self.test_csv = self.data_dir / "fecst_test.csv"
        
        # Creating csv file of the dataset
        self.process_csv()
        
        # Split train/test
        self.split_std()

        # Split train/val if requested
        if self.val_size is not None and not self.val_csv.exists():
            self.split_train_val()

        # Loading dataframes based on the split
        self.data = pd.read_csv(getattr(self, f'{self.split}_csv'))

    def __str__(self) -> str:
        return f"FetalEchoCardio_{self.split}"

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
    def videos(self) -> Set[str]:
        videos = self.data["video_id"]

        if self.target_transform:
            return self.target_transform(videos)
        else:
            return videos.tolist()

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


    def process_csv(self):
        """Create a unique CSV file with all the dataset information."""
        if self.data_csv.exists():
            return
        
        # Process data
        data_list = []
        video_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        for video_dir in tqdm(video_dirs, desc=f"Processing {self.root}", total=len(video_dirs)):
            for json_file in video_dir.glob("*.json"):
                try:
                    # Load JSON info
                    with open(json_file, "r") as f:
                        info = pd.read_json(f, typ='series')

                    # Find corresponding image (same name but .jpg)
                    image_path = json_file.with_suffix(".jpg")
                    if not image_path.exists():
                        print(f"Warning: Image {image_path} not found for {json_file}")
                        continue
                    
                    # Extract fields robustly
                    shapes = info.get("shapes", "")
                    imageHeight = info.get("imageHeight", 0)
                    imageWidth = info.get("imageWidth", 0)

                    # Extract structures from shapes
                    if isinstance(shapes, list):
                        structures = [shape.get("label", "") for shape in shapes if "label" in shape]
                    else:
                        structures = None

                    # If structures is empty, set mask_path to None, else keep the JSON path
                    mask_path = None if not structures else str(json_file.relative_to(self.root))

                    # Store info
                    data_list.append({
                        "image_path": str(image_path.relative_to(self.root)),
                        "class": "Heart",
                        "video_id": int(video_dir.name[-1]),
                        "resolution": [imageHeight, imageWidth],
                        "mask_path": mask_path,
                        "mask_structures": structures
                    })
                except Exception as e:
                    print(f"Warning: failed to process {json_file}: {e}")

        # Save to CSV
        df_final = pd.DataFrame(data_list).sort_values(by='video_id', ignore_index=True)
        df_final.to_csv(self.data_csv, index=False)
        print(f"✅ Full dataset saved in {self.data_csv}")
    

    def split_std(self):
        """Manages train and test partitioning."""
        if self.train_csv.exists() and self.test_csv.exists():
            return
        
        df = pd.read_csv(self.data_csv)
        
        video_ids = sorted(df['video_id'].unique())
        train_df = df[df['video_id'].between(video_ids[0], video_ids[-self.test_size-1])]
        test_df = df[df['video_id'].between(video_ids[-self.test_size], video_ids[-1])]

        train_df.to_csv(self.train_csv, index=False)
        test_df.to_csv(self.test_csv, index=False)

        print(f"✅ Split train/test completed and saved in {self.train_csv.parent}")


    def split_train_val(self):
        """
        Splits the training set into training and validation sets, ensuring no 
        data leakage and stratifying by the structures present in the masks.

        Args:
            val_percentage (float): Percentage of the training set to use for validation.
        """
        # Read the training data
        df = pd.read_csv(self.train_csv)
        
        video_ids = sorted(df['video_id'].unique())
        train_df = df[df['video_id'].between(video_ids[0], video_ids[-self.val_size-1])]
        val_df = df[df['video_id'].between(video_ids[-self.val_size], video_ids[-1])]

        # Save the new train and validation sets
        train_df.to_csv(self.train_csv, index=False)
        val_df.to_csv(self.val_csv, index=False)

        print(f"✅ Split train/val completed and saved in {self.val_csv.parent}")