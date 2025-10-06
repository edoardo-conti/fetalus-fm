import os
import logging
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union
from pathlib import Path

from PIL import Image
import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov3")


class FetalUS(ExtendedVisionDataset):
    """
    Custom dataset class for Fetal Ultrasound images following DINOv3 structure
    """
    
    class Split(Enum):
        TRAIN = "TRAIN"
        VAL = "VAL"
        TEST = "TEST"  # Optional
        
        @property
        def length(self) -> int:
            split_lengths = {
                FetalUS.Split.TRAIN: 129_194,   # Update with actual numbers 
                FetalUS.Split.VAL: 36_695,      # Update with actual numbers
                FetalUS.Split.TEST: 22_172,     # Update with actual numbers
            }
            return split_lengths[self]

        def get_dirname(self, class_id: Optional[str] = None) -> str:
            return self.value.lower()

        def get_image_relpath(self, class_id: Optional[str], image_id: str) -> str:
            return f"images/{image_id}"

        def get_image_list_file(self) -> str:
            split_files = {
                FetalUS.Split.TRAIN: "train.txt",
                FetalUS.Split.VAL: "val.txt",
                FetalUS.Split.TEST: "test.txt",
            }
            return split_files[self]

    def __init__(
        self,
        *,
        split: Split,
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._split = split
        self._extra_root = extra
        
        self._entries = None
        self._class_ids = None
        self._class_names = None

        # Load image paths from split file
        split_file = os.path.join(root, split.get_image_list_file())
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            self._image_files = [line.strip() for line in f.readlines()]
            
        logger.info(f"FetalUS dataset {split.value}: loaded {len(self._image_files)} images")

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._split == "TEST":
            assert False, "Class IDs are not available in TEST split"
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids
    
    def _get_class_names(self) -> np.ndarray:
        if self._split == "TEST":
            assert False, "Class names are not available in TEST split"
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        image_path = os.path.join(self.root, self._image_files[index])
        with open(image_path, mode="rb") as f:
            return f.read()
    
    def get_target(self, index: int) -> int:
        """
        Convert string class_id to integer target for training.
        Maps dataset_class combinations to unique integers.
        """
        # Fixed mapping from string class IDs to integers
        CLASS_ID_TO_INT = {
            # FPDB dataset classes
            'FPDB_ABDOMEN': 0, 'FPDB_BRAIN': 1, 'FPDB_FEMUR': 2, 'FPDB_MATERNAL-CERVIX': 3, 'FPDB_OTHER': 4, 'FPDB_THORAX': 5,
            # ACSLC dataset classes
            'ACSLC_ABDOMEN': 6,
            # FECST dataset classes
            'FECST_HEART': 7, 
            # FPLR dataset classes
            'FPLR_ABDOMEN': 8, 'FPLR_BRAIN': 9, 'FPLR_FEMUR': 10, 'FPLR_THORAX': 11,
            # HC18 dataset classes 
            # (DISABLED IN PRODUCTION BECAUSE OF LABELS MISMATCH WITH TRAIN/VAL 12 CLASSES VS 13)
            'HC18_HEAD': 12, 
        }
        
        class_id = self.get_class_id(index)
        
        # Handle cases where class_id might be None or unknown
        if class_id is None:
            logger.warning(f"No class_id found for index {index}, using fallback mapping")
            # Fallback to index if no dataset found
            return index % len(CLASS_ID_TO_INT)
        
        # Convert class_id to uppercase for consistent mapping
        class_id_upper = class_id.upper()
        
        if class_id_upper in CLASS_ID_TO_INT:
            return CLASS_ID_TO_INT[class_id_upper]
        else:
            # Log warning for unknown class_id and return fallback
            logger.warning(f"Unknown class_id: {class_id}, using fallback mapping")
            return index % len(CLASS_ID_TO_INT)

    def get_targets(self) -> Optional[np.ndarray]:
        return np.array([self.get_target(index) for index in range(len(self))])

    def get_class_id(self, index: int) -> Optional[str]:
        parts = Path(self.get_image_path(index)).parts
        for i, part in enumerate(parts):
            if part.lower() in {"train", "val", "test"}:
                return f"{parts[i - 1]}_{parts[i + 1]}"
        return None

    def get_class_name(self, index: int) -> Optional[str]:
        parts = Path(self.get_image_path(index)).parts
        for i, part in enumerate(parts):
            if part.lower() in {"train", "val", "test"}:
                return parts[i + 1]
        return None

    def get_image_id(self, index: int) -> str:
        return os.path.basename(self._image_files[index])

    def get_image_path(self, index: int) -> str:
        return os.path.join(self.root, self._image_files[index])

    # def __getitem__(self, index: int) -> Union[Tuple, Image.Image]:
    #     try:
    #         image_path = self.get_image_path(index)
    #         image = Image.open(image_path).convert("RGB")
            
    #         if self.transforms is not None:
    #             image = self.transforms(image)
                
    #         return image
    #     except Exception as e:
    #         logger.warning(f"Error loading image at index {index}: {e}")
    #         # Return a dummy image in case of error
    #         return Image.new('RGB', (224, 224), (0, 0, 0))

    def __len__(self) -> int:
        return len(self._image_files)

    def dump_extra(self) -> None:
        """
        Generate metadata files required by DINOv3
        """
        # Create entries array with image paths
        entries = np.array([self.get_image_path(i) for i in range(len(self))])
        
        # Save metadata
        extra_dir = os.path.join(self.root, "extra_dinov3")
        os.makedirs(extra_dir, exist_ok=True)
        
        split_name = self._split.value
        entries_file = os.path.join(extra_dir, f"entries-{split_name}.npy")
        np.save(entries_file, entries)
        
        # Get actual class data using dataset methods
        class_ids = np.array([self.get_class_id(i) for i in range(len(self))])
        class_names = np.array([self.get_class_name(i) for i in range(len(self))])
        
        class_ids_file = os.path.join(extra_dir, f"class-ids-{split_name}.npy")
        class_names_file = os.path.join(extra_dir, f"class-names-{split_name}.npy")
        
        np.save(class_ids_file, class_ids)
        np.save(class_names_file, class_names)
        
        logger.info(f"Saved metadata for {split_name} split: {len(entries)} entries")
