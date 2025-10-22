from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
import json
import os
from torchvision.datasets import VisionDataset

# from utils import get_label_mask, ALL_CLASSES, LABEL_COLORS_LIST

from utils.utils import DATASETS_CONFIGS, FUS_STRUCTS, FUS_STRUCTS_COLORS, FPDB_BRAIN_PLANES
from augmentations import get_cls_augmentations, get_val_augmentations

class UnifiedFetalDataset(VisionDataset):
    """Dataset unificato per immagini ecografiche fetali con maschere multi-struttura"""
    def __init__(
        self,
        root: str,
        data_path: str,
        datasets: List[str],
        split: str = 'train',
        supervised: bool = True,
        target_size: Tuple[int, int] = (644, 644),
        augmentations: Tuple[callable, callable] = None,
        task: str = 'seg',
        eval_augmentation: bool = False,
        transform: callable = None
    ):
        """
        Args:
            root: Directory root del progetto
            data_path: Directory dei dataset
            datasets: Lista di dataset da includere (es. ['hc18', 'abdominal'])
            split: Split del dataset ('train'/'test'/'val')
            target_size: Dimensione target per resize
            supervised: Se usare supervision (maschere per seg)
            augmentations: Tuple (geometric_augs, color_augs) per train set
            task: 'seg' o 'cls' per il tipo di task
            eval_augmentation: Se applicare augmentations anche su val/test (default False)
            transform: Trasformazioni torchvision da applicare
        """
        super().__init__(root, transform=transform)
        
        self.root = Path(root)
        self.csv_data_path = self.root / 'data_csv'
        self.data_path = Path(data_path)
        self.datasets = datasets
        self.target_size = target_size
        self.split = split
        self.supervised = supervised
        self.task = task
        self.eval_augmentation = eval_augmentation
        # Handle different augmentation formats
        if augmentations is None:
            self.geometric_augs = None
            self.color_augs = None
            self.combined_transform = None
        elif hasattr(augmentations, '__call__') and not isinstance(augmentations, (tuple, list)):
            # Single transform object (like CombinedTransform)
            self.geometric_augs = None
            self.color_augs = None
            self.combined_transform = augmentations
        else:
            # Tuple/list of separate transforms (legacy segmentation format)
            self.geometric_augs = augmentations[0] if len(augmentations) > 0 else None
            self.color_augs = augmentations[1] if len(augmentations) > 1 else None
            self.combined_transform = None

        # For classification, use integrated augmentation pipeline
        if self.task == 'cls':
            self.cls_augmentations = get_cls_augmentations(self.target_size)
            if self.eval_augmentation:
                self.val_augmentations = get_cls_augmentations(self.target_size)
            else:
                self.val_augmentations = None
        else:
            self.cls_augmentations = None
            self.val_augmentations = None

        self.class_values = list(range(1, len(FUS_STRUCTS)))
        self.samples = []
        self.dataframes = []
        
        # Caricamento dataset
        for ds_name in self.datasets:
            cfg = DATASETS_CONFIGS[ds_name]
            csv_path = self.csv_data_path / ds_name / f'{ds_name.lower()}_{self.split}.csv'
            
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found for {ds_name}: {csv_path}")
            
            df = pd.read_csv(csv_path)

            # filtraggio dei dati per supervised training
            df = self._filter_supervised_data(df, ds_name, cfg)

            # Filter for classification task on FPDB - only Brain class
            if self.task == 'cls' and ds_name == 'FPDB':
                df = df[df['class'] == 'Brain']
            
            df['dataset'] = ds_name
            self.dataframes.append(df)
            self.samples += [(len(self.dataframes)-1, i) for i in range(len(df))]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        df_idx, sample_idx = self.samples[idx]
        row = self.dataframes[df_idx].iloc[sample_idx]
        ds_name = row['dataset']
        
        # Load image and mask
        image, mask = getattr(self, f'_load_{ds_name.lower()}')(row, ds_name)

        # Common image and mask resizing
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Data augmentation - Unified approach for both tasks
        if self.task == 'cls':
            # For classification, apply ImageNet normalization to all splits (train/val/test)
            if self.combined_transform is not None:
                augmented = self.combined_transform(image=image)
                image = augmented['image']
        else:  # segmentation
            if self.split == 'train' or (self.eval_augmentation and self.split in ['val', 'test']):
                if self.combined_transform is not None:
                    # Use combined transform (includes Albumentations + ImageNet normalization)
                    augmented = self.combined_transform(image=image, mask=mask)
                    image = augmented['image']
                    if 'mask' in augmented:
                        mask = augmented['mask']
                elif self.task == 'seg':  # legacy segmentation
                    # Legacy segmentation: separate geometric + color, synchronized with masks
                    if self.supervised and self.geometric_augs and self.color_augs:
                        # trasformazioni geometriche sincronizzate
                        augmented = self.geometric_augs(image=image, mask=mask)
                        image, mask = augmented['image'], augmented['mask']

                        # trasformazioni di colore solo all'immagine
                        image = self.color_augs(image=image)['image']

        # image processing - handle both numpy arrays and tensors
        if isinstance(image, torch.Tensor):
            # ImageNet normalized tensor from CombinedTransform
            if image.dim() == 3 and image.shape[0] == 3:  # CHW format
                pass  # already in correct format
            elif image.dim() == 3 and image.shape[-1] == 3:  # HWC format
                image = image.permute(2, 0, 1)  # HWC to CHW
        else:
            # Traditional numpy array processing
            image = image.transpose(2, 0, 1)
            image = torch.tensor(image).float() / 255.0

        if self.task == 'cls':
            if ds_name == 'FPDB':
                brain_plane = row['brain_plane']
                name_to_abbrev = {
                    'Trans-thalamic': 'TT',
                    'Trans-ventricular': 'TV',
                    'Trans-cerebellum': 'TC'
                }
                abbrev = name_to_abbrev.get(brain_plane, 'TT')
                class_idx = FPDB_BRAIN_PLANES.index(abbrev)
            else:
                class_str = row['class']
                class_idx = FUS_STRUCTS.index(class_str.upper()) if class_str.upper() in FUS_STRUCTS else 0
            return image, class_idx
        else:
            if not self.supervised:
                return image

            # Supervised-only mask processing
            encoded_mask = self.label_mask(mask, ds_name, self.class_values)
            encoded_mask = torch.tensor(encoded_mask).long()

            return image, encoded_mask
    
    def _filter_supervised_data(self, df: pd.DataFrame, ds_name: str, cfg: dict) -> pd.DataFrame:
        """Filter dataframe for supervised training based on mask availability and dataset requirements"""
        if not self.supervised:
            return df
            
        if 'mask_path' in df.columns:
            df = df[df['mask_path'].notna()]  # Filter out rows with NaN masks

            # filtering also the classes for the FECST dataset
            if ds_name == 'fecst':
                req = set(cfg['structures'])
                df = df[df['mask_structures'].apply(lambda x: req <= set(eval(x)) if isinstance(x, str) else False)]
        else:
            df = df.iloc[0:0]  # Return empty dataframe if no mask_path column
            
        return df

    def _load_hc18(self, row, ds_name) -> Tuple[np.ndarray, np.ndarray]:
        # load specific dataset configuration
        # cfg = DATASETS_CONFIGS[ds_name]

        ds_dir = self.data_path / ds_name
        img_path = ds_dir / row['image_path']
        mask_path = ds_dir / row['mask_path']
        
        # image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # INIZIO POST-PROCESSING
        new_mask = np.zeros_like(mask)
        priority_map = np.full(mask.shape[:2], -1, dtype=int)
        
        # Applica priorità in ordine inverso per sovrascrittura
        for channel in reversed([1, 2, 0]):
            active_pixels = (mask[..., channel] > 200)
            priority_map[active_pixels] = channel
        
        # Costruzione maschera finale
        for channel in [1, 2, 0]:
            channel_mask = (priority_map == channel)
            new_mask[channel_mask, channel] = 255
        
        mask = new_mask
        # FINE POST-PROCESSING
        
        return image, mask
    
    def _load_fabd(self, row, ds_name) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / ds_name
        mask_path = ds_dir / row['mask_path']

        # sfruttare solamente file .npy che contiene sia immagine orirginale che maschere
        data = np.load(mask_path, allow_pickle=True).item()
        
        # image
        image = data['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # mask
        ds_color_map = DATASETS_CONFIGS[ds_name]['mask_color_map']
        structures = data['structures']
        combined_mask = np.zeros((*next(iter(structures.values())).shape, 3), dtype=np.uint8)
        for name, mask in structures.items():
            combined_mask[mask == 1] = ds_color_map[name.upper()]
        mask = combined_mask

        return image, mask
    
    def _load_fpdb(self, row, ds_name) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / ds_name
        img_path = ds_dir / row['image_path']

        # image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # mask (handle empty paths)
        if pd.isna(row['mask_path']):
            mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        else:
            mask_path = ds_dir / row['mask_path']
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        return image, mask
    
    def _load_ipsfh(self, row, ds_name) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / ds_name
        img_path = ds_dir / row['image_path']
        mask_path = ds_dir / row['mask_path']
        
        # image
        image = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(image)  # Convert to NumPy array
        image = np.transpose(image, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask
        mask = sitk.ReadImage(mask_path)
        mask = sitk.GetArrayFromImage(mask)  # Convert to NumPy array (256x256)
        combined_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)  # 256x256x3
        ps_color_map = DATASETS_CONFIGS[ds_name]['mask_color_map']['PS']
        fh_color_map = DATASETS_CONFIGS[ds_name]['mask_color_map']['FH']
        color_map = {1: ps_color_map, 2: fh_color_map}  # PS=red, FH=green
        for val, color in color_map.items():
            combined_mask[mask == val] = color
        mask = combined_mask

        return image, mask
    
    def _load_fplr(self, row, ds_name) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / ds_name
        img_path = ds_dir / row['image_path']

        # image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # empty mask
        mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        return image, mask
    
    def _load_acslc(self, row, ds_name) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / ds_name
        img_path = ds_dir / row['image_path']
        
        # image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # mask (handle empty paths)
        if pd.isna(row['mask_path']):
            mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        else:
            mask_path = ds_dir / row['mask_path']
            mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            
            # Post-processing: binarize mask
            new_mask = np.zeros_like(mask)
            mask_bin = (mask > 200).any(axis=2)
            new_mask[mask_bin, 0] = 255
            mask = new_mask

        return image, mask

    def _load_fecst(self, row, ds_name) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / ds_name
        img_path = ds_dir / row['image_path']

        # image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        # mask (handle empty paths)
        mask = np.zeros_like(image)
        if not pd.isna(row['mask_path']):
            json_mask_path = ds_dir / row['mask_path']
            with open(json_mask_path) as f:
                json_mask = json.load(f)
                shapes = json_mask.get('shapes', [])
            
            colors = DATASETS_CONFIGS[ds_name]['mask_color_map']
    
            for shape in shapes:
                label = shape['label']
                points = np.array(shape['points'], dtype=np.int32)
                color = colors.get(label)
                
                # Draw filled polygon
                cv2.fillPoly(mask, [points], color)

        return image, mask

    def label_mask(self, mask: np.ndarray, dataset: str, class_values: list) -> np.ndarray:
        """Create 2D mask with universal IDs based on dataset-specific colors"""
        label_map = np.zeros(mask.shape[:2], dtype=np.uint8)
        color_map = DATASETS_CONFIGS[dataset]['mask_color_map']
        
        for universal_name, c in color_map.items():
            class_id = FUS_STRUCTS.index(universal_name)
            if class_id not in class_values:
                continue
            
            # Find pixels matching this color exactly
            matches = (mask[..., 0] == c[0]) & (mask[..., 1] == c[1]) & (mask[..., 2] == c[2])
            label_map[matches] = class_id
        
        return label_map
