from pathlib import Path
from typing import List, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from torchvision.datasets import VisionDataset

from utils import get_label_mask, ALL_CLASSES, LABEL_COLORS_LIST
from utils import DATASETS_CONFIG, DATASETS_COLOR_MAPPING, FUS_STRUCTURES

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
        transform: callable = None
    ):
        """
        Args:
            root: Directory root del progetto
            data_path: Directory dei dataset
            datasets: Lista di dataset da includere (es. ['hc18', 'abdominal'])
            split: Split del dataset ('train'/'test'/'val')
            target_size: Dimensione target per resize
            transform: Trasformazioni torchvision da applicare
            augmentation: Trasformazioni albumentations per data augmentation
            label_colors_list: Lista colori per le classi
            all_classes: Lista di tutte le classi
        """
        super().__init__(root, transform=transform)
        
        self.root = Path(root)
        self.csv_data_path = self.root / 'data_csv'
        self.data_path = Path(data_path)
        self.datasets = datasets
        self.target_size = target_size
        self.split = split
        self.supervised = supervised
        self.geometric_augs = augmentations[0] if augmentations else None
        self.color_augs = augmentations[1] if augmentations else None
        self.class_values = list(range(1, len(FUS_STRUCTURES)))
        self.samples = []
        self.dataframes = []
        
        # Caricamento dataset
        for ds_name in self.datasets:
            cfg = DATASETS_CONFIG[ds_name]
            csv_path = self.csv_data_path / cfg["dir_name"] / f'{cfg["csv_name"]}_{self.split}.csv'
            
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found for {ds_name}: {csv_path}")
            
            df = pd.read_csv(csv_path)
            if self.supervised:
                if 'mask_path' in df.columns:
                    df = df[df['mask_path'].notna()]  # Filter out rows with NaN masks
                else:
                    continue  # Skip datasets without mask_path column in supervised mode
            df['dataset'] = ds_name
            self.dataframes.append(df)
            self.samples += [(len(self.dataframes)-1, i) for i in range(len(df))]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        df_idx, sample_idx = self.samples[idx]
        row = self.dataframes[df_idx].iloc[sample_idx]
        ds_name = row['dataset']
        cfg = DATASETS_CONFIG[ds_name]

        # Load image and mask
        image, mask = getattr(self, f'_load_{ds_name}')(row, cfg)
        # print(f'_load_{ds_name}') # debugging
        # Common image and mask resizing
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Data augmentation
        if self.split == 'train' and (self.geometric_augs and self.color_augs):
            if self.supervised:
                # trasformazioni geometriche sincronizzate
                augmented = self.geometric_augs(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask']
                
                # trasformazioni di colore solo all'immagine
                image = self.color_augs(image=image)['image']
            else:
                image = self.geometric_augs(image=image)['image']
                image = self.color_augs(image=image)['image']
        
        # image processing
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image).float() / 255.0 
        
        if not self.supervised:
            return image
        
        # Supervised-only mask processing
        encoded_mask = self.label_mask(mask, ds_name, self.class_values)
        encoded_mask = torch.tensor(encoded_mask).long()
        
        return image, encoded_mask
    
    def _load_hc18(self, row, cfg) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / cfg['dir_name']
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

        # print('_load_hc18')
        
        return image, mask
    
    def _load_abdominal(self, row, cfg) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / cfg['dir_name']
        mask_path = ds_dir / row['mask_path']

        # sfruttare solamente file .npy che contiene sia immagine orirginale che maschere
        data = np.load(mask_path, allow_pickle=True).item()
        
        # image
        image = data['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # mask
        structures = data['structures']
        combined_mask = np.zeros((*next(iter(structures.values())).shape, 3), dtype=np.uint8)
        for name, mask in structures.items():
            combined_mask[mask == 1] = DATASETS_COLOR_MAPPING['abdominal'][name.upper()]
        mask = combined_mask

        # print('_load_abdominal')

        return image, mask
    
    def _load_planesdb(self, row, cfg) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / cfg['dir_name']
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

        # print('_load_planesdb')

        return image, mask
    
    def _load_psfh(self, row, cfg) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / cfg['dir_name']
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
        color_map = {1: [255, 0, 0], 2: [0, 255, 0]}  # PS=red, FH=green
        for val, color in color_map.items():
            combined_mask[mask == val] = color
        mask = combined_mask

        # print('_load_psfh')

        return image, mask
    
    def _load_planesafrica(self, row, cfg) -> Tuple[np.ndarray, np.ndarray]:
        ds_dir = self.data_path / cfg['dir_name']
        img_path = ds_dir / row['image_path']

        # image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # empty mask
        mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # print('_load_planesafrica')

        return image, mask

    def label_mask(self, mask: np.ndarray, dataset: str, class_values: list) -> np.ndarray:
        """Create 2D mask with universal IDs based on dataset-specific colors"""
        label_map = np.zeros(mask.shape[:2], dtype=np.uint8)
        color_map = DATASETS_COLOR_MAPPING[dataset]
        
        for universal_name, c in color_map.items():
            class_id = FUS_STRUCTURES.index(universal_name)
            if class_id not in class_values:
                continue
            
            # Find pixels matching this color exactly
            matches = (mask[..., 0] == c[0]) & (mask[..., 1] == c[1]) & (mask[..., 2] == c[2])
            label_map[matches] = class_id
        
        return label_map
