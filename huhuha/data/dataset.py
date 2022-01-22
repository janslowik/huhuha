import os
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

from huhuha.settings import RAW_DATA_DIR


class AvalancheDataset(Dataset):
    def __init__(self, df: pd.DataFrame, resize_size: Optional[int] = 224, normalize: bool = True):
        super().__init__()

        topomap_images = [read_image(
            path=os.path.join(RAW_DATA_DIR / 'opentopomap_tiles' / 'zoom_16' / tile_name),
            mode=ImageReadMode.RGB,
        ).float() for tile_name in tqdm(df["tile_filename_zoom_16"], desc='Loading TopoMap tiles')]

        if resize_size is not None:
            resize = T.Resize((resize_size, resize_size))
            topomap_images = [resize(img) for img in tqdm(topomap_images, desc='Resizing TopoMap tiles')]

        if normalize is not None:
            normalize_func = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            topomap_images = [normalize_func(img) for img in tqdm(topomap_images, desc='Normalizing TopoMap tiles')]

        self.topo_images = topomap_images
        self.elevations = df['elevations'].values.astype(np.float32)
        self.labels = df['Avalanche'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        batch_data = {
            'topo_tile_img': self.topo_images[index],
            'numeric_features': self.elevations[index]
        }
        batch_y = self.labels[index]
        return batch_data, batch_y
