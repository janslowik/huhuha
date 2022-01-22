import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

from huhuha.settings import RAW_DATA_DIR


class AvalancheDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()

        resize = T.Resize((224, 224))
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        topomap_original_images = [read_image(
            path=os.path.join(RAW_DATA_DIR / 'opentopomap_tiles' / 'zoom_16' / tile_name),
            mode=ImageReadMode.RGB,
        ).float() for tile_name in tqdm(df["tile_filename_zoom_16"], desc='Loading TopoMap tiles')]

        topomap_original_images = [normalize(resize(img)) for img
                                   in tqdm(topomap_original_images, desc='Transforming TopoMap tiles')]

        self.topo_images = topomap_original_images
        self.elevations = df['elevations'].values
        self.labels = df['Avalanche'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        batch_data = {
            'topo_tile': self.topo_images[index],
            'elevations': self.elevations[index]
        }
        batch_y = self.labels[index]
        return batch_data, batch_y
