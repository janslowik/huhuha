import os
from typing import Optional, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.io import read_image, ImageReadMode
from tqdm import tqdm

from huhuha.settings import DATA_DIR, RAW_DATA_DIR


class AvalancheDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        zoom: List[int] = [15],
        image_source: List[str] = ["opentopomap"],
        resize_size: Optional[int] = 224,
        normalize: bool = True,
        label: str = "TRAIN"
    ):
        super().__init__()

        images = {}

        for src in image_source:
            images[src] = {}

            for z in zoom:
                images[src][z] = [
                    read_image(
                        path=os.path.join(
                            DATA_DIR
                            / "center_tiles"
                            / "avalanches"
                            / src
                            / str(z)
                            / f"{_id}.png"
                        ),
                        mode=ImageReadMode.RGB,
                    ).float()
                    for _id in tqdm(
                        df["id"],
                        desc=f"[{label.ljust(5, ' ')}][Loading    ] [tiles from {src}] [zoom level {z}]",
                    )
                ]

        if resize_size is not None:
            resize = T.Resize((resize_size, resize_size))

            for src in image_source:
                for z in zoom:
                    images[src][z] = [
                        resize(img)
                        for img in tqdm(
                            images[src][z],
                            desc=f"[{label.ljust(5, ' ')}][Resizing   ] [tiles from {src}] [zoom level {z}]",
                        )
                    ]

        if normalize is not None:
            normalize_func = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

            for src in image_source:
                for z in zoom:
                    images[src][z] = [
                        normalize_func(img)
                        for img in tqdm(
                            images[src][z],
                            desc=f"[{label.ljust(5, ' ')}][Normalizing] [tiles from {src}] [zoom level {z}]",
                        )
                    ]

        for src in image_source:
            for z in zoom:
                setattr(self, f"images_src_{src}_z_{z}", images[src][z])

        self.elevations = df["elevations"].values.astype(np.float32)
        self.labels = df["Avalanche"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        images_attrs = [p for p in dir(self) if p.startswith("images_src_")]

        images_dict = {attr: getattr(self, attr)[index] for attr in images_attrs}

        batch_data = {
            **images_dict,
            "numeric_features": self.elevations[index],
        }
        batch_y = self.labels[index]
        return batch_data, batch_y
