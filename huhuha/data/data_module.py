import os

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from huhuha.data.dataset import AvalancheDataset
from huhuha.settings import DATA_DIR


class AvalancheDataModule(pl.LightningDataModule):
    def __init__(
            self,
            batch_size: int = 64,
            seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size

        df = pd.read_csv(DATA_DIR / 'avalanches-dataset.csv')
        train_df, test_df = train_test_split(
            df,
            train_size=0.7,
            random_state=seed,
            stratify=df['Avalanche']
        )
        val_df, test_df = train_test_split(
            test_df,
            train_size=0.5,
            random_state=seed,
            stratify=test_df['Avalanche']
        )
        self.datasets = {
            "train": AvalancheDataset(train_df),
            "val": AvalancheDataset(val_df),
            "test": AvalancheDataset(test_df)
        }

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._dataloader("test")

    def _dataloader(self, split: str) -> DataLoader:
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=split == "train",
            num_workers=int(os.environ.get("NUM_WORKERS", 0)),
        )
