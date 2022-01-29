import os
from typing import Optional, List, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from huhuha.data.data_module import AvalancheDataModule
from huhuha.learning.classifier import Classifier
from huhuha.settings import LOGS_DIR, CHECKPOINTS_DIR
from huhuha.utils import dictionary_to_json


def evaluate_model(
        checkpoint_dir: str,
        datamodule: AvalancheDataModule,
        use_cuda: bool = False,
        trainer_kwargs=None,
        **kwargs,
):
    test_loader = datamodule.test_dataloader()

    model = Classifier.load_from_checkpoint(checkpoint_dir)

    _use_cuda = use_cuda and torch.cuda.is_available()
    trainer_kwargs = trainer_kwargs or {}
    trainer = pl.Trainer(
        gpus=1 if _use_cuda else 0,
        **trainer_kwargs,
    )
    results = trainer.test(model=model, dataloaders=test_loader)
    # dictionary_to_json(
    #     results[0],
    #     LOGS_DIR / logger.name / f"version_{logger.version}" / "test_results.json",
    # )

    return results[0]
