import os
import re
from typing import Optional, List, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report

from huhuha.data.data_module import AvalancheDataModule
from huhuha.learning.classifier import Classifier
from huhuha.settings import LOGS_DIR, CHECKPOINTS_DIR
from huhuha.utils import dictionary_to_json

def get_res(model, dataloader):
    y_pred_all = []
    y_true_all = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in dataloader:

            outputs = model(X_batch)
            _, y_pred = torch.max(outputs.data , 1)
            # y_pred = (y_pred > threshold).int()

            y_pred_all.extend(y_pred.tolist())
            y_true_all.extend(y_batch.tolist())

    res = {
        'y_pred': y_pred_all,
        'y_true': y_true_all,
        'latitudes': list(dataloader.dataset.latitudes),
        'longitudes': list(dataloader.dataset.longitudes)
    }

    return res

def train_test(
    datamodule: AvalancheDataModule,
    model: torch.nn.Module,
    epochs: int = 6,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    name: Optional[str] = None,
    hparams: Optional[Dict] = None,
    use_cuda: bool = False,
    early_stopping_patience: Optional[int] = None,
    custom_callbacks: Optional[List[Callback]] = None,
    trainer_kwargs=None,
    pred=False,
    **kwargs,
):
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    if name is None:
        name = model.__class__.__name__

    logger = TensorBoardLogger(name=name, save_dir=LOGS_DIR, default_hp_metric=False)
    if hparams is not None:
        logger.log_hyperparams(params=hparams)

    model = Classifier(
        model=model,
        num_classes=datamodule.num_classes,
        learning_rate=lr,
        weight_decay=weight_decay,
    )

    checkpoint_dir = CHECKPOINTS_DIR / logger.name / f"version_{logger.version}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    callbacks = [checkpoint_callback]

    if early_stopping_patience is not None:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            verbose=True,
        )
        callbacks += [early_stopping]

    if custom_callbacks:
        callbacks += custom_callbacks

    _use_cuda = use_cuda and torch.cuda.is_available()
    trainer_kwargs = trainer_kwargs or {}
    trainer = pl.Trainer(
        gpus=1 if _use_cuda else 0,
        max_epochs=epochs,
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        **trainer_kwargs,
    )
    trainer.fit(model, train_loader, val_loader)
    results = trainer.test(dataloaders=test_loader)
    dictionary_to_json(
        results[0],
        LOGS_DIR / logger.name / f"version_{logger.version}" / "test_results.json",
    )

    preds = {}
    if pred:
        preds = {
            "test": get_res(model, test_loader),
            "val": get_res(model, val_loader),
            "train": get_res(model, train_loader)
        }


    return results[0], preds
