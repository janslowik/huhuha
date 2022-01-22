from typing import Any, Dict, Tuple, Optional

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torchmetrics import (
    Accuracy,
    F1,
    Precision,
    Recall,
)


class Classifier(LightningModule):
    def __init__(
            self, model: torch.nn.Module, num_classes: int,
            learning_rate: float, weight_decay: float = 0.0
    ):
        super(LightningModule, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        metrics = {}
        for split in ["train", "val", "test"]:
            self.metrics[f"{split}_accuracy"] = Accuracy()
            if num_classes:
                metrics[f"{split}_macro_f1"] = F1(
                    average="macro", num_classes=num_classes
                )
                metrics[f"{split}_f1"] = F1(
                    average="none", num_classes=num_classes
                )
                metrics[f"{split}_precision"] = Precision(
                    average="none", num_classes=num_classes
                )
                metrics[f"{split}_recall"] = Recall(
                    average="none", num_classes=num_classes
                )
        self.metrics = torch.nn.ModuleDict(metrics)

    def forward(self, x: Any) -> torch.Tensor:
        x = self.model(x)
        return x

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        output, y_true, loss = self._shared_step(batch)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        output, y_true, loss = self._shared_step(batch)
        self.log("val_loss", loss.item(), on_epoch=True, prog_bar=True)
        self.log_all_metrics(output, y_true, "val")
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        self.log_class_metrics_at_epoch_end('valid')

    def test_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        output, y_true, loss = self._shared_step(batch)
        self.log("test_loss", loss.item(), on_epoch=True, prog_bar=True)
        self.log_all_metrics(output, y_true, "test")
        return {"test_loss": loss}

    def test_epoch_end(self, outputs) -> None:
        self.log_class_metrics_at_epoch_end('test')

    def predict_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int,
            dataloader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        output, y_true, loss = self._shared_step(batch)
        return output

    def _shared_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y_true = batch
        y_true = y_true.long()
        output = self.forward(x)
        loss = F.cross_entropy(output, y_true)
        return output, y_true, loss

    def configure_optimizers(self):
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_all_metrics(self, output, y, split, on_step=None, on_epoch=None):
        output = torch.softmax(output, dim=1)
        log_dict = {}
        for metric_type in self.metric_types:
            metric_key = f'{split}_{metric_type}'
            metric_value = self.metrics[metric_key](output.float(), y.int())

            if not metric_value.size():
                # Log only metrics with single value (e.g. accuracy or metrics averaged over classes)
                log_dict[metric_key] = self.metrics[metric_key]

        self.log_dict(log_dict, on_step=on_step, on_epoch=on_epoch, prog_bar=True)

    def log_class_metrics_at_epoch_end(self, split):
        log_dict = {}
        for metric_type in self.metric_types:
            metric_key = f'{split}_{metric_type}'
            metric = self.metrics[metric_key]

            if metric.average in [None, 'none']:
                metric_value = self.metrics[metric_key].compute()
                for idx in range(metric_value.size(dim=0)):
                    log_dict[f'{metric_key}_{idx}'] = metric_value[idx]

                self.metrics[metric_key].reset()

        self.log_dict(log_dict)
