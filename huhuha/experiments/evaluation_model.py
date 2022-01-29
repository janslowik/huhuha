import os
from itertools import product
import pandas as pd
from statistics import mode
import click
from importlib_metadata import re

from huhuha.data.data_module import AvalancheDataModule
from huhuha.learning.evaluate_model import evaluate_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def print_results(df: pd.DataFrame):
    print()
    print(f"f1_score___0: {df['mean'].loc['test_f1_0']}")
    print(f"f1_score___1: {df['mean'].loc['test_f1_1']}")
    print("-------------------")
    print(f"f1_____macro: {df['mean'].loc['test_macro_f1']}")
    print(f"loss________: {df['mean'].loc['test_loss']}")
    print(f"accuracy____: {df['mean'].loc['test_accuracy']}")
    print(f"precision__0: {df['mean'].loc['test_precision_0']}")
    print(f"precision__1: {df['mean'].loc['test_precision_1']}")
    print(f"recall_____0: {df['mean'].loc['test_recall_0']}")
    print(f"recall_____1: {df['mean'].loc['test_recall_1']}")


@click.group()
def cli_group():
    """Perform CLI operations."""
    pass


@cli_group.command()
@click.option("--model_checkpoint")
@click.option("--image-src", default=["opentopomap"], multiple=True)
@click.option("--zoom", default=[15], multiple=True)
@click.option("--resize-size", default=32)
@click.option("--batch-size", default=32)
def run(model_checkpoint: str, image_src, zoom, resize_size, batch_size):
    use_cuda = True

    data_module = AvalancheDataModule(
        batch_size=batch_size,
        resize_size=resize_size,
        image_source=image_src,
        zoom=zoom,
    )

    _results = evaluate_model(
        checkpoint_dir=model_checkpoint,
        datamodule=data_module,
        use_cuda=use_cuda
    )


if __name__ == "__main__":
    cli_group()
