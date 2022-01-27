import os
from itertools import product
import pandas as pd
from statistics import mode
import click
from importlib_metadata import re

from huhuha.data.data_module import AvalancheDataModule
from huhuha.learning.train_test import train_test
from huhuha.models.CNN_SEP_MLP import CNN_SEP_MLP
from huhuha.models.CNN_AUG_MLP import CNN_AUG_MLP
from huhuha.settings import RESULTS_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

models_dict = {
    "CNN_SEP_MLP": CNN_SEP_MLP,
    "CNN_AUG_MLP": CNN_AUG_MLP,
}


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
@click.option("--image-src", default=["opentopomap"], multiple=True)
@click.option("--zoom", default=[15], multiple=True)
@click.option("--rep-num", default=1)
@click.option("--epochs-list", default=[20], multiple=True)
@click.option("--model-names", default=["CNN_SEP_MLP"], multiple=True)
@click.option("--resize-size", default=32)
def run(model_names, image_src, zoom, rep_num, resize_size, epochs_list):

    # i left this as alist just for compatibility
    batch_sizes = [32]
    lr_list = [1e-3]
    weight_decay_list = [0.01]
    pretrained_list = [False]
    use_cuda = True

    for m in model_names:

        model_cls = models_dict[m]

        name = f"{m}___src_{'_'.join(image_src)}___zoom_f{'_'.join([str(z) for z in zoom])}"

        results = []

        for batch_size in batch_sizes:
            data_module = AvalancheDataModule(
                batch_size=batch_size,
                resize_size=resize_size,
                image_source=image_src,
                zoom=zoom,
            )
            output_dim = data_module.num_classes

            for epochs, lr, wd, pretrained in product(
                epochs_list, lr_list, weight_decay_list, pretrained_list
            ):
                hparams = {
                    "name": name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "resize_size": resize_size,
                }
                for _ in range(rep_num):
                    model = model_cls(
                        output_dim=output_dim,
                        pretrained=pretrained,
                        additional_features=1,
                        zoom=zoom,
                        image_source=image_src,
                    )
                    _results = train_test(
                        data_module,
                        model,
                        epochs=epochs,
                        lr=lr,
                        weight_decay=wd,
                        name=name,
                        hparams=hparams,
                        use_cuda=use_cuda,
                    )

                    results.append(_results)

        df = pd.DataFrame(results).describe().loc[["mean", "std"]].round(3).transpose()

        df.to_csv(RESULTS_DIR / f"{name}_num_runs_{rep_num}.csv")

        print_results(df)


if __name__ == "__main__":
    cli_group()
