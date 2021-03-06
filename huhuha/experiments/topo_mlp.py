import os
from itertools import product

from huhuha.data.data_module import AvalancheDataModule
from huhuha.learning.train_test import train_test
from huhuha.models.MLP import MLP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NUM_WORKERS"] = "0"

if __name__ == "__main__":
    rep_num = 1

    name = "MLP_TopoMap"
    model_cls = MLP
    resize_size = 32

    batch_sizes = [32]
    epochs_list = [20]
    lr_list = [1e-3]
    weight_decay_list = [0.01]

    use_cuda = True

    for batch_size in batch_sizes:
        data_module = AvalancheDataModule(
            batch_size=batch_size, resize_size=resize_size
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
                    additional_features=1,
                    input_dim=3 * resize_size * resize_size,
                )
                train_test(
                    data_module,
                    model,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=wd,
                    name=name,
                    hparams=hparams,
                    use_cuda=use_cuda,
                )
