import torch
import torch.nn as nn
import torchvision.models as M
from typing import Optional, List


class CNN_MLP_SEP(nn.Module):
    def __init__(
        self,
        output_dim=2,
        pretrained: bool = False,
        additional_features: int = 1,
        zoom: List[int] = [15],
        image_source: List[str] = ["opentopomap"],
    ):
        super().__init__()
        self.additional_features = additional_features

        self.zoom = zoom
        self.image_source = image_source

        print("zoom", len(self.zoom))
        print("image_source", len(self.image_source))

        # TODO: check two approaches -> separate resnet for each zoom level or treat zoom level as augmentation

        for src in image_source:
            for z in zoom:
                layer_name = f"resnet_{src}_{z}"

                _resnet_layer = M.resnet18(pretrained=pretrained)
                _resnet_layer.fc = nn.Identity()

                setattr(self, layer_name, _resnet_layer)

        self.fc = nn.Linear(
            512 * len(zoom) * len(image_source) + additional_features, output_dim
        )

    def forward(self, batch_data):

        image_outputs = []

        for src in self.image_source:
            for z in self.zoom:

                resnet_for_src_z = getattr(self, f"resnet_{src}_{z}")
                data_for_src_z = batch_data[f"images_src_{src}_z_{z}"]
                out_for_src_z = resnet_for_src_z(data_for_src_z)

                image_outputs.append(out_for_src_z)

        x_numeric = batch_data["numeric_features"]
        x_numeric = x_numeric.view(-1, self.additional_features)

        out = torch.cat([*image_outputs, x_numeric], dim=1)
        out = self.fc(out)

        return out
