import torch
import torch.nn as nn
import torchvision.models as M
from typing import Optional, List


class CNN_AUG_MLP(nn.Module):
    def __init__(
        self,
        output_dim=2,
        pretrained: bool = False,
        additional_features: int = 1,
        zoom: List[int] = [15],
        image_source: List[str] = ["opentopomap"],
        **_,
    ):
        super().__init__()
        self.additional_features = additional_features

        self.zoom = zoom
        self.image_source = image_source

        for src in image_source:
            layer_name = f"resnet_{src}"

            _resnet_layer = M.resnet18(pretrained=pretrained)
            _resnet_layer.fc = nn.Identity()

            setattr(self, layer_name, _resnet_layer)

        self.fc = nn.Linear(
            512 * len(image_source) + additional_features, output_dim
        )

    def forward(self, batch_data):

        image_outputs = []

        for src in self.image_source:
            resnet_for_src = getattr(self, f"resnet_{src}")
            partial = []

            for z in self.zoom:
                data_for_src_z = batch_data[f"images_src_{src}_z_{z}"]
                partial.append(resnet_for_src(data_for_src_z))


            mean_for_src = torch.mean(torch.stack(partial), dim=0)

            image_outputs.append(mean_for_src)

        x_numeric = batch_data["numeric_features"]
        x_numeric = x_numeric.view(-1, self.additional_features)

        out = torch.cat([*image_outputs, x_numeric], dim=1)
        out = self.fc(out)

        return out
