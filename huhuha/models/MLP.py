import torch
import torch.nn as nn
import torchvision.models as M
from typing import Optional, List


class MLP(nn.Module):
    def __init__(
        self,
        output_dim=2,
        additional_features: int = 1,
        hidden_dim: int = 512,
        zoom: List[int] = [15],
        image_source: List[str] = ["opentopomap"],
        resize_size: int = 224,
        **_,
    ):
        super().__init__()

        self.zoom = zoom
        self.image_source = image_source

        self.additional_features = additional_features
        self.input_img_dim = 3 * resize_size * resize_size

        self.fc1 = nn.Linear(
            self.input_img_dim * len(image_source) * len(zoom) + additional_features,
            hidden_dim,
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch_data):

        image_data = []

        for src in self.image_source:
            for z in self.zoom:

                data_for_src_z = batch_data[f"images_src_{src}_z_{z}"]

                data_for_src_z = data_for_src_z.view(-1, self.input_img_dim)

                image_data.append(data_for_src_z)

        x_numeric = batch_data["numeric_features"]
        x_numeric = x_numeric.view(-1, self.additional_features)

        x = torch.cat([*image_data, x_numeric], dim=1)

        out = self.fc1(x)
        out = self.fc2(out)
        return out
