import torch
import torch.nn as nn
import torchvision.models as M


class MLP(nn.Module):
    def __init__(
        self,
        output_dim=2,
        input_dim: int = 3 * 224 * 224,
        additional_features: int = 1,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.additional_features = additional_features
        self.input_img_dim = input_dim

        self.fc1 = nn.Linear(input_dim + additional_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch_data):
        x_img = batch_data["topo_tile_img"]
        x_img = x_img.view(-1, self.input_img_dim)

        x_numeric = batch_data["numeric_features"]
        x_numeric = x_numeric.view(-1, self.additional_features)
        x = torch.cat([x_img, x_numeric], dim=1)

        out = self.fc1(x)
        out = self.fc2(out)
        return out
