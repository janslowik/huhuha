import torch
import torch.nn as nn
import torchvision.models as M


class CNN_MLP(nn.Module):
    def __init__(self, output_dim=2, pretrained: bool = False, additional_features: int = 1):
        super().__init__()
        self.additional_features = additional_features
        self.resnet = M.resnet18(pretrained=pretrained)
        self.resnet_a = M.resnet18(pretrained=pretrained) 
        self.resnet.fc = nn.Identity()
        self.resnet_a.fc = nn.Identity()
        self.fc = nn.Linear(512*2 + additional_features, output_dim)

    def forward(self, batch_data):
        x_img = batch_data['topo_tile_img']
        x_img_a = batch_data['arcgis_tile_img']
        x_numeric = batch_data['numeric_features']
        x_numeric = x_numeric.view(-1,  self.additional_features)

        out = self.resnet(x_img)
        out_a = self.resnet_a(x_img_a)
        out = torch.cat([out, out_a, x_numeric], dim=1)
        out = self.fc(out)
        return out
