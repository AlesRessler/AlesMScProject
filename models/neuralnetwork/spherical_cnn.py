import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import sph_harm

class SphericalConvolution(torch.nn.Module):
    def __init__(self, l_max, c_in, c_out):
        super().__init__()
        self.l_max = l_max
        self.c_in = c_in
        self.c_out = c_out
        ls = torch.zeros(n_coeffs, dtype=int)
        for l in range(0, l_max + 1, 2):
            for m in range(-l, l + 1):
                ls[int(0.5 * l * (l + 1) + m)] = l
        self.register_buffer("ls", ls)
        self.weights = torch.nn.Parameter(
            torch.Tensor(self.c_out, self.c_in, int(self.l_max / 2) + 1)
        )
        torch.nn.init.uniform_(self.weights)

    def forward(self, x):
        weights_exp = self.weights[:, :, (self.ls / 2).long()]
        ys = torch.sum(
            torch.sqrt(
                math.pi / (2 * self.ls.unsqueeze(0).unsqueeze(0).unsqueeze(0) + 1)
            )
            * weights_exp.unsqueeze(0)
            * x.unsqueeze(1),
            dim=2,
        )
        return ys


class SimpleSphericalCNN(nn.Module):
    def __init__(self, num_classes, max_l):
        super(SimpleSphericalCNN, self).__init__()
        self.spherical_conv1 = SphericalConvolution(in_channels=3, out_channels=16, max_l=max_l)
        self.spherical_conv2 = SphericalConvolution(in_channels=16, out_channels=32, max_l=max_l)
        self.fc1 = nn.Linear(32 * ((max_l + 1) ** 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input):
        x = F.relu(self.spherical_conv1(input))
        x = F.relu(self.spherical_conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SCNNModel(torch.nn.Module):
    def __init__(self, n_in, n_out, l_max):
        super().__init__()
        self.conv1 = SphericalConvolution(l_max, n_in, 16)
        self.conv2 = SphericalConvolution(l_max, 16, 32)
        self.conv3 = SphericalConvolution(l_max, 32, 64)
        self.fc1 = torch.nn.Linear(64, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.fc3 = torch.nn.Linear(128, n_out)
        self.register_buffer("sft", sft)
        self.register_buffer("isft", isft)

    def nonlinearity(self, x):
        return (
            self.sft @ torch.nn.functional.relu(self.isft @ x.unsqueeze(-1))
        ).squeeze(-1)

    def global_pooling(self, x):
        return torch.mean(self.isft @ x.unsqueeze(-1), dim=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = self.conv2(x)
        x = self.nonlinearity(x)
        x = self.conv3(x)
        x = self.nonlinearity(x)
        x = self.global_pooling(x)
        x = x.squeeze(2)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x