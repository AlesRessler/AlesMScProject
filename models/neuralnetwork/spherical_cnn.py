import torch
import numpy as np

from sphericalharmonics.utils import get_number_of_coefficients


class SphericalConvolution(torch.nn.Module):
    def __init__(self, max_degree, number_of_input_channels, number_of_output_channels, number_of_sh_coefficients):
        super().__init__()
        self.max_degree = max_degree
        self.number_of_input_channels = number_of_input_channels
        self.number_of_output_channels = number_of_output_channels
        ls = torch.zeros(number_of_sh_coefficients, dtype=int)

        for l in range(0, max_degree + 1, 2):
            for m in range(-l, l + 1):
                ls[int(0.5 * l * (l + 1) + m)] = l

        self.register_buffer("ls", ls)
        self.weights = torch.nn.Parameter(
            torch.Tensor(self.number_of_output_channels, self.number_of_input_channels, int(self.max_degree / 2) + 1)
        )
        torch.nn.init.uniform_(self.weights)

    def forward(self, x):
        weights_exp = self.weights[:, :, (self.ls / 2).long()]
        ys = torch.sum(
            torch.sqrt(
                np.pi / (2 * self.ls.unsqueeze(0).unsqueeze(0).unsqueeze(0) + 1)
            )
            * weights_exp.unsqueeze(0)
            * x.unsqueeze(1),
            dim=2,
        )
        return ys


class SCNNModel(torch.nn.Module):
    def __init__(self, number_of_shells, output_size, max_degree, spherical_fourier_transform,
                 inverse_spherical_fourier_transform):
        super().__init__()

        number_of_sh_coefficients = get_number_of_coefficients(max_degree)

        self.conv1 = SphericalConvolution(max_degree, number_of_shells, 16,
                                          number_of_sh_coefficients=number_of_sh_coefficients)
        self.conv2 = SphericalConvolution(max_degree, 16, 32, number_of_sh_coefficients=number_of_sh_coefficients)
        self.conv3 = SphericalConvolution(max_degree, 32, 64, number_of_sh_coefficients=number_of_sh_coefficients)
        self.fc1 = torch.nn.Linear(2880, 1000)
        self.bn1 = torch.nn.BatchNorm1d(1000)
        self.fc2 = torch.nn.Linear(1000, 500)
        self.bn2 = torch.nn.BatchNorm1d(500)
        self.fc3 = torch.nn.Linear(500, output_size)
        self.register_buffer("sft", spherical_fourier_transform)
        self.register_buffer("isft", inverse_spherical_fourier_transform)

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
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x
