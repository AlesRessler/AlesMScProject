import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import sph_harm

class SphericalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, max_l):
        super(SphericalConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_l = max_l  # Maximum l value for spherical harmonics
        self.coefficients = nn.Parameter(torch.randn(out_channels, in_channels, (max_l + 1) ** 2))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, input):
        batch_size, in_channels, num_points = input.size()

        # Calculate spherical harmonics coefficients for input data
        input_sph_coeffs = []
        for i in range(batch_size):
            input_sph_coeffs.append([])
            for j in range(in_channels):
                input_sph_coeffs[i].append([])
                for l in range(self.max_l + 1):
                    for m in range(-l, l + 1):
                        input_sph_coeffs[i][j].append(torch.sum(
                            input[i, j] * torch.tensor(sph_harm(m, l, 0, 0), dtype=input.dtype, device=input.device)
                        ))

        input_sph_coeffs = torch.stack(
            [torch.stack([torch.cat(channel) for channel in sample]) for sample in input_sph_coeffs])

        # Calculate spherical harmonics coefficients for convolution kernel
        kernel_sph_coeffs = self.coefficients

        # Perform convolution in spherical harmonics domain
        output_sph_coeffs = torch.einsum('bilm,bijm->bjlm', kernel_sph_coeffs, input_sph_coeffs)

        # Convert spherical harmonics coefficients back to spatial domain
        output = torch.zeros(batch_size, self.out_channels, num_points, dtype=input.dtype, device=input.device)


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