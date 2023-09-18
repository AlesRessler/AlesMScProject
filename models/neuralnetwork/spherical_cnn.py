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

        for degree in range(0, max_degree + 1, 2):
            for order in range(-degree, degree + 1):
                ls[int(0.5 * degree * (degree + 1) + order)] = degree

        self.register_buffer("ls", ls)
        self.weights = torch.nn.Parameter(
            torch.Tensor(self.number_of_output_channels, self.number_of_input_channels, int(self.max_degree / 2) + 1)
        )
        # torch.nn.init.uniform_(self.weights)
        torch.nn.init.xavier_uniform_(self.weights)
        print(f'[output_channels, input_channels, sh_coefficients]: {self.weights.shape}')

    def forward(self, x):
        weights_expanded = self.weights[:, :, (self.ls / 2).long()]

        spherical_convolution = torch.sqrt(
            np.pi / (2 * self.ls.unsqueeze(0).unsqueeze(0).unsqueeze(0) + 1)
        ) * weights_expanded.unsqueeze(0) * x.unsqueeze(1)

        spherical_convolution = torch.sum(spherical_convolution, dim=2)

        return spherical_convolution


class SCNNModel(torch.nn.Module):
    def __init__(self, number_of_shells, output_size, max_degree, spherical_fourier_transform,
                 inverse_spherical_fourier_transform):
        super().__init__()

        number_of_sh_coefficients = get_number_of_coefficients(max_degree)

        self.encoder1 = SphericalConvolution(max_degree, number_of_shells, 16, number_of_sh_coefficients=number_of_sh_coefficients)
        self.encoder2 = SphericalConvolution(max_degree, 16, 32, number_of_sh_coefficients=number_of_sh_coefficients)
        self.encoder3 = SphericalConvolution(max_degree, 32, 64, number_of_sh_coefficients=number_of_sh_coefficients)
        self.encoder4 = SphericalConvolution(max_degree, 64, 128, number_of_sh_coefficients=number_of_sh_coefficients)

        self.decoder1 = SphericalConvolution(max_degree, 128, 64, number_of_sh_coefficients=number_of_sh_coefficients)
        self.decoder2 = SphericalConvolution(max_degree, 128, 32, number_of_sh_coefficients=number_of_sh_coefficients)
        self.decoder3 = SphericalConvolution(max_degree, 64, 16, number_of_sh_coefficients=number_of_sh_coefficients)
        self.decoder4 = SphericalConvolution(max_degree, 32, 16, number_of_sh_coefficients=number_of_sh_coefficients)
        self.decoder5 = SphericalConvolution(max_degree, 16, output_size, number_of_sh_coefficients=number_of_sh_coefficients)

        self.register_buffer("sft", spherical_fourier_transform)
        self.register_buffer("isft", inverse_spherical_fourier_transform)

    def nonlinearity(self, x):
        return (
                self.sft @ torch.nn.functional.relu(self.isft @ x.unsqueeze(-1))
        ).squeeze(-1)

    def forward(self, x):
        # Encoder
        encoded1 = self.encoder1(x)
        encoded1 = self.nonlinearity(encoded1)

        encoded2 = self.encoder2(encoded1)
        encoded2 = self.nonlinearity(encoded2)

        encoded3 = self.encoder3(encoded2)
        encoded3 = self.nonlinearity(encoded3)

        encoded4 = self.encoder4(encoded3)
        encoded4 = self.nonlinearity(encoded4)

        # Decoder
        decoded1 = self.decoder1(encoded4)
        decoded1 = torch.cat([decoded1, encoded3], dim=1)
        decoded1 = self.nonlinearity(decoded1)

        decoded2 = self.decoder2(decoded1)
        decoded2 = torch.cat([decoded2, encoded2], dim=1)
        decoded2 = self.nonlinearity(decoded2)

        decoded3 = self.decoder3(decoded2)
        decoded3 = torch.cat([decoded3, encoded1], dim=1)
        decoded3 = self.nonlinearity(decoded3)

        decoded4 = self.decoder4(decoded3)
        decoded4 = self.nonlinearity(decoded4)

        decoded5 = self.decoder5(decoded4)
        decoded5 = torch.squeeze(decoded5)
        return decoded5
