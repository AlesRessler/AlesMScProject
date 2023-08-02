import numpy as np
import matplotlib.pyplot as plt

from sphericalharmonics.spherical_fourier_transform import get_spherical_fourier_transform
from sphericalharmonics.spherical_fourier_transform import get_design_matrix
from sphericalharmonics.spherical_fourier_transform import get_inverse_spherical_fourier_transform

def plot_reconstruction_error_using_spherical_harmonics_up_to_degree(measurements, thetas, phis, max_degree_upper_limit, comparison_measurements):
    all_expansion_coefficients = []
    all_inverse_spherical_fourier_transforms = []

    all_design_matrices = []

    for max_degree in range(max_degree_upper_limit):
        design_matrix = get_design_matrix(max_degree=max_degree, number_of_samples=len(measurements), thetas=thetas, phis=phis)
        all_design_matrices.append(design_matrix)
        spherical_fourier_transform = get_spherical_fourier_transform(design_matrix)

        expansion_coefficients = spherical_fourier_transform @ measurements
        inverse_spherical_fourier_transform = get_inverse_spherical_fourier_transform(design_matrix)

        all_expansion_coefficients.append(expansion_coefficients)
        all_inverse_spherical_fourier_transforms.append(inverse_spherical_fourier_transform)

    all_reconstructed_signals = []

    for max_degree in range(max_degree_upper_limit):
        reconstructed_signal = all_inverse_spherical_fourier_transforms[max_degree] @ all_expansion_coefficients[
            max_degree]
        all_reconstructed_signals.append(reconstructed_signal)

    all_reconstruction_errors = []

    for max_degree in range(max_degree_upper_limit):
        reconstruction_errors = comparison_measurements - all_reconstructed_signals[max_degree]
        all_reconstruction_errors.append(reconstruction_errors)

    all_mean_reconstruction_errors = []

    for max_degree in range(max_degree_upper_limit):
        mean_reconstruction_error = np.mean(np.absolute(all_reconstruction_errors[max_degree]))
        all_mean_reconstruction_errors.append(mean_reconstruction_error)

    print(all_mean_reconstruction_errors)

    even_degree_mean_reconstruction_errors = []

    for i in range(len(all_mean_reconstruction_errors)):
        if (i % 2 == 0):
            even_degree_mean_reconstruction_errors.append(all_mean_reconstruction_errors[i])

    plt.plot(range(0, max_degree_upper_limit, 2), even_degree_mean_reconstruction_errors)
    plt.xticks(range(0, max_degree_upper_limit, 2))
    plt.xlabel("Maximum SH degree")
    plt.ylabel("Mean reconstruction error")
    plt.show()