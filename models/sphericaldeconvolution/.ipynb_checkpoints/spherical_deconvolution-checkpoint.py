from functools import partial
import numpy as np

from sampling.spherical_function_sampling import random_sampling
from preprocessing.data_transformations import convert_coords_from_cartesian_to_spherical
from sphericalharmonics.spherical_fourier_transform import get_spherical_harmonics_expansion_coefficients

def spherical_deconvolution_fit(fibre_response_function, measurements, qhat, max_degree, diffusion_time, diffusivity, number_of_samples_of_fibre_response_function):
    """
    """
    design_matrix = np.zeros((len(measurements), max_degree))

    for i in range(len(measurements)):
        fibre_response_function_partial = partial(fibre_response_function, b_vector=qhat.T[i], diffusion_time=diffusion_time, diffusivity=diffusivity)
        
        fibre_response_function_vectors, fibre_response_function_samples = random_sampling(function=fibre_response_function_partial, number_of_samples=number_of_samples_of_fibre_response_function, coordinates='cartesian', seed=1)

        thetas, phis = convert_coords_from_cartesian_to_spherical(fibre_response_function_vectors.T)

        fibre_response_function_spherical_harmonics_expansion_coefficients = get_spherical_harmonics_expansion_coefficients(samples=fibre_response_function_samples, thetas=thetas, phis=phis, max_degree=max_degree)

        design_matrix[i] = fibre_response_function_spherical_harmonics_expansion_coefficients

    design_matrix_pseudoinverse = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T

    fibre_orientation_density_function_spherical_harmonics_expansion_coefficients = design_matrix_pseudoinverse @ measurements

    return fibre_orientation_density_function_spherical_harmonics_expansion_coefficients
