from functools import partial
import numpy as np

from dataloader.simulation_noise import add_rician_noise
from sampling.spherical_function_sampling import random_sampling
from preprocessing.data_transformations import convert_coords_from_cartesian_to_spherical
from sphericalharmonics.spherical_fourier_transform import get_spherical_harmonics_expansion_coefficients
from sphericalharmonics.utils import get_number_of_coefficients

def spherical_deconvolution_fit(fibre_response_function, measurements, qhat, b_value, max_degree, number_of_samples_of_fibre_response_function):
    """
    Computes spherical harmonics expansion coefficient of a fibre orientation density function (fODF).

    Parameters:
    fibre_response_function (function): function that computes samples of a fibre response function (FRF). It must accept arguments of b-vector (np.array(1x3)) and (np.array(Nx3)) that specifies unit vectors describing the orientations of the fibres (where N is the number samples to be taken)
    measurements (np.array(1xM)): measured signals (where M is the number of measurements)
    qhat (np.array(3xM): 2D array containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component
    max_degree (int): maximum degree of spherical harmonics to be used for expansion
    number_of_samples_of_fibre_response_function (int): how many samples of the FRF should be taken to compute its spherical harmonics expansion coefficients

    Returns:
    (np.array(1xnumber_of_coefficients)): spherical harmonics expansion coefficients of the fODF
    """
    number_of_spherical_harmonics_expansion_coefficients = get_number_of_coefficients(max_degree)
    design_matrix = np.zeros((len(measurements), number_of_spherical_harmonics_expansion_coefficients))

    for i in range(len(measurements)):
        fibre_response_function_partial = partial(fibre_response_function, b_vector=qhat.T[i]*b_value)
        
        fibre_response_function_vectors, fibre_response_function_samples = random_sampling(function=fibre_response_function_partial, number_of_samples=number_of_samples_of_fibre_response_function, coordinates='cartesian', seed=1)

        #fibre_response_function_samples = np.array([add_rician_noise(measurement=fibre_response_function_sample, signal_to_noise_ratio=0.01) for fibre_response_function_sample in fibre_response_function_samples])

        thetas, phis = convert_coords_from_cartesian_to_spherical(fibre_response_function_vectors.T)

        fibre_response_function_spherical_harmonics_expansion_coefficients = get_spherical_harmonics_expansion_coefficients(samples=fibre_response_function_samples, thetas=thetas, phis=phis, max_degree=max_degree)

        design_matrix[i] = fibre_response_function_spherical_harmonics_expansion_coefficients

    design_matrix_pseudoinverse = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T

    fibre_orientation_density_function_spherical_harmonics_expansion_coefficients = design_matrix_pseudoinverse @ measurements

    return fibre_orientation_density_function_spherical_harmonics_expansion_coefficients
