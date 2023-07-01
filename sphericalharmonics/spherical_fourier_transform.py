import numpy as np
from sphericalharmonics.basis_functions import real_and_antipodal_spherical_harmonic_basis
from sphericalharmonics.utils import get_storage_index

def get_design_matrix(max_degree, number_of_samples, thetas, phis):
    number_of_coefficients = int(0.5 * (max_degree + 1) * (max_degree + 2))

    design_matrix = np.zeros((number_of_samples, number_of_coefficients))

    for l in range(0, max_degree + 1, 2):
        for m in range(-l, l + 1):
            design_matrix[:, get_storage_index(l,m)] = real_and_antipodal_spherical_harmonic_basis(l, m, thetas, phis)
    
    return design_matrix

def get_spherical_fourier_transform(design_matrix):
    spherical_fourier_transform = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T
    
    return spherical_fourier_transform

def get_inverse_spherical_fourier_transform(design_matrix):
    return design_matrix