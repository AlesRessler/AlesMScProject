import numpy as np
from sphericalharmonics.basis_functions import real_and_antipodal_spherical_harmonic_basis
from sphericalharmonics.utils import get_storage_index, get_number_of_coefficients

def get_design_matrix(max_degree, number_of_samples, thetas, phis):
    """
    Computes design matrix B for least squares SH coefficients estimation
    
    Parameters:
    max_degree (int): maximum degree of the spherical harmonic to be estimated, l>0
    number_of_samples (int): number of samples available for estimation
    thetas (np.array): 1D array of polar angles
    phis (np.array): 1D array of azimuthal angles
    
    Returns:
    (np.array): 2D design matrix
    """
    number_of_coefficients = get_number_of_coefficients(max_degree)

    design_matrix = np.zeros((number_of_samples, number_of_coefficients))
    
    # Odd degree spherical harmonics can be skipped thus the step of 2 is used
    for l in range(0, max_degree + 1, 2):
        for m in range(-l, l + 1):
            design_matrix[:, get_storage_index(l,m)] = real_and_antipodal_spherical_harmonic_basis(l, m, thetas, phis)
    
    return design_matrix

def get_spherical_fourier_transform(design_matrix):
    """
    Computes spherical fourier transform matrix
    
    Parameters:
    design_matrix (np.array): 2D design matrix (can be computed by the get_design_matrix function)
    
    Returns:
    (np.array): 2D spherical fourier transform matrix
    """
    spherical_fourier_transform = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T
    
    return spherical_fourier_transform

def get_inverse_spherical_fourier_transform(design_matrix):
    """
    Computes inverse spherical fourier transform matrix
    
    Parameters:
    design_matrix (np.array): 2D design matrix (can be computed by the get_design_matrix function)
    
    Returns:
    (np.array): 2D inverse spherical fourier transform matrix
    """
    return design_matrix

def get_spherical_harmonics_expansion_coefficients(samples, thetas, phis, max_degree):
    """
    Computes spherical harmonics expansion coefficients from function samples.

    Parameters:
    samples (np.array(1xN)): array of N function samples (where N is the number of samples)
    thetas (np.array(1xN): array of N collatidute angles
    phis (np.array(1xN): array of N longitude angles
    max_degree (int): maximum degree of spherical harmonics to be used for expansion

    Returns:
    (np.array(max_degree)) Spherical harmonics expansion coefficients
    """

    design_matrix = get_design_matrix(max_degree=max_degree,number_of_samples=len(samples), thetas=thetas, phis=phis)
    
    spherical_fourier_transform = get_spherical_fourier_transform(design_matrix)

    expansion_coefficients = spherical_fourier_transform @ samples

    return expansion_coefficients