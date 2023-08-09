import numpy as np

from sphericalharmonics.basis_functions import real_and_antipodal_spherical_harmonic_basis
from sphericalharmonics.utils import get_storage_index, get_number_of_coefficients
from preprocessing.data_transformations import convert_coords_from_cartesian_to_spherical


def load_fodf_simulated(max_degree, fibre_orientations=None):
    """
    Computes spherical harmonics expansion coefficients of fODF with given fibre orientations

    Parameters:
    fibre_orientations np.array(Nx3): where N is the number of fibre populations
    max_degree: maximum degree of spherical harmonics used in the expansion

    Returns:
    (np.array(1 x number_of_coefficients)): spherical harmonics expansion coefficients of the fODF
    """

    if (fibre_orientations is None):
        fibre_orientations = np.array([[1, 0, 0], [0, 1, 0],[0.80178373, 0.26726124, 0.53452248]])

    thetas, phis = convert_coords_from_cartesian_to_spherical(fibre_orientations.T)

    number_of_coefficients = get_number_of_coefficients(max_degree)
    spherical_harmonics_coefficients = np.zeros(number_of_coefficients)

    for i in range(len(fibre_orientations)):
        for l in range(0, max_degree + 1, 2):
            for m in range(-l, l + 1):
                spherical_harmonics_coefficients[
                    get_storage_index(l, m)] += np.sin(thetas[i]) * real_and_antipodal_spherical_harmonic_basis(l,
                                                                                                            m,
                                                                                                            thetas[i],
                                                                                                            phis[i])
    return spherical_harmonics_coefficients
