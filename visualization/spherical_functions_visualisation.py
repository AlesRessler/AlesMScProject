import numpy as np

from sphericalharmonics.utils import get_storage_index
from sphericalharmonics.basis_functions import real_and_antipodal_spherical_harmonic_basis


def get_spherical_function_values_from_spherical_expansion(expansion_coefficients, max_degree, resolution=50,
                                                           min_value=None, max_value=None, normalize=True):
    """
    Computes values of spherical function that is passed as an argument at a specified resolution. These values can be used for plotting.
    
    Parameters:
    expansion_coefficients (np.array): array of spherical harmonics expansion coefficients representing the function to compute the values of
    max_degree (int): maximum degree of spherical harmonics used for the expansion
    resolution (int): specifies resolution of the visualization
    min_value (int): minimum value of the function used for normalization to colormap scale (if None is given the minimum is calculated from the sample points)
    max_value (int): maximum value of the function used for normalization to colormap scale (if None is given the maximum is calculated from the sample points)
    
    Returns:
    (np.array,np.array,np.array,np.array) The first three arrays correspond to the x,y,z coordinates on the sphere and the last array contains the function values (the function values are scaled to range [0,1]).
    """

    thetas = np.linspace(0, np.pi, resolution)
    phis = np.linspace(0, 2 * np.pi, resolution)
    thetas, phis = np.meshgrid(thetas, phis)

    # The Cartesian coordinates of the unit sphere
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)

    fcolors = np.zeros((resolution, resolution))

    for l in range(0, max_degree + 1, 2):
        for m in range(-l, l + 1):
            coefficient_index = get_storage_index(l, m)

            fcolors += real_and_antipodal_spherical_harmonic_basis(l, m, thetas, phis) * expansion_coefficients[
                coefficient_index]

    # Normalize to range [0,1]
    fmin = None
    fmax = None

    if (min_value is None):
        fmin = fcolors.min()
    else:
        fmin = min_value

    if (max_value is None):
        fmax = fcolors.max()
    else:
        fmax = max_value

    if(normalize):
        fcolors = (fcolors - fmin) / (fmax - fmin)
        fcolors = np.nan_to_num(fcolors, nan=0.0)

    return (x, y, z, fcolors)
