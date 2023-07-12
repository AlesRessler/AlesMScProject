from scipy.special import sph_harm
import numpy as np

def real_and_antipodal_spherical_harmonic_basis(l, m, thetas, phis):
    """
    Computes the value of a real and antipodal spherical harmonic basis of degree l and order m evaluated at spherical coordinates theta and phi.
    
    Parameters
    l (int): degree of the psherical harmonic, l>0
    m (int): order of the spherical harmonic, -l <= m <= l
    thetas (np.array): 1D array of polar angles [0, pi]
    phis (np.array): 1D array of azimuthal angles [0, 2pi]
    
    Returns
    (np.array): 1D array of the values of the spherical harmonics evaluated at thetas and phis
    """
    if l % 2 == 1:
        return np.zeros(len(thetas))
    if m < 0:
        return np.sqrt(2) * sph_harm(-m, l, phis, thetas).imag
    if m == 0:
        return sph_harm(m, l, phis, thetas).real
    if m > 0:
        return np.sqrt(2) * sph_harm(m, l, phis, thetas).real