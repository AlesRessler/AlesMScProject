from scipy.special import sph_harm
import numpy as np

def real_and_antipodal_spherical_harmonic_basis(l, m, thetas, phis):
    if l % 2 == 1:
        return np.zeros(len(thetas))
    if m < 0:
        return np.sqrt(2) * sph_harm(-m, l, phis, thetas).imag
    if m == 0:
        return sph_harm(m, l, phis, thetas).real
    if m > 0:
        return np.sqrt(2) * sph_harm(m, l, phis, thetas).real