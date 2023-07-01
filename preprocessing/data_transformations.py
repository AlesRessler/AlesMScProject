import numpy as np

def convert_coords_from_cartesian_to_spherical(qhat):
    thetas = np.arccos(qhat[2])
    phis = np.arctan2(qhat[1], qhat[0]) + np.pi / 2
    
    return (thetas, phis)