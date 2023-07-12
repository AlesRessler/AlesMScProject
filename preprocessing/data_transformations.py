import numpy as np

def convert_coords_from_cartesian_to_spherical(qhat):
    """
    Converts 2D array of gradient directions in cartesian coordinates to spherical coordinates.
    
    Parameters:
    qhat (np.array): 2D array containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component
    
    Returns:
    (np.array thetas, np.array phis): thetas is a 1D array of polar angles ranging from 0 to pi, phis is a 1D array of azimuthal angles ranging from 0 to 2pi
    """
    thetas = np.arccos(qhat[2])
    phis = np.arctan2(qhat[1], qhat[0])
    
    phis[phis < 0] += 2*np.pi
    
    return (thetas, phis)