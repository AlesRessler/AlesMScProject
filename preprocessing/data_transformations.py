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

    phis[phis < 0] += 2 * np.pi

    #thetas = np.arccos(qhat[2])
    #phis = np.arctan2(qhat[1], qhat[0]) + np.pi / 2

    return (thetas, phis)


def scale_to_range(data, new_min=-1, new_max=1):
    """
    Scales data to desired range.

    Parameters:
    data (np.array(n x m)): dataset to normalize, where n is the number of data points and m is the data dimensionality
    new_min (int): desired new range minimum
    new_max (int): desired new range maximum

    Returns:
    (np.array(n x m)): normalized dataset
    """

    old_min = np.min(data)
    old_max = np.max(data)

    normalized_data = new_min + ((data - old_min) * (new_max - new_min)) / (old_max - old_min)
    return normalized_data
