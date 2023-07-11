from sphericalharmonics.utils import get_storage_index

def get_spherical_function_values_from_spherical_expansion(expansion_coefficients, resolution=50)
    """
    Computes values of spherical function that is passed as an argument at a specified resolution. These values can be used for plotting.
    
    Parameters:
    expansion_coefficients (np.array): array of spherical harmonics expansion coefficients representing the function to compute the values of
    resolution (int): specifies resolution of the visualization
    
    Returns:
    (np.array,np.array,np.array,np.array) The first three arrays correspond to the x,y,z coordinates on the sphere and the last array contains the function values (the function values are scaled to range [0,1]).
    """
    
    phis = np.linspace(0, np.pi, resolution)
    thetas = np.linspace(0, 2*np.pi, resolution)
    phis, thetas = np.meshgrid(phis, thetas)

    # The Cartesian coordinates of the unit sphere
    x = np.sin(phis) * np.cos(thetas)
    y = np.sin(phis) * np.sin(thetas)
    z = np.cos(phis)
    
    fcolors = sph_harm(order, degree, thetas, phis).real
    fmax, fmin = fcolors.max(), fcolors.min()
    fcolors = (fcolors - fmin)/(fmax - fmin)
    fcolors = np.nan_to_num(fcolors,nan=0.0)