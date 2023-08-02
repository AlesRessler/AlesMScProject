import numpy as np

def simple_fibre_response_function(b_vector, fibre_orientations, diffusion_time, diffusivity):
    """
    Compute value(s) of fibre response function defined below.
    
    Parameters:
    b_vector (np.array(1x3)): specifies vector describing the orientation and magnitude of the magnetic gradient
    fibre_orientations (np.array(Nx3)): specifies unit vectors describing the orientations of the fibres (where N is the number samples to be taken)
    diffusion_time (int): diffusion time in miliseconds
    diffusivity (int): represents diffusivity
    
    Returns:
    np.array(1xN): Calculated values
    """
    
    return np.exp(-diffusion_time*diffusivity*(b_vector @ fibre_orientations.T)**2)