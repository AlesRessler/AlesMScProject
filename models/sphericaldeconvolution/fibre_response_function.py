import numpy as np

def simple_fibre_response_function(b_vector, fibre_orientation, diffusion_time, diffusivity):
    """
    Compute value(s) of fibre response function defined below.
    
    Parameters:
    b_vector (3-tuple) or np.array((3-tuple)): specifies vector describing the orientation and magnitude of the magnetic gradient
    fibre_orientation (3-tuple) or np.array((3-tuple)): specifies unit vector describing the orientation of the fibres
    diffusion_time (int): diffusion time in miliseconds
    diffusivity (int): represents diffusivity
    
    Returns:
    (3-tuple) or np.array((3-tuple)): Calculated values
    """
    
    return np.exp(-diffusion_time*diffusivuty(b_vector @ fibre_orientation)**2)