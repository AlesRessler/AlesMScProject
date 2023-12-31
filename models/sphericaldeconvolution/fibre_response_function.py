import numpy as np

from dataloader.load_dt_simulated import compute_diffusion_tensor, simulate_normalized_signal
from mathematics.gram_schmidt_orthonormalization import gram_schmidt_orthonormalization


def simple_fibre_response_function(b_vector, fibre_orientations, diffusion_time, diffusivity):
    """
    Compute normalized value(s) of stick fibre response function defined below.
    
    Parameters:
    b_vector (np.array(1x3)): specifies vector describing the orientation and magnitude of the magnetic gradient
    fibre_orientations (np.array(Nx3)): specifies unit vectors describing the orientations of the fibres (where N is the number samples to be taken)
    diffusion_time (int): diffusion time in seconds
    diffusivity (int): represents diffusivity
    
    Returns:
    np.array(1xN): Calculated values
    """

    return np.exp(-diffusion_time * diffusivity * (b_vector @ fibre_orientations.T) ** 2)


def diffusion_tensor_response_function(b_vector, fibre_orientations, b_value,
                                       diffusion_tensor_eigenvalues=(0.003, 0.0002, 0.0002), b_0_signal=3000):
    """
    Compute normalised values of fibre response function using diffusion tensor model.
    Args:
        b_vector:
        fibre_orientations:
        b_value:
        diffusion_tensor_eigenvalues:
        b_0_signal:

    Returns: (np.array): Simulated normalised signals

    """
    responses = []

    gradient_orientation = b_vector / np.linalg.norm(b_vector)

    for fibre_orientation in fibre_orientations:
        eigenvectors = gram_schmidt_orthonormalization(fibre_orientation)

        diffusion_tensor = compute_diffusion_tensor(eigenvalues=diffusion_tensor_eigenvalues, eigenvectors=eigenvectors)

        response = simulate_normalized_signal(b_value=b_value, gradient=gradient_orientation, b_0_signal=b_0_signal,
                                              diffusion_tensor=diffusion_tensor)
        responses.append(response)

    return responses
