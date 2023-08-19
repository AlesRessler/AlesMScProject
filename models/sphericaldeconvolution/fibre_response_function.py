import numpy as np

from dataloader.load_dt_simulated import compute_diffusion_tensor, simulate_signal


def simple_fibre_response_function(b_vector, fibre_orientations, diffusion_time, diffusivity):
    """
    Compute value(s) of fibre response function defined below.
    
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
                                       diffusion_tensor_eigenvalues=(0.003, 0.0002, 0.0002)):
    responses = []

    gradient_orientation = b_vector / np.linalg.norm(b_vector)

    for fibre_orientation in fibre_orientations:
        eigenvectors = gram_schmidt_orthonormalization(fibre_orientation)

        diffusion_tensor = compute_diffusion_tensor(eigenvalues=diffusion_tensor_eigenvalues, eigenvectors=eigenvectors)

        response = simulate_signal(b_value=b_value, gradient=gradient_orientation, b_0_signal=3000,
                                   diffusion_tensor=diffusion_tensor)
        responses.append(response)

    return responses


def gram_schmidt_orthonormalization(vector):
    """
    Computes a 3D orthonormal basis given one 3D vector.

    Parameters:
    vector (np.array(1x3)): 3D vector

    Returns:
    (np.array(3x3)) 3D orthonormal basis with basis vectors as columns
    """
    vector1 = vector / np.linalg.norm(vector)

    vector2 = None
    while (True):
        vector2 = np.random.rand(3)
        vector2 /= np.linalg.norm(vector2)

        if (not np.array_equal(vector1, vector2)):
            break

    vector2 -= np.dot(vector2, vector1) * vector1
    vector2 /= np.linalg.norm(vector2)

    vector3 = np.cross(vector1, vector2)

    return np.array([vector1, vector2, vector3]).T
