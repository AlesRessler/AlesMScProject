import numpy as np
from dataloader.load_dt_simulated import generate_random_unit_vector

def random_sampling(function, number_of_samples, coordinates='cartesian', seed=1):
    """
    Generates specified number of random samples of given spherical function.
    
    Parameters:
    function (function): spherical function to be sampled. The function must accept an np.array(Nx3) where N is the number of samples to be generated.
    number_of_samples (int): number of samples to be generated
    seed (int): random generator seed

    Returns:
    (np.array(Nx3), np.array(N)): unit vectors the function was evaluated at, the function values
    """
    
    generator = np.random.default_rng(seed)

    if(coordinates == 'cartesian'):
        random_unit_vectors = []
    
        for i in range(number_of_samples):
            random_unit_vector = generate_random_unit_vector(3, generator)
            random_unit_vectors.append(random_unit_vector)
    
        random_unit_vectors = np.array(random_unit_vectors)
    
        values = function(random_unit_vectors)
    
        return (random_unit_vectors, values)
    else
        raise Exception('Invalid coordinate system')