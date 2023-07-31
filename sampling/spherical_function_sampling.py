import numpy as np
from dataloader.load_dt_simulated import generate_random_unit_vector

def random_sampling(function, number_of_samples, coordinates='cartesian' seed=1, args*):
    """
    Generates specified random samples of given spherical function.
    
    Parameters:
    function (function): spherical function to be sampled
    number_of_samples (int): number of samples to be generated
    seed (int): random generator seed
    """
    
    generator = np.random.default_rng(seed)
    random_unit_vector = generate_random_unit_vector(3, generator)