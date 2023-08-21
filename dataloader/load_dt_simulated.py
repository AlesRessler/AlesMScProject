import numpy as np
from numpy.linalg import pinv, norm

from dataloader import simulation_noise
from dataloader.load_fodf_simulated import load_fodf_simulated
from mathematics.gram_schmidt_orthonormalization import gram_schmidt_orthonormalization


def load_dt_simulated(number_of_data_points=90, b_value=1000, b_0_signal=3000, include_b_0=False,
                      signal_to_noise_ratio=30, eigenvalues=(1, 0, 0), eigenvectors=None, noise_type='rician',
                      noise_generator=None, gradient_generator=None):
    """
    Returns dataset simulated from the diffusion tensor model with specified number of data points and noise standard deviation.
    
    Parameters:
    number_of_data_points (int): number of data points to be simulated
    b_value (int): non-zero b-value used in the simulation
    b_0_signal (int): non-diffusion weighted signal
    include_b_0 (bool): determines whether the data points should include b=0 measurements
    signal_to_noise_ratio (number): signal-to-noise ratio
    eigenvalues (3-tuple): eigenvalues of the eigenvectors of the diffusion tensor
    eigenvectors (np.array(3x3)): eigenvectors (column vectors) of the diffusion tensor, if None then vectors (1,0,0),(0,1,0),(0,0,1) are used
    noise_type (string): type of noise to be added to the measurements. Supported noise types: rician, none
    noise_generator (np.Generator): generator used to create noise (if None given then a new Generator is instantiated)
    gradient_generator (np.Generator): generator used to create gradient directions (if None given then a new Generator with seed 1 is instantiated)
    
    Returns:
    (np.array, np.array, np.array): The first array is 1D containg the b-values, 2D array containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component, third array is 1D containing DWI signals
    """

    if eigenvectors is None:
        eigenvectors = np.zeros((3, 3))

        eigenvectors[0, 0] = 1.0
        eigenvectors[1, 1] = 1.0
        eigenvectors[2, 2] = 1.0

    diffusion_tensor = compute_diffusion_tensor(eigenvalues, eigenvectors)

    if (noise_generator is None):
        noise_generator = np.random.default_rng()
    if (gradient_generator is None):
        # The seed is fixed here to ensure equal gradients across multiple runs even when
        # generator is not specified
        gradient_generator = np.random.default_rng(1)

    supported_noise_types = {'rician', 'none'}

    if (not (noise_type in supported_noise_types)):
        raise Exception('Invalid noise type')

    # b-values
    bvals = []
    # Gradient orientations
    qhat = [[], [], []]
    # Measured signals
    measurements = []

    for i in range(number_of_data_points):

        # If include_b_0 is true then every 10th measurement is b=0 measurement
        if (include_b_0 and ((i + 1) % 10 == 0)):
            bvals.append(0)

            qhat[0].append(0.0)
            qhat[1].append(0.0)
            qhat[2].append(0.0)

            noisy_measurement = None

            if (noise_type == 'rician'):
                noisy_measurement = simulation_noise.add_rician_noise(measurement=b_0_signal,
                                                                      signal_to_noise_ratio=signal_to_noise_ratio,
                                                                      generator=noise_generator)
            elif (noise_type == 'none'):
                noisy_measurement = b_0_signal

            measurements.append(noisy_measurement)
        else:
            bvals.append(b_value)

            random_unit_vector = generate_random_unit_vector(3, gradient_generator)

            qhat[0].append(random_unit_vector[0])
            qhat[1].append(random_unit_vector[1])
            qhat[2].append(random_unit_vector[2])

            measurement = simulate_signal(b_value, random_unit_vector, b_0_signal, diffusion_tensor)

            noisy_measurement = None

            if (noise_type == 'rician'):
                noisy_measurement = simulation_noise.add_rician_noise(measurement=measurement,
                                                                      signal_to_noise_ratio=signal_to_noise_ratio,
                                                                      generator=noise_generator)
            elif (noise_type == 'none'):
                noisy_measurement = measurement

            measurements.append(noisy_measurement)

    bvals = np.array(bvals)
    qhat = np.array(qhat)
    measurements = np.array(measurements)

    return (bvals, qhat, measurements)


def compute_diffusion_tensor(eigenvalues, eigenvectors):
    """
    Computes diffusion tensor from eigenvalues and eigenvectors.
    
    Parameters:
    eigenvalues (3-tuple): eigenvalues of the eigenvectors of the diffusion tensor
    eigenvectors (np.array(3x3)): eigenvectors (column vectors) of the diffusion tensor
    
    Returns:
    np.array(3x3): computed diffusion tensor
    """

    diagonal_eigenvalue_matrix = np.zeros((3, 3))

    # Insert eigenvalues to the diagonal
    for i in range(len(eigenvalues)):
        diagonal_eigenvalue_matrix[i, i] = eigenvalues[i]

    # Compute the diffusion tensor according to the eigendecomposition
    diffusion_tensor = eigenvectors @ diagonal_eigenvalue_matrix @ pinv(eigenvectors)

    return diffusion_tensor


def simulate_signal(b_value, gradient, b_0_signal, diffusion_tensor):
    """
    Produce signal measurement according to the diffusion model.
    
    Parameters:
    b-value (int): non-zero b-value used in the simulation
    gradient (np.array(3x1)): unit (column) vector of the gradient direction
    b_0_signal (int): non-diffusion weighted signal
    diffusion_tensor (np.array(3x3)): diffusion tensor to be used for simulation
    """

    signal = b_0_signal * np.exp(-b_value * (gradient.T @ diffusion_tensor @ gradient))

    return signal


def generate_random_unit_vector(dimension, generator):
    """
    Generates random unit vector (distributed uniformly on the n-sphere) of specified dimension.
    
    Parameters:
    dimension (int): determines dimension of the vector
    generator (np.random.Generator): generator to be used
    
    
    Returns:
    np.array(dimension x 1): random unit vector
    """

    random_vector = generator.normal(loc=0.0, scale=1.0, size=dimension)
    random_unit_vector = random_vector / norm(random_vector)

    return random_unit_vector


def load_dt_simulated_multiple_populations(number_of_data_points=90, b_value=1000, b_0_signal=3000, include_b_0=False,
                                           signal_to_noise_ratio=30, eigenvalues=[(1, 0, 0), (0, 1, 0)],
                                           eigenvectors=[None, None], fractions=[0.5, 0.5], noise_type='rician',
                                           noise_generator_seed=1,
                                           gradient_generator_seed=1):
    """
    Returns dataset simulated from the diffusion tensor model with specified number of data points, noise standard deviation and multiple fibre populations.
    
    Parameters:
    number_of_data_points (int): number of data points to be simulated
    b_value (int): non-zero b-value used in the simulation
    b_0_signal (int): non-diffusion weighted signal
    include_b_0 (bool): determines whether the data points should include b=0 measurements
    signal_to_noise_ratio (number): signal-to-noise ratio
    eigenvalues [(3-tuple)]: eigenvalues of the eigenvectors of the diffusion tensors (one 3-tuple for each fibre population)
    eigenvectors [(np.array(3x3))]: eigenvectors (column vectors) of the diffusion tensor, if None then vectors (1,0,0),(0,1,0),(0,0,1) are used (one 3x3 matrix for each fibre population)
    fractions [int]: volume fractions of each fibre population
    noise_type (string): type of noise to be added to the measurements. Supported noise types: rician, none
    noise_generator_seed (int): generator seed used to create noise (if None given then a new Generator is instantiated)
    gradient_generator_seed (int): generator seed used to create gradient directions (if None given then a new Generator with seed 1 is instantiated)
    
    Returns:
    (np.array, np.array, np.array): The first array is 1D containg the b-values, 2D array containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component, third array is 1D containing DWI signals
    """

    noise_generator = np.random.default_rng(noise_generator_seed)
    gradient_generator = np.random.default_rng(gradient_generator_seed)

    supported_noise_types = {'rician', 'none'}

    if (not (noise_type in supported_noise_types)):
        raise Exception('Invalid noise type')

    # b-values
    bvals = None
    # Gradient orientations
    qhat = None
    # Signal measurements
    measurements = np.zeros(number_of_data_points)

    # Simulate signal for each volume fraction (fibre population)
    for i in range(len(fractions)):
        # Simulate measurements without noise
        bvals_temp, qhat_temp, measurements_temp = load_dt_simulated(number_of_data_points=number_of_data_points,
                                                                     b_value=b_value, b_0_signal=b_0_signal,
                                                                     include_b_0=include_b_0,
                                                                     signal_to_noise_ratio=0,
                                                                     eigenvalues=eigenvalues[i],
                                                                     eigenvectors=eigenvectors[i], noise_type='none',
                                                                     noise_generator=noise_generator,
                                                                     gradient_generator=gradient_generator)

        if (i == 0):
            bvals = bvals_temp
            qhat = qhat_temp

        # Weight the population signal by its volume fraction
        measurements_temp = measurements_temp * fractions[i]

        measurements = measurements + measurements_temp

        # Reset RNG to produce the same gradient orientations in next iteration
        gradient_generator = np.random.default_rng(gradient_generator_seed)

    # Add noise
    if (noise_type == 'rician'):
        measurements = np.array([simulation_noise.add_rician_noise(measurement=measurement,
                                                                   signal_to_noise_ratio=signal_to_noise_ratio,
                                                                   generator=noise_generator) for measurement in
                                 measurements])
    elif (noise_type == 'none'):
        measurements = measurements

    return (bvals, qhat, measurements)


def load_dt_simulated_dataset(dataset_size=1000, number_of_fibre_populations=2, max_degree=8,
                              fibre_population_eigenvalues=None, number_of_data_points=90, b_value=1000,
                              b_0_signal=3000, include_b_0=False,
                              signal_to_noise_ratio=30, noise_type='rician',
                              noise_generator_seed=1,
                              gradient_generator_seed=1,
                              fibre_orientation_generator_seed=1, no_rotation=False):
    """
    Parameters:
    dataset_size (int): number of datapoints to be generated
    number_of_fibre_populations (int): number of fibre populations in one voxel
    max_degree (int): maximum degree of spherical harmonics to be used to represent fODF
    fibre_population_eigenvalues (3-tuple of numbers): eigenvalues characterising diffusion tensor of single fibre population (the smaller two eigenvalues should be the same) (if None given then (0.003, 0.0002, 0.0002) is used)
    number_of_data_points (int): number of data points to be simulated
    b_value (int): non-zero b-value used in the simulation
    b_0_signal (int): non-diffusion weighted signal
    include_b_0 (bool): determines whether the data points should include b=0 measurements
    signal_to_noise_ratio (number): signal-to-noise ratio
    noise_type (string): type of noise to be added to the measurements. Supported noise types: rician, none
    noise_generator_seed (int): generator seed used to create noise (if None given then a new Generator is instantiated)
    gradient_generator_seed (int): generator seed used to create gradient directions (if None given then a new Generator with seed 1 is instantiated)
    fibre_orientation_generator_seed (int): generator seed used to generate fibre orientations

    Returns:
    (np.array(n x number_of_coefficients), list_of_diffusion_weighted_data, np.array(nx3)): first object is an array of SH coefficients for each simulated fODF, second object is a list of diffusion-weighted data as defined in load_dt_simulated_multiple_populations function, third object contains row vectors of fibre orientations
    """

    fODF_expansion_coefficients = []
    fibre_orientations = []
    diffusion_weighted_data = []
    eigenvectors = []
    volume_fractions = []

    fibre_orientation_generator = np.random.default_rng(fibre_orientation_generator_seed)

    # Generate random fibre orientations and volume fractions
    for i in range(dataset_size):
        fibre_orientations.append([])

        for j in range(number_of_fibre_populations):
            random_direction = None

            if(no_rotation):
                pass
            else:
                random_direction = generate_random_unit_vector(dimension=3, generator=fibre_orientation_generator)

            fibre_orientations[i].append(random_direction)

        fractions = fibre_orientation_generator.uniform(low=0.05, high=1, size=number_of_fibre_populations)
        fractions = fractions / np.sum(fractions)
        fractions = fractions.tolist()
        volume_fractions.append(fractions)

        fibre_orientations[i] = np.array(fibre_orientations[i])

    # Generate SH expansion coefficients of the fODFs from fibre orientations
    for i in range(dataset_size):
        fODF_expansion_coefficients_temp = load_fodf_simulated(max_degree=max_degree,
                                                               fibre_orientations=fibre_orientations[i],
                                                               fibre_fractions=volume_fractions[i])
        fODF_expansion_coefficients.append(fODF_expansion_coefficients_temp)

    fODF_expansion_coefficients = np.array(fODF_expansion_coefficients)

    eigenvalues = []

    # Set the diffusion tensor eigenvalues for each fibre population
    for i in range(number_of_fibre_populations):
        if (fibre_population_eigenvalues is None):
            eigenvalues.append((0.003, 0.0002, 0.0002))
        else:
            eigenvalues.append(fibre_population_eigenvalues)

    # Generate diffusion tensor eigenvectors from fibre orientations
    for i in range(dataset_size):
        eigenvectors.append([])
        for j in range(number_of_fibre_populations):
            eigenvectors[i].append(gram_schmidt_orthonormalization(fibre_orientations[i][j]))

    # Simulate diffusion-weighted signals using the diffusion tensor model
    for i in range(dataset_size):
        diffusion_weighted_data_temp = load_dt_simulated_multiple_populations(
            number_of_data_points=number_of_data_points, b_value=b_value,
            b_0_signal=b_0_signal, include_b_0=include_b_0,
            signal_to_noise_ratio=signal_to_noise_ratio,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors[i],
            fractions=volume_fractions[i],
            noise_type=noise_type,
            noise_generator_seed=noise_generator_seed + i,
            gradient_generator_seed=gradient_generator_seed)

        diffusion_weighted_data.append(diffusion_weighted_data_temp)

    fibre_orientations = np.array(fibre_orientations)

    return (fODF_expansion_coefficients, diffusion_weighted_data, fibre_orientations)


def save_dt_simulated_dataset(dataset, path):

    np.save(path + "/fODF_sh_coefficients", dataset[0])

    b_values = []
    gradient_orientations = []
    diffusion_weighted_signals = []

    for element in dataset[1]:
        b_values.append(element[0])
        gradient_orientations.append(element[1])
        diffusion_weighted_signals.append(element[2])

    b_values = np.array(b_values)
    gradient_orientations = np.array(gradient_orientations)
    diffusion_weighted_signals = np.array(diffusion_weighted_signals)

    np.save(path + "/b_values", b_values)
    np.save(path + "/gradient_orientations", gradient_orientations)
    np.save(path + "/diffusion_weighted_signals", diffusion_weighted_signals)

    np.save(path + "/fibre_orientations", dataset[2])
