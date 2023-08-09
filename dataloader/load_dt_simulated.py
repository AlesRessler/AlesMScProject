import numpy as np
from numpy.linalg import pinv, norm

from dataloader import simulation_noise


def load_dt_simulated(number_of_data_points=90, b_value=1000, b_0_signal=3000, include_b_0=False,
                      noise_standard_deviation=100, eigenvalues=(1, 0, 0), eigenvectors=None, noise_type='rician',
                      noise_generator=None, gradient_generator=None):
    """
    Returns dataset simulated from the diffusion tensor model with specified number of data points and noise standard deviation.
    
    Parameters:
    number_of_data_points (int): number of data points to be simulated
    b_value (int): non-zero b-value used in the simulation
    b_0_signal (int): non-diffusion weighted signal
    include_b_0 (bool): determines whether the data points should include b=0 measurements
    noise_standard_deviation (int): standard deviation of the noise component
    eigenvalues (3-tuple): eigenvalues of the eigenvectors of the diffusion tensor
    eigenvectors (np.array(3x3)): eigenvectors (column vectors) of the diffusion tensor, if None then vectors (1,0,0),(0,1,0),(0,0,1) are used
    noise_type (string): type of noise to be added to the measurements. Supported noise types: gaussian, rician, none
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
        gradient_generator = np.random.default_rng(1)

    supported_noise_types = {'gaussian', 'rician', 'none'}

    if (not (noise_type in supported_noise_types)):
        raise Exception('Invalid noise type')

    bvals = []
    qhat = [[], [], []]
    measurements = []

    for i in range(number_of_data_points):

        # If include_b_0 is true then every 10th measurement is b=0 measurement
        if (include_b_0 and ((i + 1) % 10 == 0)):
            bvals.append(0)

            qhat[0].append(0.0)
            qhat[1].append(0.0)
            qhat[2].append(0.0)

            noisy_measurement = None

            if (noise_type == 'gaussian'):
                noisy_measurement = simulation_noise.add_gaussian_noise(measurement=b_0_signal,
                                                                        noise_standard_deviation=noise_standard_deviation,
                                                                        generator=noise_generator)
            elif (noise_type == 'rician'):
                noisy_measurement = simulation_noise.add_rician_noise(measurement=b_0_signal,
                                                                      noise_standard_deviation=noise_standard_deviation,
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

            if (noise_type == 'gaussian'):
                noisy_measurement = simulation_noise.add_gaussian_noise(measurement=measurement,
                                                                        noise_standard_deviation=noise_standard_deviation,
                                                                        generator=noise_generator)
            elif (noise_type == 'rician'):
                noisy_measurement = simulation_noise.add_rician_noise(measurement=measurement,
                                                                      noise_standard_deviation=noise_standard_deviation,
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
    Generates random unit vector of specified dimension.
    
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
                                           noise_standard_deviation=100, eigenvalues=[(1, 0, 0), (0, 1, 0)],
                                           eigenvectors=[None, None], fractions=[0.5, 0.5], noise_type='rician', noise_generator_seed=1,
                                           gradient_generator_seed=1):
    """
    Returns dataset simulated from the diffusion tensor model with specified number of data points, noise standard deviation and multiple fibre populations.
    
    Parameters:
    number_of_data_points (int): number of data points to be simulated
    b_value (int): non-zero b-value used in the simulation
    b_0_signal (int): non-diffusion weighted signal
    include_b_0 (bool): determines whether the data points should include b=0 measurements
    noise_standard_deviation (int): standard deviation of the noise component
    eigenvalues [(3-tuple)]: eigenvalues of the eigenvectors of the diffusion tensors (one 3-tuple for each fibre population)
    eigenvectors [(np.array(3x3))]: eigenvectors (column vectors) of the diffusion tensor, if None then vectors (1,0,0),(0,1,0),(0,0,1) are used (one 3x3 matrix for each fibre population)
    fractions [int]: volume fractions of each fibre population
    noise_type (string): type of noise to be added to the measurements. Supported noise types: gaussian, rician, none
    noise_generator (np.Generator): generator used to create noise (if None given then a new Generator is instantiated)
    gradient_generator (np.Generator): generator used to create gradient directions (if None given then a new Generator with seed 1 is instantiated)
    
    Returns:
    (np.array, np.array, np.array): The first array is 1D containg the b-values, 2D array containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component, third array is 1D containing DWI signals
    """

    noise_generator = np.random.default_rng(noise_generator_seed)
    gradient_generator = np.random.default_rng(gradient_generator_seed)

    supported_noise_types = {'gaussian', 'rician', 'none'}

    if (not (noise_type in supported_noise_types)):
        raise Exception('Invalid noise type')

    bvals = None
    qhat = None
    measurements = np.zeros(number_of_data_points)

    for i in range(len(fractions)):
        # Simulate measurements without noise
        bvals_temp, qhat_temp, measurements_temp = load_dt_simulated(number_of_data_points=number_of_data_points,
                                                                     b_value=b_value, b_0_signal=b_0_signal,
                                                                     include_b_0=include_b_0,
                                                                     noise_standard_deviation=noise_standard_deviation,
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
    if (noise_type == 'gaussian'):
        measurements = np.array([simulation_noise.add_gaussian_noise(measurement=measurement,
                                                                noise_standard_deviation=noise_standard_deviation,
                                                                generator=noise_generator) for measurement in measurements])
    elif (noise_type == 'rician'):
        measurements = np.array([simulation_noise.add_rician_noise(measurement=measurement,
                                                              noise_standard_deviation=noise_standard_deviation,
                                                              generator=noise_generator) for measurement in measurements])
    elif (noise_type == 'none'):
        measurements = measurements

    return (bvals, qhat, measurements)
