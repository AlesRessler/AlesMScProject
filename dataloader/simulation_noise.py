import numpy as np

def add_gaussian_noise(measurement, noise_standard_deviation, generator=None):
    """
    Adds Gaussian noise (with specified standard deviation) to the measurement.

    Parameters:
    measurement (number): measurement to which the noise should be added
    noise_standard_deviation (number): standard deviation of the noise
    generator (np.Generator): rng generator (if None given new generator is instantiated)

    Returns:
    (number): noisy measurement
    """

    if(generator is None):
        generator = np.random.default_rng()

    noisy_measurement = measurement + generator.normal(loc=0.0, scale=noise_standard_deviation)

    return noisy_measurement

def add_rician_noise(measurement, noise_standard_deviation, generator=None):
    """
    Adds Rician noise (with specified standard deviation) to the measurement.

    Parameters:
    measurement (number): measurement to which the noise should be added
    noise_standard_deviation (number): standard deviation of the noise
    generator (np.Generator): rng generator (if None given new generator is instantiated)

    Returns:
    (number): noisy measurement
    """

    if(generator is None):
        generator = np.random.default_rng()

    random_normal_number1 = generator.normal(loc=0.0, scale=noise_standard_deviation)
    random_normal_number2 = generator.normal(loc=0.0, scale=noise_standard_deviation)

    noisy_measurement = np.sqrt((measurement + random_normal_number1)**2 + random_normal_number2**2)
    return noisy_measurement