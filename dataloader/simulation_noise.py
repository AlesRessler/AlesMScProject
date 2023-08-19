import numpy as np

def add_rician_noise(measurement, signal_to_noise_ratio=30, generator=None):
    """
    Adds Rician noise (with specified standard deviation) to the measurement.

    Parameters:
    measurement (number): measurement to which the noise should be added
    signal_to_noise_ratio (number): signal-to-noise ratio
    generator (np.Generator): rng generator (if None given new generator is instantiated)

    Returns:
    (number): noisy measurement
    """

    if(generator is None):
        generator = np.random.default_rng()

    noise_standard_deviation = 1/signal_to_noise_ratio

    random_normal_number1 = generator.normal(loc=0.0, scale=noise_standard_deviation)
    random_normal_number2 = generator.normal(loc=0.0, scale=noise_standard_deviation)

    noisy_measurement = np.sqrt((measurement + random_normal_number1)**2 + random_normal_number2**2)
    return noisy_measurement