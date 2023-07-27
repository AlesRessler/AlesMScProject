import numpy as np
from numpy.linalg import pinv, norm

def load_dt_simulated(number_of_data_points=90, b_value=1000, b_0_signal=3000, include_b_0=False, noise_standard_deviation=100, eigenvalues=(1,0,0), eigenvectors=None, seed=1):
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
    seed (int): random generator seed
    
    Returns:
    (np.array, np.array, np.array): The first array is 1D containg the b-values, 2D array containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component, third array is 1D containing DWI signals
    """
    
    if eigenvectors is None:
        eigenvectors = np.zeros((3,3))
        
        eigenvectors[0,0] = 1.0
        eigenvectors[1,1] = 1.0
        eigenvectors[2,2] = 1.0
    
    diffusion_tensor = compute_diffusion_tensor(eigenvalues, eigenvectors)
    generator = np.random.default_rng(seed)
    
    bvals = []
    qhat = [[],[],[]]
    measurements = []
    
    for i in range(number_of_data_points):
        
        # If include_b_0 is true then every 10th measurement is b=0 measurement
        if(include_b_0 and ((i+1) % 10 == 0)):
            bvals.append(0)
            
            qhat[0].append(0.0)
            qhat[1].append(0.0)
            qhat[2].append(0.0)
            
            noisy_measurement = b_value + generator.normal(loc=0.0, scale=noise_standard_deviation)
            
            measurements.append(noisy_measurement)
        else:
            bvals.append(0)
            
            random_unit_vector = generate_random_unit_vector(3, generator)
            
            qhat[0].append(random_unit_vector[0])
            qhat[1].append(random_unit_vector[1])
            qhat[2].append(random_unit_vector[2])
            
            measurement = simulate_signal(b_value, random_unit_vector, b_0_signal, diffusion_tensor)
            
            noisy_measurement = measurement + generator.normal(loc=0.0, scale=noise_standard_deviation)
            
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
    
    diagonal_eigenvalue_matrix = np.zeros((3,3))
    
    # Insert eigenvalues to the diagonal
    for i in range(len(eigenvalues)):
        diagonal_eigenvalue_matrix[i,i] = eigenvalues[i]
    
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
    
    random_vector = generator.uniform(low=-1.0, high=1.0, size=dimension)
    random_unit_vector = random_vector / norm(random_vector)
    
    return random_unit_vector
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    