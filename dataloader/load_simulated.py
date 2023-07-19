def load_simulated(number_of_data_points=90, b-value=1000, include_b_0=False, noise_variance=10000, eigenvalues=(1,0,0)):
    """
    Returns dataset simulated from the diffusion tensor model with specified number of data points and noise variance.
    
    Parameters:
    number_of_data_points (int): number of data points to be simulated
    b-value (int): non-zero b-value used in the simulation
    include_b_0 (bool): determines whether the data points should include b=0 measurements
    noise_variance (int): variance of the noise component
    eigenvalues (3-tuple): eigenvalues of the eigenvectors of the diffusion tensor
    eigenvectors (np.array(3x3)): eigenvectors (row vectors) of the diffusion tensor
    
    Returns:
    (int, np.array, np.array): b-values, 2D array containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component, array of DWI signals
    """