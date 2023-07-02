import numpy as np

def remove_b_0_measurements(bvals, qhat, dwis):
    """
    Removes data points corresponding to the b=0 measurements from the dataset.
    
    Parameters:
    bvals (np.array): 1D array of b-values
    qhat (np.array): 2D array containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component
    dwis (np.array): 4D array containing the measured signals where the first dimension corresponds to the test subject, second and third dimensions correspond to the x and y coordinates of a slice, and the fourth dimension determines the slice
    
    Returns:
    (np.array, np.array, np.array): outputs have the same format as the inputs except the b=0 measurements are removed
    """
    number_of_samples = len(bvals)
    
    bvals_new = np.delete(bvals, np.arange(0, number_of_samples, 6))
    
    qhat_new = np.delete(qhat, np.arange(0, number_of_samples, 6), axis=1)
    
    dwis_new = np.delete(dwis, np.arange(0, number_of_samples, 6), axis=0)
    
    return (bvals_new, qhat_new, dwis_new)