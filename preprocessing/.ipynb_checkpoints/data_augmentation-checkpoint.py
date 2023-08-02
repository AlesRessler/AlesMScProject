import numpy as np

def extend_dataset_with_origin_reflections(bvals, qhat, dwis):
    """
    Extends a given dataset with reflections through origin i.e. for each sample [x,y,z] samples [-x,-y,-z] is added
    
    Parameters:
    bvals (np.array): 1D array of b-values
    qhat (np.array): 2D array containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component
    dwis (np.array): 4D array containing the measured signals where the first dimension corresponds to the test subject, second and third dimensions correspond to the x and y coordinates of a slice, and the fourth dimension determines the slice
    
    Returns:
    (np.array, np.array, np.array): outputs have the same format as the inputs except they are extended by the reflections through the origin
    """
    bvals_new = np.append(bvals, bvals, axis=0)
    
    qhat_new = np.append(qhat, -qhat, axis=1)
    
    dwis_new = np.append(dwis, dwis, axis=0)
    
    return (bvals_new, qhat_new, dwis_new)