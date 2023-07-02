import scipy
import numpy as np

def load_hcp(data_path='./data/data.mat', bvecs_path='./data/bvecs'):
    """
    Loads Human Connectome Dataset

    Parameters:
    data_path (string): Path to the file containg the signal measurements
    bvecs_path (string): Path to the file containg the gradient orientations

    Returns:
    (np.array, np.array, np.array): The first array is 1D containg the b values, second array is 2D containing the gradient orientations such that the first dimension determines the gradient component i.e arr[0]=x_component, arr[1]=y_component and arr[2]=z_component, third array is 4D containing the measured signals where the first dimension corresponds to the test subject, second and third dimensions correspond to the x and y coordinates of a slice, and the fourth dimension determines the slice.
    """
    
    # Load data and permute axes
    matlab_contents = scipy.io.loadmat(data_path)
    dwis = matlab_contents['dwis']
    dwis = np.transpose(dwis, (3,0,1,2))
    [Dc, Dx, Dy, Dz] = dwis.shape

    # Load gradient directions
    bvecs = open(bvecs_path, 'r')
    qhat = bvecs.readlines()
    bvecs.close()

    # Remove newline characters
    qhat = [line.strip('\n') for line in qhat]

    #Split lines into components
    qhat = [line.split() for line in qhat]

    # Convert to double
    qhat = np.asarray(qhat, dtype=np.double)

    bvals = 1000*np.sum(np.multiply(qhat, qhat) , axis=0);
    
    return (bvals, qhat, dwis)