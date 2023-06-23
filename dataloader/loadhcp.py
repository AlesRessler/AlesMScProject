import scipy
import numpy as np

def load_hcp(data_path='./data/data.mat', bvecs_path='./data/bvecs'):
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