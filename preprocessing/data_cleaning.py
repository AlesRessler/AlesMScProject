import numpy as np

def remove_b_0_measurements(bvals, qhat, dwis):
    number_of_samples = len(bvals)
    
    bvals_new = np.delete(bvals, np.arange(0, number_of_samples, 6))
    
    qhat_new = np.delete(qhat, np.arange(0, number_of_samples, 6), axis=1)
    
    dwis_new = np.delete(dwis, np.arange(0, number_of_samples, 6), axis=0)
    
    return (bvals_new, qhat_new, dwis_new)