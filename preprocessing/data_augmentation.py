import numpy as np

def extend_dataset_with_origin_reflections(bvals, qhat, dwis):
    bvals_new = np.append(bvals, bvals, axis=0)
    
    qhat_new = np.append(qhat, -qhat, axis=1)
    
    dwis_new = np.append(dwis, dwis, axis=0)
    
    return (bvals_new, qhat_new, dwis_new)