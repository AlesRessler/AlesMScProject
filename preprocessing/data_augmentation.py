import numpy as np

def extend_dataset_with_origin_reflections(bvals, qhat, dwis):
    """
    Extends a given dataset with reflections through origin i.e. for each sample [x,y,z] samples [-x,-y,-z] is added
    
    Parameters:
    
    """
    bvals_new = np.append(bvals, bvals, axis=0)
    
    qhat_new = np.append(qhat, -qhat, axis=1)
    
    dwis_new = np.append(dwis, dwis, axis=0)
    
    return (bvals_new, qhat_new, dwis_new)