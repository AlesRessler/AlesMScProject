def get_storage_index(l,m):
    """
    Computes the index of the column of the design matrix that contains evaluations of the spherical harmonic of degree l and order m
    
    Parameters
    l (int): degree of the psherical harmonic, l>0
    m (int): order of the spherical harmonic, -l <= m <= l
    
    Returns:
    (int): the index of the column of the design matrix that contains evaluations of the spherical harmonic of degree l and order m
    """
    return int(0.5 * l * (l + 1) + m)