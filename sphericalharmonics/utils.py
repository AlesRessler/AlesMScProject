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

def get_number_of_coefficients(max_degree):
    """
    Computes the number of coefficients in an expansion from the maximum degree of spherical harmonics used. The spherical harmonics of odd orders are skipped
    
    Parameters
    max_degree (int): maximum spherical harmonics degree used in the expansion
    
    Returns:
    (int): the number of coefficients in an expansion with maximum degree max_degree
    """
    
    number_of_coefficients = 0
    
    if((max_degree % 2) == 0):
        number_of_coefficients = int(0.5 * (max_degree + 1) * (max_degree + 2))
    else:
        number_of_coefficients = int(0.5 * (max_degree) * (max_degree + 1))
    
    return number_of_coefficients