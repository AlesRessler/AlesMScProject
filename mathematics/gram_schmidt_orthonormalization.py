import numpy as np

def gram_schmidt_orthonormalization(vector):
    """
    Computes a 3D orthonormal basis given one 3D vector.

    Parameters:
    vector (np.array(1x3)): 3D vector

    Returns:
    (np.array(3x3)) 3D orthonormal basis with basis vectors as columns
    """
    vector1 = vector / np.linalg.norm(vector)

    vector2 = None

    while (True):
        vector2 = np.random.rand(3)
        vector2 /= np.linalg.norm(vector2)

        if (not np.array_equal(vector1, vector2)):
            break

    vector2 -= np.dot(vector2, vector1) * vector1
    vector2 /= np.linalg.norm(vector2)

    vector3 = np.cross(vector1, vector2)

    return np.array([vector1, vector2, vector3]).T