import numpy as np
from numpy import sin, cos


def yaw_pitch_roll_matrix(alpha, beta, gamma):
    yaw_pitch_roll_matrix = [[cos(alpha) * cos(beta), cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
                              cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma)],
                             [sin(alpha) * cos(beta), sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
                              sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma)],
                             [-sin(beta), cos(beta) * sin(gamma), cos(beta) * cos(gamma)]]

    yaw_pitch_roll_matrix = np.array(yaw_pitch_roll_matrix)

    return yaw_pitch_roll_matrix
