import numpy as np
from scipy.ndimage.interpolation import rotate


def get_angles(n_angles):
    return np.linspace(0, 360, n_angles + 1)[:-1]


def apply_angle(mask, angle):
    return rotate(mask, angle=angle, reshape=False, mode='constant', order=5)
