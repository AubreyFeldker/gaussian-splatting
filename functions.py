import math
import numpy as np

def get_view_matrix(r, t):
    view_mat = np.zeros([4,4])
    view_mat[:3,:3] = np.transpose(r)
    view_mat[:3,3] = t
    view_mat[3,3] = 1.0
    return view_mat

def get_projection_matrix(fov_x, fov_y, z_near = .01, z_far = 100):
    top = math.tan(fov_y/2) * z_near
    right = math.tan(fov_x/2) * z_near

    proj = np.zeros([4,4])
    proj[0,0] = z_near / right
    proj[1,1] = z_near / top
    proj[2,2] = z_far / (z_far - z_near)
    proj[2,3] = -(z_far * z_near) / (z_far - z_near)
    proj[3,2] = 1.0

    return proj

def get_jacobian_matrix(fov_x, fov_y, focal_x, focal_y, center):
    lim_x = 1.3 * math.tan(fov_x/2)
    lim_y = 1.3 * math.tan(fov_y/2)
    mathed_center = np.array([min(lim_x, max(-lim_x, center[0]/center[2])) * center[2],
                              min(lim_y, max(-lim_y, center[1]/center[2])) * center[2], center[2]])

    return np.matrix([
        [focal_x / mathed_center[2], 0.0, -(focal_x * mathed_center[0]) / (mathed_center[2] * mathed_center[2])],
        [0.0, focal_y / mathed_center[2], -(focal_y * mathed_center[1]) / (mathed_center[2] * mathed_center[2])],
        [0,0,0]
    ])

def focal_to_fov(focal_len, pixels):
    return 2 * math.atan(pixels / (2*focal_len))
def get_pixel(num, scale):
    return ((num+1.0) * scale - 1.0) * .5

# Activation functions & inverse activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def inv_sigmoid(x):
    return -1 * np.log((1/x) - 1) 