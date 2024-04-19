import numpy as np, quaternion
import math, multiprocessing

# can definitely multiproc this later
def forward_pass(camera, camera_r, camera_t, gaussians, result_size=[1280,720]):
    gaus_num = len(gaussians.center)
    mod_covariances = np.zeros([gaus_num, 3])
    mod_centers = np.zeros_like(gaussians.center)
    radii = np.zeros(gaus_num)
    tiles_touched = np.zeros(gaus_num)

    r = np.transpose(camera_r)
    t = -1 * camera_t

    fov_x = focal_to_fov(camera.params[0], camera.width)
    fov_y = focal_to_fov(camera.params[1], camera.height)
    proj_mat = get_projection_matrix(fov_x, fov_y)

    zero_dets = 0

    # this is where we get the covariance & center matrices for each gaussian in coordinance with where the camera is pointing to
    # as well as perform the projection onto them
    for i in range(len(gaussians.center)):
        view_changed_center = np.pad((r @ gaussians.center[i]) + t, (0,1), 'constant', constant_values=(1.0,))
        proj_center = proj_mat @ view_changed_center
        mod_w = 1.0 / (proj_center[3] + .00000001)
        mod_centers[i] = np.array(proj_center[:3] / mod_w)

        # snipe gaussians too close to the camera
        if((mod_centers[i][2]) < .2):
            continue

        #compute the 2D (splatted) covariance matrices from the rotation mat & scaling vector
        rot_matrix = quaternion.as_rotation_matrix(quaternion.as_quat_array(gaussians.rotation[i]))
        M = rot_matrix @ gaussians.scaling[i][:, np.newaxis]
        cov_matrix =  M @ np.transpose(M)

        j_matrix = get_jacobian_matrix(fov_x, fov_y, camera.params[0], camera.params[1], proj_center)
        W = j_matrix @ r
        cov_2d = W @ cov_matrix @ np.transpose(W)
        cov_2d_vals = [cov_2d[0,0], cov_2d[0,1],cov_2d[1,1]]

        # GAUSSIAN DILATION filter
        cov_2d_vals[0] += .3
        cov_2d_vals[2] += .3
        # yay we got the covariances (just 3 values ultimately) awesome
        #print(cov_2d_vals)

        #invert covariance
        det = (cov_2d_vals[0] * cov_2d_vals[2] - cov_2d_vals[1] * cov_2d_vals[1])
        det_inv = 1.0 / det
        conic = [cov_2d_vals[2] * det_inv, -cov_2d_vals[1] * det_inv, cov_2d_vals[0] * det_inv]

        mod_covariances[i] = conic

        # calculate the maximum radius of the gaussian based off its
        middle = .5 * (cov_2d_vals[0] + cov_2d_vals[2])
        lambda1 = middle + math.sqrt(max(.1, middle * middle - det))
        lambda2 = middle - math.sqrt(max(.1, middle * middle - det))
        radius = math.ceil(3 * math.sqrt(max(lambda1, lambda2)))

        center_on_screen = [get_pixel(mod_centers[i,0], result_size[0]), get_pixel(mod_centers[i,1], result_size[1])]
        if(center_on_screen[0] >= 0 and center_on_screen[0] < result_size[0] and center_on_screen[1] >= 0 and center_on_screen[0] < result_size[1]):
            zero_dets+=1

        
        #this is where i add in getting colors from the SHs later but i dont wanna do that rn

    print(zero_dets)

def get_view_matrix(r, t):
    view_mat = np.zeros((4,4))
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
    mathed_center = np.array([min(lim_x, max(-lim_x, center[0]/center[2])) * center[2], min(lim_y, max(-lim_y, center[1]/center[2])) * center[2], center[2]])

    return np.matrix([
        [focal_x / mathed_center[2], 0.0, -(focal_x * mathed_center[0]) / (mathed_center[2] * mathed_center[2])],
        [0.0, focal_y / mathed_center[2], -(focal_y * mathed_center[1]) / (mathed_center[2] * mathed_center[2])],
        [0,0,0]
    ])

def focal_to_fov(focal_len, pixels):
    return 2 * math.atan(pixels / (2*focal_len))
def get_pixel(num, scale):
    return ((num+1.0) * scale - 1.0) * .5