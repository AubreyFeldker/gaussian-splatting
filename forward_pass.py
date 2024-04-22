import numpy as np, quaternion
import math, multiprocessing

#SH coefficients
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = np.array([1.0925484305920792,
	-1.0925484305920792,
	0.31539156525252005,
	-1.0925484305920792,
	0.5462742152960396])
SH_C3 = np.array([-0.5900435899266435,
	2.890611442640554,
	-0.4570457994644658,
	0.3731763325901154,
	-0.4570457994644658,
	1.445305721320277,
	-0.5900435899266435])

# can definitely multiproc this later
# update: lol. lmao
def forward_pass(camera, camera_r, camera_t, gaussians, result_size=[979,546], tile_size=16):
    gaus_num = len(gaussians.center)
    mod_covariances = np.zeros([gaus_num, 3])
    mod_centers = np.zeros([gaus_num, 2])
    mod_depths = np.zeros(gaus_num)
    mod_colors = np.zeros([gaus_num, 3])
    clamped = np.zeros([gaus_num, 3])
    radii = np.zeros(gaus_num)
    tiles_touched = np.zeros([gaus_num, 2, 2], dtype=np.int32)

    r = np.transpose(camera_r)
    t = camera_t

    fov_x = focal_to_fov(camera.params[0], camera.width)
    fov_y = focal_to_fov(camera.params[1], camera.height)
    proj_mat = get_projection_matrix(fov_x, fov_y)

    zero_dets = 0
    take_1 = 0
    take_2 = 0

    # this is where we get the covariance & center matrices for each gaussian in coordinance with where the camera is pointing to
    # as well as perform the projection onto them
    for i in range(len(gaussians.center)):
        #view_changed_center = np.pad((r @ gaussians.center[i]) + t, (0,1), 'constant', constant_values=(1.0,))
        view_changed_center = np.pad(gaussians.center[i], (0,1), 'constant', constant_values=(1.0,))
        proj_center = proj_mat @ view_changed_center
        mod_w = 1.0 / (proj_center[3] + .00000001)
        center = proj_center[:3] / mod_w

        # snipe gaussians too close to the camera
        depth = (r @ center + t)[2]
        if(depth <= .02):
            take_1+=1
            continue

        mod_depths[i] = depth

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

        center_on_screen = [get_pixel(center[0], result_size[0]), get_pixel(center[1], result_size[1])]
        tiles_touched[i] = np.array([
            [int(min(result_size[0], max(0, center_on_screen[0] - radius)) / tile_size),
             int(min(result_size[1], max(0, center_on_screen[1] - radius)) / tile_size)],
            [int(min(result_size[0], max(0, center_on_screen[0] + radius + tile_size - 1)) / tile_size),
             int(min(result_size[1], max(0, center_on_screen[1] + radius + tile_size - 1)) / tile_size)] 
        ])

        if (center_on_screen[0] + radius < 0 or center_on_screen[0] - radius > result_size[0] or center_on_screen[1] + radius < 0 or center_on_screen[1] - radius > result_size[1]):
            take_2+=1
            continue

        zero_dets+=1
        mod_colors[i], clamped[i] = compute_color(gaussians.degrees, gaussians.center[i], camera_t, gaussians.spherical_harmonics[i])

        radii[i] = radius
        mod_centers[i] = center_on_screen # screen-based center of the gaussian

    print(zero_dets)
    print(take_1)
    print(take_2)
    return mod_centers, mod_depths, mod_colors, mod_covariances, clamped, tiles_touched, radii

def compute_color(degrees, center, camera_position, sh):
    #Normalized direction to gaus center
    direction = center - camera_position
    direction = direction/np.linalg.norm(direction) 
    result = SH_C0 * sh[0]

    if (degrees > 0):
        x = direction[0]
        y = direction[1]
        z = direction[2]
        result += (-y * sh[1] + z * sh[2] - x * sh[3]) * SH_C1

        if (degrees > 1):
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z

            result += (SH_C2[0] * xy * sh[4] +
                       SH_C2[1] * yz * sh[5] +
                       SH_C2[2] * (2.0 * zz - xx - yy) * sh[6] +
                       SH_C2[3] * xz * sh[7] +
                       SH_C2[4] * (xx - yy) * sh[8])

            if (degrees > 2):
                result += (SH_C3[0] * y * (3.0 * xx - yy) * sh[9] +
                SH_C3[1] * xy * z * sh[10] +
                SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11] +
                SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12]+
                SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13] +
                SH_C3[5] * z * (xx - yy) * sh[14] +
                SH_C3[6] * x * (xx - 3.0 * yy) * sh[15])

    result += np.array([.5,.5,.5])

    clamped = np.zeros(3, dtype=bool)

    for i in range(3):
        if (result[i] < 0):
            result[i] = 0
            clamped[i] = True

    return result, clamped


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