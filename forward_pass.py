import numpy as np, quaternion
import math
from functions import *
from ctypes import c_int32
import pyopencl as cl

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
def forward_pass(ctx, queue, program, camera, camera_r, camera_t, gaussians, result_size=[979,546], tile_size=16, training=True):
    gaus_num = len(gaussians.center)
    # saving these values prevents having to recompute them during the backwards pass
    if(training):
        mod_Ms = np.zeros([gaus_num, 3, 3])
        mod_Ts = np.zeros([gaus_num, 3])
        mod_2d_covs = np.zeros([gaus_num, 3])
        mod_3d_covs = np.zeros([gaus_num, 6])
        mod_Ws = np.zeros_like(mod_3d_covs)
    mod_dirs = np.zeros([gaus_num, 3])
    mod_conics = np.zeros([gaus_num, 3])
    mod_centers = np.zeros([gaus_num, 2])
    mod_depths = np.zeros(gaus_num,dtype=np.int32)
    mod_colors = np.zeros([gaus_num, 3])
    clamped = np.zeros([gaus_num, 3])
    radii = np.zeros(gaus_num)
    tiles_touched = np.zeros([gaus_num, 2, 2], dtype=np.int32)

    r = np.transpose(camera_r)
    t = camera_t

    fov_x = focal_to_fov(camera.params[0], camera.width)
    fov_y = focal_to_fov(camera.params[1], camera.height)
    proj_mat = get_projection_matrix(fov_x, fov_y)
    view_mat = get_view_matrix(r, t)

    activated_scales = np.exp(gaussians.scaling)

    num_nan = 0

    # this is where we get the covariance & center matrices for each gaussian in coordinance with where the camera is pointing to
    # as well as perform the projection onto them
    for i in range(len(gaussians.center)):
        #view_changed_center = np.pad((r @ gaussians.center[i]) + t, (0,1), 'constant', constant_values=(1.0,))
        view_changed_center = np.pad(gaussians.center[i], (0,1), 'constant', constant_values=(1.0,))
        proj_center = proj_mat @ view_changed_center
        mod_w = 1.0 / (proj_center[3] + .00000001)
        center = proj_center / mod_w

        # snipe gaussians too close to the camera
        view_adj_center = view_mat @ center
        #if(view_adj_center[2] <= .2 or math.isnan(view_adj_center[2]) ):
        if(math.isnan(view_adj_center[2]) ):
            continue

        # maximize fidelity of the depth for key sorting
        d = int(view_adj_center[2]*100000)
        mod_depths[i] = (d if (-2 ** 31 <= d < 2 ** 31) else c_int32(d).value)

        #compute the 2D (splatted) covariance matrices from the rotation mat & scaling vector
        rot_matrix = quaternion.as_rotation_matrix(quaternion.as_quat_array(gaussians.rotation[i]))
        M = rot_matrix @ (np.asarray([[activated_scales[i,0],1,1],[1,activated_scales[i,1],1],[1,1,activated_scales[i,2]]]) * .0020833)
        cov_matrix =  M @ np.transpose(M)

        t = view_mat @ np.pad(gaussians.center[i], (0,1), 'constant', constant_values=(1.0,))
        j_matrix = get_jacobian_matrix(fov_x, fov_y, camera.params[0], camera.params[1], t)
        W = j_matrix @ r
        cov_2d = W @ cov_matrix @ np.transpose(W)
        cov_2d_vals = [cov_2d[0,0], cov_2d[0,1],cov_2d[1,1]]

        # GAUSSIAN DILATION filter
        cov_2d_vals[0] += .3
        cov_2d_vals[2] += .3
        # yay we got the covariances (just 3 values ultimately) awesome
        #print(cov_2d_vals)
        if(training):
            mod_Ms[i] = M
            mod_Ws[i] = [W[0,0], W[0,1], W[0,2], W[1,0], W[1,1], W[1,2]]
            mod_Ts[i] = t[0:3]
            mod_2d_covs[i] = cov_2d_vals
            mod_3d_covs[i] = [cov_matrix[0,0], cov_matrix[0,1], cov_matrix[0,2], cov_matrix[1,1], cov_matrix[1,2], cov_matrix[2,2]]

        #invert covariance
        det = (cov_2d_vals[0] * cov_2d_vals[2] - cov_2d_vals[1] * cov_2d_vals[1])
        if (det == 0):
            continue
        det_inv = 1.0 / det
        conic = [cov_2d_vals[2] * det_inv, -cov_2d_vals[1] * det_inv, cov_2d_vals[0] * det_inv]

        mod_conics[i] = conic

        # calculate the maximum radius of the gaussian based off its
        middle = .5 * (cov_2d_vals[0] + cov_2d_vals[2])
        lambda1 = middle + math.sqrt(max(.1, middle * middle - det))
        lambda2 = middle - math.sqrt(max(.1, middle * middle - det))
        val = np.sqrt(max(lambda1, lambda2, 0))
        if (np.isnan(val)):
            num_nan+=1
            continue
        radius = math.ceil(3 * val)

        center_on_screen = [get_pixel(center[0], result_size[0]), get_pixel(center[1], result_size[1])]
        tiles_touched[i] = np.array([
            [int(min(result_size[0], max(0, center_on_screen[0] - radius)) / tile_size),
             int(min(result_size[1], max(0, center_on_screen[1] - radius)) / tile_size)],
            [int(min(result_size[0], max(0, center_on_screen[0] + radius + tile_size - 1)) / tile_size),
             int(min(result_size[1], max(0, center_on_screen[1] + radius + tile_size - 1)) / tile_size)] 
        ])

        if (center_on_screen[0] + radius < 0 or center_on_screen[0] - radius > result_size[0] or center_on_screen[1] + radius < 0 or center_on_screen[1] - radius > result_size[1]):
            continue

        radii[i] = radius
        mod_centers[i] = center_on_screen # screen-based center of the gaussian
        mod_colors[i], clamped[i], mod_dirs[i] = py_compute_color(gaussians.degrees, gaussians.center[i], camera_t, gaussians.spherical_harmonics[i])

        print('%3.2f percent done with forward_pass' % ((100.0 * i) / len(gaussians.center)), end='\r')

    print()
    print(num_nan)
    #mod_colors, clamped, mod_dirs = compute_color(ctx, queue, program, gaussians.degrees, gaussians.center, camera_t, gaussians.spherical_harmonics)

    if(training):
        return mod_centers, mod_depths, mod_colors, mod_conics, clamped, tiles_touched, radii, mod_Ws, mod_Ms, mod_Ts, mod_2d_covs, mod_3d_covs, mod_dirs
    else:
        return mod_centers, mod_depths, mod_colors, mod_conics, tiles_touched, radii
    
def compute_color(ctx, queue, program, degrees, centers, camera_position, shs):
    #Normalized direction to gaus center
    dirs = (centers - camera_position).astype(np.float32)
    for i in range(len(centers)):
        dirs[i] /= np.linalg.norm(dirs[i]) 

    colors = np.empty_like(centers, dtype=np.float32)
    clampeds = np.zeros_like(centers, dtype='bool')
    dgr = np.asarray(degrees)

    mf = cl.mem_flags
    dirs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dirs.flatten())
    shs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shs.flatten())
    sh2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SH_C2)
    sh3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SH_C3)
    dgr_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dgr)

    colors_g = cl.Buffer(ctx, mf.WRITE_ONLY, colors.nbytes)
    clampeds_g = cl.Buffer(ctx, mf.WRITE_ONLY, clampeds.nbytes)

    color_kernel = program.compute_color
    color_kernel(queue, np.empty([len(centers)]).shape, None,
              dirs_g, shs_g, sh2_g, sh3_g, dgr_g,
              colors_g, clampeds_g)
    
    cl.enqueue_copy(queue, colors, colors_g)
    cl.enqueue_copy(queue, clampeds, clampeds_g)

    return colors.astype(np.float64), clampeds, dirs.astype(np.float64)

def py_compute_color(degrees, center, camera_position, sh):
    #Normalized direction to gaus center
    direction = center - camera_position
    direction = direction/np.linalg.norm(direction) 
    sh = np.transpose(sh)
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

    result += .5

    clamped = np.zeros(3, dtype=bool)

    for i in range(3):
        if (result[i] < 0):
            result[i] = 0
            clamped[i] = True

    return result, clamped, direction