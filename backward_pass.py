import numpy as np, quaternion
import math
import pyopencl as cl
from functions import *

def backward_pass(ctx, queue, program, camera, camera_r, camera_t, gaussians,
                  radii, covs_2d, covs_3d, ws, ts,
                  d_colors, d_2d_centers, d_conics, d_opacity):
    r = np.transpose(camera_r)
    t = camera_t

    fov_x = focal_to_fov(camera.params[0], camera.width)
    fov_y = focal_to_fov(camera.params[1], camera.height)

    limits_and_focals = np.asarray([1.3 * math.tan(fov_x/2), 1.3 * math.tan(fov_y/2), camera.params[0], camera.params[1]])

    d_covs, d_means = cov_2d_backpass(ctx, queue, program, radii, d_conics, covs_2d, covs_3d, ws, ts, r, limits_and_focals)

    print(d_covs.shape)
    print(d_means)

def cov_2d_backpass(ctx, queue, program, radii, d_conics, covs_2d, covs_3d, ws, ts, r, limits_and_focals):
    num_gaus = len(radii)
    d_covs = np.zeros([num_gaus, 6])
    d_means = np.zeros([num_gaus, 3])

    print(covs_2d)
    
    mf = cl.mem_flags
    radii_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=radii)
    d_conics_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d_conics.flatten())
    covs_2d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=covs_2d.flatten())
    covs_3d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=covs_3d.flatten())
    ws_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ws.flatten())
    ts_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts.flatten())
    view_mat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r.flatten())
    l_f_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=limits_and_focals)

    d_covs_g = cl.Buffer(ctx, mf.WRITE_ONLY, d_covs.nbytes)
    d_means_g = cl.Buffer(ctx, mf.WRITE_ONLY, d_means.nbytes)

    cov_2_kernel = program.cov_2d_grads
    
    cov_2_kernel(queue, radii.shape, None,
                 radii_g, d_conics_g, covs_2d_g, covs_3d_g,
                 ws_g, ts_g, view_mat_g, l_f_g,
                 d_covs_g, d_means_g)
    
    cl.enqueue_copy(queue, d_covs, d_covs_g)
    cl.enqueue_copy(queue, d_means, d_means_g)

    return d_covs, d_means
            