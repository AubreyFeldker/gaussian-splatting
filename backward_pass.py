import numpy as np, quaternion
import math
import pyopencl as cl
from functions import *

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

def backward_pass(ctx, queue, program, camera, camera_r, camera_t, gaussians,
                  radii, covs_2d, covs_3d, ws, ms, ts, dirs, clampeds,
                  d_colors, d_2d_centers, d_conics):
    r = np.transpose(camera_r)
    t = camera_t

    fov_x = focal_to_fov(camera.params[0], camera.width)
    fov_y = focal_to_fov(camera.params[1], camera.height)
    proj_mat = get_projection_matrix(fov_x, fov_y)

    limits_and_focals = np.asarray([1.3 * math.tan(fov_x/2), 1.3 * math.tan(fov_y/2), camera.params[0], camera.params[1]])

    d_covs, d_centers = cov_2d_backpass(ctx, queue, program, radii, d_conics, covs_2d, covs_3d,
                                       ws, ts, r, limits_and_focals,
                                       proj_mat, gaussians.center, d_2d_centers)
    
    d_colors, d_centers, d_shs = sh_backpass(ctx, queue, program, dirs, gaussians.spherical_harmonics, clampeds, gaussians.degrees,
                                             d_colors, d_centers)
    d_scales = np.empty([len(radii), 3])
    d_rots = np.empty([len(radii), 4])

    for i in range(len(radii)):
        rot = gaussians.rotation[i]
        r = rot[0]
        x = rot[1]
        y = rot[2]
        z = rot[3]
        d_cov = d_covs[i]

        d_sigma = np.asarray([[d_cov[0], d_cov[1] * .5, d_cov[2] * .5],
                              [d_cov[1] * .5, d_cov[3], d_cov[4] * .5],
                              [d_cov[2] * .5, d_cov[4] * .5, d_cov[5]]])

        Rt = np.transpose(quaternion.as_rotation_matrix(quaternion.as_quat_array(gaussians.rotation[i] / np.linalg.norm(gaussians.rotation[i], np.inf))))
        dMt = np.transpose(d_sigma @ ms[i] * 2.0)

        d_scales[i,0] = np.dot(Rt[0], dMt[0])
        d_scales[i,1] = np.dot(Rt[1], dMt[1])
        d_scales[i,2] = np.dot(Rt[2], dMt[2])

        dMt[0] *= gaussians.scaling[i,0]
        dMt[1] *= gaussians.scaling[i,1]
        dMt[2] *= gaussians.scaling[i,2]

        d_rots[i,0] = 2 * z * (dMt[0][1] - dMt[1][0]) + 2 * y * (dMt[2][0] - dMt[0][2]) + 2 * x * (dMt[1][2] - dMt[2][1])
        d_rots[i,1] = 2 * y * (dMt[1][0] + dMt[0][1]) + 2 * z * (dMt[2][0] + dMt[0][2]) + 2 * r * (dMt[1][2] - dMt[2][1]) - 4 * x * (dMt[2][2] + dMt[1][1])
        d_rots[i,2] = 2 * x * (dMt[1][0] + dMt[0][1]) + 2 * r * (dMt[2][0] - dMt[0][2]) + 2 * z * (dMt[1][2] + dMt[2][1]) - 4 * y * (dMt[2][2] + dMt[0][0])
        d_rots[i,3] = 2 * r * (dMt[0][1] - dMt[1][0]) + 2 * x * (dMt[2][0] + dMt[0][2]) + 2 * y * (dMt[1][2] + dMt[2][1]) - 4 * z * (dMt[1][1] + dMt[0][0])

    return d_colors, d_centers, d_shs, d_scales, d_rots

def sh_backpass(ctx, queue, program, dirs, shs, clampeds, degree,
                d_colors, d_centers):
    num_gaus = len(dirs)
    d_shs = np.zeros([num_gaus, 16, 3])
    dgr = np.asarray(degree)

    mf = cl.mem_flags
    dirs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dirs.flatten())
    shs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=shs.flatten())
    clampeds_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=clampeds.flatten())
    sh2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SH_C2)
    sh3_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SH_C3)
    dgr_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dgr)
    d_colors_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=d_colors.flatten())
    d_centers_g = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=d_centers.flatten())

    d_shs_g = cl.Buffer(ctx, mf.WRITE_ONLY, d_shs.nbytes)

    sh_kernel = program.sh_grads
    sh_kernel(queue, np.empty([len(dirs)]).shape, None,
              dirs_g, shs_g, clampeds_g, sh2_g, sh3_g, dgr_g,
              d_colors_g, d_centers_g, d_shs_g)
    
    cl.enqueue_copy(queue, d_colors, d_colors_g)
    cl.enqueue_copy(queue, d_centers, d_centers_g)
    cl.enqueue_copy(queue, d_shs, d_shs_g)

    return d_colors, d_centers, d_shs

def cov_2d_backpass(ctx, queue, program, radii, d_conics, covs_2d, covs_3d,
                     ws, ts, r, limits_and_focals,
                    proj_mat, centers, d_2d_centers):
    num_gaus = len(radii)
    d_covs = np.zeros([num_gaus, 6])
    d_centers = np.zeros([num_gaus, 3])
    
    mf = cl.mem_flags
    radii_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=radii)
    d_conics_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d_conics.flatten())
    covs_2d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=covs_2d.flatten())
    covs_3d_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=covs_3d.flatten())
    ws_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ws.flatten())
    ts_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ts.flatten())
    view_mat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r.flatten())
    l_f_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=limits_and_focals)
    proj_mat_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=proj_mat.flatten())
    centers_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=centers.flatten())
    d_2d_centers_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d_2d_centers.flatten())

    d_covs_g = cl.Buffer(ctx, mf.WRITE_ONLY, d_covs.nbytes)
    d_centers_g = cl.Buffer(ctx, mf.WRITE_ONLY, d_centers.nbytes)

    cov_2_kernel = program.cov_2d_grads
    
    cov_2_kernel(queue, radii.shape, None,
                 radii_g, d_conics_g, covs_2d_g, covs_3d_g,
                 ws_g, ts_g, view_mat_g, l_f_g,
                 proj_mat_g, centers_g, d_2d_centers_g,
                 d_covs_g, d_centers_g)
    
    cl.enqueue_copy(queue, d_covs, d_covs_g)
    cl.enqueue_copy(queue, d_centers, d_centers_g)

    return d_covs, d_centers
            