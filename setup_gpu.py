import pyopencl as cl

def setup_gpu():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    program = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
    void atomicAdd_g_f(volatile __global double *addr, double val)
    {
        union {
            unsigned long u64;
            double f64;
            } next, expected, current;
        current.f64 = *addr;
        do {
            expected.f64 = current.f64;
            next.f64 = expected.f64 + val;
            current.u64 = atom_cmpxchg( (volatile __global unsigned long *)addr, expected.u64, next.u64);
        } while( current.u64 != expected.u64);
    }

    __kernel void tile_match(
        __global const int *tiles_touched, __global const int *depths, __global const int *tile_dims_ng, __global const int *radii,
        __global ulong *key_mapper)
    {
        int gid = get_global_id(0);
        int grid_x = gid / tile_dims_ng[1];
        int grid_y = gid % tile_dims_ng[1];
        int s = (grid_x*tile_dims_ng[1]+grid_y) * tile_dims_ng[2];
        ulong key;

        for(int gaus = 0; gaus < tile_dims_ng[2]; gaus++) {
            if(radii[gaus] == 0)
                key_mapper[s+gaus] = 0;
                
            key = (ulong)gaus << 32;
            if(grid_x >= tiles_touched[gaus*4+0] && grid_x < tiles_touched[gaus*4+2]
            && grid_y >= tiles_touched[gaus*4+1] && grid_y < tiles_touched[gaus*4+3])
                key_mapper[s+gaus] = key + depths[gaus];
            else
                key_mapper[s+gaus] = key;
        }
    }

    __kernel void rasterize(
        __global const double *centers, __global const double *colors, __global const double *opacities, __global const double *conics,
        __global const ulong *gaus_stack, __global const double *background_color, __global const int *other_data,
        __global double *image_chunk, __global double *b_colors, __global double *b_centers, __global double *b_conics, __global double *b_opacities)
    {
        int gid = get_global_id(0);
        const int loc_x = gid / 32;
        const int loc_y = gid % 32;
        const int x = other_data[2] + loc_x;
        const int y = other_data[3] + loc_y;

        if(x >= other_data[4] || y >= other_data[5])
            return;

        for(int ch = 0; ch < 3; ch++) {
            image_chunk[gid*3+ch] = 0.0;
        }
        
        int contributor = 0;
        int gaus;

        double T = 1.0;
        double dist_x, dist_y, power, alpha, test_T;

        for (int i = 0; i < other_data[0]; i++) {
            contributor += 1;

            gaus = gaus_stack[i] >> 32;

            dist_x = centers[gaus*2] - x;
            dist_y = centers[gaus*2+1] - y;

            power = -.5 * (conics[gaus*3] * dist_x * dist_x + conics[gaus*3+2] * dist_y * dist_y) - conics[gaus*3+1] * dist_x * dist_y;

            if (power > 0)
                continue;

            alpha = fmin(.99, opacities[gaus] * exp(power));
            if (alpha < (1.0/255))
                continue;

            test_T = T * (1.0 - alpha);
            if (test_T < .0001)
                break;

            for(int ch = 0; ch < 3; ch++) {
                image_chunk[gid*3+ch] += colors[gaus*3+ch] * alpha * T;
            }
            T = test_T;
        }

        //printf("%d %d %f",x, y, T);
        for(int ch = 0; ch < 3; ch++) {
            image_chunk[gid*3+ch] += background_color[ch] * T;
        }

        if(other_data[0] == 0) 
            return;

        //Backwards pass starts here

        double dchannel_dcolor, dL_dalpha, c, dL_dchannel, dL_dG, gdx, gdy, dG_ddelx, dG_ddely;
        double last_alpha = 0.0;
        double T_final = T;
        double3 accum_rec, dL_dpixel, last_color = (double3)(0.0);

        for(int ch = 0; ch < 3; ch++) {
            dL_dpixel = image_chunk[gid*3+ch];
        }

        for(int j = contributor; j >= 0; j--) {
            gaus = gaus_stack[j];

            dist_x = centers[gaus*2] - x + .5;
            dist_y = centers[gaus*2+1] - y + .5;

            power = -.5 * (conics[gaus*3] * dist_x * dist_x + conics[gaus*3+2] * dist_y * dist_y) - conics[gaus*3+1] * dist_x * dist_y;

            if (power > 0)
                continue;

            alpha = fmin(.99, opacities[gaus] * exp(power));
            if (alpha < (1.0/255))
                continue;

            test_T = T / (1.0 - alpha);
            dchannel_dcolor = alpha * T;

            dL_dalpha = 0.0;
            for(int ch = 0; ch < 3; ch++) {                
                accum_rec[ch] = last_alpha * last_color[ch] + (1.0 - last_alpha) * accum_rec[ch];
                last_color[ch] = colors[gaus*3+ch];

                dL_dalpha += (colors[gaus*3+ch] - accum_rec[ch]) * image_chunk[gid*3+ch];
                atomicAdd_g_f(&(b_colors[gaus*3+ch]), dchannel_dcolor * image_chunk[gid*3+ch]);
            }

            dL_dalpha *= T;
            last_alpha = alpha;

            double bg_inclusion = 0.0;
            for(int ch = 0; ch < 3; ch++)
                bg_inclusion += background_color[ch] * image_chunk[gid*3+ch];
            dL_dalpha += (-T_final / (1.0 - alpha)) * bg_inclusion;

            dL_dG = opacities[gaus] * dL_dalpha;
            gdx = exp(power) * dist_x;
            gdy = exp(power) * dist_y;
            dG_ddelx = -gdx * conics[gaus*3] - gdy * conics[gaus*3+1];
            dG_ddely = -gdy * conics[gaus*3] - gdy * conics[gaus*3+1];

            atomicAdd_g_f(&(b_centers[gaus*3]), dL_dG * dG_ddelx * (.5 * other_data[4]));
            atomicAdd_g_f(&(b_centers[gaus*3+1]), dL_dG * dG_ddely * (.5 * other_data[5]));

            atomicAdd_g_f(&(b_conics[gaus*3]),-.5 * gdx * dist_x * dL_dG);
            atomicAdd_g_f(&(b_conics[gaus*3+1]),-.5 * gdx * dist_y * dL_dG);
            atomicAdd_g_f(&(b_conics[gaus*3+2]),-.5 * gdy * dist_y * dL_dG);

            atomicAdd_g_f(&(b_opacities[gaus]), exp(power) * dL_dalpha);
        }
    }   

    __kernel void cov_2d_grads(
        __global const int *radii, __global const double *d_conics, __global const double *covs_2d, __global const double *covs_3d,
        __global const double *ws, __global const double *ts, __global const double *view_mat, __global const double *limits_and_focals,
        __global const double *proj_mat, __global const double *centers, __global const double *d_2d_centers,
        global double *d_covs, global double *d_centers)    
    {
        int gid = get_global_id(0);

        if (radii[gid] <= 0)
            return;
        
        double a = covs_2d[gid*3];
        double b = covs_2d[gid*3+1];
        double c = covs_2d[gid*3+2];

        double3 d_conic = (double3)(d_conics[gid*3], d_conics[gid*3+1], d_conics[gid*3+2]);

        double denom = a * c - b * b;
        double d_da, d_db, d_dc = 0.0;
        double inv_denom = 1.0 / ((denom * denom) + 0.0000001);

        int w = gid*6;

        if (inv_denom != 0) {
            d_da = inv_denom * (-c * c * d_conic[0] + 2 * b * c * d_conic[1] + (denom - a * c) * d_conic[2]);
            d_db = inv_denom * (-a * a * d_conic[2] + 2 * a * b * d_conic[1] + (denom - a * c) * d_conic[0]);
            d_dc = inv_denom * 2 * (b * c * d_conic[0] - (denom + 2 * b * b) * d_conic[1] + a * b * d_conic[2]);

            d_covs[w  ] = ws[w] * ws[w] * d_da + ws[w] * ws[w+3] * d_db + ws[w+3] * ws[w+3] * d_dc;
            d_covs[w+3] = ws[w+1] * ws[w+1] * d_da + ws[w+1] * ws[w+4] * d_db + ws[w+4] * ws[w+4] * d_dc;
            d_covs[w+5] = ws[w+2] * ws[w+2] * d_da + ws[w+2] * ws[w+5] * d_db + ws[w+5] * ws[w+5] * d_dc;

            d_covs[w+1] = 2 * ws[w] * ws[w+1] * d_da + (ws[w] * ws[w+4] + ws[w+1] * ws[w+3]) * d_db + 2 * ws[w+3] * ws[w+4] * d_dc;
            d_covs[w+2] = 2 * ws[w] * ws[w+2] * d_da + (ws[w] * ws[w+5] + ws[w+2] * ws[w+3]) * d_db + 2 * ws[w+3] * ws[w+5] * d_dc;
            d_covs[w+2] = 2 * ws[w+2] * ws[w+1] * d_da + (ws[w+1] * ws[w+5] + ws[w+2] * ws[w+4]) * d_db + 2 * ws[w+4] * ws[w+5] * d_dc;
        }

        double d_dW00 = 2 * (ws[w  ] * covs_3d[w  ] + ws[w+1] * covs_3d[w+1] + ws[w+2] * covs_3d[w+2]) * d_da +
                            (ws[w+3] * covs_3d[w  ] + ws[w+4] * covs_3d[w+1] + ws[w+5] * covs_3d[w+2]) * d_db;
        double d_dW01 = 2 * (ws[w  ] * covs_3d[w+1] + ws[w+1] * covs_3d[w+3] + ws[w+2] * covs_3d[w+4]) * d_da +
                            (ws[w+3] * covs_3d[w+1] + ws[w+4] * covs_3d[w+3] + ws[w+5] * covs_3d[w+4]) * d_db;
        double d_dW02 = 2 * (ws[w  ] * covs_3d[w+2] + ws[w+1] * covs_3d[w+4] + ws[w+2] * covs_3d[w+5]) * d_da +
                            (ws[w+3] * covs_3d[w+2] + ws[w+4] * covs_3d[w+4] + ws[w+5] * covs_3d[w+5]) * d_db;
        double d_dW10 = 2 * (ws[w+3] * covs_3d[w  ] + ws[w+4] * covs_3d[w+1] + ws[w+5] * covs_3d[w+2]) * d_da +
                            (ws[w  ] * covs_3d[w  ] + ws[w+1] * covs_3d[w+1] + ws[w+2] * covs_3d[w+2]) * d_db;
        double d_dW11 = 2 * (ws[w+3] * covs_3d[w+1] + ws[w+4] * covs_3d[w+3] + ws[w+5] * covs_3d[w+4]) * d_da +
                            (ws[w  ] * covs_3d[w+1] + ws[w+1] * covs_3d[w+3] + ws[w+2] * covs_3d[w+4]) * d_db;
        double d_dW12 = 2 * (ws[w+3] * covs_3d[w+2] + ws[w+4] * covs_3d[w+4] + ws[w+5] * covs_3d[w+5]) * d_da +
                            (ws[w  ] * covs_3d[w+2] + ws[w+1] * covs_3d[w+4] + ws[w+2] * covs_3d[w+5]) * d_db;

        double d_dJ00 = view_mat[0] * d_dW00 + view_mat[1] * d_dW01 + view_mat[2] * d_dW02;
        double d_dJ02 = view_mat[6] * d_dW00 + view_mat[7] * d_dW01 + view_mat[8] * d_dW02;
        double d_dJ11 = view_mat[3] * d_dW10 + view_mat[4] * d_dW11 + view_mat[5] * d_dW12;
        double d_dJ12 = view_mat[6] * d_dW10 + view_mat[7] * d_dW11 + view_mat[8] * d_dW12;
        
        int x_mul = ((ts[gid*3] / ts[gid*3+2]) < (limits_and_focals[0] * -1) || (ts[gid*3] / ts[gid*3+2]) > limits_and_focals[0] ) ? 0 : 1;
        int y_mul = ((ts[gid*3+1] / ts[gid*3+2]) < (limits_and_focals[1] * -1) || (ts[gid*3+1] / ts[gid*3+2]) > limits_and_focals[1] ) ? 0 : 1;

        double tz = 1.0 / ts[gid*3+2];
        double tz2 = tz * tz;
        double tz3 = tz2 * tz;

        double d_mx = x_mul * -limits_and_focals[2] * tz2 * d_dJ02;
        double d_my = y_mul * -limits_and_focals[3] * tz2 * d_dJ12;
        double d_mz = -limits_and_focals[2] * tz2 * d_dJ00
                            - limits_and_focals[3] * tz2 * d_dJ11
                            + (2 * limits_and_focals[2] * ts[gid*3]) * tz3 * d_dJ02
                            + (2 * limits_and_focals[3] * ts[gid*3+1]) * tz3 * d_dJ12;

        d_centers[gid*3  ] = view_mat[0] * d_mx + view_mat[3] * d_my + view_mat[6] * d_mz;
        d_centers[gid*3+1] = view_mat[1] * d_mx + view_mat[4] * d_my + view_mat[7] * d_mz;
        d_centers[gid*3+2] = view_mat[2] * d_mx + view_mat[5] * d_my + view_mat[8] * d_mz;

        // Now doing the means based transformation from screenspace points
        int s = gid*3;
        double m_w = 1.0 / (proj_mat[3] * centers[s] + proj_mat[7] * centers[s+1] + proj_mat[11] * centers[s+2] + proj_mat[15] + .0000001);
        double mult1 = (proj_mat[0] * centers[s] + proj_mat[4] * centers[s+1] + proj_mat[8] * centers[s+2] * proj_mat[12]) * m_w * m_w;
        double mult2 = (proj_mat[1] * centers[s] + proj_mat[5] * centers[s+1] + proj_mat[9] * centers[s+2] * proj_mat[13]) * m_w * m_w;

        d_centers[gid*3  ] += (proj_mat[0] * m_w - proj_mat[3] * mult1) * d_2d_centers[gid*2] +
                            (proj_mat[1] * m_w - proj_mat[3] * mult2) * d_2d_centers[gid*2+1];
        d_centers[gid*3+1] += (proj_mat[4] * m_w - proj_mat[7] * mult1) * d_2d_centers[gid*2] +
                            (proj_mat[5] * m_w - proj_mat[7] * mult2) * d_2d_centers[gid*2+1];
        d_centers[gid*3+2] += (proj_mat[8] * m_w - proj_mat[11] * mult1) * d_2d_centers[gid*2] +
                            (proj_mat[9] * m_w - proj_mat[11] * mult2) * d_2d_centers[gid*2+1];
    }

    // SH computations taken from gaussian splatting paper (obviously)
    double3 dnormvdv(double3 v, double3 dv)
    {
        double sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
        double invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

        double3 dnormvdv;
        dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
        dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
        dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
        return dnormvdv;
    }

    __kernel void sh_grads(
        __global const double3 *dirs, __global const double3 *shs, __global const bool *clampeds, __global const double *SH_C2, __global const double *SH_C3, __global const int *degree,
        __global double3 *d_colors, __global double3 *d_means, __global double3 *d_shs)
    {
        const double SH_C0 = 0.28209479177387814;
        const double SH_C1 = 0.4886025119029199;

        int gid = get_global_id(0);
        int g = gid*3;
        int s = gid*16;

        d_colors[g].x = clampeds[g  ] ? 0 : 1;
        d_colors[g].y = clampeds[g+1] ? 0 : 1;
        d_colors[g].z = clampeds[g+2] ? 0 : 1;

        double3 d_col_x, d_col_y, d_col_z = (double3)(0,0,0);
        double x = dirs[g].x;
        double y = dirs[g].y;
        double z = dirs[g].z;

        d_shs[s] = SH_C0 * d_colors[g];
        if(*degree > 0) {
            d_shs[s+1] = (-SH_C1 * y) * d_colors[g];
            d_shs[s+2] = ( SH_C1 * z) * d_colors[g];
            d_shs[s+3] = (-SH_C1 * x) * d_colors[g];

            d_col_x = -SH_C1 * shs[s+3];
            d_col_y = -SH_C1 * shs[s+1];
            d_col_z = -SH_C1 * shs[s+2];
        }

        double3 d_dir = (double3)(d_col_x.x * d_colors[g].x + d_col_x.y * d_colors[g].y + d_col_x.z * d_colors[g].z,
                                    d_col_y.x * d_colors[g].x + d_col_y.y * d_colors[g].y + d_col_y.z * d_colors[g].z,
                                    d_col_z.x * d_colors[g].x + d_col_z.y * d_colors[g].y + d_col_z.z * d_colors[g].z);
        d_means[g] += dnormvdv(dirs[g], d_dir);
    }

    __kernel void compute_color(
        __global const float3 *dirs, __global const float3 *shs, __global const float *SH_C2, __global const float *SH_C3, __global const int *degree,
        __global float3 *colors, __global bool *clampeds)
    {
        const float SH_C0 = 0.28209479177387814;
        const float SH_C1 = 0.4886025119029199;

        int gid = get_global_id(0);
        int s = gid*16;
        int degrees = degree[0];

        float3 result = SH_C0 * shs[s];

        if (degrees > 0) {
            float x = dirs[gid].x;
            float y = dirs[gid].y;
            float z = dirs[gid].z;
            result += (-y * shs[s+1] + z * shs[s+2] - x * shs[s+3]) * SH_C1;

            if (degrees > 1) {
                float xx = x * x;
                float yy = y * y;
                float zz = z * z;
                float xy = x * y;
                float yz = y * z;
                float xz = x * z;

                result += (SH_C2[0] * xy * shs[s+4] +
                        SH_C2[1] * yz * shs[s+5] +
                        SH_C2[2] * (2.0f * zz - xx - yy) * shs[s+6] +
                        SH_C2[3] * xz * shs[s+7] +
                        SH_C2[4] * (xx - yy) * shs[s+8]);

                if (degrees > 2) {
                    result += (SH_C3[0] * y * (3.0f * xx - yy) * shs[s+9] +
                    SH_C3[1] * xy * z * shs[s+10] +
                    SH_C3[2] * y * (4.0f * zz - xx - yy) * shs[s+11] +
                    SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * shs[s+12]+
                    SH_C3[4] * x * (4.0f * zz - xx - yy) * shs[s+13] +
                    SH_C3[5] * z * (xx - yy) * shs[s+14] +
                    SH_C3[6] * x * (xx - 3.0f * yy) * shs[s+15]);
                }
            }
        }

        result += .5f;

        if (result.x < 0) {
            result.x = 0;
            clampeds[gid*3  ] = true;
        }
        if (result.y < 0) {
            result.y = 0;
            clampeds[gid*3+1] = true;
        }
        if (result.z < 0) {
            result.z = 0;
            clampeds[gid*3+2] = true;
        }

        colors[gid] = result;

    }
    """).build()

    return ctx, queue, program