import pyopencl as cl

def setup_gpu():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    program = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
    void atomicAdd_g_f(volatile __global double *addr, double val)
    {
        union {
            unsigned int u32_a;
            unsigned int u32_b;
            double f64;
            } next, expected, current;
        current.f64 = *addr;
        do {
            expected.f64 = current.f64;
            next.f64 = expected.f64 + val;
            current.u32_a = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32_a, next.u32_a);
            current.u32_b = atomic_cmpxchg( (volatile __global unsigned int *)addr+32, expected.u32_b, next.u32_b);
        } while( current.u32_a != expected.u32_a || current.u32_b != expected.u32_b );
    }

    __kernel void rasterize(
        __global const double *centers, __global const double *colors, __global const double *opacities, __global const double *conics,
        __global const int *gaus_stack, __global const double *background_color, __global const int *other_data,
        __global double *image_chunk, __global double *b_colors, __global double *b_centers, __global double *b_conics, __global double *b_opacities)
    {
        int gid = get_global_id(0);
        const int loc_x = gid / 16;
        const int loc_y = gid % 16;
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
            gaus = gaus_stack[i];

            dist_x = centers[gaus*2] - x + .5;
            dist_y = centers[gaus*2+1] - y + .5;

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
    """).build()

    return ctx, queue, program